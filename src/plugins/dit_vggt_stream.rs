// WIP 
// DiT-StreamVGGT Plugin - Real-time gaussian generation from image / DiT streams
// Uses shared memory (mmap-sync) for zero-copy communication with Python StreamVGGT

use bevy::{
    app::{App, Plugin, Startup, Update},
    asset::Assets,
    ecs::{
        component::Component,
        resource::Resource,
        system::{Commands, Local, Query, Res, ResMut},
    },
    log::{debug, error, info, warn},
    render::camera::Camera,
    time::Time,
    transform::components::GlobalTransform,
};
use bevy_gaussian_splatting::{
    Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle, Planar,
    gaussian::f32::{PositionVisibility, Rotation, ScaleOpacity},
    material::spherical_harmonics::SphericalHarmonicCoefficients,
};
use mmap_sync::synchronizer::Synchronizer;
use rkyv::{Archive, Deserialize, Serialize, Infallible};
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    process::{Child, Command},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::plugins::image_copy::FrameData;

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct SharedFrame {
    pub width: u32,
    pub height: u32,
    pub timestamp_ms: u64,
    pub camera_position: [f32; 3],
    pub camera_rotation: [f32; 4], // quaternion
    pub camera_forward: [f32; 3],  // Camera forward direction
    pub frame_data: Vec<u8>,       // RGBA data
    pub is_outpaint_request: bool, // Request outpainting for sparse area
    pub outpaint_seed: i32,        // Fixed seed for consistent outpainting
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct SharedGaussian {
    pub position: [f32; 3],
    pub quaternion: [f32; 4],
    pub scale: [f32; 3],
    pub opacity: f32,
    pub color: [f32; 3],
    pub confidence: f32,
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct SharedGaussianBatch {
    pub count: u32,
    pub timestamp_ms: u64,
    pub camera_position: [f32; 3], // Position where these gaussians were generated
    pub gaussians: Vec<SharedGaussian>,
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct ControlSignal {
    pub command: String,
    pub timestamp_ms: u64,
}

// ============================================================================
// Resources
// ============================================================================

#[derive(Resource, Clone)]
pub struct VGGTConfig {
    pub enabled: bool,
    // Enable scene extension via outpainting
    pub extend_scene: bool, 
    pub merge_distance_threshold: f32,
    pub confidence_threshold: f32,
    pub max_gaussians: usize,
    // Process every N frames
    pub frame_interval: u32, 
    // Minimum gaussian density to trigger outpainting
    pub sparse_threshold: f32, 
    pub python_script_path: PathBuf,
    pub mmap_base_path: PathBuf,
    pub model_checkpoint: Option<PathBuf>,
    pub replicate_api_token: Option<String>,
}

impl Default for VGGTConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            extend_scene: false,
            merge_distance_threshold: 0.01,
            confidence_threshold: 0.3,
            max_gaussians: 500_000,
            frame_interval: 30, // Process every 30 frames (2 FPS at 60 FPS)
            sparse_threshold: 10.0, // Minimum gaussians per cubic unit
            python_script_path: PathBuf::from("src/plugins/vggt_worker.py"),
            mmap_base_path: PathBuf::from("/tmp/vggt"),
            model_checkpoint: None,
            replicate_api_token: std::env::var("REPLICATE_API_TOKEN").ok(),
        }
    }
}

#[derive(Resource)]
struct VGGTMemorySync {
    frame_writer: Arc<Mutex<Synchronizer>>,
    gaussian_reader: Arc<Mutex<Synchronizer>>,
    control_writer: Arc<Mutex<Synchronizer>>,
    last_write_time: Arc<AtomicU64>,
    last_read_time: Arc<AtomicU64>,
}

#[derive(Resource)]
struct PythonWorker {
    process: Child,
    is_running: Arc<AtomicBool>,
}

#[derive(Resource)]
struct GaussianMerger {
    spatial_index: HashMap<(i32, i32, i32), Vec<usize>>, 
    grid_resolution: f32,
}

impl GaussianMerger {
    fn new(grid_resolution: f32) -> Self {
        Self {
            spatial_index: HashMap::new(),
            grid_resolution,
        }
    }
    
    fn count_gaussians_in_frustum(&self, camera_pos: &[f32; 3], camera_forward: &[f32; 3], gaussians: &[Gaussian3d], range: f32) -> usize {
        // Simple frustum check - count gaussians in front of camera within range
        let mut count = 0;
        for gaussian in gaussians {
            let pos = &gaussian.position_visibility.position;
            let to_gaussian = [
                pos[0] - camera_pos[0],
                pos[1] - camera_pos[1],
                pos[2] - camera_pos[2],
            ];
            
            // Check if in front of camera (dot product > 0)
            let dot = to_gaussian[0] * camera_forward[0] + 
                      to_gaussian[1] * camera_forward[1] + 
                      to_gaussian[2] * camera_forward[2];
            
            if dot > 0.0 {
                let dist_sq = to_gaussian[0].powi(2) + to_gaussian[1].powi(2) + to_gaussian[2].powi(2);
                if dist_sq < range * range {
                    count += 1;
                }
            }
        }
        count
    }

    fn position_to_grid(&self, pos: &[f32; 3]) -> (i32, i32, i32) {
        (
            (pos[0] / self.grid_resolution).floor() as i32,
            (pos[1] / self.grid_resolution).floor() as i32,
            (pos[2] / self.grid_resolution).floor() as i32,
        )
    }

    fn build_index(&mut self, gaussians: &[Gaussian3d]) {
        self.spatial_index.clear();
        for (idx, gaussian) in gaussians.iter().enumerate() {
            let pos = [
                gaussian.position_visibility.position[0],
                gaussian.position_visibility.position[1],
                gaussian.position_visibility.position[2],
            ];
            let grid = self.position_to_grid(&pos);
            self.spatial_index.entry(grid).or_default().push(idx);
        }
    }

    fn find_nearby(&self, position: &[f32; 3], gaussians: &[Gaussian3d], threshold: f32) -> Vec<usize> {
        let mut nearby = Vec::new();
        let center_grid = self.position_to_grid(position);
        
        // Check 3x3x3 grid around position
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let grid = (
                        center_grid.0 + dx,
                        center_grid.1 + dy,
                        center_grid.2 + dz,
                    );
                    
                    if let Some(indices) = self.spatial_index.get(&grid) {
                        for &idx in indices {
                            let g_pos = &gaussians[idx].position_visibility.position;
                            let dist_sq = 
                                (g_pos[0] - position[0]).powi(2) +
                                (g_pos[1] - position[1]).powi(2) +
                                (g_pos[2] - position[2]).powi(2);
                            
                            if dist_sq < threshold * threshold {
                                nearby.push(idx);
                            }
                        }
                    }
                }
            }
        }
        
        nearby
    }

    fn merge_gaussians(
        &mut self,
        existing: Vec<Gaussian3d>,
        new_batch: SharedGaussianBatch,
        config: &VGGTConfig,
    ) -> Vec<Gaussian3d> {
        let mut result = existing.clone();
        self.build_index(&result);
        
        let mut replacements = Vec::new();
        let mut additions = Vec::new();
        
        for shared_gaussian in new_batch.gaussians.iter() {
            if shared_gaussian.confidence < config.confidence_threshold {
                continue;
            }
            
            let nearby = self.find_nearby(
                &shared_gaussian.position,
                &result,
                config.merge_distance_threshold,
            );
            
            if nearby.is_empty() {
                // No nearby gaussians, add as new
                additions.push(shared_gaussian.clone());
            } else {
                // Find lowest confidence nearby gaussian
                let mut min_conf_idx = None;
                let mut min_conf = f32::MAX;
                
                for &idx in &nearby {
                    let opacity = result[idx].scale_opacity.opacity;
                    if opacity < min_conf {
                        min_conf = opacity;
                        min_conf_idx = Some(idx);
                    }
                }
                
                // Replace if new gaussian has higher confidence
                if let Some(idx) = min_conf_idx {
                    if shared_gaussian.confidence > min_conf {
                        replacements.push((idx, shared_gaussian.clone()));
                    }
                }
            }
        }
        
        // Apply replacements
        for (idx, shared_gaussian) in replacements {
            result[idx] = self.shared_to_gaussian3d(&shared_gaussian);
        }
        
        // Add new gaussians
        for shared_gaussian in additions {
            result.push(self.shared_to_gaussian3d(&shared_gaussian));
        }
        
        // Limit total count by removing lowest confidence gaussians
        if result.len() > config.max_gaussians {
            result.sort_by(|a, b| {
                b.scale_opacity.opacity
                    .partial_cmp(&a.scale_opacity.opacity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            result.truncate(config.max_gaussians);
        }
        
        info!(
            "Merged gaussians: {} existing + {} new = {} total",
            existing.len(),
            new_batch.gaussians.len(),
            result.len()
        );
        
        result
    }
    
    fn shared_to_gaussian3d(&self, shared: &SharedGaussian) -> Gaussian3d {
        Gaussian3d {
            position_visibility: PositionVisibility {
                position: shared.position,
                visibility: 1.0,
            },
            rotation: Rotation {
                rotation: shared.quaternion,
            },
            scale_opacity: ScaleOpacity {
                scale: shared.scale,
                opacity: shared.opacity,
            },
            spherical_harmonic: SphericalHarmonicCoefficients {
                coefficients: {
                    let mut coeffs = [0.0; 48];
                    // Set DC component to color
                    coeffs[0] = shared.color[0];
                    coeffs[1] = shared.color[1];
                    coeffs[2] = shared.color[2];
                    coeffs
                },
            },
        }
    }
}

#[derive(Component)]
pub struct VGGTCamera;

pub struct VGGTStreamPlugin {
    config: VGGTConfig,
}

impl VGGTStreamPlugin {
    pub fn new() -> Self {
        Self {
            config: VGGTConfig::default(),
        }
    }
    
    pub fn with_config(config: VGGTConfig) -> Self {
        Self { config }
    }
    
    pub fn with_scene_extension(mut self, enabled: bool) -> Self {
        self.config.extend_scene = enabled;
        self
    }
}

impl Plugin for VGGTStreamPlugin {
    fn build(&self, app: &mut App) {
        info!("Initializing VGGT Stream Plugin");
        
        // Ensure mmap directory exists
        fs::create_dir_all(&self.config.mmap_base_path)
            .expect("Failed to create mmap directory");
        
        app.insert_resource(self.config.clone());
        app.insert_resource(GaussianMerger::new(0.05)); // 5cm grid resolution
        
        app.add_systems(Startup, setup_vggt_system);
        app.add_systems(
            Update,
            (
                capture_and_send_frames,
                receive_and_merge_gaussians,
                monitor_python_worker,
            ),
        );
    }
}

fn setup_vggt_system(
    mut commands: Commands,
    config: Res<VGGTConfig>,
) {
    info!("Setting up VGGT memory synchronization");
    
    let frame_path = config.mmap_base_path.join("frames");
    let gaussian_path = config.mmap_base_path.join("gaussians");
    let control_path = config.mmap_base_path.join("control");
    
    // Initialize synchronizers
    let frame_writer = Synchronizer::new(frame_path.as_os_str());
    let gaussian_reader = Synchronizer::new(gaussian_path.as_os_str());
    let control_writer = Synchronizer::new(control_path.as_os_str());
    
    let memory_sync = VGGTMemorySync {
        frame_writer: Arc::new(Mutex::new(frame_writer)),
        gaussian_reader: Arc::new(Mutex::new(gaussian_reader)),
        control_writer: Arc::new(Mutex::new(control_writer)),
        last_write_time: Arc::new(AtomicU64::new(0)),
        last_read_time: Arc::new(AtomicU64::new(0)),
    };
    
    commands.insert_resource(memory_sync);
    
    launch_python_worker(&mut commands, &config);
}

fn launch_python_worker(commands: &mut Commands, config: &VGGTConfig) {
    info!("Launching Python VGGT worker");
    
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg(&config.python_script_path)
        .env("VGGT_FRAME_PATH", config.mmap_base_path.join("frames"))
        .env("VGGT_GAUSSIAN_PATH", config.mmap_base_path.join("gaussians"))
        .env("VGGT_CONTROL_PATH", config.mmap_base_path.join("control"))
        .env("VGGT_CONFIDENCE_THRESHOLD", config.confidence_threshold.to_string())
        .env("VGGT_EXTEND_SCENE", config.extend_scene.to_string());
    
    if let Some(checkpoint) = &config.model_checkpoint {
        cmd.env("VGGT_MODEL_CHECKPOINT", checkpoint);
    }
    
    if let Some(token) = &config.replicate_api_token {
        cmd.env("REPLICATE_API_TOKEN", token);
    }
    
    match cmd.spawn() {
        Ok(child) => {
            info!("Python worker launched with PID: {:?}", child.id());
            let worker = PythonWorker {
                process: child,
                is_running: Arc::new(AtomicBool::new(true)),
            };
            commands.insert_resource(worker);
        }
        Err(e) => {
            error!("Failed to launch Python worker: {}", e);
        }
    }
}

fn capture_and_send_frames(
    frame_data: Res<FrameData>,
    cameras: Query<(&Camera, &GlobalTransform), bevy::ecs::query::With<VGGTCamera>>,
    gaussian_handles: Query<&PlanarGaussian3dHandle>,
    gaussian_assets: Res<Assets<PlanarGaussian3d>>,
    merger: Res<GaussianMerger>,
    memory_sync: Option<Res<VGGTMemorySync>>,
    config: Res<VGGTConfig>,
    mut frame_counter: Local<u32>,
    _time: Res<Time>,
) {
    if !config.enabled {
        return;
    }
    
    let Some(memory_sync) = memory_sync else {
        return;
    };
    
    *frame_counter += 1;
    if *frame_counter % config.frame_interval != 0 {
        return;
    }
    
    // Get latest frame data
    let Some(frame_bytes) = frame_data.get_latest_frame() else {
        return;
    };
    
    if frame_bytes.is_empty() {
        return;
    }
    
    // Get camera transform and forward direction
    let (camera_pos, camera_rot, camera_forward) = cameras
        .iter()
        .next()
        .map(|(_, transform)| {
            let pos = transform.translation();
            let rot = transform.rotation();
            let forward = transform.forward();
            (
                [pos.x, pos.y, pos.z],
                [rot.x, rot.y, rot.z, rot.w],
                [forward.x, forward.y, forward.z],
            )
        })
        .unwrap_or(([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0]));
    
    // Check gaussian density if scene extension is enabled
    let mut is_sparse_area = false;
    if config.extend_scene {
        // Find existing gaussian cloud
        for handle in gaussian_handles.iter() {
            if let Some(cloud) = gaussian_assets.get(handle.0.id()) {
                let gaussians: Vec<Gaussian3d> = cloud.to_interleaved();
                let density = merger.count_gaussians_in_frustum(&camera_pos, &camera_forward, &gaussians, 20.0);
                
                if (density as f32) < config.sparse_threshold {
                    is_sparse_area = true;
                    info!("Sparse area detected: {} gaussians in frustum (threshold: {})", 
                          density, config.sparse_threshold);
                }
                break;
            }
        }
    }
    
    // Create shared frame
    let shared_frame = SharedFrame {
        width: 3840,  // TODO: Get from config
        height: 2160, // TODO: Get from config
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        camera_position: camera_pos,
        camera_rotation: camera_rot,
        camera_forward,
        frame_data: frame_bytes.to_vec(),
        is_outpaint_request: is_sparse_area,
        outpaint_seed: 101001, // Fixed seed for consistency
    };
    
    // Write to shared memory
    if let Ok(mut writer) = memory_sync.frame_writer.lock() {
        match writer.write(&shared_frame, Duration::from_millis(10)) {
            Ok((bytes_written, _)) => {
                debug!("Wrote frame to shared memory: {} bytes", bytes_written);
                memory_sync.last_write_time.store(
                    shared_frame.timestamp_ms,
                    Ordering::Relaxed,
                );
            }
            Err(e) => {
                warn!("Failed to write frame to shared memory: {:?}", e);
            }
        }
    }
}

fn receive_and_merge_gaussians(
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
    gaussian_handles: Query<&PlanarGaussian3dHandle>,
    memory_sync: Option<Res<VGGTMemorySync>>,
    mut merger: ResMut<GaussianMerger>,
    config: Res<VGGTConfig>,
    mut last_merge_time: Local<f64>,
    time: Res<Time>,
) {
    if !config.enabled {
        return;
    }
    
    let Some(memory_sync) = memory_sync else {
        return;
    };
    
    // Rate limit merging to avoid performance issues
    let current_time = time.elapsed_secs_f64();
    if current_time - *last_merge_time < 0.5 {
        // Merge at most every 500ms
        return;
    }
    
    // Read gaussian batch from shared memory
    let gaussian_batch = if let Ok(mut reader) = memory_sync.gaussian_reader.lock() {
        match unsafe { reader.read::<SharedGaussianBatch>(false) } {
            Ok(result) => {
                // ReadResult derefs to Archived<T>, need to deserialize
                use rkyv::Deserialize as RkyvDeserialize;
                let archived = &*result;
                let batch: SharedGaussianBatch = archived.deserialize(&mut Infallible).unwrap();
                memory_sync.last_read_time.store(
                    batch.timestamp_ms,
                    Ordering::Relaxed,
                );
                Some(batch)
            }
            Err(e) => {
                warn!("Failed to read gaussians from shared memory: {:?}", e);
                None
            }
        }
    } else {
        None
    };
    
    let Some(batch) = gaussian_batch else {
        return;
    };
    
    info!("Received {} gaussians from Python worker", batch.gaussians.len());
    *last_merge_time = current_time;
    
    // Find existing gaussian cloud to merge with
    for handle in gaussian_handles.iter() {
        if let Some(cloud) = gaussian_assets.get_mut(handle.0.id()) {
            let mut gaussians: Vec<Gaussian3d> = cloud.to_interleaved();
            
            gaussians = merger.merge_gaussians(gaussians, batch.clone(), &config);
            
            *cloud = PlanarGaussian3d::from(gaussians);
            
            info!("Updated gaussian cloud with {} total gaussians", cloud.len());
            break; 
        }
    }
}

fn monitor_python_worker(
    worker: Option<Res<PythonWorker>>,
    mut commands: Commands,
    config: Res<VGGTConfig>,
    mut check_interval: Local<f64>,
    time: Res<Time>,
) {
    let current_time = time.elapsed_secs_f64();
    if current_time - *check_interval < 5.0 {
        return;
    }
    *check_interval = current_time;
    
    if let Some(worker) = worker {
        if !worker.is_running.load(Ordering::Relaxed) {
            warn!("Python worker is not running, attempting to restart...");
            launch_python_worker(&mut commands, &config);
        }
    }
}

impl Drop for PythonWorker {
    fn drop(&mut self) {
        info!("Shutting down Python VGGT worker");
        self.is_running.store(false, Ordering::Relaxed);
        let _ = self.process.kill();
    }
}