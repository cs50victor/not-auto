pub mod plugins;

use std::time::Duration;

use bevy::asset::{Assets, RenderAssetUsages};
use bevy::core_pipeline::core_3d::Camera3d;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::component::Component;
use bevy::ecs::system::{Commands, Query, Res, ResMut};
use bevy::image::{BevyDefault, Image};
use bevy::math::primitives::{Circle, Cuboid};
use bevy::math::{Quat, Vec3};
use bevy::pbr::{MeshMaterial3d, PointLight, StandardMaterial};
use bevy::render::camera::{Camera, RenderTarget};
use bevy::render::mesh::{Mesh, Mesh3d};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::Backends;
use bevy::time::Time;
use bevy::transform::components::Transform;
use bevy::{animation::AnimationPlugin, app::{App, PanicHandlerPlugin, ScheduleRunnerPlugin, Startup, TaskPoolPlugin, Update}, asset::AssetPlugin, color::Color, core_pipeline::CorePipelinePlugin, diagnostic::{DiagnosticsPlugin, FrameCountPlugin}, gizmos::GizmoPlugin, log::LogPlugin, pbr::PbrPlugin, remote::{http::RemoteHttpPlugin, RemotePlugin}, render::{camera::ClearColor, settings::{PowerPreference, RenderCreation, WgpuSettings}, texture::ImagePlugin, RenderPlugin}, state::app::StatesPlugin, time::TimePlugin, transform::TransformPlugin, utils::default};

use bevy::render::render_resource::TextureFormat;
use bevy_gaussian_splatting::{
    random_gaussians_3d, CloudSettings, GaussianCamera, GaussianSplattingPlugin, PlanarGaussian3d, PlanarGaussian3dHandle
};
use crate::plugins::capture_frame::{CaptureFramePlugin, ImageToSave, SceneController, SceneState};
use crate::plugins::image_copy::{update_frame_data, FrameData, ImageCopyPlugin, ImageCopier};
use crate::plugins::gstreamer_livekit::GStreamerLiveKitPlugin;

struct AppConfig {
    width: u32,
    height: u32,
    stream_port: u16,
    stream_host: String,
}

#[derive(Component)]
struct RotatingCamera {
    rotation_speed: f32, // Radians per second
    radius: f32,         // Distance from center point
    center: Vec3,        // Center point of rotation
    height: f32,         // Height of camera above center point
    elapsed: f32,        // Track elapsed time for rotation
}

impl Default for RotatingCamera {
    fn default() -> Self {
        Self {
            rotation_speed: 0.5, // About 1/4 rotation per second
            radius: 50.0,        // Distance from center - even further for full view
            center: Vec3::new(0.0, 0.0, 0.0),  // Center at origin
            height: 20.0,        // Height above ground - looking down more
            elapsed: 0.0,        // Starting time
        }
    }
}


fn main() {
    dotenvy::from_filename_override(".env.local").ok();    
    
    let config = AppConfig {
        width: 3840,
        height: 2160,
        stream_port: 5000,
        stream_host: std::env::var("STREAM_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
    };
   

    App::new()
        .add_plugins(PanicHandlerPlugin)
        .add_plugins(LogPlugin::default())
        .add_plugins(TaskPoolPlugin::default())
        .add_plugins(FrameCountPlugin)
        .add_plugins(TimePlugin)
        .add_plugins(TransformPlugin)
        .add_plugins(DiagnosticsPlugin)
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            // Run 60 times per second.
            Duration::from_secs_f64(1.0 / 60.0),
        ))
        .add_plugins(bevy::window::WindowPlugin {
            primary_window: None,
            exit_condition: bevy::window::ExitCondition::DontExit,
            close_when_requested: true,
        })
        .add_plugins(AssetPlugin::default())
        .add_plugins(StatesPlugin)
        .add_plugins(
            RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    power_preference: PowerPreference::HighPerformance,
                    #[cfg(target_os = "macos")]
                    backends: Some(Backends::METAL),
                    #[cfg(not(target_os = "macos"))]
                    backends: Some(Backends::VULKAN),
                    ..default()
                }),
                ..default()
            }
        )
        .add_plugins(ImagePlugin::default_nearest())
        .add_plugins(CorePipelinePlugin)
        .add_plugins(PbrPlugin::default())
        .add_plugins(AnimationPlugin)
        .add_plugins(GizmoPlugin)
        .add_plugins(ImageCopyPlugin)
        .add_plugins(CaptureFramePlugin)
        .add_plugins(GaussianSplattingPlugin)
        // Using GStreamer-based LiveKit streaming
        .add_plugins(GStreamerLiveKitPlugin::new(config.width, config.height))
        .add_plugins(RemotePlugin::default())
        // jsonrpc server
        .add_plugins(RemoteHttpPlugin::default().with_port(8080).with_headers(
            bevy::remote::http::Headers::new()
                    .insert("Access-Control-Allow-Origin", "*")
                    .insert("Access-Control-Allow-Headers", "Content-Type")
        ))
        // .add_plugins(InputDispatchPlugin)
        // .add_plugins(GltfPlugin::default())
        // .add_plugins(GilrsPlugin::default())
        // headless frame capture
        // ScheduleRunnerPlugin provides an alternative to the default bevy_winit app runner, which
        // manages the loop without creating a window.
        .init_resource::<FrameData>()
        .insert_resource(SceneController::new(
            config.width,
            config.height,
        ))
        .insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)))
        .init_resource::<SceneController>()
        .add_systems(Startup, setup)
        .add_systems(Update, (update_frame_data, rotate_camera))
        .run();
}

fn rotate_camera(time: Res<Time>, mut query: Query<(&mut Transform, &mut RotatingCamera)>) {
    for (mut transform, mut camera) in query.iter_mut() {
        camera.elapsed += time.delta_secs();

        // new position
        let angle = camera.elapsed * camera.rotation_speed;
        let x = camera.radius * angle.cos();
        let z = camera.radius * angle.sin();

        // update camera position and look at center
        transform.translation = Vec3::new(x, camera.height, z) + camera.center;
        transform.look_at(camera.center, Vec3::Y);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scene_controller: ResMut<SceneController>,
    render_device: Res<RenderDevice>,
    mut gaussian_assets: ResMut<Assets<PlanarGaussian3d>>,
){
    let pre_roll_frames = 40;
    let scene_name = "main_scene".into();
    
    let render_target = {
        let size = Extent3d {
            width: scene_controller.width,
            height: scene_controller.height,
            ..Default::default()
        };
    
        // This is the texture that will be rendered to.
        let mut render_target_image = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0; 4],
            TextureFormat::bevy_default(),
            RenderAssetUsages::default(),
        );
        
        render_target_image.texture_descriptor.usage |=
            TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let render_target_image_handle = images.add(render_target_image);
    
        // This is the texture that will be copied to.
        let cpu_image = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0; 4],
            TextureFormat::bevy_default(),
            RenderAssetUsages::default(),
        );
        let cpu_image_handle = images.add(cpu_image);
    
        commands.spawn(ImageCopier::new(render_target_image_handle.clone(), size, &render_device));
    
        commands.spawn(ImageToSave(cpu_image_handle));
    
        scene_controller.state = SceneState::Render(pre_roll_frames);
        scene_controller.name = scene_name;
        RenderTarget::Image(render_target_image_handle.into())
    };
    
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    
    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
    
    // light
    commands.spawn((
        PointLight { shadows_enabled: true, ..default() },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));


    let cloud = gaussian_assets.add(random_gaussians_3d(10_000));
    
    commands.spawn((
        PlanarGaussian3dHandle(cloud),
        CloudSettings::default(),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            target: render_target,
            ..default()
        },
        Tonemapping::None,
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        RotatingCamera::default(),
        // Required for gaussian splatting rendering
        GaussianCamera::default(),  
    ));
    
}

