use anyhow::{Context, Result};
use bevy::{
    app::{App, Plugin, Update},
    ecs::{
        resource::Resource,
        system::{Local, Res},
    },
    log::{error, info, warn},
    time::Time,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video::{VideoFormat, VideoInfo};
use livekit_api::services::room::{RoomClient, CreateRoomOptions};

use crate::plugins::image_copy::FrameData;

async fn ensure_room_exists(config: &GStreamerLiveKitConfig) -> Result<()> {
    info!("Checking if LiveKit room '{}' exists...", config.room_name);
    
    // Extract host from ws_url (remove ws:// or wss:// prefix)
    let api_url = if config.ws_url.starts_with("wss://") {
        config.ws_url.replace("wss://", "https://")
    } else if config.ws_url.starts_with("ws://") {
        config.ws_url.replace("ws://", "http://")
    } else {
        config.ws_url.clone()
    };
    
    let room_client = RoomClient::with_api_key(&api_url, &config.api_key, &config.secret_key);
    
    // Try to list rooms to check if our room exists
    match room_client.list_rooms(vec![config.room_name.clone()]).await {
        Ok(rooms) => {
            if rooms.is_empty() {
                info!("Room '{}' does not exist, creating it...", config.room_name);
                
                let create_options = CreateRoomOptions::default();
                
                match room_client.create_room(&config.room_name, create_options).await {
                    Ok(room) => {
                        info!("Successfully created room: {}", room.name);
                    }
                    Err(e) => {
                        // Room might already exist (race condition) or other error
                        warn!("Failed to create room (it may already exist): {}", e);
                    }
                }
            } else {
                info!("Room '{}' already exists", config.room_name);
            }
        }
        Err(e) => {
            warn!("Failed to list rooms: {}. Attempting to create room anyway...", e);
            
            let create_options = CreateRoomOptions::default();
            
            match room_client.create_room(&config.room_name, create_options).await {
                Ok(room) => {
                    info!("Successfully created room: {}", room.name);
                }
                Err(e) => {
                    warn!("Failed to create room (it may already exist): {}", e);
                }
            }
        }
    }
    
    Ok(())
}

#[derive(Clone)]
pub struct GStreamerLiveKitConfig {
    ws_url: String,
    api_key: String,
    secret_key: String,
    room_name: String,
    participant_identity: String,
    participant_name: String,
    width: u32,
    height: u32,
}

impl GStreamerLiveKitConfig {
    pub fn from_env(width: u32, height: u32) -> Result<Self> {
        let livekit_url = std::env::var("LIVEKIT_URL")
            .context("LIVEKIT_URL environment variable must be set")?;
        
        // Convert https:// or wss:// to ws:// for signalling
        let ws_url = if livekit_url.starts_with("https://") {
            livekit_url.replace("https://", "wss://")
        } else if livekit_url.starts_with("http://") {
            livekit_url.replace("http://", "ws://")
        } else {
            livekit_url
        };
        
        Ok(Self {
            ws_url,
            api_key: std::env::var("LIVEKIT_API_KEY")
                .context("LIVEKIT_API_KEY environment variable must be set")?,
            secret_key: std::env::var("LIVEKIT_API_SECRET")
                .context("LIVEKIT_API_SECRET environment variable must be set")?,
            room_name: std::env::var("LIVEKIT_ROOM_NAME")
                .unwrap_or_else(|_| "bevy_render_room".to_string()),
            participant_identity: std::env::var("LIVEKIT_PARTICIPANT_IDENTITY")
                .unwrap_or_else(|_| "bevy_spatial".to_string()),
            participant_name: std::env::var("LIVEKIT_PARTICIPANT_NAME")
                .unwrap_or_else(|_| "bevy_spatial_renderer".to_string()),
            width,
            height,
        })
    }
}

#[derive(Resource)]
pub struct GStreamerLiveKitPipeline {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    width: u32,
    height: u32,
}

impl GStreamerLiveKitPipeline {
    pub fn new(config: GStreamerLiveKitConfig) -> Result<Self> {
        info!("Creating GStreamer LiveKit pipeline...");
        
        // LiveKit API requires Tokio runtime, so we need to create one
        // This is only for the room creation, the actual streaming uses GStreamer
        let rt = tokio::runtime::Runtime::new()
            .context("Failed to create Tokio runtime")?;
        
        // Ensure the room exists
        rt.block_on(async {
            ensure_room_exists(&config).await
        })?;
        
        // Create the pipeline string for LiveKit WebRTC streaming
        // Using the correct property names based on documentation
        let pipeline_str = format!(
            "appsrc name=video_src format=time is-live=true do-timestamp=true ! \
            video/x-raw,format=RGBA,width={},height={},framerate=60/1 ! \
            queue ! \
            videoconvert ! \
            video/x-raw,format=I420 ! \
            queue ! \
            x264enc tune=zerolatency speed-preset=ultrafast bitrate=5000 key-int-max=60 ! \
            video/x-h264,profile=baseline ! \
            queue ! \
            livekitwebrtcsink name=livekit \
                signaller::ws-url={} \
                signaller::api-key={} \
                signaller::secret-key={} \
                signaller::room-name={} \
                signaller::identity={} \
                signaller::participant-name=\"{}\" \
                video-caps=\"video/x-h264\"",
            config.width,
            config.height,
            config.ws_url,
            config.api_key,
            config.secret_key,
            config.room_name,
            config.participant_identity,
            config.participant_name
        );
        
        info!("Pipeline: {}", pipeline_str);
        
        // Try to create the pipeline
        let pipeline = match gst::parse_launch(&pipeline_str) {
            Ok(pipeline) => {
                info!("Successfully created LiveKit WebRTC pipeline");
                pipeline
            }
            Err(e) => {
                error!("Failed to create LiveKit WebRTC pipeline: {}", e);
                
                // Check if the livekitwebrtcsink element is available
                if gst::ElementFactory::find("livekitwebrtcsink").is_none() {
                    error!("livekitwebrtcsink element not found. Please install gst-plugins-rs with livekit feature enabled.");
                    error!("On macOS: brew install gst-plugins-rs");
                    error!("Or build from source: https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs");
                }
                
                return Err(anyhow::anyhow!("Failed to create LiveKit pipeline: {}", e));
            }
        };
        
        let pipeline = pipeline.downcast::<gst::Pipeline>()
            .map_err(|_| anyhow::anyhow!("Failed to cast to pipeline"))?;
        
        // Get the appsrc element
        let appsrc = pipeline
            .by_name("video_src")
            .ok_or_else(|| anyhow::anyhow!("Could not get appsrc element"))?
            .downcast::<gst_app::AppSrc>()
            .map_err(|_| anyhow::anyhow!("Not an appsrc"))?;
        
        // Configure the appsrc
        appsrc.set_property("format", gst::Format::Time);
        appsrc.set_property("is-live", true);
        appsrc.set_property("do-timestamp", true);
        
        // Set caps for RGBA format
        let video_info = VideoInfo::builder(VideoFormat::Rgba, config.width, config.height)
            .fps(gst::Fraction::new(60, 1))
            .build()
            .context("Failed to create video info")?;
        
        let caps = video_info.to_caps()
            .context("Failed to create caps from video info")?;
        appsrc.set_caps(Some(&caps));
        
        // Set up bus watch for pipeline messages
        let bus = pipeline.bus().ok_or_else(|| anyhow::anyhow!("Pipeline has no bus"))?;
        
        // Add a bus watch to handle messages
        let _bus_watch = bus.add_watch(move |_bus, msg| {
            match msg.view() {
                gst::MessageView::Error(err) => {
                    error!(
                        "Pipeline error from {:?}: {} ({:?})",
                        err.src().map(|s| s.path_string()),
                        err.error(),
                        err.debug()
                    );
                }
                gst::MessageView::Warning(warning) => {
                    warn!(
                        "Pipeline warning from {:?}: {} ({:?})",
                        warning.src().map(|s| s.path_string()),
                        warning.error(),
                        warning.debug()
                    );
                }
                gst::MessageView::StateChanged(state_changed) => {
                    if state_changed.src().map(|s| s == msg.src().unwrap()).unwrap_or(false) {
                        info!(
                            "Pipeline state changed from {:?} to {:?}",
                            state_changed.old(),
                            state_changed.current()
                        );
                    }
                }
                gst::MessageView::Eos(_) => {
                    info!("End of stream");
                }
                _ => {}
            }
            gst::glib::ControlFlow::Continue
        })
        .context("Failed to add bus watch")?;
        
        pipeline.set_state(gst::State::Playing)
            .context("Failed to set pipeline to playing state")?;
        
        info!("GStreamer LiveKit pipeline started successfully");
        
        Ok(Self {
            pipeline,
            appsrc,
            width: config.width,
            height: config.height,
        })
    }
    
    pub fn push_frame(&self, frame_data: &[u8]) -> Result<()> {
        let buffer_size = frame_data.len();
        if buffer_size == 0 {
            return Ok(());
        }
        
        let mut buffer = gst::Buffer::with_size(buffer_size)
            .context("Could not allocate buffer")?;
        
        {
            let buffer_ref = buffer.get_mut().unwrap();
            
            // Copy the frame data to the buffer
            let mut map = buffer_ref.map_writable()
                .context("Could not map buffer writable")?;
            map.copy_from_slice(frame_data);
        }
        
        match self.appsrc.push_buffer(buffer) {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to push buffer: {:?}", e);
                Err(anyhow::anyhow!("Failed to push buffer: {:?}", e))
            }
        }
    }
}

impl Drop for GStreamerLiveKitPipeline {
    fn drop(&mut self) {
        info!("Shutting down GStreamer LiveKit pipeline");
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

fn stream_frames_to_livekit(
    frame_data: Res<FrameData>,
    pipeline: Option<Res<GStreamerLiveKitPipeline>>,
    time: Res<Time>,
    mut last_frame_count: Local<u64>,
    mut last_push_timestamp: Local<f64>,
) {
    // If pipeline is not configured, skip
    let Some(pipeline) = pipeline else {
        return;
    };
    
    let current_frame = frame_data.get_frame_count();
    
    // Only process new frames
    if current_frame <= *last_frame_count {
        return;
    }
    
    if let Some(frame_data) = frame_data.get_latest_frame() {
        if frame_data.is_empty() {
            return;
        }
        
        // Push frame to GStreamer pipeline
        match pipeline.push_frame(&frame_data) {
            Ok(_) => {
                let current_time = time.elapsed_secs_f64();
                let frame_latency = if *last_push_timestamp > 0.0 {
                    ((current_time - *last_push_timestamp) * 1000.0) as u64
                } else {
                    0
                };
                
                info!("Pushed frame {} to GStreamer LiveKit pipeline ({}ms since last frame)",
                    current_frame, frame_latency);
                
                *last_push_timestamp = current_time;
            }
            Err(e) => {
                error!("Failed to push frame to pipeline: {}", e);
            }
        }
        
        *last_frame_count = current_frame;
    }
}

pub struct GStreamerLiveKitPlugin {
    width: u32,
    height: u32,
}

impl GStreamerLiveKitPlugin {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Plugin for GStreamerLiveKitPlugin {
    fn build(&self, app: &mut App) {
        info!("Initializing GStreamerLiveKitPlugin");
        
        gst::init().unwrap();
        
        let config = GStreamerLiveKitConfig::from_env(self.width, self.height).unwrap();
        let pipeline = GStreamerLiveKitPipeline::new(config).unwrap();
        app.insert_resource(pipeline);
        app.add_systems(Update, stream_frames_to_livekit);
    }
}