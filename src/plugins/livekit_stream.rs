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
use livekit::prelude::*;
use livekit::options::{TrackPublishOptions, VideoCodec};
use livekit::webrtc::{
    video_frame::{I420Buffer, VideoFrame, VideoRotation},
    video_source::native::NativeVideoSource,
    video_source::{RtcVideoSource, VideoResolution},
};
use std::sync::Arc;
use async_channel::{unbounded, Sender, Receiver};

use crate::plugins::image_copy::FrameData;

struct LiveKitConfig {
    url: String,
    api_key: String,
    api_secret: String,
    room_name: String,
    participant_identity: String,
    participant_name: String,
    width: u32,
    height: u32,
}

impl LiveKitConfig {
    fn from_env(width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            url: std::env::var("LIVEKIT_URL")
                .expect("LIVEKIT_URL environment variable must be set"),
            api_key: std::env::var("LIVEKIT_API_KEY")
                .expect("LIVEKIT_API_KEY environment variable must be set"),
            api_secret: std::env::var("LIVEKIT_API_SECRET")
                .expect("LIVEKIT_API_SECRET environment variable must be set"),
            room_name: std::env::var("LIVEKIT_ROOM_NAME")
                .unwrap_or_else(|_| "bevy_render_room".to_string()),
            participant_identity: std::env::var("LIVEKIT_PARTICIPANT_IDENTITY")
                .unwrap_or_else(|_| "bevy_renderer".to_string()),
            participant_name: std::env::var("LIVEKIT_PARTICIPANT_NAME")
                .unwrap_or_else(|_| "not-auto-renderer".to_string()),
            width,
            height,
        })
    }
}

#[derive(Resource)]
pub struct LiveKitStreamer {
    _room: Arc<Room>,
    frame_sender: Sender<Vec<u8>>,
    _video_track: LocalVideoTrack,
    _runtime_handle: tokio::runtime::Handle,
    _frame_task: tokio::task::JoinHandle<()>,
}

impl LiveKitStreamer {
    async fn new(config: LiveKitConfig) -> Result<Self> {
        info!("Creating LiveKit access token...");
        
        let token = create_access_token(&config)?;
        
        info!("Connecting to LiveKit room at {}...", config.url);
        
        let (room, mut room_events) = Room::connect(&config.url, &token, RoomOptions::default())
            .await
            .context("Failed to connect to LiveKit room")?;
        
        info!("Connected to room: {}", config.room_name);
        
        let native_source = NativeVideoSource::new(
            VideoResolution {
                width: config.width,
                height: config.height,
            }
        );
        
        let video_source = RtcVideoSource::Native(native_source.clone());
        
        let video_track = LocalVideoTrack::create_video_track(
            "bevy_video",
            video_source,
        );
        
        let publish_options = TrackPublishOptions {
            video_codec: VideoCodec::H264,
            source: TrackSource::Camera,
            ..Default::default()
        };
        
        room.local_participant()
            .publish_track(LocalTrack::Video(video_track.clone()), publish_options)
            .await
            .context("Failed to publish video track")?;
        
        info!("Video track published successfully");
        
        // Create async-channel for frame data (works across runtime boundaries)
        let (frame_sender, frame_receiver) = unbounded::<Vec<u8>>();
        
        tokio::spawn(async move {
            while let Some(event) = room_events.recv().await {
                match event {
                    RoomEvent::Connected { participants_with_tracks: _ } => {
                        info!("Room connected successfully");
                    }
                    RoomEvent::Disconnected { reason: _ } => {
                        warn!("Room disconnected");
                        break;
                    }
                    RoomEvent::ParticipantConnected(participant) => {
                        info!("Participant connected: {}", participant.identity());
                    }
                    RoomEvent::ParticipantDisconnected(participant) => {
                        info!("Participant disconnected: {}", participant.identity());
                    }
                    RoomEvent::LocalTrackPublished { publication, .. } => {
                        info!("Local track published: {}", publication.sid());
                    }
                    RoomEvent::LocalTrackUnpublished { publication, .. } => {
                        info!("Local track unpublished: {}", publication.sid());
                    }
                    _ => {}
                }
            }
        });
        
        let width = config.width;
        let height = config.height;
        let source_for_capture = native_source.clone();
        
        let frame_task = tokio::spawn(async move {
            info!("Frame sender task started - waiting for frames");
            let mut frame_count = 0i64;
            
            // Pre-allocate I420 buffer
            info!("Creating I420 buffer with dimensions {}x{}", width, height);
            let mut video_frame = match std::panic::catch_unwind(|| {
                VideoFrame {
                    rotation: VideoRotation::VideoRotation0,
                    buffer: I420Buffer::new(width, height),
                    timestamp_us: 0,
                }
            }) {
                Ok(frame) => frame,
                Err(e) => {
                    error!("Failed to create I420 buffer: {:?}", e);
                    return;
                }
            };
            
            info!("I420 buffer created successfully, starting receive loop");
            
            loop {
                info!("Waiting for frame...");
                
                // timeout to detect if we're stuck
                match tokio::time::timeout(tokio::time::Duration::from_secs(5), frame_receiver.recv()).await {
                    Ok(Ok(frame_data)) => {
                        info!("Received frame {} with {} bytes", frame_count + 1, frame_data.len());
                        
                        // Update timestamp - 60 FPS
                        video_frame.timestamp_us = frame_count * 1_000_000 / 60; 
                        
                        // Check frame data size
                        let expected_size = (width * height * 4) as usize;
                        if frame_data.len() != expected_size {
                            error!("Frame data size mismatch: got {} bytes, expected {} bytes", 
                                frame_data.len(), expected_size);
                            continue;
                        }
                        
                        rgba_to_i420(&frame_data, &mut video_frame.buffer, width, height);
                        source_for_capture.capture_frame(&video_frame);
                        frame_count += 1;
                        
                        info!("Successfully processed frame {}", frame_count);
                    }
                    Ok(Err(e)) => {
                        warn!("Frame receiver channel closed, ending frame sender task. {e:?}");
                        break;
                    }
                    Err(e) => {
                        error!("Timeout waiting for frame after 5 seconds! {e:?}");
                        break;
                        // Continue waiting instead of breaking
                    }
                }
            }
            
            warn!("Frame sender loop ended after {} frames", frame_count);
        });
        
        Ok(Self {
            _room: Arc::new(room),
            _video_track: video_track,
            frame_sender,
            _runtime_handle: tokio::runtime::Handle::current(),
            _frame_task: frame_task,
        })
    }
    
    pub fn send_frame(&self, frame_data: Vec<u8>) {
        // Use try_send for non-blocking send from sync context
        match self.frame_sender.try_send(frame_data) {
            Ok(_) => {}
            Err(async_channel::TrySendError::Full(_)) => {
                warn!("Channel is full, dropping frame");
            }
            Err(async_channel::TrySendError::Closed(_)) => {
                error!("Channel is closed - receiver has been dropped");
            }
        }
    }
}

fn create_access_token(config: &LiveKitConfig) -> Result<String> {
    use livekit_api::access_token;
    
    let token = access_token::AccessToken::with_api_key(&config.api_key, &config.api_secret)
        .with_identity(&config.participant_identity)
        .with_name(&config.participant_name)
        .with_grants(access_token::VideoGrants {
            room_join: true,
            room: config.room_name.clone(),
            can_publish: true,
            can_subscribe: true,
            can_publish_data: true,
            ..Default::default()
        })
        .to_jwt()
        .context("Failed to create JWT token")?;
    
    Ok(token)
}

// Convert RGBA to I420 (YUV420p) format
fn rgba_to_i420(rgba_data: &[u8], i420_buffer: &mut I420Buffer, width: u32, height: u32) {
    let width = width as usize;
    let height = height as usize;
    
    let (stride_y, stride_u, stride_v) = i420_buffer.strides();
    let (data_y, data_u, data_v) = i420_buffer.data_mut();
    
    // Convert each pixel from RGBA to YUV
    for y in 0..height {
        for x in 0..width {
            let rgba_idx = (y * width + x) * 4;
            let r = rgba_data[rgba_idx] as f32;
            let g = rgba_data[rgba_idx + 1] as f32;
            let b = rgba_data[rgba_idx + 2] as f32;
            
            // BT.601 conversion
            // Y = 0.299*R + 0.587*G + 0.114*B
            let y_val = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8;
            data_y[y * stride_y as usize + x] = y_val;
            
            // Subsample U and V (every 2x2 block shares U and V)
            if x % 2 == 0 && y % 2 == 0 {
                let uv_x = x / 2;
                let uv_y = y / 2;
                
                // Average the 2x2 block for chroma subsampling
                let mut r_sum = r;
                let mut g_sum = g;
                let mut b_sum = b;
                let mut count = 1.0;
                
                // Add neighboring pixels if they exist
                if x + 1 < width {
                    let idx = rgba_idx + 4;
                    r_sum += rgba_data[idx] as f32;
                    g_sum += rgba_data[idx + 1] as f32;
                    b_sum += rgba_data[idx + 2] as f32;
                    count += 1.0;
                }
                
                if y + 1 < height {
                    let idx = rgba_idx + width * 4;
                    r_sum += rgba_data[idx] as f32;
                    g_sum += rgba_data[idx + 1] as f32;
                    b_sum += rgba_data[idx + 2] as f32;
                    count += 1.0;
                    
                    if x + 1 < width {
                        let idx = idx + 4;
                        r_sum += rgba_data[idx] as f32;
                        g_sum += rgba_data[idx + 1] as f32;
                        b_sum += rgba_data[idx + 2] as f32;
                        count += 1.0;
                    }
                }
                
                let r_avg = r_sum / count;
                let g_avg = g_sum / count;
                let b_avg = b_sum / count;
                
                // U = -0.147*R - 0.289*G + 0.436*B + 128
                let u_val = (-0.147 * r_avg - 0.289 * g_avg + 0.436 * b_avg + 128.0).clamp(0.0, 255.0) as u8;
                data_u[uv_y * stride_u as usize + uv_x] = u_val;
                
                // V = 0.615*R - 0.515*G - 0.100*B + 128
                let v_val = (0.615 * r_avg - 0.515 * g_avg - 0.100 * b_avg + 128.0).clamp(0.0, 255.0) as u8;
                data_v[uv_y * stride_v as usize + uv_x] = v_val;
            }
        }
    }
}

fn stream_frame_data(
    frame_data: Res<FrameData>,
    livekit_streamer: Option<Res<LiveKitStreamer>>,
    time: Res<Time>,
    mut last_frame_count: Local<u64>,
    mut last_push_timestamp: Local<f64>,
) {
    // If LiveKit is not configured, skip
    let Some(streamer) = livekit_streamer else {
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
        
        // Send frame to LiveKit
        streamer.send_frame(frame_data.clone());
        
        let current_time = time.elapsed_secs_f64();
        let frame_latency = if *last_push_timestamp > 0.0 {
            ((current_time - *last_push_timestamp) * 1000.0) as u64
        } else {
            0
        };
        
        info!("Sent frame {} to LiveKit ({}ms since last frame)",
            current_frame, frame_latency);
        
        *last_push_timestamp = current_time;
        *last_frame_count = current_frame;
    }
}

pub struct LiveKitStreamPlugin {
    width: u32,
    height: u32,
}

impl LiveKitStreamPlugin {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Plugin for LiveKitStreamPlugin {
    fn build(&self, app: &mut App) {
        info!("Initializing LiveKitStreamPlugin");
        let lk_config = LiveKitConfig::from_env(self.width, self.height).unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let lk_streamer = runtime.block_on(LiveKitStreamer::new(lk_config)).unwrap();
        app.insert_resource(lk_streamer);
        app.add_systems(Update, stream_frame_data);
    }
}