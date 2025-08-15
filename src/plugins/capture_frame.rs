use bevy::{app::{App, AppExit, Plugin, PostUpdate}, asset::{Assets, Handle}, ecs::{component::Component, event::EventWriter, system::{Local, Query, Res, ResMut}}, image::Image, log::info, prelude::{Deref, DerefMut}, render::renderer::RenderDevice};
use bevy_gaussian_splatting::TextureFormatPixelInfo;
use bevy::ecs::resource::Resource;

use crate::plugins::image_copy::MainWorldReceiver;

#[derive(Debug, Default)]
pub enum SceneState {
    #[default]
    // State before any rendering
    BuildScene,
    // Rendering state, stores the number of frames remaining before saving the image
    Render(u32),
}

#[derive(Debug, Default, Resource)]
pub struct SceneController {
    pub state: SceneState,
    pub name: String,
    pub width: u32,
    pub height: u32,
}

impl SceneController {
    pub fn new(width: u32, height: u32) -> SceneController {
        SceneController { state: SceneState::BuildScene, name: String::from(""), width, height }
    }
}


/// CPU-side image for saving
#[derive(Component, Deref, DerefMut)]
pub struct ImageToSave(pub Handle<Image>);

pub struct CaptureFramePlugin;
impl Plugin for CaptureFramePlugin {
    fn build(&self, app: &mut App) {
        info!("Adding CaptureFramePlugin");
        app.add_systems(PostUpdate, update);
    }
}

fn update(
    images_to_save: Query<&ImageToSave>,
    receiver: Res<MainWorldReceiver>,
    mut images: ResMut<Assets<Image>>,
    mut scene_controller: ResMut<SceneController>,
    // TODO
    mut _app_exit_writer: EventWriter<AppExit>,
    mut _file_number: Local<u32>,
) {
    if let SceneState::Render(n) = scene_controller.state {
        if n < 1 {
            let mut image_data = Vec::new();
            // use try_recv to prevent blocking
            while let Ok(data) = receiver.try_recv() {
                // image generation could be faster than saving to fs,
                // that's why use only last of them
                image_data = data;
            }
            if !image_data.is_empty() {
                for image in images_to_save.iter() {
                    // Fill correct data from channel to image
                    let img_bytes = images.get_mut(image.id()).unwrap();

                    // We need to ensure that this works regardless of the image dimensions
                    // If the image became wider when copying from the texture to the buffer,
                    // then the data is reduced to its original size when copying from the buffer to the image.
                    let row_bytes = img_bytes.width() as usize
                        * img_bytes.texture_descriptor.format.pixel_size();
                    let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);
                    if row_bytes == aligned_row_bytes {
                        img_bytes.data.as_mut().unwrap().clone_from(&image_data);
                    } else {
                        // shrink data to original image size
                        img_bytes.data = Some(
                            image_data
                                .chunks(aligned_row_bytes)
                                .take(img_bytes.height() as usize)
                                .flat_map(|row| &row[..row_bytes.min(row.len())])
                                .cloned()
                                .collect(),
                        );
                    }
                }
            }
        } else {
            // clears channel for skipped frames
            while receiver.try_recv().is_ok() {}
            scene_controller.state = SceneState::Render(n - 1);
        }
    }
}