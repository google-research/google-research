pub mod colmap;
pub mod colmap_read_model;
pub mod nerf_synthetic;
pub mod scene_batch;

use std::path::{Path, PathBuf};

use anyhow::Result;
use brush_train::scene::SceneView;
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use image::{DynamicImage, GenericImageView};

pub(crate) fn normalized_path_string(path: &Path) -> String {
    Path::new(path)
        .components()
        .skip_while(|c| matches!(c, std::path::Component::CurDir))
        .collect::<PathBuf>()
        .to_str()
        .unwrap()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

pub(crate) fn clamp_img_to_max_size(image: DynamicImage, max_size: u32) -> DynamicImage {
    if image.width() <= max_size && image.height() <= max_size {
        return image;
    }

    let aspect_ratio = image.width() as f32 / image.height() as f32;
    let (new_width, new_height) = if image.width() > image.height() {
        (max_size, (max_size as f32 / aspect_ratio) as u32)
    } else {
        ((max_size as f32 * aspect_ratio) as u32, max_size)
    };
    image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
}

fn image_to_tensor<B: Backend>(
    image: &DynamicImage,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 3>> {
    let (w, h) = image.dimensions();
    let num_channels = image.color().channel_count();
    let data = if num_channels == 3 {
        image.to_rgb32f().into_vec()
    } else {
        image.to_rgba32f().into_vec()
    };

    let tensor_data = TensorData::new(data, [h as usize, w as usize, num_channels as usize]);
    Ok(Tensor::from_data(tensor_data, device))
}

pub fn read_dataset(
    zip_data: &[u8],
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<Vec<SceneView>> {
    nerf_synthetic::read_dataset(zip_data, max_frames, max_resolution)
        .or_else(move |_| colmap::read_dataset(zip_data, max_frames, max_resolution))
}
