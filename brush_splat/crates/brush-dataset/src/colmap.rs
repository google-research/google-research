use brush_render::camera::{self, Camera};
use brush_train::scene::SceneView;
use std::io::{Cursor, Read};

use crate::colmap_read_model;

#[allow(unused_variables)]
pub fn read_dataset(
    zip_data: &[u8],
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> anyhow::Result<Vec<SceneView>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(zip_data))?;
    let files: Vec<_> = archive.file_names().collect();

    let (bin, cam_path, img_path) = if archive.by_name("sparse/0/cameras.bin").is_ok() {
        (true, "sparse/0/cameras.bin", "sparse/0/images.bin")
    } else if archive.by_name("sparse/0/cameras.txt").is_ok() {
        (false, "sparse/0/cameras.txt", "sparse/0/images.txt")
    } else {
        anyhow::bail!("No COLMAP data found (either text or binary.");
    };

    let cam_model_data = {
        let mut cam_file = archive.by_name(cam_path)?;
        colmap_read_model::read_cameras(&mut cam_file, bin)?
    };

    let img_infos = {
        let img_file = archive.by_name(img_path)?;
        let mut buf_reader = std::io::BufReader::new(img_file);
        colmap_read_model::read_images(&mut buf_reader, bin)?
    };

    let mut img_info_list = img_infos
        .iter()
        .map(|(id, info)| (*id, info))
        .collect::<Vec<_>>();

    // The ids aren't guaranteed to be meaningful but seem to usually correspond
    // to capture order.
    img_info_list.sort_by_key(|(id, info)| *id);

    let mut views = Vec::with_capacity(img_info_list.len());

    for (_, img_info) in img_info_list.iter() {
        let cam = &cam_model_data[&img_info.camera_id];

        let img_path = &img_info.name;
        let image_data = archive.by_name(&format!("images/{img_path}"))?;
        let img_bytes = image_data.bytes().collect::<std::io::Result<Vec<u8>>>()?;
        let mut img = image::load_from_memory(&img_bytes)?;
        if let Some(max) = max_resolution {
            img = crate::clamp_img_to_max_size(img, max);
        }

        let translation = img_info.tvec;
        let quat = img_info.quat;

        // Convert w2c to c2w.
        let world_to_cam = glam::Mat4::from_rotation_translation(quat, translation);
        let cam_to_world = world_to_cam.inverse();

        let (_, quat, translation) = cam_to_world.to_scale_rotation_translation();

        let focal = cam.focal();
        let fovx = camera::focal_to_fov(focal.x, cam.width as u32);
        let fovy = camera::focal_to_fov(focal.y, cam.height as u32);

        let center = cam.center();
        let center_uv = center / glam::vec2(cam.width as f32, cam.height as f32);

        views.push(SceneView {
            camera: Camera::new(translation, quat, glam::vec2(fovx, fovy), center_uv),
            image: img,
        });

        if let Some(max) = max_frames {
            if views.len() >= max {
                break;
            }
        }
    }

    Ok(views)
}
