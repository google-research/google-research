use std::io::Cursor;
use std::io::Read;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use brush_render::camera;
use brush_render::camera::Camera;

use brush_train::scene::SceneView;

use crate::clamp_img_to_max_size;
use crate::normalized_path_string;

// TODO: This could be simplified with some serde-fu by creating a struct
// we deserialize into.
pub fn read_dataset(
    zip_data: &[u8],
    max_frames: Option<usize>,
    max_resolution: Option<u32>,
) -> Result<Vec<SceneView>> {
    let mut cameras = vec![];

    let mut archive = zip::ZipArchive::new(Cursor::new(zip_data))?;

    let transform_fname = archive
        .file_names()
        .find(|x| x.contains("transforms_train.json"))
        .context("No transforms file")?
        .to_owned();

    let base_path = Path::new(&transform_fname).parent().unwrap_or(Path::new("./"));

    let contents: serde_json::Value = {
        let mut transforms_file = archive.by_name(&transform_fname)?;
        let mut transform_buf = String::new();
        transforms_file.read_to_string(&mut transform_buf)?;
        serde_json::from_str(&transform_buf)?
    };

    let fovx = contents
        .get("camera_angle_x")
        .context("Camera angle x")?
        .as_f64()
        .context("Parsing camera angle")? as f32;

    let frames_array = contents
        .get("frames")
        .context("Frames arary")?
        .as_array()
        .context("Parsing frames array")?;

    for (i, frame) in frames_array.iter().enumerate() {
        // NeRF 'transform_matrix' is a camera-to-world transform
        let transform_matrix = frame
            .get("transform_matrix")
            .context("Get transform matrix")?
            .as_array()
            .context("Transform as array")?;

        let transform_matrix: Vec<f32> = transform_matrix
            .iter()
            .flat_map(|x| x.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32))
            .collect();
        let mut transform = glam::Mat4::from_cols_slice(&transform_matrix).transpose();

        // Swap basis to go from z-up, left handed (a la OpenCV) to our kernel format
        // (right-handed, y-down).
        transform.y_axis *= -1.0;
        transform.z_axis *= -1.0;
        transform = glam::Mat4::from_rotation_x(std::f32::consts::PI / 2.0) * transform;

        let (_, rotation, translation) = transform.to_scale_rotation_translation();

        let image_file_path = frame
            .get("file_path")
            .context("Get file path")?
            .as_str()
            .context("File path as str")?;

        let image_path =
            normalized_path_string(&base_path.join(image_file_path.to_owned() + ".png"));

        let img_buffer = archive.by_name(&image_path)?.bytes().collect::<Result<Vec<_>, _>>()?;

        // Create a cursor from the buffer
        let mut image = image::load_from_memory(&img_buffer)?;

        if let Some(max_resolution) = max_resolution {
            image = clamp_img_to_max_size(image, max_resolution);
        }

        let fovy = camera::focal_to_fov(camera::fov_to_focal(fovx, image.width()), image.height());

        cameras.push(SceneView {
            camera: Camera::new(
                translation,
                rotation,
                glam::vec2(fovx, fovy),
                glam::vec2(0.5, 0.5),
            ),
            image,
        });

        if let Some(max) = max_frames {
            if i == max - 1 {
                break;
            }
        }
    }

    Ok(cameras)
}

pub fn read_viewpoint_data(file: &str) -> Result<Vec<Camera>> {
    let mut cameras = vec![];

    let file = std::fs::read_to_string(file).expect("Couldn't find viewpoints file.");
    let contents: serde_json::Value = serde_json::from_str(&file).unwrap();

    let frames_array = contents.as_array().context("Parsing cameras as list")?;

    for cam in frames_array.iter() {
        // NeRF 'transform_matrix' is a camera-to-world transform
        let translation = cam
            .get("position")
            .context("Get transform matrix")?
            .as_array()
            .context("Transform as array")?;
        let translation: Vec<f32> =
            translation.iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let translation = glam::vec3(translation[0], translation[1], translation[2]);

        let rot_matrix =
            cam.get("rotation").context("Get rotation")?.as_array().context("rotation as array")?;

        let rot_matrix: Vec<f32> = rot_matrix
            .iter()
            .flat_map(|x| x.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32))
            .collect();
        let rot_matrix = glam::Mat3::from_cols_slice(&rot_matrix).transpose();

        let width = cam.get("width").context("Get width")?.as_i64().context("parse width")? as u32;

        let height =
            cam.get("height").context("Get height")?.as_i64().context("parse height")? as u32;

        let fx = cam.get("fx").context("Get fx")?.as_f64().context("parse fx")?;

        let fy = cam.get("fy").context("Get fy")?.as_f64().context("parse fy")?;

        let rotation = glam::Quat::from_mat3(&rot_matrix);
        let fovx = camera::focal_to_fov(fx as f32, width);
        let fovy = camera::focal_to_fov(fy as f32, height);
        cameras.push(Camera::new(
            translation,
            rotation,
            glam::vec2(fovx, fovy),
            glam::vec2(0.5, 0.5),
        ));
    }

    Ok(cameras)
}
