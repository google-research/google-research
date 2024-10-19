use anyhow::Result;
use brush_render::{gaussian_splats::Splats, AutodiffBackend, Backend};
use brush_train::{scene::Scene, train::TrainStepStats};
use burn::tensor::{activation::sigmoid, ElementConversion, Tensor};
use image::{DynamicImage, GenericImageView};
use rerun::{Color, FillMode, RecordingStream, TensorBuffer, TensorDimension};

pub struct VisualizeTools {
    rec: RecordingStream,
}

async fn tensor_to_image<B: Backend>(tensor: Tensor<B, 3>) -> DynamicImage {
    image::DynamicImage::from(
        image::Rgba32FImage::from_raw(
            tensor.shape().dims[1] as u32,
            tensor.shape().dims[0] as u32,
            tensor.into_data_async().await.to_vec::<f32>().unwrap(),
        )
        .unwrap(),
    )
}

impl VisualizeTools {
    pub fn new() -> Self {
        let rec = rerun::RecordingStreamBuilder::new("brush_visualize")
            .spawn()
            .expect("Failed to start rerun");

        Self { rec }
    }

    pub(crate) async fn log_splats<B: Backend>(&self, splats: &Splats<B>) -> Result<()> {
        let means = splats
            .means
            .val()
            .into_data_async()
            .await
            .to_vec::<f32>()
            .unwrap();
        let means = means.chunks(3).map(|c| glam::vec3(c[0], c[1], c[2]));

        let sh_c0 = 0.2820947917738781;
        let base_rgb = splats.sh_coeffs.val().slice([0..splats.num_splats(), 0..3]) * sh_c0 + 0.5;

        let transparency = sigmoid(splats.raw_opacity.val());

        let colors = base_rgb.into_data_async().await.to_vec::<f32>().unwrap();
        let colors = colors.chunks(3).map(|c| {
            Color::from_rgb(
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            )
        });

        // Visualize 2 sigma, and simulate some of the small covariance blurring.
        let radii = (splats.log_scales.val().exp() * transparency.unsqueeze_dim(1) * 2.0 + 0.004)
            .into_data_async()
            .await
            .to_vec()
            .unwrap();

        let rotations = splats
            .rotation
            .val()
            .into_data_async()
            .await
            .to_vec::<f32>()
            .unwrap();
        let rotations = rotations.chunks(4).map(|q| {
            let rotation = glam::Quat::from_array([q[1], q[2], q[3], q[0]]);
            rotation
        });

        let radii = radii.chunks(3).map(|r| glam::vec3(r[0], r[1], r[2]));

        self.rec.log(
            "world/splat/points",
            &rerun::Ellipsoids3D::from_centers_and_half_sizes(means, radii)
                .with_quaternions(rotations)
                .with_colors(colors)
                .with_fill_mode(FillMode::Solid),
        )?;

        Ok(())
    }

    pub(crate) fn log_scene(&self, scene: &Scene) -> Result<()> {
        let rec = &self.rec;
        rec.log_static("world", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN)?;

        for (i, data) in scene.views.iter().enumerate() {
            let path = format!("world/dataset/camera/{i}");
            let (width, height) = data.image.dimensions();
            let vis_size = glam::uvec2(width, height);
            let rerun_camera = rerun::Pinhole::from_focal_length_and_resolution(
                data.camera.focal(vis_size),
                glam::vec2(vis_size.x as f32, vis_size.y as f32),
            );
            rec.log_static(path.clone(), &rerun_camera)?;
            rec.log_static(
                path.clone(),
                &rerun::Transform3D::from_translation_rotation(
                    data.camera.position,
                    data.camera.rotation,
                ),
            )?;
            rec.log_static(
                path + "/image",
                &rerun::Image::from_dynamic_image(data.image.clone())?,
            )?;
        }
        Ok(())
    }

    pub async fn log_train_stats<B: AutodiffBackend>(
        &self,
        splats: &Splats<B>,
        stats: &TrainStepStats<B>,
        gt_image: Tensor<B, 4>,
    ) -> Result<()> {
        let rec = &self.rec;
        rec.set_time_sequence("iterations", stats.iter);
        rec.log("lr/mean", &rerun::Scalar::new(stats.lr_mean))?;
        rec.log("lr/opac", &rerun::Scalar::new(stats.lr_opac))?;
        rec.log("lr/rest", &rerun::Scalar::new(stats.lr_rest))?;

        rec.log(
            "splats/num",
            &rerun::Scalar::new(splats.num_splats() as f64).clone(),
        )?;

        let mse = (stats.pred_images.clone() - gt_image.clone())
            .powf_scalar(2.0)
            .mean();
        let psnr = mse.clone().recip().log() * 10.0 / std::f32::consts::LN_10;

        rec.log(
            "losses/main",
            &rerun::Scalar::new(stats.loss.clone().into_scalar_async().await.elem::<f64>()),
        )?;
        rec.log(
            "stats/PSNR",
            &rerun::Scalar::new(psnr.into_scalar_async().await.elem::<f64>()),
        )?;

        // Not sure what's best here, atm let's just log the first batch render only.
        // Maybe could do an average instead?
        let aux = &stats.auxes[0];

        rec.log(
            "splats/num_intersects",
            &rerun::Scalar::new(aux.read_num_intersections() as f64),
        )?;
        rec.log(
            "splats/num_visible",
            &rerun::Scalar::new(aux.read_num_visible() as f64),
        )?;

        let tile_depth = aux.read_tile_depth();
        rec.log(
            "images/tile depth",
            &rerun::Tensor::new(rerun::TensorData::new(
                tile_depth
                    .dims()
                    .map(|x| TensorDimension::unnamed(x as u64))
                    .to_vec(),
                TensorBuffer::I32(tile_depth.into_data().to_vec::<i32>().unwrap().into()),
            )),
        )?;

        let main_gt_image = tensor_to_image(gt_image.slice([0..1]).squeeze(0)).await;
        let main_pred_image =
            tensor_to_image(stats.pred_images.clone().slice([0..1]).squeeze(0)).await;

        rec.log(
            "images/predicted",
            &rerun::Image::from_image(main_pred_image.to_rgba8())?,
        )?;
        rec.log(
            "images/ground truth",
            &rerun::Image::from_image(main_gt_image.to_rgba8())?,
        )?;

        Ok(())
    }
}
