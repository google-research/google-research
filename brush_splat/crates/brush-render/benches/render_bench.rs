#![allow(clippy::single_range_in_vec_init)]
use std::collections::HashMap;
use std::path::Path;
use std::{fs::File, io::Read};

use brush_render::BurnBack;
use brush_render::{
    camera::{focal_to_fov, fov_to_focal, Camera},
    gaussian_splats::Splats,
};
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::tensor::Tensor;
use burn_wgpu::WgpuDevice;
use safetensors::SafeTensors;

fn main() {
    divan::main();
}

type DiffBack = Autodiff<brush_render::BurnBack>;

const BENCH_DENSITIES: [f32; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const DENSE_MULT: f32 = 0.25;

const LOW_RES: glam::UVec2 = glam::uvec2(512, 512);
const HIGH_RES: glam::UVec2 = glam::uvec2(1024, 1024);

const TARGET_SAMPLE_COUNT: u32 = 5;
const INTERNAL_ITERS: u32 = 4;


fn generate_bench_data() -> anyhow::Result<()> {
    <DiffBack as burn::prelude::Backend>::seed(4);
    let num_points = 2usize.pow(21); //  Maxmimum number of splats to bench.

    let device = WgpuDevice::BestAvailable;
    let means = Tensor::<DiffBack, 2>::random([num_points, 3], burn::tensor::Distribution::Uniform(-0.5, 0.5), &device) * 10000.0;
    let log_scales = Tensor::<DiffBack, 2>::random([num_points, 3], burn::tensor::Distribution::Uniform(0.05, 15.0), &device).log();
    let coeffs = Tensor::<DiffBack, 3>::random([num_points, 1, 3], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);

    let u = Tensor::<DiffBack, 2>::random([num_points, 1], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);
    let v = Tensor::<DiffBack, 2>::random([num_points, 1], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);
    let w = Tensor::<DiffBack, 2>::random([num_points, 1], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);

    let v = v * 2.0 * std::f32::consts::PI;
    let w = w * 2.0 * std::f32::consts::PI;

    let quats = Tensor::cat(
        vec![
            Tensor::sqrt(-u.clone() + 1.0) * Tensor::sin(v.clone()),
            Tensor::sqrt(-u.clone() + 1.0) * Tensor::cos(v.clone()),
            Tensor::sqrt(u.clone()) * Tensor::sin(w.clone()),
            Tensor::sqrt(u.clone()) * Tensor::cos(w.clone()),
        ],
        1,
    );
    let opacities = Tensor::<DiffBack, 1>::random([num_points], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);

    let bytes = means.to_data().bytes;
    let means =  safetensors::tensor::TensorView::new(safetensors::Dtype::F32, means.shape().dims.to_vec(), &bytes)?;
    let bytes = log_scales.to_data().bytes;
    let log_scales =  safetensors::tensor::TensorView::new(safetensors::Dtype::F32, log_scales.shape().dims.to_vec(), &bytes)?;
    let bytes = quats.to_data().bytes;
    let quats =  safetensors::tensor::TensorView::new(safetensors::Dtype::F32, quats.shape().dims.to_vec(), &bytes)?;
    let bytes = coeffs.to_data().bytes;
    let coeffs =  safetensors::tensor::TensorView::new(safetensors::Dtype::F32, coeffs.shape().dims.to_vec(), &bytes)?;
    let bytes = opacities.to_data().bytes;
    let opacities =  safetensors::tensor::TensorView::new(safetensors::Dtype::F32, opacities.shape().dims.to_vec(), &bytes)?;

    let tensors = HashMap::from([
        ("means", means),
        ("scales", log_scales),
        ("coeffs", coeffs),
        ("quats", quats),
        ("opacities", opacities),
    ]);

    safetensors::serialize_to_file(&tensors, &None, Path::new("./test_cases/bench_data.safetensors"))?;
    Ok(())
}


fn bench_general(
    bencher: divan::Bencher,
    dens: f32,
    mean_mult: f32,
    resolution: glam::UVec2,
    grad: bool,
) {
    if !Path::new("./test_cases/bench_data.safetensors").exists() {
        generate_bench_data().expect("Failed to generate bench data");
    }

    let device = WgpuDevice::BestAvailable;
    let mut buffer = Vec::new();
    let _ = File::open("./test_cases/bench_data.safetensors")
        .unwrap()
        .read_to_end(&mut buffer)
        .unwrap();
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    let splats = Splats::<DiffBack>::from_safetensors(&tensors, &device).unwrap();
    let num_points = (splats.num_splats() as f32 * dens) as usize;
    let splats = Splats::from_data(
        (splats.means.val() * mean_mult).slice([0..num_points]),
        splats.sh_coeffs.val().slice([0..num_points]),
        splats.rotation.val().slice([0..num_points]),
        splats.raw_opacity.val().slice([0..num_points]),
        splats.log_scales.val().slice([0..num_points]),
        &device,
    );
    let [w, h] = resolution.into();
    let fov = std::f32::consts::PI * 0.5;
    let focal = fov_to_focal(fov, w);
    let fov_x = focal_to_fov(focal, w);
    let fov_y = focal_to_fov(focal, h);
    let camera = Camera::new(
        glam::vec3(0.0, 0.0, -8.0),
        glam::Quat::IDENTITY,
        glam::vec2(fov_x, fov_y),
        glam::vec2(0.5, 0.5),
    );

    if grad {
        bencher.bench_local(move || {
            for _ in 0..INTERNAL_ITERS {
                let out = splats.render(&camera, resolution, glam::vec3(0.0, 0.0, 0.0), false);
                let _ = out.0.mean().backward();
            }
            // Wait for GPU work.
            <BurnBack as burn::prelude::Backend>::sync(
                &WgpuDevice::BestAvailable,
                burn::tensor::backend::SyncType::Wait,
            );
        });
    } else {
        // Run with no autodiff graph.
        let splats = splats.valid();

        bencher.bench_local(move || {
            for _ in 0..INTERNAL_ITERS {
                let _ = splats.render(&camera, resolution, glam::vec3(0.0, 0.0, 0.0), true);
            }
            // Wait for GPU work.
            <BurnBack as burn::prelude::Backend>::sync(
                &WgpuDevice::BestAvailable,
                burn::tensor::backend::SyncType::Wait,
            );
        });
    }
}

#[divan::bench_group(max_time = 20, sample_count = TARGET_SAMPLE_COUNT, sample_size = 1)]
mod fwd {
    use super::*;

    #[divan::bench(args = BENCH_DENSITIES)]
    fn base(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, LOW_RES, false);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn dense(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, DENSE_MULT, LOW_RES, false);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn hd(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, HIGH_RES, false);
    }
}

#[divan::bench_group(max_time = 20, sample_count = TARGET_SAMPLE_COUNT, sample_size = 1)]
mod bwd {
    use super::*;

    #[divan::bench(args = BENCH_DENSITIES)]
    fn base(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, LOW_RES, true);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn dense(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, DENSE_MULT, LOW_RES, true);
    }

    #[divan::bench(args = BENCH_DENSITIES)]
    fn hd(bencher: divan::Bencher, dens: f32) {
        bench_general(bencher, dens, 1.0, HIGH_RES, true);
    }
}
