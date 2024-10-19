use async_stream::try_stream;
use brush_render::{render::num_sh_coeffs, Backend};
use burn::{
    module::{Param, ParamId},
    tensor::{Tensor, TensorData},
};
use futures_lite::Stream;
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};
use std::io::BufRead;
use tracing::info_span;

use anyhow::{Context, Result};
use brush_render::gaussian_splats::Splats;

pub(crate) struct GaussianData {
    means: [f32; 3],
    scale: [f32; 3],
    opacity: f32,
    rotation: [f32; 4],
    sh_dc: [f32; 3],
    sh_coeffs: Vec<f32>,
}

fn inv_sigmoid(v: f32) -> f32 {
    (v / (1.0 - v)).ln()
}

const SH_C0: f32 = 0.28209479;

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData {
            means: [0.0; 3],
            scale: [0.0; 3],
            opacity: 0.0,
            rotation: [0.0; 4],
            sh_dc: [0.0, 0.0, 0.0],
            sh_coeffs: Vec::new(),
        }
    }

    fn set_property(&mut self, key: &str, property: Property) {
        if let Property::Float(v) = property {
            match key {
                "x" => self.means[0] = v,
                "y" => self.means[1] = v,
                "z" => self.means[2] = v,
                "scale_0" => self.scale[0] = v,
                "scale_1" => self.scale[1] = v,
                "scale_2" => self.scale[2] = v,
                "opacity" => self.opacity = v,
                "rot_0" => self.rotation[0] = v,
                "rot_1" => self.rotation[1] = v,
                "rot_2" => self.rotation[2] = v,
                "rot_3" => self.rotation[3] = v,
                "f_dc_0" => self.sh_dc[0] = v,
                "f_dc_1" => self.sh_dc[1] = v,
                "f_dc_2" => self.sh_dc[2] = v,
                _ if key.starts_with("f_rest_") => {
                    if let Ok(idx) = key["f_rest_".len()..].parse::<u32>() {
                        if idx >= self.sh_coeffs.len() as u32 {
                            self.sh_coeffs.resize(idx as usize + 1, 0.0);
                        }
                        self.sh_coeffs[idx as usize] = v;
                    }
                }
                _ => (),
            }
        }
    }
}

fn update_splats<B: Backend>(
    splats: &mut Option<Splats<B>>,
    means: Vec<f32>,
    sh_coeffs: Vec<f32>,
    rotation: Vec<f32>,
    opacity: Vec<f32>,
    scales: Vec<f32>,
    device: &B::Device,
) {
    let n_splats = means.len() / 3;
    let n_coeffs = sh_coeffs.len() / n_splats;

    let new_means = Tensor::from_data(TensorData::new(means, [n_splats, 3]), device);
    let new_coeffs = Tensor::from_data(TensorData::new(sh_coeffs, [n_splats, n_coeffs]), device);
    let new_rots = Tensor::from_data(TensorData::new(rotation, [n_splats, 4]), device);
    let new_opac = Tensor::from_data(TensorData::new(opacity, [n_splats]), device);
    let new_scales = Tensor::from_data(TensorData::new(scales, [n_splats, 3]), device);

    if let Some(splats) = splats.as_mut() {
        splats.concat_splats(new_means, new_rots, new_coeffs, new_opac, new_scales);
        splats.norm_rotations();
    } else {
        let mut init = Splats {
            means: Param::initialized(ParamId::new(), new_means),
            sh_coeffs: Param::initialized(ParamId::new(), new_coeffs),
            rotation: Param::initialized(ParamId::new(), new_rots),
            raw_opacity: Param::initialized(ParamId::new(), new_opac),
            log_scales: Param::initialized(ParamId::new(), new_scales),
            xys_dummy: Tensor::zeros([n_splats, 2], device),
        };
        init.norm_rotations();
        // Create a new splat instance if it hasn't been initialzized yet.
        *splats = Some(init);
    }
}

pub fn ply_count(ply_data: &[u8]) -> Result<usize> {
    let mut reader = std::io::Cursor::new(ply_data);
    let gaussian_parser = Parser::<GaussianData>::new();
    let header = gaussian_parser.read_header(&mut reader)?;
    header
        .elements
        .iter()
        .find(|e| e.name == "vertex")
        .map(|e| e.count)
        .context("Invalid ply file")
}

fn interleave_coeffs(sh_dc: [f32; 3], sh_rest: &[f32]) -> Vec<f32> {
    let channels = 3;
    let coeffs_per_channel = sh_rest.len() / channels;
    let mut result = Vec::with_capacity(sh_rest.len() + 3);
    result.extend(sh_dc);

    for i in 0..coeffs_per_channel {
        for j in 0..channels {
            let index = j * coeffs_per_channel + i;
            result.push(sh_rest[index]);
        }
    }
    result
}

pub fn load_splat_from_ply<B: Backend>(
    ply_data: &[u8],
    device: B::Device,
) -> impl Stream<Item = Result<Splats<B>>> + '_ {
    // set up a reader, in this case a file.
    let mut reader = std::io::Cursor::new(ply_data);

    let mut splats: Option<Splats<B>> = None;

    let update_every = 50000;

    let _span = info_span!("Read splats").entered();

    let gaussian_parser = Parser::<GaussianData>::new();

    try_stream! {
        let header = gaussian_parser.read_header(&mut reader)?;

        for element in &header.elements {
            if element.name == "vertex" {
                let min_props = [
                    "x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1",
                    "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
                ];

                if !min_props
                    .iter()
                    .all(|p| element.properties.iter().any(|x| &x.name == p))
                {
                    Err(anyhow::anyhow!("Invalid splat ply. Missing properties!"))?
                }

                let sh_coeffs_per_channel = (1 + element
                    .properties
                    .iter()
                    .filter_map(|x| {
                        x.name
                            .strip_prefix("f_rest_")
                            .and_then(|x| x.parse::<u32>().ok())
                    })
                    .max()
                    .unwrap_or(0)
                    / 3) as usize;

                let mut means = Vec::with_capacity(update_every * 3);
                let mut sh_coeffs = Vec::with_capacity(update_every * sh_coeffs_per_channel * 3);
                let mut rotation = Vec::with_capacity(update_every * 4);
                let mut opacity = Vec::with_capacity(update_every);
                let mut scales = Vec::with_capacity(update_every * 3);

                for i in 0..element.count {
                    let splat = match header.encoding {
                        ply_rs::ply::Encoding::Ascii => {
                            let mut line = String::new();
                            reader.read_line(&mut line)?;
                            gaussian_parser.read_ascii_element(&line, element)?
                        }
                        ply_rs::ply::Encoding::BinaryBigEndian => {
                            gaussian_parser.read_big_endian_element(&mut reader, element)?
                        }
                        ply_rs::ply::Encoding::BinaryLittleEndian => {
                            gaussian_parser.read_little_endian_element(&mut reader, element)?
                        }
                    };

                    let mut sh_coeffs_interleaved = interleave_coeffs(splat.sh_dc, &splat.sh_coeffs);

                    // Limit the numer of SH channels for now.
                    let max_sh_len = num_sh_coeffs(2) * 3;
                    if sh_coeffs_interleaved.len() > max_sh_len {
                        sh_coeffs_interleaved.truncate(max_sh_len);
                    }

                    means.extend(splat.means);
                    sh_coeffs.extend(sh_coeffs_interleaved);
                    rotation.extend(splat.rotation);
                    opacity.push(splat.opacity);
                    scales.extend(splat.scale);

                    // Occasionally send some updated splats.
                    if i % update_every == update_every - 1 {
                        update_splats(
                            &mut splats,
                            means.clone(),
                            sh_coeffs.clone(),
                            rotation.clone(),
                            opacity.clone(),
                            scales.clone(),
                            &device,
                        );
                        means.clear();
                        sh_coeffs.clear();
                        rotation.clear();
                        opacity.clear();
                        scales.clear();

                        yield splats.clone().context("Failed to update splats")?;
                    }
                }

                update_splats(
                    &mut splats,
                    means,
                    sh_coeffs,
                    rotation,
                    opacity,
                    scales,
                    &device,
                );

                if let Some(splats) = splats.as_ref() {
                    if splats.num_splats() == 0 {
                        Err(anyhow::anyhow!("No splats found"))?;
                    }
                }

                yield splats.clone().context("Invalid ply file.")?;
            }
        }
    }
}
