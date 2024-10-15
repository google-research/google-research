#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]

mod burn_texture;
mod orbit_controls;
mod splat_import;

#[cfg(feature = "rerun")]
mod visualize;

pub mod viewer;
pub mod wgpu_config;
