# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for handling configurations."""
# pylint: disable=line-too-long
# pylint: disable=g-importing-member

import dataclasses
from dataclasses import field
from os import path
from typing import Any, Optional, Tuple

from absl import flags
from flax.core import FrozenDict
import gin
import jax
import jax.numpy as jnp
from smerf.internal import schedule
from smerf.internal import utils


# Note that this is intended to complete the original list in
configurables = {
    'jax.nn': [jax.nn.elu],
}

for module, configurables in configurables.items():
  for configurable in configurables:
    gin.config.external_configurable(configurable, module=module)


# By setting eq=False, this object's hash becomes its id(). This is a
# prerequisite for its use as a static_argnum argument in jax.jit() or
# jax.pmap().
@gin.configurable()
@dataclasses.dataclass(eq=False)
class Config:
  """Configuration flags for everything."""

  # Paths.
  checkpoint_dir: Optional[str] = None  # Where to log checkpoints.
  data_dir: Optional[str] = None  # Input data directory.
  baking_checkpoint_dir: Optional[str] = None  # Where to log baked models.
  # Append WID to checkpoint_dir in HParam sweeps.
  one_checkpoint_dir_per_work_unit: bool = True
  # Append WID to baking_checkpoint_dir in HParam sweeps.
  one_baking_checkpoint_dir_per_work_unit: bool = True

  # Representation.
  triplane_resolution: int = 2048  # Planes will have dimensions (T, T) where
  # T = triplane_resolution.
  sparse_grid_resolution: int = 512  # Voxel grid will have dimensions (S, S, S)
  # Number of submodels to use. Use this to create subvolumes, each owning
  # its own model parameters. Set to 1 to reproduce MERF.
  submodel_grid_resolution: int = 1

  # where S = sparse_grid_resolutiondistill_teacher_use_rng.
  num_samples_per_voxel: int = 1  # Only affects rendering from the baked
  # representation.
  data_block_size: int = 8  # Block size for the block-sparse 3D grid
  # (see SNeRG).
  range_features: Tuple[float, float] = field(
      default_factory=lambda: (-7.0, 7.0)
  )  # Value range for appearance features.
  range_density: Tuple[float, float] = field(
      default_factory=lambda: (-14.0, 14.0)
  )  # Value range for density features.
  # Controls if teacher is exposure-aware. If None, takes on the same value as
  # use_exposure_in_deferred_mlp for backward compatibility.
  use_exposure_in_teacher: Optional[bool] = None
  # If True, the student model is exposure-aware.
  use_exposure_in_deferred_mlp: bool = False
  # How the outputs of the DeferredMLP are used in generating the final color
  # for a pixel.
  deferred_rendering_mode: str = 'snerg'  # [snerg, vfr]
  # If True, use the last entry in the sparse grid feature vector as a
  # weight for the triplane contribution to the feature preactivation.
  use_low_res_features_as_weights: bool = False
  # If True, use the last entry in the sparse grid feature vector as a
  # weight for the triplane contribution to the density preactivation.
  use_triplane_weights_for_density: bool = False
  # How interpolated triplane & sparse voxel feature vectors are combined
  # before the activation function.
  merge_features_combine_op: str = 'sum'  # [sum, coarse_sum, cross_product_sum]

  # Extra latent feature grids for better view dependence.
  num_viewdir_features: int = 0
  num_origin_features: int = 0
  viewdir_grid_size: int = 16
  origin_grid_size: int = 4

  # Submodel options
  submodel_enable_multimlp: bool = True  # Each submodel has its own MLPs.
  submodel_idx_replace_percent: float = 0.0  # Probability of replacing a ray's sm_idx
  submodel_idx_replace_percent_3d: float = 0.0  # Probability of replacing a point's sm_idx
  submodel_idx_override: Optional[int] = None  # Train one submodel here.
  submodel_scale_factor: float = 1.0  # Scale for world-to-submodel transform.

  # Control flow.
  max_steps: int = 25000  # Number of optimization steps.
  batch_size: int = 65536  # The number of rays/pixels in each batch.
  render_chunk_size: int = 65536  # Chunk size for whole-image renderings.
  checkpoint_every: int = 5000  # Steps to save a checkpoint.
  print_every: int = 100  # Steps between printing losses.
  train_render_every: int = 500  # Steps between validation renders
  cast_rays_in_train_step: bool = True  # If True, compute rays in train step.
  gradient_accumulation_steps: int = 8  # Increase this value when running OOM.
  model_seed: int = 6550634  # This seed is used to initalize model parameters.
  path_video_every: int = 1  # DEPRECATED. Do not use.
  render_path_video_every: int = 1  # Render every nth frame in render_path video
  enable_train: bool = True  # Train model
  enable_eval: bool = True  # Render test set
  enable_video: bool = True  # Render ellipse video
  enable_render_path_video: bool = True  # Render poses from "render_path_file.npy"
  enable_baking: bool = True  # Bake for realtime viewer
  enable_baked_export: bool = True  # Export baked assets for realtime viewer
  enable_baked_eval: bool = True  # Render baked submodels.
  enable_baked_video: bool = True  # Render baked ellipse videos.

  # Video rendering
  baked_render_path_video_every: int = 1  # Render every nth frame in baked render path video.
  baked_render_path_all_cameras: bool = False  # Render all cameras with all submodels

  # Baking
  baking_force_alpha_culling: Optional[bool] = None  # Enable/disable alpha culling.
  baking_max_distance_to_submodel_origin: Optional[float] = None  # world units, inf-norm
  baking_max_images_to_render: Optional[int] = None  # Render no more than this many images for alive_voxels.
  baking_alive_voxels_median_filter_size: Optional[int] = None  # Size of filter to apply to alive_voxels.
  baking_alive_voxels_median_chunk_size: int = 128  # Apply median filter to cubes of shape size^3.
  baking_alive_voxels_max_chunk_size: int = 128  # Apply max filter to cubes of shape size^3.
  baking_triplane_features_buffer: Optional[int] = None  # Number of texels around each alive texel to activate.
  baking_enable_ray_jitter: bool = False  # Jitter rays during baking.
  baking_subsample_factor: Optional[int] = None  # Further downsample images at baking time
  baking_occupancy_grid_downsample_factors: tuple[int, Ellipsis] = (2, 4, 8, 16, 32, 64, 128)
  baking_distance_grid_downsample_factors: tuple[int, Ellipsis] = (4, 8, 16, 32, 64, 128)

  # Export
  export_apply_zdiff_to_slices: bool = False
  export_array_format: str = 'png'  # [png, raw.gz, npy.gz]
  export_slice_occupancy_and_distance_grids: bool = False  # Write grids as slices.
  export_pad_occupancy_and_distance_grids: bool = True  # Replicate each entry to RGBA
  export_store_deferred_mlp_separately: bool = False  # store DeferredMLP params in a separate file.
  export_store_rgb_and_density_separately: bool = False  # store RGB and density in two separate arrays.

  # Loss weights.
  data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
  charb_padding: float = 0.001  # The padding used for Charbonnier loss.
  interlevel_loss_mult: float = 1.0  # Mult. for the loss on the proposal MLP.
  distortion_loss_mult: float = 0.01  # Multiplier on the distortion loss.
  yu_sparsity_loss_mult: Optional[schedule.Schedule] = schedule.ConstSchedule(
      0.01
  )  # Multiplier for sparsity loss.
  distill_teacher_ckpt_dir: Optional[str] = None  # Where to find the teacher.
  distill_teacher_use_rng: bool = False  # Use rng during teacher rendering?
  distill_use_teacher_tdist: bool = True  # Use tdist values from teacher. This is for backwards compatibility.
  distill_use_teacher_exposure: bool = True  # Use exposure values from teacher if available. This is for backwards compatibility.
  distill_geometry_loss_mult_init: float = 0.0
  distill_geometry_loss_mult_final: float = 0.0
  distill_geometry_loss_mode: str = 'both'  # lower, upper, or both.
  distill_rgb_loss_mult: float = 0.0
  distill_rgb_loss_fn: str = 'rmse'  # [rmse, l2, l1, cauchy]
  distill_ssim_loss_mult: float = 0.0
  distill_weight_power: float = 0.0  # Downweight pixels deep in contracted space.
  distill_shift_origins_forward: bool = False  # Move ray origins forward along camera ray?
  distill_shift_origins_forward_buffer: float = 1.0  # How far forward can the camera move?
  distill_shift_origins_forward_jitter_viewdirs: bool = False  # Jitter directions again?
  antiflicker_rgb_loss_mult: float = 0.0
  antiflicker_enable_stopgrad: bool = False
  antiflicker_features_loss_mult: float = 0.0
  num_random_samples: int = 2**17  # For sparsity loss
  num_random_student_intervals_per_ray: int = 0  # For floater removal
  alpha_threshold: Optional[schedule.Schedule] = schedule.LogLerpSchedule(
      start=10000, end=20000, v0=0.0005, v1=0.005, zero_before_start=True
  )  # Multiplier for alpha-culling-simulation loss.
  # (mult, acc_fn, alpha, scale) adds the following to the loss,
  #   loss += mult * acc_fn(lossfun(param, alpha, scale)
  # for each parameter with a prefix matching the given key.
  param_regularizers: FrozenDict[str, Any] = FrozenDict({
      'MultiDensityAndFeaturesMLP_0/MultiHashEncoding_0': (
          0.03,
          jnp.mean,
          2,
          1,
      ),
      'MultiPropMLP_0/MultiPropHashEncoding_0': (
          0.03,
          jnp.mean,
          2,
          1,
      ),
  })  # Fine-grained parameter regularization strength.
  # Multiplier for a regularizer specifically designed for the DeferredMLP.
  # Penalizes parameters from spatially adjacent MLPs from varying.
  deferred_mlp_tv_regularizer_loss_mult: float = 0.0
  deferred_mlp_tv_regularizer_loss_fn: str = 'l1'  # [l1, l2]

  # Optimization.
  lr_init: float = 1e-2  # The initial learning rate.
  lr_final: float = 1e-3  # The final learning rate.
  lr_delay_steps: int = 100  # The number of "warmup" learning steps.
  lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
  adam_beta1: float = 0.9  # Adam's beta2 hyperparameter.
  adam_beta2: float = 0.99  # Adam's beta2 hyperparameter.
  adam_eps: float = 1e-15  # Adam's epsilon hyperparameter.
  grad_max_norm: float = 0.001  # Gradient clipping magnitude, disabled if == 0.
  grad_max_val: float = 0.0  # Gradient clipping value, disabled if == 0.
  enable_ray_jitter: bool = False  # Jitter ray origin, direction, and viewdir.
  ray_jitter_origin_stddev: float = 0.01
  ray_jitter_origin_patch_aware: bool = True  # Apply same noise to entire patch
  ray_jitter_viewdir_stddev: float = 0.03
  ray_jitter_viewdir_patch_aware: bool = True  # Apply same noise to entire patch

  # Data loading.
  dataset_loader: str = 'llff'  # The type of dataset loader to use.
  batching: str = 'all_images'  # Batch composition, [single_image, all_images].
  patch_size: int = 1  # Resolution of patches sampled for training batches.
  factor: int = 4  # The downsample factor of images, 0 for no downsampling.
  # Load images in COLMAP vs alphabetical ordering (affects heldout test set).
  load_alphabetical: bool = True
  forward_facing: bool = False  # Set to True for forward-facing LLFF captures.
  llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
  # If true, use all input images for training.
  llff_load_from_poses_bounds: bool = False  # If True, load camera poses of
  # LLFF data from poses_bounds.npy.
  llff_use_all_images_for_training: bool = False
  use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
  randomized: bool = True  # Use randomized stratified sampling.
  near: float = 0.2  # Near plane distance.
  far: float = 1e6  # Far plane distance.
  vocab_tree_path: Optional[str] = None  # Path to vocab tree for COLMAP.

  # DO NOT SET THIS DIRECTLY! It should be populated by
  # grid_utils.initialize_grid_config()
  grid_config: FrozenDict[str, Any] = FrozenDict({})
  exposure_config: FrozenDict[str, Any] = FrozenDict({})


def define_common_flags():
  flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
  flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')


def load_config(save_config=True):
  """Load the config, and optionally checkpoint it."""
  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True
  )
  config = Config()
  if save_config and jax.host_id() == 0:
    utils.makedirs(config.checkpoint_dir)
    with utils.open_file(
        path.join(config.checkpoint_dir, 'config.gin'), 'w'
    ) as f:
      f.write(gin.config_str())
  return config


def use_exposure_in_teacher(config):
  """Determines whether or not teacher is allowed to access exposure info."""
  if config.use_exposure_in_teacher is not None:
    return config.use_exposure_in_teacher
  return config.use_exposure_in_deferred_mlp

