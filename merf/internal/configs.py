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

  # pylint: disable=g-importing-member

import dataclasses
from dataclasses import field
from os import path
from typing import Any, Optional, Tuple

from absl import flags
from flax.core import FrozenDict
import gin
from internal import schedule
from internal import utils
import jax
import jax.numpy as jnp

configurables = {
    'jnp': [
        jnp.reciprocal,
        jnp.log,
        jnp.log1p,
        jnp.exp,
        jnp.sqrt,
        jnp.square,
        jnp.sum,
        jnp.mean,
    ],
    'jax.nn': [jax.nn.relu, jax.nn.softplus, jax.nn.silu],
    'jax.nn.initializers.he_normal': [jax.nn.initializers.he_normal()],
    'jax.nn.initializers.he_uniform': [jax.nn.initializers.he_uniform()],
    'jax.nn.initializers.glorot_normal': [jax.nn.initializers.glorot_normal()],
    'jax.nn.initializers.glorot_uniform': [
        jax.nn.initializers.glorot_uniform()
    ],
}

for module, configurables in configurables.items():
  for configurable in configurables:
    gin.config.external_configurable(configurable, module=module)


@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""

  # Paths.
  checkpoint_dir: Optional[str] = None  # Where to log checkpoints.
  data_dir: Optional[str] = None  # Input data directory.

  # Representation.
  triplane_resolution: int = 2048  # Planes will have dimensions (T, T) where
  # T = triplane_resolution.
  sparse_grid_resolution: int = 512  # Voxel grid will have dimensions (S, S, S)
  # where S = sparse_grid_resolution.
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

  # Control flow.
  max_steps: int = 25000  # Number of optimization steps.
  batch_size: int = 65536  # The number of rays/pixels in each batch.
  render_chunk_size: int = 65536  # Chunk size for whole-image renderings.
  checkpoint_every: int = 5000  # Steps to save a checkpoint.
  print_every: int = 100  # Steps between printing losses.
  train_render_every: int = 500  # Steps between validation renders
  cast_rays_in_train_step: bool = True  # If True, compute rays in train step.
  gradient_accumulation_steps: int = 8  # Increase this value when running OOM.
  stop_after_training: bool = False
  stop_after_testing: bool = False
  stop_after_compute_alive_voxels: bool = False
  render_train_set: bool = False
  model_seed: int = 6550634  # This seed is used to initalize model parameters.

  # Loss weights.
  data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
  charb_padding: float = 0.001  # The padding used for Charbonnier loss.
  interlevel_loss_mult: float = 1.0  # Mult. for the loss on the proposal MLP.
  distortion_loss_mult: float = 0.01  # Multiplier on the distortion loss.
  yu_sparsity_loss_mult: Optional[schedule.Schedule] = schedule.ConstSchedule(
      0.01
  )  # Multiplier for sparsity loss.
  num_random_samples: int = 2**17  # For sparsity loss
  alpha_threshold: Optional[schedule.Schedule] = schedule.LogLerpSchedule(
      start=10000, end=20000, v0=0.0005, v1=0.005, zero_before_start=True
  )  # Multiplier for alpha-culling-simulation loss.
  param_regularizers: FrozenDict[str, Any] = FrozenDict({
      'DensityAndFeaturesMLP_0/HashEncoding_0': (0.03, jnp.mean, 2, 1),
      'PropMLP_0/PropHashEncoding_0': (0.03, jnp.mean, 2, 1),
  })  # Fine-grained parameter regularization strength.

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
