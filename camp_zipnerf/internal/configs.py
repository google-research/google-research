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

from collections.abc import Mapping, Sequence
import dataclasses
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from absl import flags
import flax
import gin
from internal import camera_delta
from internal import math
from internal import utils
import jax
import jax.numpy as jnp
import optax


# A bounding box defined as a tuple containing (min_coord, max_coord).
BboxType = tuple[tuple[float, float, float], tuple[float, float, float]]
FrozenDict = flax.core.frozen_dict.FrozenDict

configurables = {
    'math': [math.noop, math.power_ladder, math.create_learning_rate_decay],
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
    'optax': [
        optax.adam,
        optax.sgd,
        optax.adamw,
        optax.warmup_exponential_decay_schedule,
        optax.warmup_cosine_decay_schedule,
        optax.linear_schedule,
        optax.constant_schedule,
        optax.polynomial_schedule,
        optax.join_schedules,
        optax.piecewise_constant_schedule,
        optax.piecewise_interpolate_schedule,
    ],
    'camera_delta': camera_delta.CAMERA_DELTA_CLASSES,
}

for module, configurables in configurables.items():
  for configurable in configurables:
    gin.config.external_configurable(configurable, module=module)


# CallDef is a construct that makes it easier to use callables with arguments
# in Gin configs. A CallDef is simply a tuple containing a callable and keyword
# arguments the callable should be called with.
#
# See: `parse_call_def` and `parse_call_def_partial`.
#
# Example:
#   ```
#   >> def add(a, b):
#   >>   return a + b
#
#   >> call_def = (add, {'a': 1, 'b': 2})
#   >> config_utils.parse_call_def(call_def)
#   3
#   ```
CallDef = tuple[Callable[Ellipsis, Any], Mapping[str, Any]]


def parse_call_def(call_def):
  """Parses a function call definition.

  Args:
    call_def: A tuple containing (fn, kwargs).

  Returns:
    The result of `fn(**kwargs)`.
  """
  fn, kwargs = call_def
  return fn(**kwargs)


def parse_call_def_partial(call_def):
  """Parses a function call definition partially.

  Parses a CallDef, but instead of evaluating the function immediately,
  return a partial function with the given kwargs.

  Args:
    call_def: A tuple containing (fn, kwargs).

  Returns:
    A partial function `fn(**kwargs)`.
  """
  fn, kwargs = call_def
  return functools.partial(fn, **kwargs)


@gin.configurable
def join_schedule_defs(
    schedule_defs, boundaries
):
  """A gin configurable wrapper around `optax.join_schedules`."""
  schedules = [parse_call_def(s) for s in schedule_defs]
  return optax.join_schedules(schedules, boundaries)


@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""

  debug_mode: bool = False  # If True, compute some expensive debug outputs.
  dataset_loader: str = 'llff'  # The type of dataset loader to use.
  batching: str = 'all_images'  # Batch composition, [single_image, all_images].
  batch_size: int = 16384  # The number of rays/pixels in each batch.
  patch_size: int = 1  # Resolution of patches sampled for training batches.
  factor: int = 0  # The downsample factor of images, 0 for no downsampling.
  # Integer downsampling factors to use for multiscale training.
  # Note 1 is included by default! Use [2, 4, 8] for mip-NeRF 2021 convention.
  multiscale_train_factors: Optional[List[int]] = None
  # Load images in COLMAP vs alphabetical ordering (affects heldout test set).
  load_alphabetical: bool = True
  forward_facing: bool = False  # Set to True for forward-facing LLFF captures.
  # Function for transforming loaded poses in non-forward-facing scenes.
  transform_poses_fn: Optional[Callable[Ellipsis, Any]] = None
  # If True, training cameras will be set to the identity.
  use_identity_cameras: bool = False
  use_perturbed_cameras: bool = False
  camera_perturb_sigma_look_at: float = 0.0
  camera_perturb_sigma_position: float = 0.0
  camera_perturb_sigma_dolly_z: float = 0.0
  camera_perturb_sigma_focal_length: float = 0.0
  camera_perturb_intrinsic_single: bool = True
  camera_perturb_zero_distortion: bool = False
  camera_perturb_dolly_use_average: bool = False

  render_path: bool = False  # If True, render a path. Used only by LLFF.
  llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
  # If true, use all input images for training.
  llff_load_from_poses_bounds: bool = False  # If True, load camera poses of
  # LLFF data from poses_bounds.npy.
  llff_use_all_images_for_training: bool = False
  load_ngp_format_poses: bool = False  # Use `transforms.json` file for poses.
  # Use `metadata.json` for new ARCore poses, `original_metadata.json` for old.
  arcore_format_pose_file: Optional[str] = None
  colmap_subdir: Optional[str] = None  # Where to find COLMAP pose data.
  image_subdir: Optional[str] = None  # Where to find image data.
  load_colmap_points: bool = False
  use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
  use_exrs: bool = False  # If True, use EXR images. Used only by Blender.
  compute_disp_metrics: bool = False  # If True, load and compute disparity MSE.
  compute_normal_metrics: bool = False  # If True, load and compute normal MAE.
  gc_every: int = 10000  # The number of steps between garbage collections.
  disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
  dtu_light_cond: int = 3  # Which DTU dataset lighting condition to load.

  randomized: bool = True  # Use randomized stratified sampling.
  near: float = 2.0  # Near plane distance.
  far: float = 6.0  # Far plane distance.
  # Bounding box for object. Either a single float x representing box from -x to
  # +x on each axis, or a pair of 3D points representing the box corners.
  scene_bbox: None | float | BboxType = None
  # Near and far plane distance in meters. If not None, calibration images are
  # used for conversion to scene units.
  near_plane_meters: Optional[float] = None
  far_plane_meters: Optional[float] = None
  checkpoint_dir: Optional[str] = None  # Where to log checkpoints.
  render_dir: Optional[str] = None  # Output rendering directory.
  data_dir: Optional[str] = None  # Input data directory.
  vocab_tree_path: Optional[str] = None  # Path to vocab tree for COLMAP.
  render_chunk_size: int = 32768
  num_showcase_images: int = 5  # The number of test-set images to showcase.
  deterministic_showcase: bool = True  # If True, showcase the same images.
  vis_num_rays: int = 16  # The number of rays to visualize.
  # Decimate images for tensorboard (ie, x[::d, ::d]) to conserve memory usage.
  vis_decimate: int = 0

  # Only used by train.py:
  max_steps: int = 250000  # The number of optimization steps.
  early_exit_steps: Optional[int] = None  # Early stopping, for debugging.
  visualize_every: int = 25000  # How many steps between model visualizations.
  checkpoint_every: int = 25000  # The number of steps between checkpoint saves.
  checkpoint_keep: int = 2  # Keep the last N checkpoints saved to disk.
  checkpoint_init: bool = False  # If True, checkpoint upon initialization.
  print_every: int = 100  # The number of steps between reports to tensorboard.
  print_camera_every: int = 500  # The number of steps between camera reports.
  train_render_every: int = 0  # Steps between test set renders when training
  jax_rng_seed: int = 20200823  # The seed that JAX's RNG uses.
  np_rng_seed: int = 20201473  # The seed that Numpy's RNG uses.
  donate_args_to_train: bool = True  # Train step overwrites previous state.
  disable_pmap_and_jit: bool = False  # If True disable the training pmap.
  cast_rays_in_train_step: bool = False  # If True, compute rays in train step.
  cast_rays_in_eval_step: bool = False  # If True, compute rays in eval step.
  data_loss_type: str = 'charb'  # What kind of loss to use ('mse' or 'charb').
  charb_padding: float = 0.001  # The padding used for Charbonnier loss.
  data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
  data_coarse_loss_mult: float = 0.0  # Multiplier for the coarser data terms.
  spline_interlevel_params: FrozenDict[str, Any] = FrozenDict(
      {'mults': 0.01, 'blurs': (0.03, 0.003)}
  )

  orientation_loss_mult: float = 0.0  # Multiplier on the orientation loss.
  orientation_coarse_loss_mult: float = 0.0  # Coarser orientation loss weights.
  # What that loss is imposed on, options are 'normals' or 'normals_pred'.
  orientation_loss_target: str = 'normals_pred'
  predicted_normal_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
  # Mult. on the coarser predicted normal loss.
  predicted_normal_coarse_loss_mult: float = 0.0
  param_regularizers: FrozenDict[str, Any] = FrozenDict({})
  # An example of total L2 loss (weight decay) on the NeRF MLP and average
  # Geman-McClure loss on the first layer of the proposal MLP:
  #   param_regularizers = {
  #       'NerfMLP_0': (0.00001, @jnp.sum, 2, 1),
  #       'PropMLP_0/Dense_0': (0.01, @jnp.mean, -2, 1),
  #   }
  # Any model parameter that isn't specified gets a multiplier of 0. See the
  # train_weight_l2_* parameters in TensorBoard to know what can be regularized.
  # The hyperparameters are of the form (mult, alpha, scale) that parameterize
  # a general robust loss, see https://arxiv.org/abs/1701.03077 for details.
  robust_loss_scale: float = 0.01

  eikonal_loss_mult: float = 0.0  # Multiplier on the eikonal loss.
  eikonal_coarse_loss_mult: float = 0.0  # Multiplier on the coarser eikonal.
  lr_init: float = 0.002  # The initial learning rate.
  lr_final: float = 0.00002  # The final learning rate.
  lr_init_grid: Optional[float] = None  # Initial learning rate for grid vars.
  lr_final_grid: Optional[float] = None  # Final learning rate for grid vars.
  lr_delay_steps: int = 512  # The number of "warmup" learning steps.
  lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
  adam_beta1: float = 0.9  # Adam's beta2 hyperparameter.
  adam_beta2: float = 0.999  # Adam's beta2 hyperparameter.
  adam_eps: float = 1e-6  # Adam's epsilon hyperparameter.
  grad_max_norm: float = 0.001  # Gradient clipping magnitude, disabled if == 0.
  grad_max_val: float = 0.0  # Gradient clipping value, disabled if == 0.
  distortion_loss_target: str = 'sdist'  # The distance that distortion uses.
  distortion_loss_mult: float = 0.01  # Multiplier on the distortion loss.
  # The curve applied to distortion_loss_target before computing distortion of
  # the form (fn, **kwargs), like (@math.power_ladder, {'p':-2, 'premult':10}).
  distortion_loss_curve_fn: Optional[
      Tuple[Callable[Ellipsis, Any], Dict[str, Any]]
  ] = None

  # Only used by eval.py:
  eval_checkpoint_wait_timeout_sec: float = float(
      'inf'
  )  # Wait for a new checkpoint and die if there is no checkpoint.
  eval_only_once: bool = True  # If True evaluate the model only once, ow loop.
  eval_save_output: bool = True  # If True save predicted images to disk.
  eval_save_ray_data: bool = False  # If True save individual ray traces.
  eval_render_interval: int = 1  # The interval between images saved to disk.
  eval_dataset_limit: int = jnp.iinfo(jnp.int32).max  # Num test images to eval.
  eval_quantize_metrics: bool = True  # If True, run metrics on 8-bit images.
  eval_crop_borders: int = 0  # Ignore c border pixels in eval (x[c:-c, c:-c]).

  # Only used by render.py
  render_video_fps: int = 60  # Framerate in frames-per-second.
  render_video_crf: int = 18  # Constant rate factor for ffmpeg video quality.
  render_path_frames: int = 120  # Number of frames in render path.
  z_variation: float = 0.0  # How much height variation in render path.
  z_phase: float = 0.0  # Phase offset for height variation in render path.
  rad_mult_min: float = 1.0  # How close to get to the object, relative to 1.
  rad_mult_max: float = 1.0  # How far to get from the object, relative to 1.
  render_rotate_xaxis: float = 0.0  # Rotate camera around x axis.
  render_rotate_yaxis: float = 0.0  # Rotate camera around y axis.
  lock_up: bool = False  # If True, locks the up axis (good for sideways paths).
  render_dist_adaptive: bool = False  # If False, use (config.near, config.far).
  render_dist_percentile: float = 0.5  # The near/far percentile, when adaptive.
  # Parameters for math.power_ladder that curve distance before visualization.
  render_dist_vis_params: FrozenDict[str, Any] = FrozenDict(
      {'p': -1.5, 'premult': 2}
  )
  render_path_file: Optional[str] = None  # Numpy render pose file to load.
  render_rgb_only: bool = False  # Render spherical 360 panoramas.
  # Render resolution, as (width, height).
  render_resolution: Optional[Tuple[int, int]] = None
  render_focal: Optional[float] = None  # Render focal length.
  render_camtype: Optional[str] = None  # 'perspective', 'fisheye', or 'pano'.
  render_spherical: bool = False  # Render spherical 360 panoramas.
  # Text file containing names of images to be used as spline keyframes, OR
  # directory containing those images.
  render_spline_keyframes: Optional[str] = None
  # Comma-separated list of possible values for option
  # "render_spline_keyframes". If set, the render pipeline will be executed
  # once per entry, overwriting "render_spline_keyframes" in the process.
  render_spline_keyframes_choices: Optional[str] = None
  render_spline_n_interp: int = 30  # Num. frames to interpolate per keyframe.
  render_spline_degree: int = 5  # Polynomial degree of B-spline interpolation.
  render_spline_lock_up: bool = False  # If True, no up/down tilt in path.
  # B-spline smoothing factor, 0 for exact interpolation of keyframes.
  # Interpolate per-frame exposure value from spline keyframes.
  render_spline_smoothness: float = 0.03
  # Weight associated with rotation dimensions. Higher values means preserving
  # view direction is more important than camera position. Must be >0.
  render_spline_rot_weight: float = 0.1
  render_spline_interpolate_exposure_smoothness: int = 20
  render_spline_interpolate_exposure: bool = False
  render_spline_lookahead_i: Optional[int] = None
  render_spline_fixed_up: bool = False
  render_spline_meters_per_sec: Optional[float] = None
  # If both parameters below are specified, spline keyframes that are far from
  # their neighbors will be ignored.
  render_spline_outlier_keyframe_quantile: Optional[float] = None
  render_spline_outlier_keyframe_multiplier: Optional[float] = None
  # Text file or directory with image pairs for calibrating metric scale.
  render_calibration_keyframes: Optional[str] = None
  render_calibration_distance: float = 3.0  # Default calibration is 3 meters.
  save_calibration_to_disk: bool = False  # Save calibration to a text file.
  render_spline_const_speed: bool = False  # Retime spline to have const speed.
  render_spline_n_buffer: Optional[int] = None  # Extra keyframes for path.
  # A tuple of video formats to render. Accepted formats: 'mp4', 'webm', 'gif'.
  render_video_exts: Tuple[str, Ellipsis] = ('mp4',)
  # Whether or not to delete the still images after rendering a video.
  render_delete_images_when_done: bool = True
  # Whether or not videos should be rendered looped (going forwards then the
  # same way backwards)
  render_looped_videos: bool = False

  # Only used by MetricHarness.
  metric_harness_train_config: FrozenDict[str, Any] = FrozenDict({
      'disable_ssim': True,
  })
  metric_harness_eval_config: FrozenDict[str, Any] = FrozenDict({})

  # Parameters for the local color correction used in evaluating the color
  # corrected error metrics. Note that increasing any of these numbers is
  # virtually guaranteed to improve all error metrics, so these parameter should
  # be tuned by visual inspection.
  color_correction_config: FrozenDict[str, Union[int, Tuple[int, int]]] = (
      FrozenDict({
          'num_spatial_bins': [6, 10],
          'num_luma_bins': 9,
          'num_chroma_bins': 3,
      })
  )

  # Flags for raw datasets.
  rawnerf_mode: bool = False  # Load raw images and train in raw color space.
  exposure_percentile: float = 97.0  # Image percentile to expose as white.
  # During training, discard N-pixel border around each input image.
  num_border_pixels_to_mask: int = 0
  autoexpose_renders: bool = False  # During rendering, autoexpose each image.
  # For raw test scenes, use affine raw-space color correction.
  eval_raw_affine_cc: bool = False

  # Flags for aerial datasets.
  world_scale: float = 1.0  # Camera positions are divided by this quantity.
  z_min: Optional[float] = None  # Rays end at this Z value.
  z_max: Optional[float] = None  # Rays start at this Z value.

  # Loss-scaling related values
  enable_loss_scaler: bool = False
  loss_scale: float = 1000.0

  optimize_cameras: bool = False
  camera_delta_cls: Type[camera_delta.CameraDelta] = (
      camera_delta.FocalPoseCameraDelta
  )
  camera_optimizer: Callable[Ellipsis, Any] = optax.adam
  camera_optimizer_kwargs: Mapping[str, Any] = FrozenDict({})
  camera_lr_schedule_def: CallDef = (
      math.create_learning_rate_decay,
      {
          'lr_init': 1e-3,
          'lr_final': 1e-4,
          'lr_delay_steps': 2500,
          'lr_delay_mult': 1e-8,
          'max_steps': 25000,
      },
  )
  camera_lr_fn: Callable[Ellipsis, Any] = optax.warmup_cosine_decay_schedule
  camera_lr_fn_kwargs: Mapping[str, Any] = FrozenDict({
      'init_value': 0.0,
      'peak_value': 1e-4,
      'warmup_steps': 200,
      'decay_steps': 5800,
      'end_value': 1e-4,
  })

  # If True, use coarse-to-fine which will be applied to the grid optimizer as
  # an additional scale to the learning rate.
  enable_grid_c2f: bool = False
  # The grid size containing the whole -2 to 2 volume, including the contracted
  # area.
  grid_c2f_resolution_schedule_def: CallDef = (
      optax.linear_schedule,
      {
          'init_value': 1024,
          'end_value': 8192,
          'transition_steps': 2500,
          'transition_begin': 0,
      },
  )
  grid_c2f_weight_method: str = 'cosine_sequential'

  focal_length_var_loss_mult: float = 0.0
  principal_point_var_loss_mult: float = 0.0
  principal_point_reg_loss_mult: float = 0.0
  radial_distortion_var_loss_mult: float = 0.0

  # Flags for test time camera alignment.
  optimize_test_cameras: bool = False
  optimize_test_cameras_for_n_steps: int = 200  # Gradient descent iterations.
  optimize_test_cameras_lr: float = 0.001
  optimize_test_cameras_batch_size: int = 10000
  test_camera_delta_cls: Type[camera_delta.CameraDelta] = (
      camera_delta.SE3CameraDelta
  )
  compute_procrustes_metric: bool = False


def define_common_flags():
  # Define the flags used by both train.py and eval.py
  flags.DEFINE_string('mode', None, 'Required by GINXM, not used.')
  flags.DEFINE_string('base_folder', None, 'Required by GINXM, not used.')
  flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
  flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')
  flags.DEFINE_bool('is_xm_sweep', False, 'Whether the run is an xm sweep.')


def load_config(save_config = True):
  """Loads the config, and optionally checkpoints it."""
  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True
  )
  config = Config()
  if save_config and jax.host_id() == 0:
    utils.makedirs(config.checkpoint_dir)
    with utils.open_file(config.checkpoint_dir + '/config.gin', 'w') as f:
      f.write(gin.config_str())
  return config
