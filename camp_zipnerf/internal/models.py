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

"""NeRF and its MLPs, with helper functions for construction and rendering."""

import functools
import time
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging
from flax import linen as nn
import gin
from internal import configs
from internal import coord
from internal import geopoly
from internal import grid_utils
from internal import image_utils
from internal import math
from internal import ref_utils
from internal import render
from internal import stepfun
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections


gin.config.external_configurable(math.safe_exp, module='math')
gin.config.external_configurable(math.laplace_cdf, module='math')
gin.config.external_configurable(math.scaled_softplus, module='math')
gin.config.external_configurable(math.power_ladder, module='math')
gin.config.external_configurable(math.inv_power_ladder, module='math')
gin.config.external_configurable(coord.contract, module='coord')


def random_split(rng):
  if rng is None:
    key = None
  else:
    key, rng = random.split(rng)
  return key, rng


@gin.configurable
class Model(nn.Module):
  """A mip-Nerf360 model containing all MLPs."""

  config: Any = None  # A Config class, must be set upon construction.
  # A list of tuples (mlp_idx, grid_idx, num_samples) for each sampling round.
  # This code defaults to what the mip-NeRF 360 codebase used, which was three
  # rounds of sampling using one "proposal" MLP and one "NeRF" MLP and no grids.
  sampling_strategy: Tuple[Tuple[int, int, int], Ellipsis] = (
      (0, None, 64),
      (0, None, 64),
      (1, None, 32),
  )
  # The specific parameters for the MLPs + grids used by this model. The length
  # of these tuples also determines how many MLPs/grids will get constructed.
  # The user must ensure that the number of MLPs/grids matches the config in
  # `sampling_strategy` or else this code will not run.
  mlp_params_per_level: Tuple[ml_collections.FrozenConfigDict, Ellipsis] = (
      {'disable_rgb': True},
      {'disable_rgb': False},
  )
  # Grid is disabled by default.
  grid_params_per_level: Tuple[ml_collections.FrozenConfigDict, Ellipsis] = ()
  bg_intensity_range: Tuple[float, float] = (1.0, 1.0)  # Background RGB range.
  anneal_slope: float = 10  # Higher = more rapid annealing.
  stop_level_grad: bool = True  # If True, don't backprop across levels.
  use_viewdirs: bool = True  # If True, use view directions as input.
  ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
  disable_integration: bool = False  # If True, use PE instead of IPE.
  single_jitter: bool = True  # If True, jitter whole rays instead of samples.
  num_glo_features: int = 0  # GLO vector length, disabled if 0.
  num_glo_embeddings: int = 10000  # Upper bound on max number of train images.
  learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
  near_anneal_rate: Optional[float] = None  # How fast to anneal in near bound.
  near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
  resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
  # The following hyperparameters control beta, which is the scale parameter
  # used for the transformation from SDF to density (see equations 2 and 3 in
  # https://arxiv.org/pdf/2106.12052.pdf).
  scheduled_beta: bool = False  # If True, scheduleding beta rather learning it.
  # The final beta value at each level of sampling.
  final_betas: Tuple[float, Ellipsis] = (1.5e-2, 3.0e-3, 1.0e-3)
  rate_beta: float = 0.75  # Rate for scheduling beta.
  # The curve used for ray distances. Can be just a function like @jnp.log,
  # or can be of the form (fn, fn_inv, **kwargs), like
  # (@math.power_ladder, @math.inv_power_ladder, {'p': -2, 'premult': 10})
  raydist_fn: Union[Tuple[Callable[Ellipsis, Any], Ellipsis], Callable[Ellipsis, Any]] = None
  max_exposure: float = 1.0

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac,
      compute_extras,
      zero_glo=True,
      percentiles = (5, 50, 95),
      train = True,
  ):
    """The mip-NeRF Model.

    Args:
      rng: random number generator (or None for deterministic output).
      rays: util.Rays, a pytree of ray data and metadata.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.
      percentiles: depth will be returned for these percentiles.
      train: Set to True when training.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """

    mlps = [
        MLP(name=f'MLP_{i}', **params)
        for i, params in enumerate(self.mlp_params_per_level)
    ]

    grids = [
        grid_utils.HashEncoding(name=f'grid_{i}', **params)
        for i, params in enumerate(self.grid_params_per_level)
    ]

    if self.num_glo_features > 0:
      if not zero_glo:
        # Construct/grab GLO vectors for the cameras of each input ray.
        glo_vecs = nn.Embed(self.num_glo_embeddings, self.num_glo_features)
        cam_idx = rays.cam_idx[Ellipsis, 0]
        glo_vec = glo_vecs(cam_idx)
      else:
        glo_vec = jnp.zeros(rays.origins.shape[:-1] + (self.num_glo_features,))
    else:
      glo_vec = None

    if self.learned_exposure_scaling:
      # Setup learned scaling factors for output colors.
      # TODO(bmild): fix use of `num_glo_embeddings` here.
      max_num_exposures = self.num_glo_embeddings
      # Initialize the learned scaling offsets at 0.
      init_fn = jax.nn.initializers.zeros
      exposure_scaling_offsets = nn.Embed(
          max_num_exposures,
          features=3,
          embedding_init=init_fn,
          name='exposure_scaling_offsets',
      )

    # Define the mapping from normalized to metric ray distance.
    if isinstance(self.raydist_fn, tuple):
      fn, fn_inv, kwargs = self.raydist_fn  # pylint: disable=unpacking-non-sequence
      _, s_to_t = coord.construct_ray_warps(
          functools.partial(fn, **kwargs),
          rays.near,
          rays.far,
          fn_inv=functools.partial(fn_inv, **kwargs),
      )
    else:
      _, s_to_t = coord.construct_ray_warps(
          self.raydist_fn, rays.near, rays.far
      )

    exposure_values = rays.exposure_values

    # Initialize the range of (normalized) distances for each ray to [0, 1],
    # and assign that single interval a weight of 1. These distances and weights
    # will be repeatedly updated as we proceed through sampling levels.
    # `near_anneal_rate` can be used to anneal in the near bound at the start
    # of training, eg. 0.1 anneals in the bound over the first 10% of training.
    if self.near_anneal_rate is None:
      init_s_near = 0.0
    else:
      init_s_near = jnp.clip(
          1 - train_frac / self.near_anneal_rate, 0, self.near_anneal_init
      )
    init_s_far = 1.0
    sdist = jnp.concatenate(
        [
            jnp.full_like(rays.near, init_s_near),
            jnp.full_like(rays.far, init_s_far),
        ],
        axis=-1,
    )
    weights = jnp.ones_like(rays.near)

    ray_history = []
    renderings = []
    mlp_was_used = [False] * len(mlps)
    grid_was_used = [False] * len(grids)
    for i_level, (i_mlp, i_grid, num_samples) in enumerate(
        self.sampling_strategy
    ):
      mlp = mlps[i_mlp]
      mlp_was_used[i_mlp] = True
      if i_grid is None:
        grid = None
      else:
        grid = grids[i_grid]
        grid_was_used[i_grid] = True

      # Optionally anneal the weights as a function of training iteration.
      if self.anneal_slope > 0:
        # Schlick's bias function, see https://arxiv.org/abs/2010.09714
        bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
        anneal = bias(train_frac, self.anneal_slope)
      else:
        anneal = 1.0

      # A slightly more stable way to compute weights**anneal. If the distance
      # between adjacent intervals is zero then its weight is fixed to 0.
      logits_resample = jnp.where(
          sdist[Ellipsis, 1:] > sdist[Ellipsis, :-1],
          anneal * math.safe_log(weights + self.resample_padding),
          -jnp.inf,
      )

      # Draw sampled intervals from each ray's current weights.
      key, rng = random_split(rng)
      sdist = stepfun.sample_intervals(
          key,
          sdist,
          logits_resample,
          num_samples,
          single_jitter=self.single_jitter,
          domain=(init_s_near, init_s_far),
      )

      # Optimization will usually go nonlinear if you propagate gradients
      # through sampling.
      if self.stop_level_grad:
        sdist = jax.lax.stop_gradient(sdist)

      # Convert normalized distances to metric distances.
      tdist = s_to_t(sdist)

      # Cast our rays, by turning our distance intervals into Gaussians.
      gaussians = render.cast_rays(
          tdist,
          rays.origins,
          rays.directions,
          rays.radii,
          self.ray_shape,
          diag=False,
      )

      if self.disable_integration:
        # Setting the covariance of our Gaussian samples to 0 disables the
        # "integrated" part of integrated positional encoding.
        gaussians = (gaussians[0], jnp.zeros_like(gaussians[1]))

      # Push our Gaussians through the MLP.
      key, rng = random_split(rng)
      curr_beta = None
      if self.scheduled_beta:
        if len(self.final_betas) != len(self.sampling_strategy) or (
            any([beta <= 0.0 for beta in self.final_betas])
        ):
          raise ValueError(
              'Scheduled betas should be given to each level and positive.'
          )
        curr_beta = self.get_scheduled_beta(i_level, train_frac)
      ray_results = mlp(
          key,
          gaussians,
          viewdirs=rays.viewdirs if self.use_viewdirs else None,
          imageplane=rays.imageplane,
          glo_vec=glo_vec,
          exposure=jax.lax.stop_gradient(exposure_values),
          curr_beta=curr_beta,
          grid=grid,
          rays=rays,
          tdist=tdist,
          train=train,
      )

      # Compute "rectified" versions of all all normals, where surfaces facing
      # away from the camera have their sign flipped so that they face the
      # camera (note that flipping the sign of a normal has no effect on the
      # mirrored directions used by ref-nerf).
      rectified = {}
      for key, val in ray_results.items():
        if key.startswith('normals') and val is not None:
          p = jnp.sum(val * rays.viewdirs[Ellipsis, None, :], axis=-1, keepdims=True)
          rectified[key + '_rectified'] = val * jnp.where(p > 0, -1, 1)
      ray_results.update(rectified)

      # Get the weights used by volumetric rendering (and our other losses).
      weights = render.compute_alpha_weights(
          ray_results['density'], tdist, rays.directions
      )

      # Define or sample the background color for each ray.
      if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
        # If the min and max of the range are equal, just take it.
        bg_rgbs = self.bg_intensity_range[0]
      elif rng is None:
        # If rendering is deterministic, use the midpoint of the range.
        bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
      else:
        # Sample RGB values from the range for each ray.
        key, rng = random_split(rng)
        bg_rgbs = random.uniform(
            key,
            shape=weights.shape[:-1] + (3,),
            minval=self.bg_intensity_range[0],
            maxval=self.bg_intensity_range[1],
        )

      # RawNeRF exposure logic.
      if (ray_results['rgb'] is not None) and (rays.exposure_idx is not None):
        # Scale output colors by the exposure.
        ray_results['rgb'] *= rays.exposure_values[Ellipsis, None, :]
        if self.learned_exposure_scaling:
          exposure_idx = rays.exposure_idx[Ellipsis, 0]
          # Force scaling offset to always be zero when exposure_idx is 0.
          # This constraint fixes a reference point for the scene's brightness.
          mask = exposure_idx > 0
          # Scaling is parameterized as an offset from 1.
          scaling = 1 + mask[Ellipsis, None] * exposure_scaling_offsets(exposure_idx)
          ray_results['rgb'] *= scaling[Ellipsis, None, :]

      # Render each ray.
      extras_to_render = ['roughness']

      rendering = render.volumetric_rendering(
          ray_results['rgb'],
          weights,
          tdist,
          bg_rgbs,
          compute_extras,
          extras={
              k: v
              for k, v in ray_results.items()
              if k.startswith('normals') or k in extras_to_render
          },
          percentiles=percentiles,
      )

      if compute_extras:
        # Collect some rays to visualize directly. By naming these quantities
        # with `ray_` they get treated differently downstream --- they're
        # treated as bags of rays, rather than image chunks.
        n = self.config.vis_num_rays
        rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
        rendering['ray_weights'] = weights.reshape([-1, weights.shape[-1]])[
            :n, :
        ]
        rgb = ray_results['rgb']
        if rgb is not None:
          rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[
              :n, :, :
          ]
        else:
          rendering['ray_rgbs'] = None

      renderings.append(rendering)
      ray_results['tdist'] = jnp.copy(tdist)
      ray_results['sdist'] = jnp.copy(sdist)
      ray_results['weights'] = jnp.copy(weights)
      ray_history.append(ray_results)

    if compute_extras:
      # Because the proposal network doesn't produce meaningful colors, for
      # easier visualization we replace their colors with the final average
      # color.
      weights = [r['ray_weights'] for r in renderings]
      rgbs = [r['ray_rgbs'] for r in renderings]
      final_rgb = jnp.sum(rgbs[-1] * weights[-1][Ellipsis, None], axis=-2)
      avg_rgbs = [
          jnp.broadcast_to(final_rgb[:, None, :], w.shape + (3,))
          for w in weights[:-1]
      ]
      for i, avg_rgb in enumerate(avg_rgbs):
        renderings[i]['ray_rgbs'] = avg_rgb

    if not all(mlp_was_used):
      s = ', '.join([f'{i}' for i, v in enumerate(mlp_was_used) if not v])
      raise ValueError(f'MLPs {s} not used by the sampling strategy.')
    if not all(grid_was_used):
      s = ', '.join([f'{i}' for i, v in enumerate(grid_was_used) if not v])
      raise ValueError(f'Grids {s} not used by the sampling strategy.')
    return renderings, ray_history

  def get_scheduled_beta(self, i_level, train_frac=1.0):
    """Scheduling the scale beta for the VolSDF density.

    Args:
      i_level: int, the index of the sampling level.
      train_frac: float in [0, 1], what fraction of training is complete.

    Returns:
      curr_beta: float, the current scale beta.
    """
    min_beta = self.final_betas[i_level]
    max_beta = 0.5
    curr_beta = max_beta * (
        1.0
        / (
            1.0
            + ((max_beta - min_beta) / min_beta) * train_frac**self.rate_beta
        )
    )
    return curr_beta


def construct_model(rng, rays, config, dataset=None):
  """Construct a mip-NeRF 360 model.

  Args:
    rng: jnp.ndarray. Random number generator.
    rays: an example of input Rays.
    config: A Config class.
    dataset: Dataset, used to set max_exposure.

  Returns:
    model: initialized nn.Module, a NeRF model with parameters.
    init_variables: flax.Module.state, initialized NeRF model parameters.
  """
  # Grab just 10 rays, to minimize memory overhead during construction.
  ray = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:10], rays
  )
  model_kwargs = {}
  if dataset is not None and dataset.max_exposure is not None:
    model_kwargs['max_exposure'] = dataset.max_exposure
  model = Model(config=config, **model_kwargs)
  init_variables = model.init(
      rng,  # The RNG used by flax to initialize random weights.
      rng=None,  # The RNG used by sampling within the model.
      rays=ray,
      train_frac=1.0,
      compute_extras=False,
      zero_glo=model.num_glo_features == 0,
  )
  return model, init_variables


@gin.configurable
class MLP(nn.Module):
  """A PosEnc MLP."""

  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  bottleneck_width: int = 256  # The width of the bottleneck vector.
  net_depth_viewdirs: int = 1  # The depth of the second part of ML.
  net_width_viewdirs: int = 128  # The width of the second part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 12  # Max degree of positional encoding for 3D points.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
  num_rgb_channels: int = 3  # The number of RGB channels.
  deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
  use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
  use_directional_enc: bool = False  # If True, use IDE to encode directions.
  # If False and if use_directional_enc is True, use zero roughness in IDE.
  enable_pred_roughness: bool = False
  # Roughness activation function.
  roughness_activation: Callable[Ellipsis, Any] = nn.softplus
  roughness_bias: float = -1.0  # Shift added to raw roughness pre-activation.
  use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
  use_specular_tint: bool = False  # If True, predict tint.
  use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
  bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
  density_activation: Callable[Ellipsis, Any] = nn.softplus  # Density activation.
  density_bias: float = -1.0  # Shift added to raw densities pre-activation.
  density_noise: float = (
      0.0  # Standard deviation of noise added to raw density.
  )
  density_as_sdf: bool = False  # if True, use volsdf representation.
  sphere_init: bool = False  # if True, initialize the mlp as sdf to a sphere.
  sphere_radius: float = 1.0  # The radius for the sphere initialization.
  rgb_premultiplier: float = 1.0  # Premultiplier on RGB before activation.
  rgb_activation: Callable[Ellipsis, Any] = nn.sigmoid  # The RGB activation.
  rgb_bias: float = 0.0  # The shift added to raw colors pre-activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  enable_pred_normals: bool = False  # If True compute predicted normals.
  disable_density_normals: bool = False  # If True don't compute normals.
  disable_rgb: bool = False  # If True don't output RGB.
  isotropize_gaussians: bool = False  # If True, make Gaussians isotropic.
  warp_fn: Callable[Ellipsis, Any] = None
  basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
  basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).
  use_learned_vignette_map: bool = False
  use_exposure_at_bottleneck: bool = False
  unscented_mip_basis: str = 'mean'  # Which unscented transform basis to use.
  unscented_scale_mult: float = 0.0  # Unscented scale, 0 == disabled.
  # GLO vectors can either be 'concatenate'd onto the `bottleneck` or used to
  # construct an 'affine' transformation on the `bottleneck``.
  glo_mode: str = 'concatenate'
  # The MLP architecture used to transform the GLO codes before they are used.
  # Setting to () is equivalent to not using an MLP.
  glo_mlp_arch: Tuple[int, Ellipsis] = tuple()
  glo_mlp_act: Callable[Ellipsis, Any] = nn.silu  # The activation for the GLO MLP.
  glo_premultiplier: float = 1.0  # Premultiplier on GLO vectors before process.
  beta_init: float = 0.1  # If density_as_sdf is True, used for volsdf rep.
  beta_min: float = 0.0001  # If density_as_sdf is True, used for volsdf rep.
  squash_before: bool = False  # Apply squash before computing density gradient.
  # If True, concatenate the posenc features even if the grid is enabled.
  use_posenc_with_grid: bool = False
  # A scalar scale for the positional encoding features. This can be used to
  # scale the positional encoding features, e.g., to down-weight them relative
  # to the grid features.
  posenc_feature_scale: float = 1.0
  # Use bottleneck as affine transformation of directional encoding, instead of
  # concatenation.
  use_affine_dir_enc_transform: bool = False
  skip_final_density_layer: bool = False  # Use one grid feature as log-density.
  extra_grid_kwargs: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict()
  )

  def setup(self):
    # Make sure that normals are computed if reflection direction is used.
    if self.use_reflections and not (
        self.enable_pred_normals or not self.disable_density_normals
    ):
      raise ValueError('Normals must be computed for reflection directions.')

    # Precompute and store (the transpose of) the basis being used.
    self.pos_basis_t = jnp.array(
        geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)
    ).T

    # Precompute and define viewdir or refdir encoding function.
    if self.use_directional_enc:
      self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
    else:

      def dir_enc_fn(direction, _):
        return coord.pos_enc(
            direction, min_deg=0, max_deg=self.deg_view, append_identity=True
        )

      self.dir_enc_fn = dir_enc_fn

  @nn.compact
  def __call__(
      self,
      rng,
      gaussians,
      viewdirs=None,
      imageplane=None,
      glo_vec=None,
      exposure=None,
      curr_beta=None,
      grid=None,
      rays=None,
      tdist=None,
      train = True,
  ):
    """Evaluate the MLP.

    Args:
      rng: jnp.ndarray. Random number generator.
      gaussians: a tuple containing:                                           /
        - mean: [..., n, 3], coordinate means, and                             /
        - cov: [..., n, 3{, 3}], coordinate covariance matrices.
      viewdirs: jnp.ndarray(float32), [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane: jnp.ndarray(float32), [batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.
      curr_beta: float, beta to be used in the sdf to density transformation, if
        None then using the learned beta.
      grid: Callable, a function that computes a grid-like feature embeddding
        for a spatial position.
      rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
      tdist: jnp.ndarray(float32), with a shape of [..., n+1] containing the
        metric distances of the endpoints of each mip-NeRF interval.
      train: Boolean flag. Set to True when training.

    Returns:
      rgb: jnp.ndarray(float32), with a shape of [..., num_rgb_channels].
      density: jnp.ndarray(float32), with a shape of [...].
      normals: jnp.ndarray(float32), with a shape of [..., 3], or None.
      normals_pred: jnp.ndarray(float32), with a shape of [..., 3], or None.
      roughness: jnp.ndarray(float32), with a shape of [..., 1], or None.
    """

    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )
    view_dependent_dense_layer = functools.partial(
        nn.Dense,
        kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    # Initialize the density dense layers so that it approximate a sign distance
    # function to a sphere, theorem 1 from https://arxiv.org/pdf/1911.10414.pdf.
    if self.sphere_init:
      density_dense_layer = functools.partial(
          nn.Dense,
          kernel_init=jax.nn.initializers.normal(
              jnp.sqrt(2.0) / jnp.sqrt(self.net_width)
          ),
          bias_init=jax.nn.initializers.zeros,
      )

      # We zero-out the positional encoding at initialization, to model function
      # of the position solely.
      posenc_dense_layer = functools.partial(
          nn.Dense,
          self.net_width,
          kernel_init=jax.nn.initializers.zeros,
          bias_init=jax.nn.initializers.zeros,
      )

      # Final dense layer of the initialization based on the sphere radius.
      # Alternatively, could initialized last layer as constant with value as
      # 'init_mean', but for stochasticism we add a small variation.
      init_mean = jnp.sqrt(jnp.pi) / jnp.sqrt(self.net_width)
      init_std = 0.0001
      kernel_init = lambda *args: init_mean + random.normal(*args) * init_std
      final_density_dense_layer = functools.partial(
          nn.Dense,
          kernel_init=kernel_init,
          bias_init=jax.nn.initializers.constant(-self.sphere_radius),
      )
    else:
      density_dense_layer = dense_layer
      final_density_dense_layer = dense_layer

    density_key, rng = random_split(rng)
    grid_key, rng = random_split(rng)

    def predict_density(means, covs, **kwargs):
      """Helper function to output density."""

      x = []
      # Encode input positions.
      if grid is not None:
        control_offsets = kwargs['control_offsets']
        control = means[Ellipsis, None, :] + control_offsets
        perp_mag = kwargs['perp_mag']

        scale = None
        if not self.squash_before and self.warp_fn is not None:
          if perp_mag is not None and self.unscented_scale_mult > 0:
            if self.warp_fn.__wrapped__ == coord.contract:
              # We can accelerate the contraction a lot by special-casing
              # on the contraction and computing the cube root of the
              # determinant of the Jacobian directly.
              s = coord.contract3_isoscale(control)
              scale = self.unscented_scale_mult * (perp_mag * s)[Ellipsis, None]
              control = self.warp_fn(control)  # pylint: disable=not-callable
            else:
              control, perp_mag = coord.track_isotropic(
                  self.warp_fn, control, perp_mag
              )
              scale = self.unscented_scale_mult * perp_mag[Ellipsis, None]
          else:
            control = self.warp_fn(control)  # pylint: disable=not-callable

        # Gathering/scattering from a grid is impossibly slow on TPU.
        if utils.device_is_tpu():
          raise ValueError('Hash Encodings should not be used on a TPU.')
        x.append(
            grid(
                control,
                x_scale=scale,
                per_level_fn=math.average_across_multisamples,
                train=train,
                rng=grid_key,
                **self.extra_grid_kwargs,
            )
        )

      if grid is None or self.use_posenc_with_grid:
        # Encode using the strategy used in mip-NeRF 360.
        if not self.squash_before and self.warp_fn is not None:
          means, covs = coord.track_linearize(self.warp_fn, means, covs)

        lifted_means, lifted_vars = coord.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )
        x.append(
            self.posenc_feature_scale
            * coord.integrated_pos_enc(
                lifted_means,
                lifted_vars,
                self.min_deg_point,
                self.max_deg_point,
            )
        )

      x = jnp.concatenate(x, axis=-1)

      inputs = x
      # Evaluate network to produce the output density.
      for i in range(self.net_depth):
        if self.sphere_init and (
            i == 0 or ((i - 1) % self.skip_layer == 0 and i > 1)
        ):
          # For the first and skip connection layers we zero-out the positional
          # encoding part at initialization, for the sphere initialization to be
          # correct as a function of the position (means).
          if i == 0:
            x = means
          elif (i - 1) % self.skip_layer == 0 and i > 1:
            x = x[Ellipsis, : -inputs.shape[-1]]
            # To preserves the input norm in skip connection, we need to average
            # the norms of the previous layer and the concatenated input.
            # More information is in https://arxiv.org/pdf/1911.10414.pdf.
            x = jnp.concatenate([x, means], axis=-1) / jnp.sqrt(2.0)
          x = density_dense_layer(self.net_width)(x) + (
              posenc_dense_layer(self.net_width)(inputs)
          )
        else:
          x = density_dense_layer(self.net_width)(x)
        x = self.net_activation(x)
        if i % self.skip_layer == 0 and i > 0:
          x = jnp.concatenate([x, inputs], axis=-1)

      # Density is hardcoded to a single channel.
      if self.skip_final_density_layer:
        if x.shape[-1] != 1:
          raise ValueError(f'x has {x.shape[-1]} channels, but must have 1.')
        raw_density = x[Ellipsis, 0]
      else:
        raw_density = final_density_dense_layer(1)(x)[Ellipsis, 0]

      # Add noise to regularize the density predictions if needed.
      if (density_key is not None) and (self.density_noise > 0):
        raw_density += self.density_noise * random.normal(
            density_key, raw_density.shape
        )
      return raw_density, x

    means, covs = gaussians
    # Encode input positions
    if self.squash_before and self.warp_fn is not None:
      means, covs = coord.track_linearize(self.warp_fn, means, covs)

    predict_density_kwargs = {}
    if grid is not None:
      # Grid/hash structures don't give us an easy way to do closed-form
      # integration with a Gaussian, so instead we sample each Gaussian
      # according to an unscented transform (or something like it) and average
      # the sampled encodings.
      control_points_key, rng = random_split(rng)
      control, perp_mag = coord.compute_control_points(
          means,
          covs,
          rays,
          tdist,
          control_points_key,
          self.unscented_mip_basis,
          self.unscented_scale_mult,
      )
      control_offsets = control - means[Ellipsis, None, :]
      predict_density_kwargs['control_offsets'] = control_offsets
      predict_density_kwargs['perp_mag'] = perp_mag

    if self.disable_density_normals:
      raw_density, x = predict_density(means, covs, **predict_density_kwargs)
      raw_grad_density = None
      normals = None
    else:
      # Flatten the input so value_and_grad can be vmap'ed.
      n_flatten = len(means.shape) - 1
      gaussians_flat, pd_kwargs_flat = jax.tree_util.tree_map(
          lambda x: x.reshape((-1,) + x.shape[n_flatten:]),
          ((means, covs), predict_density_kwargs),
      )

      # Evaluate the network and its gradient on the flattened input.
      predict_density_and_grad_fn = jax.vmap(
          jax.value_and_grad(predict_density, has_aux=True),
      )
      (raw_density_flat, x_flat), raw_grad_density_flat = (
          predict_density_and_grad_fn(*gaussians_flat, **pd_kwargs_flat)
      )

      # Unflatten the output.
      raw_density = raw_density_flat.reshape(means.shape[:-1])
      x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
      raw_grad_density = raw_grad_density_flat.reshape(means.shape)

      # Compute normal vectors as negative normalized density gradient.
      # We normalize the gradient of raw (pre-activation) density because
      # it's the same as post-activation density, but is more numerically stable
      # when the activation function has a steep or flat gradient.
      # Note: In VolSDF normals are proportional to +grad, in NeRF it's -grad.
      if self.density_as_sdf:
        normals = ref_utils.l2_normalize(raw_grad_density)
      else:
        normals = -ref_utils.l2_normalize(raw_grad_density)

    if self.enable_pred_normals:
      grad_pred = dense_layer(3)(x)

      # Normalize negative predicted gradients to get predicted normal vectors.
      normals_pred = -ref_utils.l2_normalize(grad_pred)
      normals_to_use = normals_pred
    else:
      grad_pred = None
      normals_pred = None
      normals_to_use = normals

    # Apply bias and activation to raw density
    if self.density_as_sdf:
      # Use learned beta or given beta.
      if curr_beta is None:
        beta = self.param('beta', nn.initializers.constant(self.beta_init), ())
        curr_beta = jnp.abs(beta) + self.beta_min
      density = self.density_activation(
          raw_density + self.density_bias, curr_beta
      )
    else:
      density = self.density_activation(raw_density + self.density_bias)

    roughness = None
    if self.disable_rgb:
      rgb = None
    else:
      if viewdirs is not None or glo_vec is not None:
        # Predict diffuse color.
        if self.use_diffuse_color:
          raw_rgb_diffuse = dense_layer(self.num_rgb_channels)(x)

        if self.use_specular_tint:
          tint = nn.sigmoid(dense_layer(3)(x))

        if self.enable_pred_roughness:
          raw_roughness = dense_layer(1)(x)
          roughness = self.roughness_activation(
              raw_roughness + self.roughness_bias
          )

        # Output of the first part of MLP.
        if self.bottleneck_width > 0:
          bottleneck = dense_layer(self.bottleneck_width)(x)

          # Add bottleneck noise.
          if (rng is not None) and (self.bottleneck_noise > 0):
            key, rng = random_split(rng)
            bottleneck += self.bottleneck_noise * random.normal(
                key, bottleneck.shape
            )

          # Incorporate exposure in the style of HDR-NeRF, where we assume the
          # bottleneck is scaled proportional to log radiance and thus we can
          # scale it appropriately by adding log of the exposure value.
          if self.use_exposure_at_bottleneck and exposure is not None:
            bottleneck += jnp.log(exposure)[Ellipsis, None, :]

          x = [bottleneck]
        else:
          x = []

        if viewdirs is not None:
          # Encode view (or reflection) directions.
          if self.use_reflections:
            # Compute reflection directions. Note that we flip viewdirs before
            # reflecting, because they point from the camera to the point,
            # whereas ref_utils.reflect() assumes they point toward the camera.
            # Returned refdirs then point from the point to the environment.
            refdirs = ref_utils.reflect(-viewdirs[Ellipsis, None, :], normals_to_use)
            # Encode reflection directions.
            dir_enc = self.dir_enc_fn(refdirs, roughness)
          else:
            # Encode view directions.
            dir_enc = self.dir_enc_fn(viewdirs, roughness)

            dir_enc = jnp.broadcast_to(
                dir_enc[Ellipsis, None, :],
                bottleneck.shape[:-1] + (dir_enc.shape[-1],),
            )

          # Append view (or reflection) direction encoding to bottleneck vector.
          x.append(dir_enc)

        # Append dot product between normal vectors and view directions.
        if self.use_n_dot_v:
          dotprod = jnp.sum(
              normals_to_use * viewdirs[Ellipsis, None, :], axis=-1, keepdims=True
          )
          x.append(dotprod)

        # Predicting color based on IDR representation.
        if self.density_as_sdf and not self.use_reflections:
          x.append(normals)
          x.append(means)

        # Use the GLO vector, if it's available.
        if glo_vec is not None:
          # Pass the GLO vector (optionally) through a small MLP.
          y = glo_vec * self.glo_premultiplier
          for wi, w in enumerate(self.glo_mlp_arch):
            y = self.glo_mlp_act(nn.Dense(w, name=f'GLO_MLP_{wi}')(y))

          if self.glo_mode == 'concatenate':
            # Concatenate the transformed GLO vector onto the bottleneck.
            shape = bottleneck.shape[:-1] + y.shape[-1:]
            x.append(jnp.broadcast_to(y[Ellipsis, None, :], shape))
          elif self.glo_mode == 'affine':
            if self.bottleneck_width <= 0:
              # The user probably shouldn't use this mode if the bottleneck
              # is non-existent.
              raise ValueError('Bottleneck must have a non-zero width.')
            # Turn the transformed GLO vector into an affine transformation on
            # the bottleneck, and replace the bottleneck with that.
            y = nn.Dense(
                2 * bottleneck.shape[-1],
                name=f'GLO_MLP_{len(self.glo_mlp_arch)}',
            )(y)
            log_a, b = tuple(
                jnp.moveaxis(y.reshape(y.shape[:-1] + (-1, 2)), -1, 0)
            )
            a = math.safe_exp(log_a)
            bottleneck = a[Ellipsis, None, :] * bottleneck + b[Ellipsis, None, :]
            x[0] = bottleneck  # clobber the bottleneck at x[0].

        # Concatenate bottleneck, directional encoding, and GLO.
        x = jnp.concatenate(x, axis=-1)

        # Output of the second part of MLP.
        inputs = x
        for i in range(self.net_depth_viewdirs):
          x = view_dependent_dense_layer(self.net_width_viewdirs)(x)
          x = self.net_activation(x)
          if i % self.skip_layer_dir == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)

      # If using diffuse/specular colors, then `rgb` is treated as linear
      # specular color. Otherwise it's treated as the color itself.
      rgb = self.rgb_activation(
          self.rgb_premultiplier
          * view_dependent_dense_layer(self.num_rgb_channels)(x)
          + self.rgb_bias
      )

      if self.use_learned_vignette_map:
        vignette_weights = self.param(
            'VignetteWeights',
            lambda x: jax.nn.initializers.zeros(x, shape=[3, 3]),
        )
        vignette = image_utils.compute_vignette(imageplane, vignette_weights)
        # Account for the extra dimensions from ray samples.
        rgb *= vignette[Ellipsis, None, :]

      if self.use_diffuse_color:
        # Initialize linear diffuse color around 0.25, so that the combined
        # linear color is initialized around 0.5.
        diffuse_linear = nn.sigmoid(raw_rgb_diffuse - jnp.log(3.0))
        if self.use_specular_tint:
          specular_linear = tint * rgb
        else:
          specular_linear = 0.5 * rgb

        # Combine specular and diffuse components and tone map to sRGB.
        rgb = jnp.clip(
            image_utils.linear_to_srgb(specular_linear + diffuse_linear),
            0.0,
            1.0,
        )

      # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

    if grid is not None:
      warped_means = means
      if self.warp_fn is not None:
        warped_means = self.warp_fn(means)
      # Set the density of points outside the valid area to zero.
      # If there is a contraction, this mask will be true for all points. On the
      # other hand, if there isn't, invalid points that fall outside of the
      # bounding box will have their density set to zero.
      density_is_valid = jnp.all(
          (warped_means > grid.bbox[0]) & (warped_means < grid.bbox[1]), axis=-1
      )
      density = jnp.where(density_is_valid, density, 0.0)

    ray_results = dict(
        density=density,
        rgb=rgb,
        raw_grad_density=raw_grad_density,
        grad_pred=grad_pred,
        normals=normals,
        normals_pred=normals_pred,
        roughness=roughness,
    )

    return ray_results


def render_image(
    render_fn,
    rays,
    rng,
    config,
    return_all_levels = False,
    verbose = True,
):
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rng, rays) -> pytree.
    rays: a `Rays` pytree, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    config: A Config class.
    return_all_levels: return image buffers from ALL levels of nerf resampling.
    verbose: print progress indicators.

  Returns:
    rgb: jnp.ndarray, rendered color image_utils.
    disp: jnp.ndarray, rendered disparity image_utils.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  height, width = rays.pixels.shape[:2]
  num_rays = height * width
  rays = jax.tree_util.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  host_id = jax.process_index()
  chunks = []
  idx0s = range(0, num_rays, config.render_chunk_size)
  last_chunk_idx = None
  for i_chunk, idx0 in enumerate(idx0s):
    # pylint: disable=cell-var-from-loop
    if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
      if last_chunk_idx is None:
        logging.info('Rendering chunk %d/%d', i_chunk + 1, len(idx0s))
      else:
        rays_per_sec = (
            (i_chunk - last_chunk_idx)
            * config.render_chunk_size
            / (time.time() - start_chunk_time)
        )
        logging.info(
            'Rendering chunk %d/%d, %0.0f rays/sec',
            i_chunk + 1,
            len(idx0s),
            rays_per_sec,
        )
      start_chunk_time = time.time()
      last_chunk_idx = i_chunk
    chunk_rays = jax.tree_util.tree_map(
        lambda r: r[idx0 : idx0 + config.render_chunk_size], rays
    )

    actual_chunk_size = chunk_rays.pixels.shape[0]
    rays_remaining = actual_chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining

      def pad_fn(r):
        return jnp.pad(r, [(0, padding)] + [(0, 0)] * (r.ndim - 1), mode='edge')

      chunk_rays = jax.tree_util.tree_map(pad_fn, chunk_rays)
    else:
      padding = 0

    # After padding the number of chunk_rays is always divisible by host_count.
    rays_per_host = chunk_rays.pixels.shape[0] // jax.process_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = jax.tree_util.tree_map(
        lambda r: utils.shard(r[start:stop]), chunk_rays
    )
    # TODO(barron): There should be some optimization on the table here from
    # not computing the second output argument of render_fn.
    chunk_renderings, _ = render_fn(rng, chunk_rays)

    # Unshard the renderings.
    chunk_renderings = jax.tree_util.tree_map(
        lambda v: utils.unshard(v[0], padding), chunk_renderings
    )

    # Transpose the tree from list of dicts to dict of lists.
    chunk_renderings = {
        k: [z[k] for z in chunk_renderings if k in z]
        for k in chunk_renderings[-1].keys()
    }

    if not return_all_levels:
      # Throw away all but the final level for each image buffer.
      for k in chunk_renderings:
        if not k.startswith('ray_'):
          chunk_renderings[k] = chunk_renderings[k][-1]

    # Move to CPU.
    # Note: This used to be jax.block_until_read(), but as of cl/581302290,
    # this leads to stalls. This can be reverted once a solution is found.
    chunk_renderings = jax.device_get(chunk_renderings)

    chunks.append(chunk_renderings)

  # Concatenate all chunks within each leaf of a single pytree.
  rendering = jax.tree_util.tree_map(
      lambda *args: jnp.concatenate(args), *chunks
  )

  keys = [k for k in rendering if k.startswith('ray_')]
  if keys:
    num_rays = rendering[keys[0]][0].shape[0]
    ray_idx = random.permutation(random.PRNGKey(0), num_rays)
    ray_idx = ray_idx[: config.vis_num_rays]

  def reshape_fn(key):
    if key.startswith('ray_'):
      # Grab random sampling for a ray vis buffer.
      return lambda x: x[ray_idx]
    else:
      # Reshape to original resolution for an image buffer.
      return lambda x: x.reshape((height, width) + x.shape[1:])

  rendering = {
      k: jax.tree_util.tree_map(reshape_fn(k), z) for k, z in rendering.items()
  }
  if return_all_levels:
    # Throw away useless RGB buffers from proposal network.
    rendering['rgb'] = rendering['rgb'][-1]

  return rendering
