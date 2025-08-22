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

"""Models used during training of MERF.

During training we parameterize all grids (triplane or sparse 3D grid) with help
of an MLP to save memory during training.
"""
# pylint: disable=g-long-lambda
# pylint: disable=line-too-long

import functools
from typing import Any, Callable, Optional

from absl import logging
import chex
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from smerf.internal import configs
from smerf.internal import coord
from smerf.internal import grid_utils
from smerf.internal import hash_encoding
from smerf.internal import math
from smerf.internal import quantize
from smerf.internal import render
from smerf.internal import stepfun
from smerf.internal import utils


def random_split(rng):
  if rng is None:
    key = None
  else:
    key, rng = random.split(rng)
  return key, rng


# We fix the number of feature channels to 7. 3 channels are used for diffuse
# RGB and the remaining 4 channels model view-dependent effects. The baking
# pipeline and real-time renderer are hardcoded against this value. Changing
# will therefore result in baked models that cannot be loaded by the renderer.
NUM_CHANNELS = 7


def one_round_of_hierarchical_sampling(
    rng,
    i_level,
    num_samples,
    prod_num_samples,
    train_frac,
    init_s_near,
    init_s_far,
    rays,
    sdist,
    weights,
    tdist_override,
    config,
    grid_config,
):
  """Performs one round of hierarchical sampling.

  Args:
    rng: random number generator seed.
    i_level: level in coarse-to-fine sampling hierarchy.
    num_samples: int. Number of samples per ray. Denoted K' below.
    prod_num_samples: ...
    train_frac: percentage of training completed thus far. In [0, 1].
    init_s_near: lower bound on s-dist.
    init_s_far: upper bound on s-dist.
    rays: rays to sample intervals along in t-coordinates.
    sdist: f32[..., K+1]. Start and end of each interval in s-coordinates.
    weights: f32[..., K]. Weights assigned to each sdist interval.
    tdist_override: f32[..., K+1]. Start and end of each tdist interval. If
      defined, this replaces resampled sdist intervals.
    config: Config instance.
    grid_config: grid configuration.

  Returns:
    rng: updated random number generator.
    t_positions: f32[..., K']. positions in t-coordinates.
    prod_num_samples: int. updated number of samples.
    sdist: f32[..., K'+1]. Start and end of each new interval in squash
      space.
    tdist: f32[..., K'+1]. Start and end of each new interval in world
      coordinates.
  """
  # Performs one round of hierarchical sampling based on the volume rendering
  # weights from the previous round. More samples will be placed in segements
  # along the ray with high volume rendering weights, which effectively
  # leads to a concentration of samples at surfaces at the later rounds.
  #
  # This function also computes the distance of the interval endpoints to
  # the ray origin: sdist and tdist
  # The normalized `sdist` lie within [0, 1], whereas the `tdist` are
  # actual world space distances in [near, far].
  #
  # The resulting tensors will have the shapes
  #   positions: UxKx3, sdist: UxK+1x3, tdist: U+K+1x3
  # where U is the batch shape and K is the number of samples along the ray.

  # Dilate by some multiple of the expected span of each current interval,
  # with some bias added in.
  dilation_multiplier = 0.5
  dilation_bias = 0.0025
  dilation = (
      dilation_bias
      + dilation_multiplier * (init_s_far - init_s_near) / prod_num_samples
  )

  # Record the product of the number of samples seen so far.
  prod_num_samples *= num_samples

  # After the first level (where dilation would be a no-op)
  # dilate the interval weights along each ray slightly so that they're
  # overestimates, which can reduce aliasing.
  if i_level > 0:
    sdist, weights = stepfun.max_dilate_weights(
        sdist,
        weights,
        dilation,
        domain=(init_s_near, init_s_far),
        renormalize=True,
    )
    sdist = sdist[Ellipsis, 1:-1]
    weights = weights[Ellipsis, 1:-1]

  # Anneal the weights as a function of training iteration.
  anneal_slope = 10
  # Schlick's bias function, see https://arxiv.org/abs/2010.09714
  bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
  anneal = bias(train_frac, anneal_slope)

  # A slightly more stable way to compute weights**anneal. If the distance
  # between adjacent intervals is zero then its weight is fixed to 0.
  logits_resample = jnp.where(
      sdist[Ellipsis, 1:] > sdist[Ellipsis, :-1],
      anneal * jnp.log(jnp.maximum(1e-30, weights)),
      -jnp.inf,
  )

  # Draw sampled intervals from each ray's current weights.
  key, rng = random_split(rng)
  sdist = stepfun.sample_intervals(
      key,
      sdist,
      logits_resample,
      num_samples,
      single_jitter=True,
      domain=(init_s_near, init_s_far),
  )

  # Optimization will usually go nonlinear if you propagate gradients
  # through sampling.
  sdist = jax.lax.stop_gradient(sdist)

  if tdist_override is not None:
    # Override sampled sdist values
    tdist = tdist_override
    sdist = coord.tdist_to_sdist(tdist, rays.near, rays.far)
  else:
    # Convert squashed distances to world distances.
    # Note: This has no relation to world/submodel/squash coordinates.
    tdist = coord.sdist_to_tdist(sdist, rays.near, rays.far)

  # Compute sampling positions along ray in world coordinates.
  t_positions = render.get_sample_positions_along_ray(
      tdist, rays.origins, rays.directions, rays.radii
  )

  return rng, t_positions, prod_num_samples, sdist, tdist


def query_representation(
    sm_idxs,
    s_positions,
    viewdirs,
    config,
    grid_config,
    density_and_features_mlp,
    post_processing_mlp,
    merge_features_and_density_fn,
):
  """Query our representation (triplane + sparse grid) at given 3D positions.

  Accounts for spatial discretization with triplanes and voxel grids. Also
  accounts for discretization of feature and density values.

  Args:
    sm_idxs: i32[..., 1] tensor. Submodel indices for each s_position.
    s_positions: f32[..., 3] tensor. XYZ positions in s-coordinates.
    viewdirs: f32[..., 3]. Unit-norm view directions in world coordinates.
    config: Config object.
    grid_config: Grid config spec.
    density_and_features_mlp: MLP for querying density, features.
    post_processing_mlp: MLP for processing interpolated feature vectors.
    merge_features_and_density_fn: Function for merging triplane & sparse grid
      features and density preactivations.

  Returns:
    features: f32[..., c] tensor. Features for each position. Values in [0, 1]
    density: f32[..., 1] tensor. Density for each position.
  """
  # The structure of the representation is enforced by grid simulation. All grid
  # values are predicted by a single MLP (MultiDensityAndFeaturesMLP).
  chex.assert_equal_shape(
      (sm_idxs, s_positions), dims=range(len(sm_idxs.shape) - 1)
  )
  if viewdirs is not None:
    chex.assert_equal_shape((s_positions, viewdirs))

  # Flatten positions (makes our life easier).
  batch_shape = s_positions.shape[:-1]
  sm_idxs = sm_idxs.reshape(-1, 1)  # in: UxKx1, out: U*Kx1
  s_positions = s_positions.reshape(-1, 3)  # in: UxKx3, out: U*Kx3.
  if viewdirs is not None:
    viewdirs = viewdirs.reshape(-1, 3)  # in: UxKx3, out: U*Kx3.

  # Prepare grid simulation. Afterwards,
  #   sm_idxs: i32[U*K*S, 1]
  #   s_positions f32[U*K*S, 3]
  #   triplane_s_positions_local: f32[U*K, 3]
  #   sparse_grid_positions_local: f32[U*K, 3]
  (
      sm_idxs_for_mlp,
      s_positions_for_mlp,
      triplane_s_positions_local,
      sparse_grid_s_positions_local,
  ) = grid_utils.get_eval_positions_and_local_coordinates(
      sm_idxs, s_positions, config, grid_config
  )

  # Query MLP at grid corners (U*K*Sx7 and U*K*Sx1). If training one submodel
  # per host, set sm_idx=0 in order to access the one and only submodel
  # parameters present on this host.
  features, density = density_and_features_mlp(
      coord.sm_idxs_to_params_idxs(sm_idxs_for_mlp, config, grid_config),
      s_positions_for_mlp,
  )

  # Simulate quantization on MLP outputs.
  features = quantize.simulate_quantization(
      features, config.range_features[0], config.range_features[1]
  )
  density = quantize.simulate_quantization(
      density, config.range_density[0], config.range_density[1]
  )

  # Grid simulation: bi-lineary and/or tri-linearly interpolate outputs.
  triplane_features, sparse_grid_features = (
      grid_utils.interpolate_based_on_local_coordinates(
          features,
          triplane_s_positions_local,
          sparse_grid_s_positions_local,
          config,
      )
  )  # U*Kx7.
  triplane_density, sparse_grid_density = (
      grid_utils.interpolate_based_on_local_coordinates(
          density,
          triplane_s_positions_local,
          sparse_grid_s_positions_local,
          config,
      )
  )  # U*Kx1.

  # Merge contributions from triplanes and features together.
  features, density = merge_features_and_density_fn(
      triplane_features,
      sparse_grid_features,
      triplane_density,
      sparse_grid_density,
  )

  # Apply a small pointwise network to each feature vector.
  if post_processing_mlp is not None:
    features = post_processing_mlp(
        coord.sm_idxs_to_params_idxs(sm_idxs, config, grid_config),
        s_positions,
        viewdirs,
        features,
    )

  # Apply activation functions after interpolation. Doing this after
  # interpolation increases the model's representational power.
  features = math.feature_activation(features)
  density = math.density_activation(density)

  # Unflatten results.
  def unflatten(x):
    return x.reshape(*batch_shape, -1)

  features = unflatten(features)  # UxKx7.
  density = unflatten(density)  # UxKx1.
  return features, density


@gin.configurable
class Model(nn.Module):
  """Our volume rendering model maps rays to RGB colors."""

  # Rays (origins and directions) are mapped to RGB colors. During training we
  # employ hierarchical sampling using two Proposal-MLPs as in MipNeRF-360.

  config: configs.Config = (
      None  # A Config class, must be set upon construction.
  )
  num_prop_samples: int = 64  # The number of samples for each proposal level.
  num_final_samples: int = 32  # The number of samples for the final level.
  num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
  enable_prop_mlp: bool = True  # Train a PropMLP.
  enable_post_processing_mlp: bool = False  # Train a PostProcessingMLP.

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac,
      sm_idxs = None,
      s_positions = None,
      return_ray_results = False,
      alpha_threshold = 0.0,
      tdist_override = None,
  ):
    """Maps ray origins and ray directions to RGB colors via volume rendering.

    Args:
      rng: Random number seed.
      rays: Rays to use for raycasting.
      train_frac: float. Percentage of training finished. In [0, 1].
      sm_idxs: i32[..., 1]. Submodel indices for each point in s-positions. Only
        used when s_positions is defined.
      s_positions: f32[..., 3]. s-positions to query the model at. Requires
        sm_idxs to be defined.
      return_ray_results: bool. If True, return additional info.
      alpha_threshold: Alpha values below this quantity are treated as zero.
      tdist_override: f32[..., k+1]. Overrides PropMLP's ray intervals if
        specified.

    Returns:
      ...
    """

    # Alternatively when passing in `positions` the underlying representation is
    # directly queried (without volume rendering) and the given `rays` are being
    # ignored. This is currently only used for the sparsity loss that is
    # computed by sampling densities at abitrary locations in the volume.
    #
    # The passed in rays.origins and ray.directions have shape (..., 3). During
    # training this shape will be BxPxPx3, where B is the batch size
    # (Config.batch_size / #devices) and P is the patch size.
    # In our case P=1, which means we are sampling 1x1 patches
    # (just single pixels) from the training images. During inference the shape
    # will be Cx3, where C corresponds to the chunk size. Let U be the batch
    # shape(U=BxPxP or U = C)

    # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
    # being regularized.
    grid_config = self.config.grid_config
    logging.info('Instantiating MultiDensityAndFeaturesMLP...')
    density_and_features_mlp = MultiDensityAndFeaturesMLP(
        net_kernels=grid_config['num_submodel_mlp_kernels'],
        net_hash_kernels=grid_config['num_submodel_hash_encoding_kernels'],
    )

    # Avoid instantiating PropMLP if possible.
    if self.enable_prop_mlp:
      logging.info('Instantiating MultiPropMLP...')
      prop_mlp = MultiPropMLP(
          net_kernels=grid_config['num_submodel_mlp_kernels'],
          net_hash_kernels=grid_config['num_submodel_hash_encoding_kernels'],
      )
      i_levels = range(self.num_levels)
    else:
      prop_mlp = None
      i_levels = [self.num_levels - 1]

    # Instantiate MultiPostProcessingMLP.
    post_processing_mlp = None
    if self.enable_post_processing_mlp:
      logging.info('Instantiating MultiPostProcessingMLP...')
      post_processing_mlp = MultiPostProcessingMLP(
          net_kernels=grid_config['num_submodel_mlp_kernels'],
      )

    # Instantiate function for merging triplane and sparse voxel features &
    # density preactivations.
    #
    # TODO(duckworthd): Implement something more clever, like a Transformer.
    merge_features_and_density_fn = functools.partial(
        grid_utils.sum_with_triplane_weights,
        config=self.config,
    )

    # When `positions` is set we directly query the representation, which
    # is only used for computing the sparsity loss currently.
    if s_positions is not None:
      assert (
          sm_idxs is not None
      ), 'sm_idxs is required to query representation at s_positions.'
      features, density = query_representation(
          sm_idxs=sm_idxs,
          s_positions=s_positions,
          viewdirs=None,  # Not needed.
          config=self.config,
          grid_config=grid_config,
          density_and_features_mlp=density_and_features_mlp,
          post_processing_mlp=None,  # Not needed.
          merge_features_and_density_fn=merge_features_and_density_fn,
      )
      return features, density

    # Initialize the range of s-distances for each ray to [0, 1],
    # and assign that single interval a weight of 1. These distances and weights
    # will be repeatedly updated as we proceed through sampling levels.
    init_s_near = 0.0
    init_s_far = 1.0
    sdist = jnp.concatenate(
        [
            jnp.full_like(rays.near, init_s_near),
            jnp.full_like(rays.far, init_s_far),
        ],
        axis=-1,
    )  # Ux2.
    weights = jnp.ones_like(rays.near)  # Ux1
    prod_num_samples = 1

    # Compute submodel indices corresponding to each ray.
    # Resample sm_idxs for each camera ray. Only submodels present on this host
    # and neighboring the ray's home subvolume are eligible.
    sm_idxs = rays.sm_idxs  # Ux1
    rng, sm_idxs = _resample_sm_idxs(
        rng=rng,
        t_origins=rays.origins,
        sm_idxs=sm_idxs,
        prob_replace=self.config.submodel_idx_replace_percent,
        train_frac=train_frac,
        config=self.config,
        grid_config=grid_config,
    )

    ray_history = []
    for i_level in i_levels:
      is_prop = i_level < (self.num_levels - 1)
      num_samples = (
          self.num_prop_samples if is_prop else self.num_final_samples
      )  # = K

      # Obtain sampling positions via hierarchical sampling. If tdist_override
      # is specified, use those values for computing sdist/tdist instead.
      # positions: UxKx3, sdist: UxK+1, tdist: UxK+1.
      rng, t_positions, prod_num_samples, sdist, tdist = (
          one_round_of_hierarchical_sampling(
              rng=rng,
              i_level=i_level,
              num_samples=num_samples,
              prod_num_samples=prod_num_samples,
              train_frac=train_frac,
              init_s_near=init_s_near,
              init_s_far=init_s_far,
              rays=rays,
              sdist=sdist,
              weights=weights,
              tdist_override=(None if is_prop else tdist_override),
              config=self.config,
              grid_config=grid_config,
          )
      )

      # Copy sm_idx values along camera ray. [U, 1] -> [U, K, 1]
      sm_idxs_for_level = _broadcast_along_ray(sm_idxs, t_positions)

      # Resample sm_idxs for each individual point along a camera ray. Only
      # submodels present on this host and neighboring the ray's home subvolume
      # are eligible.
      rng, sm_idxs_for_level = _resample_sm_idxs(
          rng=rng,
          t_origins=_broadcast_along_ray(rays.origins, t_positions),
          sm_idxs=sm_idxs_for_level,
          prob_replace=self.config.submodel_idx_replace_percent_3d,
          train_frac=train_frac,
          config=self.config,
          grid_config=grid_config,
      )

      # Contraction: s_positions will lie within (-2, 2)^3 afterwards.
      #
      # **WARNING** s_positions must be paired with sm_idxs_for_level at all
      # times. The same t_position maps to a different s_position in each
      # submodel.
      sm_positions = coord.world_to_submodel(
          sm_idxs_for_level,
          t_positions,
          self.config,
          grid_config,
      )  # UxKx3
      s_positions = coord.contract(sm_positions)  # UxKx3

      if is_prop:
        density = prop_mlp(
            coord.sm_idxs_to_params_idxs(
                sm_idxs_for_level, self.config, grid_config
            ),
            s_positions,
        )  # UxKx1
        density = math.density_activation(density)
        features = None
      else:
        features, density = query_representation(
            sm_idxs=sm_idxs_for_level,
            s_positions=s_positions,
            viewdirs=_broadcast_along_ray(rays.viewdirs, s_positions),
            config=self.config,
            grid_config=grid_config,
            density_and_features_mlp=density_and_features_mlp,
            post_processing_mlp=post_processing_mlp,
            merge_features_and_density_fn=merge_features_and_density_fn,
        )  # UxKx7 and UxKx1.

        density = simulate_alpha_culling(
            density=density,
            sm_idxs=sm_idxs_for_level,
            sm_positions=sm_positions,
            sm_viewdirs=rays.viewdirs,
            alpha_threshold=alpha_threshold,
            config=self.config,
            grid_config=grid_config,
        )
      density = density[Ellipsis, 0]  # UxK.

      # Compute the weights used for volumetric rendering (and other losses).
      weights = render.compute_volume_rendering_weights(
          density, tdist, rays.directions
      )  # UxK.

      # Keep track of these values for distortion and interlevel losses.
      ray_results = {}
      ray_results['sm_idxs'] = jnp.copy(sm_idxs)
      ray_results['sdist'] = jnp.copy(sdist)
      ray_results['tdist'] = jnp.copy(tdist)
      ray_results['weights'] = jnp.copy(weights)
      ray_results['features'] = (
          None if features is None else jnp.copy(features)
      )  # UxKx7 or None
      ray_history.append(ray_results)

      rendering = {}
      if not is_prop:
        # Blend features based on volume rendering weights.
        features_blended = (weights[Ellipsis, None] * features).sum(axis=-2)  # Ux7

        # Accumulated opacity. In [0, 1].
        acc = weights.sum(axis=-1)  # U

        # Depth in world distances.
        depth = render.compute_depth_from_tdist(rays, tdist, weights)

        # Decide which deferred rendering method to use.
        deferred_render_fn = DEFERRED_RENDER_FNS.get(
            self.config.deferred_rendering_mode
        )
        if deferred_render_fn is None:
          raise ValueError(self.config.deferred_rendering_mode)

        # Construct deferred rendering model.
        logging.info('Instantiating DeferredMLP...')
        deferred_mlp = DeferredMLP(
            use_exposure=self.config.use_exposure_in_deferred_mlp,
            num_kernels=grid_config['num_submodel_hash_encoding_kernels'],
        )
        param_idxs = coord.sm_idxs_to_params_idxs(
            sm_idxs, self.config, grid_config
        )

        # Construct origins in DeferredMLP coordinates.
        deferred_mlp_origins = coord.world_to_deferred_mlp(
            sm_idxs, rays.origins, self.config, grid_config
        )
        deferred_mlp_viewdirs = rays.viewdirs

        deferred_mlp_viewdir_features = None
        if self.config.num_viewdir_features > 0:
          logging.info('Instantiating ViewDir Feature Grid...')
          viewdir_encoding = hash_encoding.Multi3DGrid(
              num_kernels=grid_config['num_submodel_hash_encoding_kernels'],
              num_features=self.config.num_viewdir_features,
              grid_size=self.config.viewdir_grid_size,
          )
          # Map from the unit sphere in [-1,1]^3 to [0,1]^3.
          deferred_mlp_viewdir_features = viewdir_encoding(
              param_idxs, 0.5 * (deferred_mlp_viewdirs + 1.0)
          )

        deferred_mlp_origin_features = None
        if self.config.num_origin_features > 0:
          logging.info('Instantiating Origin Feature Grid...')
          origin_encoding = hash_encoding.Multi3DGrid(
              num_kernels=grid_config['num_submodel_hash_encoding_kernels'],
              num_features=self.config.num_origin_features,
              grid_size=self.config.origin_grid_size,
          )
          # Map from the space of ray/camera origins [-1,1]^3 to [0,1]^3.
          deferred_mlp_origin_features = origin_encoding(
              param_idxs, 0.5 * (deferred_mlp_origins + 1.0)
          )

        # Apply deferred rendering.
        # pylint: disable=cell-var-from-loop
        _, rets, extras = deferred_render_fn(
            rng=rng,
            features=features_blended,
            acc=acc,
            model=lambda features: deferred_mlp(
                features=features,
                viewdirs=deferred_mlp_viewdirs,
                origins=deferred_mlp_origins,
                exposures=rays.exposure_values,
                param_idxs=param_idxs,
                viewdir_features=deferred_mlp_viewdir_features,
                origin_features=deferred_mlp_origin_features,
            ),
        )
        # pylint: enable=cell-var-from-loop

        # Update rendering.
        rendering.update(rets)
        rendering['acc'] = acc  # U
        rendering['depth'] = depth  # U

        # These quantities are required for visbility culling
        # (see compute_alive_voxels).
        if return_ray_results:
          rendering['tdist'] = tdist
          rendering['weights'] = weights
          rendering['density'] = density
          rendering.update(extras)

        return rendering, ray_history


def construct_model(rng, rays, config):
  """Constructs the model and forwards one ray for debugging."""
  # Grab just the first ray, to minimize memory overhead during construction.
  ray = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:1], rays
  )
  model = Model(config=config)
  init_variables = jax.jit(model.init)(
      rng,  # The RNG used by flax to initialize random weights.
      rng=None,  # The RNG used by sampling within the model.
      rays=ray,
      train_frac=1.0,
  )
  return model, init_variables


@gin.configurable(denylist=['net_kernels', 'net_hash_kernels'])
class MultiPropMLP(nn.Module):
  """Proposal-MLP with hash encoding.

  Maps 3D positions to densities used to guide sampling of our representation.
  """

  net_kernels: int = 1  # number of MLPs to choose from.
  net_hash_kernels: int = 1  # Number of HashEncoding kernels to choose from.
  net_depth: int = 1  # The depth of the first part of MLP.
  net_width: int = 64  # The width of the first part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(self, idxs, positions):
    # Predict density and feature vector.
    xs = hash_encoding.MultiPropHashEncoding(num_kernels=self.net_hash_kernels)(
        idxs, positions
    )
    mlp = MultiMLP(
        net_kernels=self.net_kernels,
        net_depth=self.net_depth,
        net_width=self.net_width,
        num_output_channels=1,
        net_activation=self.net_activation,
        weight_init=self.weight_init,
    )
    density = mlp(idxs, xs)
    return density


@gin.configurable(denylist=['net_kernels', 'net_hash_kernels'])
class MultiDensityAndFeaturesMLP(nn.Module):
  """This MLP parameterizes our grids."""

  # It predicts the values at gridcorners. By employing a hash encoding the MLP
  # is relatively light-weight.

  net_kernels: int = 1  # number of MLPs to choose from.
  net_hash_kernels: int = 1  # Number of HashEncoding kernels to choose from.
  net_depth: int = 1  # The depth of the first part of MLP.
  net_width: int = 64  # The width of the first part of MLP.
  num_output_channels: int = NUM_CHANNELS  # Number of outputs
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(self, idxs, positions):
    xs = hash_encoding.MultiHashEncoding(num_kernels=self.net_hash_kernels)(
        idxs, positions
    )
    mlp = MultiMLP(
        net_kernels=self.net_kernels,
        net_depth=self.net_depth,
        net_width=self.net_width,
        num_output_channels=(1 + self.num_output_channels),
        net_activation=self.net_activation,
        weight_init=self.weight_init,
    )
    density_and_features = mlp(idxs, xs)
    density = density_and_features[Ellipsis, :1]
    features = density_and_features[Ellipsis, 1:]
    return features, density


@gin.configurable(denylist=['net_kernels'])
class MultiPostProcessingMLP(nn.Module):
  """Post-processes feature vectors after interpolation."""

  net_kernels: int = 1  # number of MLPs to choose from.
  net_depth: int = 1
  net_width: int = 16
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  use_viewdirs: bool = True
  viewdir_deg_enc: int = 4
  use_positions: bool = True
  positions_deg_enc: int = 2

  @nn.compact
  def __call__(self, idxs, positions, viewdirs, features):
    """Applies MultiPostProcessingMLP.

    Args:
      idxs: i32[..., 1]. Which kernel to use.
      positions: f32[..., 3]. Positions in squash coordinates.
      viewdirs: f32[..., 3]. Unit-norm view directions in world coordinates.
      features: f32[..., d]. Feature vectors.

    Returns:
      f32[..., d]. Post-processed features.
    """
    mlp = MultiMLP(
        net_kernels=self.net_kernels,
        net_depth=self.net_depth,
        net_width=self.net_width,
        num_output_channels=features.shape[-1],
        net_activation=self.net_activation,
        weight_init=self.weight_init,
    )

    def posenc(x, deg_enc):
      return coord.pos_enc(x, min_deg=0, max_deg=deg_enc, append_identity=True)

    # Construct feature inputs.
    x = [features]

    if self.use_positions:
      # TODO(duckworthd): Positions are in squash coordinates, which can only
      # be disambiguated when net_kernels == num_submodels. Use some other
      # coordinate system.
      x.append(posenc(positions, self.positions_deg_enc))

    if self.use_viewdirs:
      x.append(posenc(viewdirs, self.viewdir_deg_enc))

    # Apply MLP
    x = jnp.concatenate(x, axis=-1)
    return mlp(idxs, x)


@gin.configurable
class DeferredMLP(nn.Module):
  """View-dependent RGB colors are predicted by this tiny MLP."""

  # This MLP only is only queried once per ray as in SNeRG
  # (https://arxiv.org/abs/2103.14645). The webviewer assumes a fixed
  # architecture and any changes to the hyperparameters below (except
  # weight_init) would require adapting the webviewer's code.
  use_exposure: bool
  exposure_combine_fn: str = 'logit_logadd'
  net_depth: int = 2
  net_width: int = 16
  deg_enc: int = 4
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  num_kernels: int = 1  # The number of coarse subdivions (for nearest fetching).
  grid_size: int = 1  # If > 1, use a grid of MLPs parameterized by ray origins.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(
      self,
      features,
      viewdirs,
      origins,
      exposures,
      param_idxs,
      viewdir_features,
      origin_features,
  ):
    """Applies DeferredMLP.

    Args:
      features: f32[..., k]. Feature vectors for each camera ray.
      viewdirs: f32[..., 3]. Unit-norm view directions for each camera ray in
        world coordinates.
      origins: f32[..., 3]. The origins of each camera ray in DeferredMLP
        coordinates. Values predominantly in [-1, 1].
      exposures: Optional f32[..., 1]. Exposure value for each camera ray.
        Approximately equal to ISOSpeedRatings * ExposureTime in seconds / 1000.
      param_idxs: i32[..., 1]. Submodel paramater indices for each ray..
      viewdir_features: f32[..., l]. Feature vectors predicted from the view
        directions.
      origin_features: f32[..., m]. Feature vectors predicted from the ray
        origins.

    Returns:
      f32[..., 3]. Offset to RGB capturing view-dependent color.
    """

    assert features.shape[:1] == viewdirs.shape[:1]

    if self.use_exposure:
      assert exposures is not None
      assert features.shape[:1] == exposures.shape[:1]
      if self.exposure_combine_fn == 'logadd':
        # Assume that features are in log-space.
        features = features + jnp.log(exposures)
      elif self.exposure_combine_fn == 'logit_logadd':
        # Move from (0, 1)-space to log space with a logit function. This
        # function is unstable when 'features' approaches 0 or 1, so values
        # are clipped. An epsilon of 1e-7 gives a dynamic range of (-15, 15)
        # to the pre-sigmoid feature vector.
        features = _clipped_logit(features, 1e-7) + jnp.log(exposures)
      elif self.exposure_combine_fn == 'multiply':
        # Assume features are in linear space.
        features = features * exposures
      else:
        raise ValueError(self.exposure_combine_fn)

    # Prepare inputs to MLP.
    viewdirs = coord.pos_enc(
        viewdirs, min_deg=0, max_deg=self.deg_enc, append_identity=True
    )
    x = [features, viewdirs]

    if origin_features is not None:
      x.append(origin_features)

    x = jnp.concatenate(x, axis=-1)
    if self.grid_size > 1:
      assert origin_features is None and viewdir_features is None, (
          'We do not yet support a grid of DeferredMLPs with features from'
          'the ray origins and/or viewdirs.'
      )

      resample_dense = functools.partial(
          ResampleDense,
          num_kernels=self.num_kernels,
          grid_size=self.grid_size,
          weight_init=self.weight_init
      )
      origins_in_01 = 0.5 * (origins + 1.0)  # [-1, 1] -> [0, 1]
      for _ in range(self.net_depth):
        layer = resample_dense(
            features=self.net_width, input_channels=x.shape[-1])
        x = layer(param_idxs, origins_in_01, x)
        x = self.net_activation(x)
      layer = resample_dense(features=3, input_channels=x.shape[-1])
      out = jax.nn.sigmoid(layer(param_idxs, origins_in_01, x))
      return out

    # This could be implemented with MultiMLP, but isn't for backward
    # compatibility with existing checkpoints.
    if self.num_kernels > 1:
      logging.warn(
          f'DeferredMLP is not ready for {self.num_kernels}>1 kernels. This'
          ' argument will be ignored.'
      )

    # This is the normal case where we only have one DeferredMLP, and can make
    # use of features fetched using the ray origins and/or viewdirs.
    dense = functools.partial(
        nn.Dense,
        kernel_init=getattr(jax.nn.initializers, self.weight_init)(),
    )
    for _ in range(self.net_depth):
      x = dense(self.net_width)(x)
      x = self.net_activation(x)

    # If we do not have any view direction features, then directly predict the
    # view dependent color.
    if viewdir_features is None:
      return jax.nn.sigmoid(dense(3)(x))

    # But if we do have view direction features, then we compute the view
    # dependent color as the sum of the MLP output above, and the weighted sum
    # of each channel in the view direction feature vector.
    diffuse_and_weights = dense(3 + viewdir_features.shape[-1] * 3)(x)
    diffuse = diffuse_and_weights[Ellipsis, 0:3]
    weights = diffuse_and_weights[Ellipsis, 3:].reshape(
        viewdir_features.shape + (3,)
    )
    residual = jnp.stack(
        [
            jnp.sum(weights[Ellipsis, 0] * viewdir_features, axis=-1),
            jnp.sum(weights[Ellipsis, 1] * viewdir_features, axis=-1),
            jnp.sum(weights[Ellipsis, 2] * viewdir_features, axis=-1),
        ],
        axis=-1,
    )

    return jax.nn.sigmoid(diffuse + residual)


class MultiDense(nn.Module):
  """Dense layer with a switch."""

  kernels: int = 1  # number of dense layers to choose from.
  features: int = 64  # The width of the layer
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(self, idxs, xs):
    kernel_init_fn = getattr(jax.nn.initializers, self.weight_init)()
    kernels = self.param(
        'kernel',
        kernel_init_fn,
        (self.kernels, xs.shape[-1], self.features),
    )
    biases = self.param(
        'bias', nn.initializers.zeros, (self.kernels, self.features)
    )

    @jax.vmap
    def _apply_kernel(idx, x):
      kernel = kernels[idx]
      bias = biases[idx]
      return jnp.matmul(x, kernel) + bias

    # Merge batch dims
    batch_shape = idxs.shape[0:-1]
    idxs_flat = idxs.reshape((-1,))
    xs_flat = xs.reshape((-1, xs.shape[-1]))

    # Apply dense layer
    if self.kernels == 1:
      # Fast codepath. By using a constant index, we avoid a scatter-update.
      xs_flat = jnp.matmul(xs_flat, kernels[0]) + biases[0]
    else:
      xs_flat = _apply_kernel(idxs_flat, xs_flat)

    # Reintroduce batch dims
    xs = jnp.reshape(xs_flat, (*batch_shape, self.features))

    return xs


class MultiMLP(nn.Module):
  """MLP composed of multiple MultiDense layers."""

  num_output_channels: int  # Number of outputs
  net_kernels: int = 1  # number of MLPs to choose from.
  net_depth: int = 1  # The depth of the first part of MLP.
  net_width: int = 64  # The width of the first part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(self, idxs, xs):
    dense_layer = lambda features: MultiDense(
        features=features,
        kernels=self.net_kernels,
        weight_init=self.weight_init,
    )

    for _ in range(self.net_depth):
      xs = dense_layer(self.net_width)(idxs, xs)
      xs = self.net_activation(xs)
    xs = dense_layer(self.num_output_channels)(idxs, xs)
    return xs


class ResampleDense(nn.Module):
  """Dense layer whose parameters are sampled from a 3D voxel grid."""

  num_kernels: int = 1  # The of coarse subdivions (for nearest fetching).
  grid_size: int = 1  # We store our parameters in a grid_size^3 grid
  features: int = 16  # The width of the layer
  input_channels: int = 8  # We cannot use shape inference with a setup method,
  # so we need to know the last dimension of the input apriori.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  def setup(self):
    kernel_init_fn = getattr(jax.nn.initializers, self.weight_init)()
    self.kernels = self.param(
        'kernel',
        kernel_init_fn,
        (
            self.num_kernels,
            self.grid_size,
            self.grid_size,
            self.grid_size,
            self.input_channels,
            self.features,
        ),
    )

    self.biases = self.param(
        'bias',
        nn.initializers.zeros,
        (
            self.num_kernels,
            self.grid_size,
            self.grid_size,
            self.grid_size,
            self.features,
        ),
    )

  def __call__(self, param_idxs, pos, xs):
    """Applies ResampleDense layer.

    Args:
      param_idxs: i32[..., 1]. Which submodel parameters to use.
      pos: f32[..., 3]. Position used for parameter trilerp. Values in [0, 1]
        result in interpolated MLP parameters. Values outside of these bounds
        are constant padded.
      xs: f32[..., c]. Feature vectors

    Returns:
      f32[..., c']. The result of applying a linear layer to each point, where
        the linear layer's weights are the result of a trilerp over a grid of
        weights using 'pos' to determine the trilerp weights.
    """
    @jax.vmap
    def _apply_kernel(param_idx, pos, x):
      """Applies ResampleDense layer to a single point.

      Args:
        param_idx: i32[1]. ...
        pos: f32[3]. ...
        x: f32[c]. ...

      Returns:
        f32[c']. ...
      """
      # [0, 1] -> [-0.5, grid_size-0.5]
      locations = pos * self.grid_size - 0.5
      floored = jnp.floor(locations)
      ceil = floored + 1.0
      positions = [
          jnp.stack(
              [floored[Ellipsis, 0], floored[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1
          ),
          jnp.stack(
              [floored[Ellipsis, 0], floored[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1
          ),
          jnp.stack(
              [floored[Ellipsis, 0], ceil[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1
          ),
          jnp.stack(
              [floored[Ellipsis, 0], ceil[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1
          ),
          jnp.stack(
              [ceil[Ellipsis, 0], floored[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1
          ),
          jnp.stack(
              [ceil[Ellipsis, 0], floored[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1
          ),
          jnp.stack(
              [ceil[Ellipsis, 0], ceil[Ellipsis, 1], floored[Ellipsis, 2]], axis=-1
          ),
          jnp.stack(
              [ceil[Ellipsis, 0], ceil[Ellipsis, 1], ceil[Ellipsis, 2]], axis=-1
          ),
      ]
      ceil_w = locations - floored
      floor_w = 1.0 - ceil_w
      weights = [
          floor_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
          floor_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
          floor_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
          floor_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
          ceil_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
          ceil_w[Ellipsis, 0] * floor_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
          ceil_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * floor_w[Ellipsis, 2],
          ceil_w[Ellipsis, 0] * ceil_w[Ellipsis, 1] * ceil_w[Ellipsis, 2],
      ]

      kernel = jnp.zeros(
          (
              *locations.shape[:-1],
              self.kernels.shape[-2],
              self.kernels.shape[-1],
          ),
          dtype=self.kernels.dtype,
      )
      bias = jnp.zeros(
          (*locations.shape[:-1], self.biases.shape[-1]),
          dtype=self.biases.dtype,
      )

      for position, weight in zip(positions, weights):
        indexes = position.astype(jnp.int32)
        indexes = jnp.maximum(indexes, 0)
        indexes = jnp.minimum(indexes, self.grid_size - 1)

        x_coordinate = indexes[Ellipsis, 0]
        y_coordinate = indexes[Ellipsis, 1]
        z_coordinate = indexes[Ellipsis, 2]
        chex.assert_equal_shape(
            (param_idx, z_coordinate, y_coordinate, x_coordinate)
        )

        gathered_kernel = self.kernels[
            param_idx, z_coordinate, y_coordinate, x_coordinate
        ]
        gathered_kernel *= weight[Ellipsis, None]
        kernel += gathered_kernel

        gathered_bias = self.biases[
            param_idx, z_coordinate, y_coordinate, x_coordinate
        ]
        gathered_bias *= weight[Ellipsis, None]
        bias += gathered_bias

      return jnp.matmul(x, kernel) + bias

    # Merge batch dims
    batch_shape = pos.shape[0:-1]
    param_idxs_flat = param_idxs.reshape((-1,))
    pos_flat = pos.reshape((-1, 3))
    xs_flat = xs.reshape((-1, xs.shape[-1]))

    # Apply dense layer
    xs_flat = _apply_kernel(param_idxs_flat, pos_flat, xs_flat)

    # Reintroduce batch dims
    xs = jnp.reshape(xs_flat, (*batch_shape, self.features))

    return xs


def sample_bg_rgbs(rng, shape):
  """Samples background colors."""
  bg_intensity_range = (0.0, 1.0)  # The range of background colors.
  # Define or sample the background color for each ray.
  if rng is None:
    # If rendering is deterministic, use the midpoint of the range.
    bg_rgbs = (bg_intensity_range[0] + bg_intensity_range[1]) / 2
  else:
    # Sample RGB values from the range for each ray.
    key, rng = random_split(rng)
    bg_rgbs = random.uniform(
        key,
        shape=shape,
        minval=bg_intensity_range[0],
        maxval=bg_intensity_range[1],
    )
  return rng, bg_rgbs


@jax.named_scope('simulate_alpha_culling')
def simulate_alpha_culling(
    density,
    sm_idxs,
    sm_positions,
    sm_viewdirs,
    alpha_threshold,
    config,
    grid_config,
):
  """Sets density to zero when alpha is below a threshold.

  Used to simulate the alpha culling process applied during baking.

  Args:
    density: f32[..., k]. Density for each ray interval.
    sm_idxs: i32[..., k, 1]. Submodel indices for each ray interval.
    sm_positions: f32[..., k, 3]. Positions in submodel coordinates.
    sm_viewdirs: f32[..., k, 3]. Unit-norm directions in submodel coordinates.
    alpha_threshold: f32. Alpha values below this threshold are set to zero.
    config: Config object.
    grid_config: See grid_utils.initialize_grid_config().

  Returns:
    density with entries set to 0 when the corresponding alpha would be 0
    during real-time rendering.
  """
  # During real-time rendering a constant step size (i.e. voxel size) is used.
  # During training a variable step size is used that can be vastly different
  # from the voxel size. When baking we discard voxels that would only
  # contribute neglible alpha values in the real-time renderer. To make this
  # lossless we already simulate the behaviour of the real-time renderer during
  # training by ignoring alpha values below the threshold.

  def zero_density_below_threshold(density):
    viewdirs_b = jnp.broadcast_to(
        sm_viewdirs[Ellipsis, None, :], sm_positions.shape
    ).reshape(-1, 3)
    sm_positions_b = sm_positions.reshape(-1, 3)
    sm_step_size = coord.sm_stepsize_from_s_stepsize(
        sm_positions_b, viewdirs_b, grid_config['voxel_size_to_use']
    )
    t_step_size = coord.sm_dist_to_world_dist(
        sm_idxs, sm_step_size, config, grid_config
    )
    t_step_size = t_step_size.reshape(density.shape)
    alpha = math.density_to_alpha(density, t_step_size)
    return jnp.where(
        alpha >= alpha_threshold, density, 0.0
    )  # density = 0 <=> alpha = 0

  return zero_density_below_threshold(density)


def render_image(
    render_fn, rays, rng, config, verbose=True, transfer_to_cpu=False
):
  """Renders an image (in test mode)."""
  height, width = rays.origins.shape[:2]
  num_rays = height * width
  rays = jax.tree_util.tree_map(lambda r: r.reshape((num_rays, -1)), rays)

  host_id = jax.process_index()
  chunks = []
  render_chunk_size = (
      config.render_chunk_size // config.gradient_accumulation_steps
  )
  idx0s = range(0, num_rays, render_chunk_size)
  for i_chunk, idx0 in enumerate(idx0s):
    # pylint: disable=cell-var-from-loop
    if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
      print(f'Rendering chunk {i_chunk+1}/{len(idx0s)-1}')
    chunk_rays = jax.tree_util.tree_map(
        lambda r: r[idx0 : idx0 + render_chunk_size], rays
    )
    actual_chunk_size = chunk_rays.origins.shape[0]
    rays_remaining = actual_chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = jax.tree_util.tree_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays
      )
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by host_count.
    rays_per_host = chunk_rays.origins.shape[0] // jax.process_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = jax.tree_util.tree_map(
        lambda r: utils.shard(r[start:stop]), chunk_rays
    )
    chunk_rendering, _ = render_fn(rng, chunk_rays)

    # Unshard the rendering.
    chunk_rendering = jax.tree_util.tree_map(
        lambda v: utils.unshard(v[0], padding), chunk_rendering
    )

    if transfer_to_cpu:
      chunk_rendering = jax.tree_util.tree_map(jax.device_get, chunk_rendering)
    chunks.append(chunk_rendering)

  # Concatenate all chunks within each leaf of a single pytree.
  xnp = np if transfer_to_cpu else jnp
  rendering = jax.tree_util.tree_map(
      lambda *args: xnp.concatenate(args), *chunks
  )
  for k, z in rendering.items():
    if not k.startswith('ray_'):
      # Reshape 2D buffers into original image shape.
      rendering[k] = z.reshape((height, width) + z.shape[1:])

  # After all of the ray bundles have been concatenated together, extract a
  # new random bundle (deterministically) from the concatenation that is the
  # same size as one of the individual bundles.
  keys = [k for k in rendering if k.startswith('ray_')]
  if keys:
    num_rays = rendering[keys[0]][0].shape[0]
    ray_idx = random.permutation(random.PRNGKey(0), num_rays)
    ray_idx = ray_idx[: config.vis_num_rays]
    for k in keys:
      rendering[k] = [r[ray_idx] for r in rendering[k]]

  return rendering


def _broadcast_along_ray(x, y):
  """Broadcast x's values along y's second-to-last dimension."""
  chex.assert_rank(x, len(y.shape) - 1)
  chex.assert_equal_shape((x, y), dims=range(len(y.shape)-2))
  batch_shape = y.shape[:-1]
  target_shape = (*batch_shape, x.shape[-1])
  return jnp.broadcast_to(x[Ellipsis, jnp.newaxis, :], target_shape)


def _resample_sm_idxs(
    rng, t_origins, sm_idxs, prob_replace, train_frac, config, grid_config
):
  """Randomly resample a subset of submodel indices.

  Only submodels present on this host are eligible for assignment.

  Args:
    rng: Random generator key.
    t_origins: f32[..., 3]. Ray origins in world coordinates.
    sm_idxs: i32[..., 1]. Submodel indices.
    prob_replace: f32[]. Probabilty of replacing a submodel index with a random
      neighbor's.
    train_frac: f32[]. Percentage of training finished. 1.0 at training
      completion.
    config: configs.Config object.
    grid_config: See grid_utils.initialize_grid_config().

  Returns:
    rng: Updated random generator key.
    sm_idxs: i32[...]. Updated submodel idxs.
  """
  chex.assert_equal_rank((sm_idxs, t_origins))
  chex.assert_equal_shape(
      (sm_idxs, t_origins), dims=range(len(sm_idxs.shape) - 1)
  )
  chex.assert_shape(sm_idxs, (Ellipsis, 1))
  chex.assert_shape(t_origins, (Ellipsis, 3))

  # No randomization.
  if rng is None:
    return rng, sm_idxs

  # No need to replace.
  if prob_replace <= 0:
    return rng, sm_idxs

  # Sample a boolean mask and new sm_idx values.
  rng, mask_rng, choice_rng = jax.random.split(rng, num=3)
  should_replace = jnp.logical_and(
      jax.random.bernoulli(mask_rng, prob_replace, shape=sm_idxs.shape),
      train_frac < 1.0,
  )

  # Sample neighbor submodel idxs.
  new_sm_idxs = coord.sample_random_neighbor_sm_idx(
      rng=choice_rng,
      t_positions=t_origins,
      config=config,
      grid_config=grid_config,
  )
  chex.assert_equal_shape((sm_idxs, new_sm_idxs))

  # Replace sm_idxs at random locations.
  sm_idxs = jnp.where(should_replace, new_sm_idxs, sm_idxs)

  return rng, sm_idxs


def _clipped_logit(y, eps):
  """A clipped version of the logit function."""
  y = jnp.clip(y, eps, 1 - eps)
  return jnp.log(y) - jnp.log(1 - y)


def deferred_render_snerg(rng, features, acc, model, bg_rgbs=None):
  """Applies SNeRG's additive deferred rendeing model.

  Args:
    rng: jax.random.RandomState instance.
    features: f32[..., d]. Per-ray feature vectors.
    acc: f32[...]. Accumulated opacity per ray. Values in [0, 1].
    model: Function converting features to specular RGB. First 3 channels
      represent diffuse RGB. Outputs values should be in [0, 1].
    bg_rgbs: f32[..., 3]. Optional background color.

  Returns:
    rng: Updated random number state.
    rendering: {str: array}. Contains predicted RGB after deferred rendering.
      Values should be in [0, 1], but this isn't guaranteed.
    ray_results: [str: array} Extra debug arrays.
  """
  # Add background to first 3 channels.
  rgb_diffuse = features[Ellipsis, :3]  # Ux3
  bg_weight = jnp.maximum(0, 1 - acc[Ellipsis, None])  # Ux1
  if bg_rgbs is None:
    rng, bg_rgbs = sample_bg_rgbs(rng, rgb_diffuse.shape)
  rgb_diffuse_with_bg = rgb_diffuse + bg_weight * bg_rgbs  # Ux3
  features = features.at[Ellipsis, :3].set(rgb_diffuse_with_bg)

  # Evaluate deferred MLP to compute view-dependent colors.
  rgb_specular = model(features)  # Ux3
  rgb = rgb_diffuse + rgb_specular  # Ux3

  # Predicted color for each pixel.
  rendering = {'rgb': rgb}

  # Debug info.
  ray_results = {'rgb_diffuse': rgb_diffuse, 'rgb_specular': rgb_specular}

  return rng, rendering, ray_results


def deferred_render_vfr(rng, features, acc, model, bg_rgbs=None):
  """Applies VFR's direct deferred rendeing model.

  Args:
    rng: jax.random.RandomState instance.
    features: f32[..., d]. Per-ray feature vectors.
    acc: f32[...]. Accumulated opacity per ray. Values in [0, 1].
    model: Function converting features to RGB. Outputs values should be in [0,
      1].
    bg_rgbs: f32[..., 3]. Optional background color.

  Returns:
    rng: Updated random number state.
    rendering: {str: array}. Contains predicted RGB after deferred rendering.
      Values should in [0, 1].
    ray_results: [str: array} Extra debug arrays.
  """
  # Predict colors directly.
  rgb = model(features)  # Ux3

  # Add background
  bg_weight = jnp.maximum(0, 1 - acc[Ellipsis, None])  # Ux1
  if bg_rgbs is None:
    rng, bg_rgbs = sample_bg_rgbs(rng, rgb.shape)
  rgb_with_bg = rgb + bg_weight * bg_rgbs  # Ux3

  # Predicted color for each pixel.
  rendering = {'rgb': rgb_with_bg}

  # Debug info.
  ray_results = {}

  return rng, rendering, ray_results


DEFERRED_RENDER_FNS = {
    'snerg': deferred_render_snerg,
    'vfr': deferred_render_vfr,
}
