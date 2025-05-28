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

import functools
from typing import Any, Callable

from flax import linen as nn
import gin
from internal import coord
from internal import grid_utils
from internal import hash_encoding
from internal import math
from internal import quantize
from internal import render
from internal import stepfun
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

gin.config.external_configurable(math.safe_exp, module='math')


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
):
  """Performs one round of hierarchical sampling."""
  # Performs one round of hierarchical sampling based on the volume rendering
  # weights from the previous round. More samples will be placed in segements
  # along the ray with high volume rendering weights, which effectively
  # leads to a concentration of samples at surfaces at the later rounds.
  #
  # This function also computes the distance of the interval endpoints to
  # the ray origin: sdist and tdist
  # The normalized `sdist` lie within [0, 1] , whereas the `tdist` are
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

  # Convert normalized distances to metric distances.
  tdist = coord.s_to_t(sdist, rays.near, rays.far)

  # Compute sampling positions along ray.
  positions = render.get_sample_positions_along_ray(
      tdist, rays.origins, rays.directions, rays.radii
  )

  return rng, positions, prod_num_samples, sdist, tdist, weights


def query_representation(
    positions, config, grid_config, density_and_features_mlp
):
  """Query our representation (triplane + sparse grid) at given 3D positions."""
  # The structure of the representation is enforced by grid simulation. All grid
  # values are predicted by a single MLP (DensityAndFeaturesMLP).

  # Flatten positions (makes our life easier).
  batch_shape = positions.shape[:-1]
  positions = positions.reshape(-1, 3)  # in: UxKx3, out: U*Kx3.

  # Prepare grid simulation, afterwards `positions` has the shape S*U*Kx3.
  # triplane_positions_local, sparse_grid_positions_local: U*Kx3.
  positions, triplane_positions_local, sparse_grid_positions_local = (
      grid_utils.get_eval_positions_and_local_coordinates(
          positions, config, grid_config
      )
  )

  # Query MLP at grid corners (S*U*Kx7 and S*U*Kx1).
  features, density = density_and_features_mlp(positions)

  # Simulate quantization on MLP outputs.
  features = quantize.simulate_quantization(
      features, config.range_features[0], config.range_features[1]
  )
  density = quantize.simulate_quantization(
      density, config.range_density[0], config.range_density[1]
  )

  # Grid simulation: bi-lineary and/or tri-linearly interpolate outputs.
  features = grid_utils.interpolate_based_on_local_coordinates(
      features, triplane_positions_local, sparse_grid_positions_local, config
  )  # U*Kx7.
  density = grid_utils.interpolate_based_on_local_coordinates(
      density, triplane_positions_local, sparse_grid_positions_local, config
  )  # U*Kx1.

  # Apply activation functions after interpolation. Doing this after
  # interpolation increases the model's representational power.
  features = nn.sigmoid(features)
  density = math.density_activation(density)

  # Unflatten results.
  def unflatten(x):
    return x.reshape(*batch_shape, -1)

  features = unflatten(features)  # UxKx7.
  density = unflatten(density)  # UxKx3.
  return features, density


@gin.configurable
class Model(nn.Module):
  """Our volume rendering model maps rays to RGB colors."""
  # Rays (origins and directions) are mapped to RGB colors. During training we
  # employ hierarchical sampling using two Proposal-MLPs as in MipNeRF-360.

  config: Any = None  # A Config class, must be set upon construction.
  num_prop_samples: int = 64  # The number of samples for each proposal level.
  num_final_samples: int = 32  # The number of samples for the final level.
  num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      train_frac,
      positions=None,
      return_ray_results=False,
      alpha_threshold=0.0,
  ):
    """Maps ray origins and ray directions to RGB colors via volume rendering."""

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
    density_and_features_mlp = DensityAndFeaturesMLP()
    prop_mlp = PropMLP()
    grid_config = grid_utils.calculate_grid_config(self.config)

    # When `positions` is set we directly query the representation, which
    # is only used for computing the sparsity loss currently.
    if positions is not None:
      _, density = query_representation(
          positions, self.config, grid_config, density_and_features_mlp
      )
      return density

    # Initialize the range of (normalized) distances for each ray to [0, 1],
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

    ray_history = []
    for i_level in range(self.num_levels):
      is_prop = i_level < (self.num_levels - 1)
      num_samples = (
          self.num_prop_samples if is_prop else self.num_final_samples
      )  # = K

      # Obtain sampling positions via hierarchical sampling
      # positions: UxKx3, sdist: UxK+1x3, tdist: U+K+1x3.
      rng, positions, prod_num_samples, sdist, tdist, _ = (
          one_round_of_hierarchical_sampling(
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
          )
      )

      # Contraction: positions will lie within (-2, 2)^3 afterwards.
      positions = coord.contract(positions)

      if is_prop:
        # Query PropMLP.
        density = prop_mlp(positions)
        density = math.density_activation(density)
      else:
        features, density = query_representation(
            positions, self.config, grid_config, density_and_features_mlp
        )  # UxKx7 and UxKx1.

        density = simulate_alpha_culling(
            density,
            positions,
            rays.viewdirs,
            alpha_threshold,
            grid_config['voxel_size_to_use'],
        )
      density = density[Ellipsis, 0]  # UxK.

      # Compute the weights used for volumetric rendering (and other losses).
      weights = render.compute_volume_rendering_weights(
          density, tdist, rays.directions
      )  # UxK.

      # Keep track of these values for distortion and interlevel losses.
      ray_results = {}
      ray_results['sdist'] = jnp.copy(sdist)
      ray_results['weights'] = jnp.copy(weights)
      ray_history.append(ray_results)

      rendering = {}
      if not is_prop:
        # Blend features based on volume rendering weights.
        acc = weights.sum(axis=-1)  # U
        bg_weight = jnp.maximum(0, 1 - acc[Ellipsis, None])  # Ux1
        features_blended = (weights[Ellipsis, None] * features).sum(axis=-2)  # Ux7

        # Blend diffuse RGB color with random (not learnable) background color.
        rgb_diffuse = features_blended[Ellipsis, :3]  # Ux3
        rng, bg_rgbs = sample_bg_rgbs(rng, rgb_diffuse.shape)  # pylint: disable=unused-variable
        rgb_diffuse = rgb_diffuse + bg_weight * bg_rgbs
        features_blended = features_blended.at[Ellipsis, :3].set(rgb_diffuse)

        # Evaluate deferred MLP to compute view-dependent colors.
        rgb_specular = DeferredMLP()(features_blended, rays.viewdirs)  # Ux3
        rendering['rgb'] = rgb_diffuse + rgb_specular  # Ux3
        rendering['acc'] = acc

        # These quantities are required for visbility culling
        # (see compute_alive_voxels).
        if return_ray_results:
          rendering['tdist'] = tdist
          rendering['weights'] = weights
          rendering['density'] = density
        return rendering, ray_history


def construct_model(rng, rays, config):
  """Constructs the model and forwards one ray for debugging."""
  # Grab just the first ray, to minimize memory overhead during construction.
  ray = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, [-1, x.shape[-1]])[:1], rays
  )
  model = Model(config=config)
  init_variables = model.init(
      rng,  # The RNG used by flax to initialize random weights.
      rng=None,  # The RNG used by sampling within the model.
      rays=ray,
      train_frac=1.0,
  )
  return model, init_variables


class PropMLP(nn.Module):
  """Proposal-MLP with hash encoding.

  Maps 3D positions to densities used to guide sampling of our representation.
  """

  net_depth: int = 1  # The depth of the first part of MLP.
  net_width: int = 64  # The width of the first part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(self, positions):
    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    # Predict density and feature vector.
    x = hash_encoding.PropHashEncoding()(positions)
    for _ in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
    density = dense_layer(1)(x)
    return density


class DensityAndFeaturesMLP(nn.Module):
  """This MLP parameterizes our grids."""
  # It predicts the values at gridcorners. By employing a hash encoding the MLP
  # is relatively light-weight.

  net_depth: int = 1  # The depth of the first part of MLP.
  net_width: int = 64  # The width of the first part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(self, positions):
    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    # Predict density and feature vector.
    x = hash_encoding.HashEncoding()(positions)
    for _ in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
    density = dense_layer(1)(x)
    features = dense_layer(NUM_CHANNELS)(x)
    return features, density


@gin.configurable
class DeferredMLP(nn.Module):
  """View-dependent RGB colors are predicted by this tiny MLP."""
  # This MLP only is only queried once per ray as in SNeRG
  # (https://arxiv.org/abs/2103.14645). The webviewer assumes a fixed
  # architecture and any changes to the hyperparameters below (except
  # weight_init) would require adapting the webviewer's code.
  net_depth: int = 2
  net_width: int = 16
  deg_enc: int = 4
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.

  @nn.compact
  def __call__(self, features, viewdirs):
    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )
    viewdirs = coord.pos_enc(
        viewdirs, min_deg=0, max_deg=self.deg_enc, append_identity=True
    )
    x = [features, viewdirs]
    x = jnp.concatenate(x, axis=-1)
    for _ in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
    return jax.nn.sigmoid(dense_layer(3)(x))


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


def simulate_alpha_culling(
    density, positions, viewdirs, alpha_threshold, voxel_size_to_use
):
  """Computes the alpha value based on a constant step size."""
  # During real-time rendering a constant step size (i.e. voxel size) is used.
  # During training a variable step size is used that can be vastly different
  # from the voxel size. When baking we discard voxels that would only
  # contribute neglible alpha values in the real-time renderer. To make this
  # lossless we already simulate the behaviour of the real-time renderer during
  # training by ignoring alpha values below the threshold.

  def zero_density_below_threshold(density):
    viewdirs_b = jnp.broadcast_to(
        viewdirs[Ellipsis, None, :], positions.shape
    ).reshape(-1, 3)
    positions_b = positions.reshape(-1, 3)
    step_size_uncontracted = coord.stepsize_in_squash(
        positions_b, viewdirs_b, voxel_size_to_use
    )
    step_size_uncontracted = step_size_uncontracted.reshape(density.shape)
    alpha = math.density_to_alpha(density, step_size_uncontracted)
    return jnp.where(
        alpha >= alpha_threshold, density, 0.0
    )  # density = 0 <=> alpha = 0

  return jax.lax.cond(
      alpha_threshold > 0.0, zero_density_below_threshold, lambda x: x, density
  )


def render_image(
    render_fn, rays, rng, config, verbose=True, transfer_to_cpu=False
):
  """Rendes an image (in test mode)."""
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
      chunk_rendering = jax.tree_util.tree_map(
          lambda x: np.array(x), chunk_rendering  # pylint: disable=unnecessary-lambda
      )
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
