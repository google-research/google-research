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

"""Training step and model creation functions."""

import collections
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple

from absl import logging
from camp_zipnerf.internal import datasets as mipnerf360_datasets
from camp_zipnerf.internal import image_utils as mipnerf360_image_utils
from camp_zipnerf.internal import utils as mipnerf360_utils
import chex
import flax
from flax.core import scope
from flax.training import train_state
import jax
from jax import random
import jax.numpy as jnp
import optax
from smerf.internal import configs
from smerf.internal import coord
from smerf.internal import datasets
from smerf.internal import grid_utils
from smerf.internal import image
from smerf.internal import math
from smerf.internal import models
from smerf.internal import render
from smerf.internal import stepfun
from smerf.internal import utils


def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
  return tree_sum(jax.tree_util.tree_map(lambda x: jnp.sum(x**2), tree))


def tree_norm(tree):
  return jnp.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
  return jax.tree_util.tree_reduce(
      lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), tree, initializer=0
  )


def tree_len(tree):
  return tree_sum(
      jax.tree_util.tree_map(lambda z: jnp.prod(jnp.array(z.shape)), tree)
  )


def summarize_tree(fn, tree, ancestry=(), max_depth=3):
  """Flatten 'tree' while 'fn'-ing values and formatting keys like/this."""
  stats = {}
  for k, v in tree.items():
    name = ancestry + (k,)
    stats['/'.join(name)] = fn(v)
    if hasattr(v, 'items') and len(ancestry) < (max_depth - 1):
      stats.update(summarize_tree(fn, v, ancestry=name, max_depth=max_depth))
  return stats


def mse(x, y):
  return jnp.square(jnp.mean(x - y))


def yu_sparsity_loss(
    sm_idxs,
    s_positions,
    viewdirs,
    density,
    config,
    grid_config,
):
  """Loss penalizing alpha for voxel-sized step sizes."""
  random_sm_positions = coord.uncontract(s_positions, allow_inf=False)
  sm_step_size = coord.sm_stepsize_from_s_stepsize(
      random_sm_positions, viewdirs, grid_config['voxel_size_to_use']
  )
  t_step_size = coord.sm_dist_to_world_dist(
      sm_idxs, sm_step_size, config, grid_config
  )
  return 1.0 - jnp.exp(-t_step_size * density).mean()


def distill_weight_from_rays(
    rays, config
):
  """Weight for each camera ray in distillation losses.

  Args:
    rays: Rays object with entries of shape f32[..., d].
    config: Config instance.

  Returns:
    weights: f32[...]. Weight assigned to each ray.
  """
  # Convert to ray origins to coordinates.
  grid_config = config.grid_config
  sm_idxs = rays.sm_idxs
  sm_origins = coord.world_to_submodel(
      sm_idxs, rays.origins, config, grid_config
  )
  s_origins = coord.contract(sm_origins)

  # Calculate how far each origin is from its nearest neighbor in uncontracted
  # space. Values in [0, 1], bigger = farther away.
  s_closest = jnp.clip(s_origins, -1, 1)
  squash_dist = jnp.linalg.norm(s_origins - s_closest, axis=-1, ord=jnp.inf)

  # This shouldn't be necessary, but just in case...
  squash_dist = jnp.clip(squash_dist, 0, 1)

  # Weights are larger where distance is small.
  weights = (1 - squash_dist) ** config.distill_weight_power

  # Divide by batch size. Now sum(weights * values) is roughly the same as
  # mean(values).
  weights = weights / weights.size
  return weights


def distill_geometry_loss(
    rays,
    teacher_ray_history,
    student_ray_history,
    train_frac,
    *,
    config,
):
  """Distillation loss for weights."""
  stats = {}

  # Extract matching quantities from teacher, student. Use ray history from
  # NerfMLP, not PropMLP.
  t_tdist = jax.lax.stop_gradient(teacher_ray_history['tdist'])
  t_weights = jax.lax.stop_gradient(teacher_ray_history['weights'])

  # Only use final ray history.
  student_ray_history = student_ray_history[-1]
  s_tdist = student_ray_history['tdist']
  s_weights = student_ray_history['weights']

  # Find lower & upper bound on teacher's weight for each interval queried by
  # the student.
  w_lower, w_upper = stepfun.inner_outer2(s_tdist, t_tdist, t_weights)

  # Penalize student from deviating from these bounds.
  loss_lower = jnp.maximum(0, w_lower - s_weights)
  loss_upper = jnp.maximum(0, s_weights - w_upper)

  # Average across all rays.
  loss_geometry_per_point = {
      'lower': loss_lower,
      'upper': loss_upper,
      'both': loss_lower + loss_upper,
  }[config.distill_geometry_loss_mode]

  # Sum over points along ray.
  loss_geometry_per_ray = jnp.sum(loss_geometry_per_point, axis=-1)

  # Weighted loss.
  weight = distill_weight_from_rays(rays, config)
  assert weight.shape == loss_geometry_per_ray.shape
  loss_geometry = jnp.sum(weight * loss_geometry_per_ray)

  stats['distill_geometry_loss'] = loss_geometry
  stats['distill_geometry_loss_weights_sum'] = jnp.sum(weight)
  stats['distill_geometry_loss_weights_mean'] = jnp.mean(weight)
  stats['distill_geometry_loss_weights_q05'] = jnp.quantile(weight, 0.05)
  stats['distill_geometry_loss_weights_q95'] = jnp.quantile(weight, 0.95)

  distill_geometry_loss_mult = (
      config.distill_geometry_loss_mult_init * (1.0 - train_frac)
      + config.distill_geometry_loss_mult_final * train_frac
  )
  weighted_loss = loss_geometry * distill_geometry_loss_mult
  return weighted_loss, stats


def distill_rgb_loss(
    rays,
    teacher_rendering,
    student_rendering,
    *,
    config,
):
  """Distillation loss for rendered quantities."""
  stats = {}

  # Compute error between teacher and student's rendered color.
  t_rgb = teacher_rendering['rgb']
  s_rgb = student_rendering['rgb']

  # Compute loss for each ray.
  loss_fn = {
      'rmse': lambda x: jnp.mean(jnp.sqrt(1e-6 + x * x), axis=-1),
      'l2': lambda x: jnp.mean(x * x, axis=-1),
      'l1': lambda x: jnp.mean(jnp.abs(x), axis=-1),
      'cauchy': lambda x: jnp.mean(jnp.log(1.0 + x * x / 0.5), axis=-1),
  }.get(config.distill_rgb_loss_fn)
  if loss_fn is None:
    raise NotImplementedError(
        f'Unrecognized distill_rgb_loss_fn: {config.distill_rgb_loss_fn}'
    )
  loss_rgb_per_ray = loss_fn(t_rgb - s_rgb)

  # Weighted loss.
  weight = distill_weight_from_rays(rays, config)
  assert weight.shape == loss_rgb_per_ray.shape
  loss_rgb = jnp.sum(weight * loss_rgb_per_ray)

  stats['distill_rgb_loss'] = loss_rgb

  weighted_loss = config.distill_rgb_loss_mult * loss_rgb
  return weighted_loss, stats


def distill_ssim_loss(
    teacher_rendering,
    student_rendering,
    *,
    config,
):
  """Distillation loss for rendered quantities."""
  stats = {}

  # Compute error between teacher and student's rendered color.
  t_rgb = teacher_rendering['rgb']
  s_rgb = student_rendering['rgb']

  ssim_per_patch = mipnerf360_image_utils.tiled_ssim(t_rgb, s_rgb)
  loss_ssim_per_patch = mipnerf360_image_utils.ssim_to_dssim(ssim_per_patch)

  # Weighted loss
  if config.distill_weight_power > 0:
    raise NotImplementedError()
  loss_ssim = jnp.mean(loss_ssim_per_patch)

  stats['distill_ssim_loss'] = loss_ssim

  weighted_loss = config.distill_ssim_loss_mult * loss_ssim
  return weighted_loss, stats


def antiflicker_rgb_loss(rendering, antiflicker_rendering, config):
  """Anti-flicker loss for rendered RGB."""
  # If enabled, do not let the antiflicker loss affect the original rendering.
  rendering_rgb = rendering['rgb']
  if config.antiflicker_enable_stopgrad:
    rendering_rgb = jax.lax.stop_gradient(rendering_rgb)

  # Compute error between rendering and anti-flicker rendering.
  residual = rendering_rgb - antiflicker_rendering['rgb']
  loss_rgb_per_ray = jnp.mean(jnp.sqrt(1e-6 + residual * residual), axis=-1)

  # Average across all rays.
  loss_rgb = jnp.mean(loss_rgb_per_ray)

  stats = {
      'antiflicker_rgb_loss': loss_rgb,
  }

  weighted_loss = config.antiflicker_rgb_loss_mult * loss_rgb
  return weighted_loss, stats


def antiflicker_features_loss(ray_history, antiflicker_ray_history, config):
  """Anti-flicker loss for feature vectors."""
  # math.feature_activation() has already been applid here. In the default
  # case, this function is a sigmoid and this quantity is bounded.
  features_v1 = ray_history[-1]['features']
  features_v2 = antiflicker_ray_history[-1]['features']
  residual = features_v1 - features_v2

  loss_per_point = jnp.mean(jnp.sqrt(1e-6 + residual * residual), axis=-1)

  # Average across all points.
  loss_features = jnp.mean(loss_per_point)

  stats = {
      'antiflicker_features_loss': loss_features,
  }

  weighted_loss = config.antiflicker_features_loss_mult * loss_features
  return weighted_loss, stats


def compute_data_loss(batch, rendering, rays, config):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  stats = collections.defaultdict(lambda: [])

  # lossmult can be used to apply a weight to each ray in the batch.
  # For example: masking out rays, applying the Bayer mosaic mask, upweighting
  # rays from lower resolution images and so on.
  lossmult = rays.lossmult
  if lossmult is None:
    lossmult = jnp.ones_like(batch.rgb[Ellipsis, :3])
  else:
    lossmult = jnp.broadcast_to(lossmult, batch.rgb[Ellipsis, :3].shape)

  resid_sq = (rendering['rgb'] - batch.rgb[Ellipsis, :3]) ** 2
  denom = lossmult.sum()
  stats['mse'] = (lossmult * resid_sq).sum() / denom

  # Charbonnier loss.
  data_loss = jnp.sqrt(resid_sq + config.charb_padding**2)
  data_loss = (lossmult * data_loss).sum() / denom

  # If data_loss_mult is 0, replace data_loss entirely. This will sever the
  # connection during backprop and reduce computation that has no effect
  # on the model parameters.
  if config.data_loss_mult != 0.0:
    data_loss *= config.data_loss_mult
  else:
    data_loss = jnp.zeros_like(data_loss)
    stats['mse'] = jnp.zeros_like(stats['mse'])

  return data_loss, stats


def interlevel_loss(ray_history, config):
  """Computes the interlevel loss defined in mip-NeRF 360."""
  # Stop the gradient from the interlevel loss onto the NeRF MLP.
  last_ray_results = ray_history[-1]
  c = jax.lax.stop_gradient(last_ray_results['sdist'])
  w = jax.lax.stop_gradient(last_ray_results['weights'])
  loss_interlevel = 0.0
  for ray_results in ray_history[:-1]:
    cp = ray_results['sdist']
    wp = ray_results['weights']
    loss_interlevel += jnp.mean(stepfun.lossfun_outer(c, w, cp, wp))
  return config.interlevel_loss_mult * loss_interlevel


def distortion_loss(ray_history, config):
  """Computes the distortion loss regularizer defined in mip-NeRF 360."""
  last_ray_results = ray_history[-1]
  c = last_ray_results['sdist']
  w = last_ray_results['weights']
  loss = jnp.mean(stepfun.lossfun_distortion(c, w, normalize=False))
  return config.distortion_loss_mult * loss


def param_regularizer_loss(variables, config):
  """Computes regularizer loss(es) over optimized parameters."""
  # Do some bookkeeping to ensure that every regularizer is valid.
  reg_used = {k: False for k in config.param_regularizers}
  params_flat = flax.traverse_util.flatten_dict(variables['params'])
  loss_reg = 0.0
  for name_tuple, param in params_flat.items():
    name = '/'.join(name_tuple)
    for prefix in config.param_regularizers:
      if name.startswith(prefix):
        reg_used[prefix] = True
        mult, acc_fn, alpha, scale = config.param_regularizers[prefix]
        loss_reg += mult * acc_fn(math.lossfun(param, alpha, scale))

  # If some regularizer was not used, the gin config is probably wrong.
  unused_regs = [k for k, v in reg_used.items() if not v]
  if unused_regs:
    logging.warning(  # pylint: disable=logging-fstring-interpolation
        'The following regularizers were not'
        f' used:\n{unused_regs}\n\nThe following variable names are'
        f' available for regularization:\n{params_flat.keys()}'
    )
  return loss_reg


def deferred_mlp_tv_regularizer_loss(variables, config):
  """Computes total variation regularizer for Deferred MLP's parameters."""
  # Loss functions for residuals.
  loss_fns = {
      'l1': jnp.abs,
      'l2': jnp.square,
  }

  zero = lambda dtype: jnp.zeros((), dtype=dtype)

  @jax.vmap
  def _tv_regularizer_loss_vmap(param):
    """Computes TV regularize on a single parameter.

    Args:
      param: f32[k, k, k, ...]. Parameter with three spatial dimensions.

    Returns:
      size: int. number of elements used to compute error.
      error: f32[]. mean of errors across all elements in this parameter.
    """
    k = param.shape[0]
    chex.assert_shape(param, (k, k, k, Ellipsis))
    if k <= 1:
      return zero(jnp.int32), zero(jnp.float32)
    loss_fn = loss_fns[config.deferred_mlp_tv_regularizer_loss_fn]
    di = loss_fn(param[:-1, :, :] - param[1:, :, :])
    dj = loss_fn(param[:, :-1, :] - param[:, 1:, :])
    dk = loss_fn(param[:, :, :-1] - param[:, :, 1:])
    size = di.size + dj.size + dk.size
    error = jnp.sum(di) + jnp.sum(dj) + jnp.sum(dk)
    return size, error

  def tv_regularizer_loss(param):
    if len(param.shape) >= 4:
      # Assume that param has shape [n,k,k,k,...].
      return _tv_regularizer_loss_vmap(param)
    # param cannot have shape [n,k,k,k,...]. TV regularizer is irrelevant.
    return zero(jnp.int32), zero(jnp.float32)

  # Parameters have shape [#submodels, k, k, k, ...] if
  # DeferredMLP.grid_size > 1 and shape [...] otherwise.
  params = variables['params']['DeferredMLP_0']
  params = flax.traverse_util.flatten_dict(params)

  # Compute sum of errors and number of elements contributing to that sum.
  # One entry per submodel.
  sizes_and_errors = list(map(tv_regularizer_loss, params.values()))
  total_size = sum([size for size, _ in sizes_and_errors], start=0)
  total_error = sum([error for _, error in sizes_and_errors], start=0.0)

  # Compute average error over all entries per submodel, then average over
  # submodels.
  losses = jnp.where(total_size > 0, total_error / total_size, 0.0)
  loss = jnp.mean(losses)

  loss = config.deferred_mlp_tv_regularizer_loss_mult * loss
  stats = {
      'deferred_mlp_tv_regularizer_size': jnp.array(total_size),
      'deferred_mlp_tv_regularizer_loss': total_error,
  }
  return loss, stats


def clip_gradients(grad, config):
  """Clips gradients of each MLP individually based on norm and max value."""
  # Clip the gradients of each MLP individually.
  grad_clipped = {'params': {}}
  for k, g in grad['params'].items():
    # Clip by value.
    if config.grad_max_val > 0:
      g = jax.tree_util.tree_map(
          lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), g
      )

    # Then clip by norm.
    if config.grad_max_norm > 0:
      mult = jnp.minimum(
          1, config.grad_max_norm / (jnp.finfo(jnp.float32).eps + tree_norm(g))
      )
      g = jax.tree_util.tree_map(lambda z: mult * z, g)  # pylint:disable=cell-var-from-loop

    grad_clipped['params'][k] = g
  grad = type(grad)(grad_clipped)
  return grad


def jitter_rays(
    # Array arguments
    rng,
    rays,
    # Constant arguments
    config,
    strict = True,
):
  """Jitter camera rays.

  Jitters the origins, directions, and viewdirs of a Rays instance. The
  relationship between directions and viewdirs is maintained.  All viewdirs
  within a single patch receive the same noise.

  Args:
    rng: Random seed.
    rays: Rays instance with values of shape f32[..., h, w, d]. The h, w
      dimensions correspond to a patch's height and width.
    config: Config instance.
    strict: Throw exception if data loss is enabled.

  Returns:
    rays_jittered: Same as rays, but with jittered ray origins, directions, and
      viewdirs. All ray directions/viewdirs in the same patch receive the same
      (approximate) noise vector.
  """
  if strict and config.data_loss_mult != 0.0:
    raise ValueError(
        'Do not use pixel supervision when ray jittering is enabled! The rays'
        ' will be pointed in the wrong direction.'
    )

  rng_origins, rng_viewdirs = jax.random.split(rng, 2)

  # Jitter ray origins.
  jittered_origins = jitter_origins(
      rng_origins,
      rays.origins,
      config.ray_jitter_origin_stddev,
      patch_aware=config.ray_jitter_origin_patch_aware,
  )

  # Apply a random rotation matrix to each patch of viewdirs.
  viewdirs_jittered = jitter_viewdirs(
      rng_viewdirs,
      rays.viewdirs,
      config.ray_jitter_viewdir_stddev,
      patch_aware=config.ray_jitter_viewdir_patch_aware,
  )

  # Maintain relationship between viewdirs, rays.
  norm = lambda x: jnp.linalg.norm(x, axis=-1, keepdims=True)
  directions_jittered = viewdirs_jittered * norm(rays.directions)

  rays_jittered = rays.replace(
      origins=jittered_origins,
      directions=directions_jittered,
      viewdirs=viewdirs_jittered,
  )
  return rays_jittered


pjitter_rays = jax.pmap(
    jitter_rays, in_axes=(0, 0), static_broadcasted_argnums=(2,)
)


def jitter_origins(
    rng,
    origins,
    stddev,
    patch_aware,
):
  """Jitter ray origins.

  Jittered origins are *NOT* guaranteed to lie in [-1, 1]^3.

  Args:
    rng: Random seed.
    origins: f32[..., h, w, 3]. Camera origins.
    stddev: float. Standard deviation for Gaussian noise.
    patch_aware: bool. If True, apply same jitter to each patch.

  Returns:
    origins_jittered: f32[..., h, w, 3]. Jittered origins vectors.
  """
  *batch_dims, h, w, c = origins.shape
  if patch_aware:
    shape = (*batch_dims, 1, 1, c)
  else:
    shape = (*batch_dims, h, w, c)
  noise = jax.random.normal(rng, shape=shape) * stddev
  origins_jittered = origins + noise
  return origins_jittered


def jitter_viewdirs(rng, viewdirs, perturbation, patch_aware=False):
  """Jitter unit-norm direction vectors.

  Samples random rotation matrices, one per patch, that encompasses a uniform
  distribution over a portion of the unit sphere.

  Args:
    rng: Random seed.
    viewdirs: f32[..., h, w, 3]. Unit-norm direction vectors for image patches.
    perturbation: float in [0, 1]. Percentage of the sphere to sample from.
      Lower values result in smaller perturbations.
    patch_aware: bool. If True, apply same jitter to each patch.

  Returns:
    viewdirs_jittered: f32[..., h, w, 3]. Jittered direction vectors.
  """
  # Extract patch size.
  *batch_dims, h, w, _ = viewdirs.shape

  # Sample random rotation matrix
  if patch_aware:
    shape = (*batch_dims, 1, 1)
  else:
    shape = (*batch_dims, h, w)
  rot = math.random_rotations(rng, perturbation=perturbation, shape=shape)
  chex.assert_shape(rot, (*shape, 3, 3))

  # Apply same rotation matrix to all rays in a single patch
  jittered_viewdirs = jnp.einsum('...ij, ...j -> ...i', rot, viewdirs)
  chex.assert_shape(jittered_viewdirs, viewdirs.shape)

  return jittered_viewdirs


def sample_random_positions(
    rng, model, variables, train_frac, config, grid_config
):
  """Samples random positions in squash space and evaluates density there."""
  # Sample a fixed number of points in s-coordinates [-2,2]^3. Assign
  # them to random submodels.
  num_random_samples = (
      config.num_random_samples // config.gradient_accumulation_steps
  )
  random_samples_keys = random.split(rng, 3)
  random_sm_idxs = jax.random.choice(
      random_samples_keys[0],
      jnp.array(grid_config['submodels_on_host']),
      shape=(num_random_samples, 1),
  ).astype(jnp.int32)
  random_s_positions = jax.random.uniform(
      random_samples_keys[1],
      shape=(num_random_samples, 3),
      dtype=jnp.float32,
      minval=grid_utils.WORLD_MIN,
      maxval=grid_utils.WORLD_MAX,
  )
  random_viewdirs = jax.random.normal(
      random_samples_keys[2], shape=(num_random_samples, 3)
  )
  random_viewdirs /= jnp.linalg.norm(
      random_viewdirs, axis=-1, keepdims=True
  )
  _, random_samples_density = model.apply(
      variables,
      rng=None,
      rays=None,
      train_frac=train_frac,
      sm_idxs=random_sm_idxs,
      s_positions=random_s_positions,
  )
  return (
      random_sm_idxs,
      random_s_positions,
      random_viewdirs,
      random_samples_density,
  )


def _shift_rays_forward(
    # arrays
    rng,  # 0
    rays,  # 1
    teacher_tdist,  # 2
    # static
    config,  # 3
):
  """Shift rays forward.

  Args:
    rng: jax.random.RandomState.
    rays: utils.Rays instance.
    teacher_tdist: tdist intervals from teacher.
    config: configs.Config instance.

  Returns:
    Updated rays.
  """
  rng_forward, rng_viewdirs = jax.random.split(rng, 2)

  # Choose a distance along the camera ray to move. Move up to, but not past,
  # the first tdist value proposed by the teacher.
  far = teacher_tdist[Ellipsis, :1]  # f32[..., 1]
  far = far * config.distill_shift_origins_forward_buffer
  near = rays.near  # f32[..., 1]
  if near is None:
    near = jnp.zeros_like(far)  # f32[..., 1]
  far = jnp.maximum(near, far)  # Make sure that far is no smaller than near.
  tdist = jax.random.uniform(rng_forward, near.shape)  # f32[..., 1] in [0, 1)
  tdist = near + tdist * (far - near)  # f32[..., 1] in [near, far)

  # Update rays origins. Origins are pushed forward along the camera ray.
  new_origins = rays.origins + tdist * rays.directions  # f32[..., 3]
  chex.assert_equal_shape((rays.origins, new_origins))

  # Update sm_idxs. Use the closest submodel to each new ray origin.
  new_sm_idxs = coord.nearest_submodel(
      t_positions=new_origins,
      config=config,
      grid_config=config.grid_config,
  )
  chex.assert_equal_shape((rays.sm_idxs, new_sm_idxs))

  # Update rays.
  rays = rays.replace(origins=new_origins, sm_idxs=new_sm_idxs)

  # Jitter view directions after advancing forward.
  if config.distill_shift_origins_forward_jitter_viewdirs:
    # Update viewdirs.
    new_viewdirs = jitter_viewdirs(
        rng_viewdirs, rays.viewdirs, config.ray_jitter_viewdir_stddev
    )
    chex.assert_equal_shape((rays.viewdirs, new_viewdirs))

    # Update directions.
    norm = lambda x: jnp.linalg.norm(x, axis=-1, keepdims=True)
    new_directions = new_viewdirs * norm(rays.directions)
    chex.assert_equal_shape((rays.directions, new_directions))

    # Update rays.
    rays = rays.replace(directions=new_directions, viewdirs=new_viewdirs)

  return rays


_pshift_rays_forward = jax.pmap(
    _shift_rays_forward,
    in_axes=(0, 0, 0),
    static_broadcasted_argnums=(3,)
)


def pshift_batch_forward(
    prng,
    pbatch,
    teacher_pstate,
    prender_teacher,
    config,
):
  """Pushes rays forward along camera ray.

  Push ray origins forward along the camera ray until the teacher model's
  first suggested ray interval.

  Ray directions may be jittered after being pushed forward.

  Args:
    prng: ...
    pbatch: ...
    teacher_pstate: ...
    prender_teacher: ...
    config: ...

  Returns:
    Updated batch.
  """
  if not config.distill_shift_origins_forward:
    return pbatch

  # TODO(duckworthd): Fix the implementation below. It's not patch-aware.
  del prng, teacher_pstate, prender_teacher
  raise NotImplementedError(
      'The code below is broken and leads to quality losses! Do not use it.'
  )

  # Query teacher model to know how far forward a ray can advance before
  # hitting a surface. Do not jitter ray intervals.
  teacher_ray_history = prender_teacher(None, teacher_pstate, pbatch)

  # Shift rays forward.
  rays = _pshift_rays_forward(
      prng, pbatch.rays, teacher_ray_history['tdist'], config
  )

  return pbatch.replace(rays=rays)


def construct_student_tdist(
    rng,
    rays,
    teacher_ray_history,
    config,
    grid_config,
):
  """Computes tdist values for student model.

  By default, uses tdist values provided by the teacher. If random intervals
  are requested, this set of tdist values is augmented by randomly-sampled
  ray intervals with a length matching that used in real-time rendering.
  These ray intervals are guaranteed to start within the interval set
  provided by the teacher.

  Args:
    rng: Random seed.
    rays: Batch of rays to render. Arrays have shape f32[..., C].
    teacher_ray_history: Quantities produced by the teacher. Must contain key
      'tdist' with a value of shape f32[..., S+1].
    config: ...
    grid_config: ...

  Returns:
    student_tdist: f32[.., S'+1] where S' = S+1 or S+2K+1, depending on the
      config. Each collection of entries per ray will be ordered from smallest
      to largest.
  """
  student_tdist = None

  if config.distill_use_teacher_tdist:
    logging.info('Using teacher tdist values...')
    # Query the student model at the teacher's intervals.
    student_tdist = teacher_ray_history['tdist']  # f32[..., S+1]

  if config.num_random_student_intervals_per_ray > 0:
    logging.info('Adding random ray intervals to tdist...')
    # Insert extra intervals before the first student tdist value.
    assert student_tdist is not None
    batch_dims = student_tdist.shape[:-1]  # f32[...]

    # Sample start distances for ray intervals.
    # We sample one extra point per ray in order to use
    # get_sample_positions_along_ray(), which will treat t_starts as a series
    # of fenceposts.
    shape = (*batch_dims, config.num_random_student_intervals_per_ray + 1)
    near = jnp.zeros_like(student_tdist[Ellipsis, :1])
    far = jnp.max(student_tdist, axis=-1, keepdims=True)
    t_starts = sample_tdist(rng, shape, near, far)

    # Get position in world coordinates. One point is returned per fencepost
    # pair. The actual distance from the ray origin to the point is somewhere
    # the two fenceposts.
    t_positions = render.get_sample_positions_along_ray(
        tdist=t_starts,
        origins=rays.origins,
        directions=rays.directions,
        radii=rays.radii,
    )  # f32[..., K, 3]

    # Compute step size in world coordinates corresponding to voxel_size_to_use
    # in squash coordinates.
    def broadcast_along_ray(x, y):
      shape = y.shape[:-1] + x.shape[-1:]
      return jnp.broadcast_to(x[Ellipsis, jnp.newaxis, :], shape)

    t_step_size = coord.t_stepsize_from_s_stepsize(
        t_positions=t_positions,
        t_viewdirs=broadcast_along_ray(rays.viewdirs, t_positions),
        sm_idxs=broadcast_along_ray(rays.sm_idxs, t_positions),
        s_stepsizes=grid_config['voxel_size_to_use'],
        config=config,
        grid_config=grid_config,
    )  # f32[..., K]

    # Compute end for new tdist intervals. Note that tdist values are
    # multiplied by rays.directions, which may have a different norm than
    # rays.viewdirs.
    directions_norm = jnp.linalg.norm(rays.directions, axis=-1, keepdims=True)
    t_ends = t_starts[Ellipsis, :-1] + t_step_size / directions_norm  # f32[..., K]

    # Merge intervals from teacher, t_starts, and t_ends. We sort to satisfy
    # tdist.
    tdists = jnp.concatenate(
        [student_tdist, t_starts[Ellipsis, :-1], t_ends], axis=-1
    )  # f32[..., S+1+2K]
    student_tdist = jnp.sort(tdists, axis=-1)

  return student_tdist


def sample_tdist(rng, shape, near, far):
  """Samples tdist values of target shape uniformly at random.

  Args:
    rng: Random seed.
    shape: tuple of ints. Output shape.
    near: float or f32 array with a shape compatible with the output shape.
      Lower bounds for intervals.
    far: float or f32 array with a shape compatible with the output shape.
      Upper bounds for intervals.

  Returns:
    tdist: f32[...] with shape matching `shape`. Last axis is sorted.
  """
  tdist = jax.random.uniform(rng, shape)  # f32[...] in [0, 1)
  tdist = near + tdist * (far - near)  # f32[...] in [near, far)
  tdist = jnp.sort(tdist, axis=-1)  # f32[...]
  return tdist


def create_train_step(
    model,
    config,
):
  """Creates the pmap'ed Nerf training function.

  Args:
    model: The linen model.
    config: The configuration.

  Returns:
    pmap'ed training function.
  """

  def train_step(
      rng,
      state,
      batch,
      teacher_ray_history,
      train_frac,
  ):
    """One optimization step.

    Args:
      rng: jnp.ndarray, random number generator.
      state: train_state.TrainState, state of the model/optimizer.
      batch: Batch, a mini-batch of data for training.
      teacher_ray_history: predictions from a teacher model.
      train_frac: float, the fraction of training that is complete.

    Returns:
      A tuple (new_state, stats, rng) with
        new_state: train_state.TrainState, new training state.
        stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        rng: jnp.ndarray, updated random number generator.
    """
    # Rays must already be cast.
    assert batch.rays.origins is not None

    step = state.step // config.gradient_accumulation_steps
    rng, render_rng, random_sample_rng, teacher_rng, antiflicker_rng = (
        random.split(rng, 5)
    )
    grid_config = config.grid_config

    def loss_fn(variables):
      rays = batch.rays

      alpha_threshold = 0.0
      if config.alpha_threshold is not None:
        alpha_threshold = config.alpha_threshold(step)

      def model_apply(rays, **kwargs):
        return model.apply(
            variables,
            render_rng if config.randomized else None,
            rays,
            train_frac=train_frac,
            alpha_threshold=alpha_threshold,
            **kwargs,
        )

      # Apply forward pass without teacher.
      rendering, ray_history = model_apply(rays)

      # Sample random positions in squash space and evaluate the density there.
      need_random_samples = config.yu_sparsity_loss_mult is not None
      random_sm_idxs = random_s_positions = random_viewdirs = None
      random_samples_density = None
      if need_random_samples:
        logging.info('Sampling random positions...')
        (
            random_sm_idxs,
            random_s_positions,
            random_viewdirs,
            random_samples_density,
        ) = sample_random_positions(
            rng=random_sample_rng,
            model=model,
            variables=variables,
            train_frac=train_frac,
            config=config,
            grid_config=grid_config,
        )

      # Predict density at same points the teacher was queried at.
      need_teacher_samples = (
          config.distill_use_teacher_tdist
          or config.distill_use_teacher_exposure
          or config.distill_rgb_loss_mult > 0
          or config.distill_ssim_loss_mult > 0
          or config.distill_geometry_loss_mult_init > 0
      ) and teacher_ray_history is not None
      student_rays = student_tdist = None
      if need_teacher_samples:
        logging.info('Making predictions for distillation...')
        # Use teacher's exposure predictions instead of ground truth if it's
        # provided. This will be the case if an ExposureMLP is in use.
        student_rays = rays

        if config.distill_use_teacher_exposure:
          teacher_exposure_values = teacher_ray_history['exposure_prediction']
          if teacher_exposure_values is not None:
            logging.info('Overriding exposure with teacher prediction...')
            student_rays = student_rays.replace(
                exposure_values=teacher_exposure_values
            )

        # Query the student model at the teacher's intervals.
        student_tdist = construct_student_tdist(
            rng=teacher_rng,
            rays=student_rays,
            teacher_ray_history=teacher_ray_history,
            config=config,
            grid_config=grid_config,
        )

        # Override 'rendering' and 'ray_history' from above.
        rendering, ray_history = model_apply(
            rays=student_rays, tdist_override=student_tdist
        )

      need_antiflicker_samples = (
          config.antiflicker_rgb_loss_mult > 0
          or config.antiflicker_features_loss_mult > 0
      )
      antiflicker_rendering = antiflicker_ray_history = None
      if need_antiflicker_samples:
        logging.info('Making predictions for antiflicker...')
        antiflicker_rays = student_rays if need_teacher_samples else rays
        antiflicker_tdist = student_tdist

        # Sample neighbor submodel idxs.
        neighbor_sm_idxs = coord.sample_random_neighbor_sm_idx(
            antiflicker_rng,
            antiflicker_rays.origins,
            config,
            config.grid_config,
        )
        chex.assert_equal_shape((antiflicker_rays.sm_idxs, neighbor_sm_idxs))
        antiflicker_rays = antiflicker_rays.replace(sm_idxs=neighbor_sm_idxs)

        # Render one more time, this time using a neighboring submodel.
        #
        # TODO(duckworthd): Only render a subset of rays.
        antiflicker_rendering, antiflicker_ray_history = model_apply(
            rays=antiflicker_rays, tdist_override=antiflicker_tdist
        )

      losses = {}

      logging.info('Applying Data Loss...')
      data_loss, stats = compute_data_loss(batch, rendering, rays, config)
      losses['data'] = data_loss

      if config.interlevel_loss_mult > 0:
        logging.info('Applying Interlevel Loss...')
        losses['interlevel'] = interlevel_loss(ray_history, config)

      if config.distortion_loss_mult > 0:
        logging.info('Applying Distortion Loss...')
        losses['distortion'] = distortion_loss(ray_history, config)

      if config.param_regularizers:
        logging.info('Applying Param Regularizers...')
        losses['regularizer'] = param_regularizer_loss(variables, config)

      if config.deferred_mlp_tv_regularizer_loss_mult > 0:
        logging.info('Applying TV regularizer for Grid MLP...')
        tv_loss, tv_stats = (
            deferred_mlp_tv_regularizer_loss(variables, config)
        )
        losses['deferred_mlp_tv_regularizer'] = tv_loss
        stats.update(tv_stats)

      if config.yu_sparsity_loss_mult is not None:
        logging.info('Applying Yu Sparsity loss...')
        losses['yu_sparsity_loss'] = config.yu_sparsity_loss_mult(
            step
        ) * yu_sparsity_loss(
            sm_idxs=random_sm_idxs,
            s_positions=random_s_positions,
            viewdirs=random_viewdirs,
            density=random_samples_density,
            config=config,
            grid_config=grid_config,
        )

      if config.distill_geometry_loss_mult_init > 0:
        logging.info('Applying Distill Geometry loss...')
        distill_loss, distill_stats = distill_geometry_loss(
            rays=rays,
            teacher_ray_history=teacher_ray_history,
            student_ray_history=ray_history,
            config=config,
            train_frac=train_frac,
        )
        losses['distill_geom_loss'] = distill_loss
        stats.update(distill_stats)

      if config.distill_rgb_loss_mult > 0:
        logging.info('Applying Distill RGB loss...')
        distill_loss, distill_stats = distill_rgb_loss(
            rays=rays,
            teacher_rendering={
                'rgb': teacher_ray_history['rendered_rgb']
            },
            student_rendering=rendering,
            config=config,
        )
        losses['distill_rgb_loss'] = distill_loss
        stats.update(distill_stats)

      if config.distill_ssim_loss_mult > 0:
        logging.info('Applying Distill SSIM loss...')
        distill_loss, distill_stats = distill_ssim_loss(
            teacher_rendering={
                'rgb': teacher_ray_history['rendered_rgb']
            },
            student_rendering=rendering,
            config=config,
        )
        losses['distill_ssim_loss'] = distill_loss
        stats.update(distill_stats)

      if config.antiflicker_rgb_loss_mult > 0:
        logging.info('Applying Anti-flicker RGB loss...')
        antiflicker_loss, antiflicker_stats = antiflicker_rgb_loss(
            rendering=rendering,
            antiflicker_rendering=antiflicker_rendering,
            config=config,
        )
        losses['antiflicker_rgb_loss'] = antiflicker_loss
        stats.update(antiflicker_stats)

      if config.antiflicker_features_loss_mult > 0:
        logging.info('Applying Anti-flicker Features loss...')
        antiflicker_loss, antiflicker_stats = antiflicker_features_loss(
            ray_history=ray_history,
            antiflicker_ray_history=antiflicker_ray_history,
            config=config,
        )
        losses['antiflicker_features_loss'] = antiflicker_loss
        stats.update(antiflicker_stats)

      stats['loss'] = jnp.sum(jnp.array(list(losses.values())))
      stats['losses'] = losses

      return stats['loss'], stats

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, stats), grad = loss_grad_fn(state.params)

    pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
    grad = pmean(grad)
    stats = pmean(stats)

    stats['grad_norms'] = summarize_tree(tree_norm, grad['params'])
    stats['grad_maxes'] = summarize_tree(tree_abs_max, grad['params'])

    grad = clip_gradients(grad, config)
    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)

    new_state = state.apply_gradients(grads=grad)

    opt_delta = jax.tree_util.tree_map(
        lambda x, y: x - y, new_state, state
    ).params['params']
    stats['opt_update_norms'] = summarize_tree(tree_norm, opt_delta)
    stats['opt_update_maxes'] = summarize_tree(tree_abs_max, opt_delta)

    stats['psnr'] = image.mse_to_psnr(stats['mse'])
    return new_state, stats, rng

  train_pstep = jax.pmap(
      train_step,
      axis_name='batch',
      in_axes=(0, 0, 0, 0, None),
      donate_argnums=(0, 1),
  )
  return train_pstep


def create_optimizer(
    config, variables
):
  """Creates optax optimizer for model training."""
  adam_kwargs = {
      'b1': config.adam_beta1,
      'b2': config.adam_beta2,
      'eps': config.adam_eps,
  }
  lr_kwargs = {
      'max_steps': config.max_steps,
      'lr_delay_steps': config.lr_delay_steps,
      'lr_delay_mult': config.lr_delay_mult,
  }

  def get_lr_fn(lr_init, lr_final):
    return functools.partial(
        math.learning_rate_decay,
        lr_init=lr_init,
        lr_final=lr_final,
        **lr_kwargs,
    )

  lr_fn_main = get_lr_fn(config.lr_init, config.lr_final)
  tx = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)

  if config.gradient_accumulation_steps > 1:
    tx = optax.MultiSteps(
        tx, every_k_schedule=config.gradient_accumulation_steps
    )

  return (
      train_state.TrainState.create(apply_fn=None, params=variables, tx=tx),
      lr_fn_main,
  )


def create_render_fn(
    model, return_ray_results, final_alpha_threshold
):
  """Creates pmap'ed function for full image rendering."""

  def render_eval_fn(variables, train_frac, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            train_frac=train_frac,
            return_ray_results=return_ray_results,
            alpha_threshold=final_alpha_threshold,
        ),
        axis_name='batch',
    )

  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, None, 0),
      axis_name='batch',
  )
  return render_eval_pfn


def setup_model(
    config,
    rng,
    dataset,
    return_ray_results = False,
):
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

  temp_rays = datasets.cam_to_rays(dataset, 0)
  temp_rays = datasets.preprocess_rays(
      rays=temp_rays, mode='test', merf_config=config, dataset=dataset
  )
  model, variables = models.construct_model(rng, temp_rays, config)
  logging.info(
      'Model Variables: size: %f MB', _estimate_memory_usage_mb(variables)
  )

  final_alpha_threshold = 0.0
  if config.alpha_threshold is not None:
    final_alpha_threshold = config.alpha_threshold(config.max_steps)

  state, lr_fn = create_optimizer(config, variables)
  logging.info('Optimizer State: size: %f MB', _estimate_memory_usage_mb(state))

  render_eval_pfn = create_render_fn(
      model, return_ray_results, final_alpha_threshold
  )
  train_pstep = create_train_step(model, config)

  return model, state, render_eval_pfn, train_pstep, lr_fn


def _safe_shape(x):
  """Returns shape of an array if possible."""
  try:
    return x.shape
  except AttributeError:
    return x


def _estimate_memory_usage_mb(x):
  """Estimates number of megabytes in a PyTree."""
  arrays, _ = jax.tree_util.tree_flatten(x)
  result = 0.0
  for array in arrays:
    try:
      array = jax.device_get(array)
      result += array.size * array.itemsize
    except AttributeError:
      pass
  return result / (2 ** 20)
