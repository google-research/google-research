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

import flax
from flax.core import scope
from flax.training import train_state
from internal import camera_utils
from internal import configs
from internal import coord
from internal import datasets
from internal import grid_utils
from internal import image
from internal import math
from internal import models
from internal import stepfun
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import optax


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


def yu_sparsity_loss(random_positions, random_viewdirs, density, voxel_size):
  step_size = coord.stepsize_in_squash(
      random_positions, random_viewdirs, voxel_size
  )
  return 1.0 - jnp.exp(-step_size * density).mean()


def compute_data_loss(batch, rendering, rays, config):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  stats = collections.defaultdict(lambda: [])

  # lossmult can be used to apply a weight to each ray in the batch.
  # For example: masking out rays, applying the Bayer mosaic mask, upweighting
  # rays from lower resolution images and so on.
  lossmult = rays.lossmult
  lossmult = jnp.broadcast_to(lossmult, batch.rgb[Ellipsis, :3].shape)

  resid_sq = (rendering['rgb'] - batch.rgb[Ellipsis, :3]) ** 2
  denom = lossmult.sum()
  stats['mse'] = (lossmult * resid_sq).sum() / denom

  # Charbonnier loss.
  data_loss = jnp.sqrt(resid_sq + config.charb_padding**2)
  data_loss = (lossmult * data_loss).sum() / denom
  data_loss *= config.data_loss_mult
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
  for reg, used in reg_used.items():
    if not used:
      raise ValueError(f'Regularizer {reg} not used.')
  return loss_reg


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


def create_train_step(
    model,
    config,
    dataset = None,
):
  """Creates the pmap'ed Nerf training function.

  Args:
    model: The linen model.
    config: The configuration.
    dataset: Training dataset.

  Returns:
    pmap'ed training function.
  """
  if dataset is None:
    camtype = camera_utils.ProjectionType.PERSPECTIVE
  else:
    camtype = dataset.camtype

  def train_step(
      rng,
      state,
      batch,
      cameras,
      train_frac,
  ):
    """One optimization step.

    Args:
      rng: jnp.ndarray, random number generator.
      state: train_state.TrainState, state of the model/optimizer.
      batch: dict, a mini-batch of data for training.
      cameras: module containing camera poses.
      train_frac: float, the fraction of training that is complete.

    Returns:
      A tuple (new_state, stats, rng) with
        new_state: train_state.TrainState, new training state.
        stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        rng: jnp.ndarray, updated random number generator.
    """
    step = state.step // config.gradient_accumulation_steps
    keys = random.split(rng)
    grid_config = grid_utils.calculate_grid_config(config)

    def loss_fn(variables):
      rays = batch.rays
      if config.cast_rays_in_train_step:
        rays = camera_utils.cast_ray_batch(cameras, rays, camtype, xnp=jnp)

      alpha_threshold = 0.0
      if config.alpha_threshold is not None:
        alpha_threshold = config.alpha_threshold(step)

      rendering, ray_history = model.apply(
          variables,
          keys[0] if config.randomized else None,
          rays,
          train_frac=train_frac,
          alpha_threshold=alpha_threshold,
      )

      need_random_samples = config.yu_sparsity_loss_mult is not None
      if need_random_samples:
        # Sample a fixed number of points within [-2,2]^3.
        num_random_samples = (
            config.num_random_samples // config.gradient_accumulation_steps
        )
        random_positions = jax.random.uniform(
            keys[1],
            shape=(num_random_samples, 3),
            dtype=jnp.float32,
            minval=grid_utils.WORLD_MIN,
            maxval=grid_utils.WORLD_MAX,
        )
        random_viewdirs = jax.random.normal(
            keys[1], shape=(num_random_samples, 3)
        )
        random_viewdirs /= jnp.linalg.norm(
            random_viewdirs, axis=-1, keepdims=True
        )
        density = model.apply(
            variables,
            rng=None,
            rays=None,
            train_frac=train_frac,
            positions=random_positions,
        )

      losses = {}

      data_loss, stats = compute_data_loss(batch, rendering, rays, config)
      losses['data'] = data_loss

      if config.interlevel_loss_mult > 0:
        losses['interlevel'] = interlevel_loss(ray_history, config)

      if config.distortion_loss_mult > 0:
        losses['distortion'] = distortion_loss(ray_history, config)

      if config.param_regularizers:
        losses['regularizer'] = param_regularizer_loss(variables, config)

      if config.yu_sparsity_loss_mult is not None:
        voxel_size_to_use = grid_config['voxel_size_to_use']
        losses['yu_sparsity_loss'] = config.yu_sparsity_loss_mult(
            step
        ) * yu_sparsity_loss(
            random_positions, random_viewdirs, density, voxel_size_to_use
        )

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
      in_axes=(0, 0, 0, None, None),
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
    dataset = None,
    return_ray_results=False,
):
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

  temp_rays = datasets.cam_to_rays(dataset, 0)
  model, variables = models.construct_model(rng, temp_rays, config)

  final_alpha_threshold = 0.0
  if config.alpha_threshold is not None:
    final_alpha_threshold = config.alpha_threshold(config.max_steps)

  state, lr_fn = create_optimizer(config, variables)
  render_eval_pfn = create_render_fn(
      model, return_ray_results, final_alpha_threshold
  )
  train_pstep = create_train_step(model, config, dataset=dataset)

  return model, state, render_eval_pfn, train_pstep, lr_fn
