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

# pylint: skip-file
"""Training step and model creation functions."""

import dataclasses
import collections
import copy
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple

import chex
import flax
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from flax.training import checkpoints
import jax
from jax import random
import jax.numpy as jnp
import jmp
import numpy as np
import optax

from google_research.yobo.internal import camera_utils
from google_research.yobo.internal import configs
from google_research.yobo.internal import coord
from google_research.yobo.internal import datasets
from google_research.yobo.internal import grid_utils
from google_research.yobo.internal import image
from google_research.yobo.internal import loss_utils
from google_research.yobo.internal import math
from google_research.yobo.internal import models
from google_research.yobo.internal import ref_utils
from google_research.yobo.internal import utils
from google_research.robust_loss_jax import general

from google_research.yobo.internal.inverse_render import render_utils


def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
  return jax.tree_util.tree_reduce(
      lambda x, y: x + jnp.sum(y**2), tree, initializer=0
  )


def tree_norm(tree):
  return jnp.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
  return jax.tree_util.tree_reduce(
      lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), tree, initializer=0
  )


def tree_len(tree):
  return jax.tree_util.tree_reduce(
      lambda x, y: x + jnp.prod(jnp.array(y.shape)), tree, initializer=0
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



def compute_unbiased_loss_charb(rendering, gt, config):
  diff = (rendering['rgb'] - gt)
  diff_nocorr = (rendering['rgb_nocorr'] - gt)
  return diff * jax.lax.stop_gradient(jnp.sign(diff_nocorr))
  # return diff * jax.lax.stop_gradient(jnp.sign(diff))


def compute_unbiased_loss(rendering, gt, config):
  diff = (rendering['rgb'] - gt)
  diff_nocorr = (rendering['rgb_nocorr'] - gt)
  return diff * jax.lax.stop_gradient(diff_nocorr)


def compute_unbiased_loss_rawnerf(
    rendering, gt, config, clip_val=1000.0, exponent=1.0, eps=1.0
):
  if 'cache_rgb' in rendering:
    rgb_clip = jnp.clip(
        rendering['cache_rgb'],
        eps,
        clip_val
    )
  else:
    rgb_clip = jnp.clip(
        rendering['rgb'],
        eps,
        clip_val
    )

  scaling_grad = 1.0 / (
      jax.lax.stop_gradient(rgb_clip)
  )

  data_loss = compute_unbiased_loss(
      rendering, gt, config
  ) * jnp.power(scaling_grad, exponent)

  return data_loss


def compute_loss_rawnerf(
    rendering, gt, config, clip_val=1000.0, exponent=1.0, eps=1.0
):
  if 'cache_rgb' in rendering:
    rgb_clip = jnp.clip(
        rendering['cache_rgb'],
        eps,
        clip_val
    )
  else:
    rgb_clip = jnp.clip(
        rendering['rgb'],
        eps,
        clip_val
    )

  scaling_grad = 1.0 / (
      jax.lax.stop_gradient(rgb_clip)
  )

  data_loss = (
      (rendering['rgb'] - gt) ** 2
  ) * jnp.power(scaling_grad, exponent)

  return data_loss


def compute_data_loss(batch, rendering, rays, config):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  data_losses = []
  stats = collections.defaultdict(lambda: [])

  # lossmult can be used to apply a weight to each ray in the batch.
  # For example: masking out rays, applying the Bayer mosaic mask, upweighting
  # rays from lower resolution images and so on.
  lossmult = rays.lossmult
  lossmult = jnp.broadcast_to(lossmult, batch.rgb[..., :3].shape)
  
  if config.convert_srgb:
    rendering = jax.tree_util.tree_map(
        lambda x: x, rendering
    )
    rendering['rgb'] = image.linear_to_srgb(rendering['rgb'])
    batch = batch.replace(rgb=image.linear_to_srgb(batch.rgb[..., :3]))

  if config.disable_multiscale_loss:
    lossmult = jnp.ones_like(lossmult)

  if batch.masks is not None:
    lossmult = lossmult * batch.masks

  if rendering['rgb'] is None:
    mse = -1.0
    sub_data_loss = 0
  else:
    resid_sq = (
        rendering['rgb'] - batch.rgb[..., :3]
    ) ** 2
    mse = (lossmult * resid_sq).sum() / lossmult.sum()
    gt = batch.rgb[..., :3]
    data_loss = 0

    if config.data_loss_type == 'mse':
      # Mean-squared error (L2) loss.
      data_loss = (rendering['rgb'] - batch.rgb[..., :3]) ** 2
    elif config.data_loss_type == 'mse_unbiased':
      data_loss = compute_unbiased_loss(
          rendering, gt, config
      )
    elif config.data_loss_type == 'rawnerf':
      data_loss = compute_loss_rawnerf(
          rendering, gt, config
      )
    elif config.data_loss_type == 'rawnerf_unbiased':
      data_loss = compute_unbiased_loss_rawnerf(
          rendering, gt, config
      )
    elif config.data_loss_type == 'rawnerf_original':
      data_loss = compute_loss_rawnerf(
          rendering, gt, config, clip_val=10.0, exponent=2.0, eps=1e-3
      )
    elif config.data_loss_type == 'rawnerf_unbiased_original':
      data_loss = compute_unbiased_loss_rawnerf(
          rendering, gt, config, clip_val=10.0, exponent=2.0, eps=1e-3
      )
    elif config.data_loss_type == 'charb':
      # Charbonnier loss.
      data_loss = jnp.sqrt(
          (rendering['rgb'] - batch.rgb[..., :3]) ** 2
          + config.charb_padding**2
      )
    elif config.data_loss_type == 'charb_unbiased':
      data_loss = compute_unbiased_loss_charb(
          rendering, gt, config
      )
    elif config.data_loss_type == 'charb_clip':
      # Charbonnier loss.
      rgb_render_clip = jnp.minimum(1.0, rendering['rgb'])
      rgb_gt_clip = jnp.minimum(1.0, batch.rgb[..., :3])
      resid_sq_clip = (rgb_render_clip - rgb_gt_clip) ** 2
      data_loss = jnp.sqrt(resid_sq_clip + config.charb_padding**2)
    else:
      assert False

    if 'bg_noise' in rendering:
      data_loss += (rendering['bg_noise'] ** 2)

    sub_data_loss = (lossmult * data_loss).mean()

  # Stats
  stats['mses'].append(mse)
  data_losses.append(sub_data_loss)

  if config.compute_disp_metrics:
    # Using mean to compute disparity, but other distance statistics can
    # be used instead.
    disp = 1 / (1 + rendering['distance_mean'])
    stats['disparity_mses'].append(((disp - batch.disps) ** 2).mean())

  if config.compute_normal_metrics:
    if 'normals' in rendering:
      weights = rendering['acc'] * batch.alphas
      normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
      normalized_normals = ref_utils.l2_normalize(rendering['normals'])
      normal_mae = ref_utils.compute_weighted_mae(
          weights, normalized_normals, normalized_normals_gt
      )
    else:
      # If normals are not computed, set MAE to -1.
      normal_mae = -1.0

    stats['normal_maes'].append(normal_mae)

  data_losses = jnp.array(data_losses)
  loss = (
      config.data_coarse_loss_mult * jnp.sum(data_losses[:-1])
      + config.data_loss_mult * data_losses[-1]
  )

  stats = {k: jnp.array(stats[k]) for k in stats}
  return loss, stats


def compute_semantic_loss(batch, renderings, rays, config, stats):
  """Computes semantic loss terms for semantic outputs."""
  loss, aux = loss_utils.semantic_loss(
      batch,
      renderings,
      rays,
      coarse_mult=config.semantic_coarse_loss_mult,
      mult=config.semantic_loss_mult,
  )
  stats.update(aux)
  return loss


def interlevel_loss(ray_history, loss_mults, loss_blurs, config):
  """Computes the interlevel loss defined in mip-NeRF 360."""
  if config.use_spline_interlevel_loss:
    return loss_utils.spline_interlevel_loss(
        ray_history,
        mults=loss_mults,
        blurs=loss_blurs,
    )
  else:
    return loss_utils.interlevel_loss(
        ray_history, mults=loss_mults
    )


def distortion_loss(ray_history, distortion_loss_mult, config):
  if config.distortion_loss_curve_fn is None:
    curve_fn = lambda x: x
  else:
    curve_fn, kwargs = config.distortion_loss_curve_fn
    curve_fn = functools.partial(curve_fn, **kwargs)
  return loss_utils.distortion_loss(
      ray_history,
      target=config.distortion_loss_target,
      mult=distortion_loss_mult,
      curve_fn=curve_fn,
      normalize=config.normalize_distortion_loss,
  )


def patch_loss(batch, renderings, config):
  """Computes a smoothing regularizer over output depth patches."""
  return loss_utils.patch_loss(
      batch,
      renderings,
      charb_padding=config.charb_padding,
      bilateral_strength=config.bilateral_strength,
      patch_variance_weighting=config.patch_variance_weighting,
      mult=config.patch_loss_mult,
  )


def orientation_loss(rays, ray_history, config):
  """Computes the orientation loss regularizer defined in ref-NeRF."""
  return loss_utils.orientation_loss(
      rays,
      ray_history,
      target=config.orientation_loss_target,
      coarse_mult=config.orientation_coarse_loss_mult,
      mult=config.orientation_loss_mult,
      stop_gradient_weights=False,
  )


def predicted_normal_loss(ray_history, config):
  """Computes the predicted normal supervision loss defined in ref-NeRF."""
  return loss_utils.predicted_normal_loss(
      ray_history,
      coarse_mult=config.predicted_normal_coarse_loss_mult,
      mult=config.predicted_normal_loss_mult,
      gt='normals_pred',
      pred='normals',
  )

def predicted_normal_reverse_loss(ray_history, config):
  """Computes the predicted normal supervision loss defined in ref-NeRF."""
  return loss_utils.predicted_normal_loss(
      ray_history,
      coarse_mult=config.predicted_normal_coarse_loss_mult,
      mult=config.predicted_normal_reverse_loss_mult,
      gt='normals',
      pred='normals_pred',
  )


def exposure_prediction_bounds_loss(
    predicted_exposure: jnp.ndarray, dataset: datasets.Dataset
) -> jnp.ndarray:
  """Compute loss for staying in bounds of the dataset exposures."""
  min_exp = np.min(np.log(dataset.exposures))
  max_exp = np.max(np.log(dataset.exposures))

  exp_pred = jnp.log(predicted_exposure)

  min_loss = jnp.maximum(0, min_exp - exp_pred) ** 2
  max_loss = jnp.maximum(0, exp_pred - max_exp) ** 2
  loss_bounds = jnp.mean(min_loss + max_loss)

  return loss_bounds


def exposure_prediction_loss(
    rays: utils.Rays,
    renderings: list[dict[str, jnp.ndarray]],
    config: configs.Config,
    dataset: datasets.Dataset,
) -> jnp.ndarray:
  """Compute loss for exposure prediction for each ray."""
  predicted_exposure = renderings[-1]['exposure_prediction']
  target_exposure = rays.exposure_values
  exposure_residuals = (predicted_exposure - target_exposure) ** 2
  loss = config.exposure_prediction_loss_mult * jnp.mean(exposure_residuals)
  if config.exposure_prediction_bounds_loss_mult > 0:
    loss += (
        config.exposure_prediction_bounds_loss_mult
        * exposure_prediction_bounds_loss(predicted_exposure, dataset)
    )
  return loss


def param_regularizer_loss(variables, config):
  """Computes regularizer loss(es) over optimized parameters."""
  # Do some bookkeeping to ensure that every regularizer is valid.
  reg_used = {k: False for k in config.param_regularizers}
  params_flat = flax.traverse_util.flatten_dict(variables['params'])
  losses = {k: 0.0 for k in config.param_regularizers.keys()}

  for name_tuple, param in params_flat.items():
    name = '/'.join(name_tuple)
    for prefix in config.param_regularizers:
      if name.startswith(prefix) or prefix in name_tuple:
        reg_used[prefix] = True
        mult, acc_fn, alpha, scale = config.param_regularizers[prefix]
        if (alpha == 2) and (scale == 1):
          # Special-casing this common setting gives a small speedup and much
          # faster compilation times.
          losses[prefix] += mult * 0.5 * acc_fn(param**2)
        else:
          losses[prefix] += mult * acc_fn(general.lossfun(param, alpha, scale))
        if not config.disable_pmap_and_jit:
          print(
              'Regularizing '
              + f'{mult}*{acc_fn.__name__}(lossfun{(name, alpha, scale)})'
          )

  # If some regularizer was not used, the gin config is probably wrong.
  for reg, used in reg_used.items():
    if not used:
      print(f'Regularizer {reg} not used.')

  return losses


def eikonal_equation(n, eps=jnp.finfo(jnp.float32).tiny):
  """Compute the satisfactory of the normals n with the eikonal equations ."""
  norm = jnp.sqrt(jnp.maximum(jnp.sum(n**2, axis=-1), eps))
  return jnp.mean((norm - 1.0) ** 2.0)


def eikonal_loss(ray_history, config):
  """Computes the eikonal normal regularization loss defined in VolSDF."""
  total_loss = 0.0
  for i, ray_results in enumerate(ray_history):
    n = ray_results['normals']
    if n is None:
      raise ValueError('Gradient normals cannot be None if eikonal loss is on.')
    loss = eikonal_equation(n)
    if i < len(ray_history) - 1:
      total_loss += config.eikonal_coarse_loss_mult * loss
    else:
      total_loss += config.eikonal_loss_mult * loss
  return total_loss


def clip_gradients(grad, config):
  """Clips gradients of each MLP individually based on norm and max value."""
  # Clip the gradients of each MLP individually.
  grad_clipped = flax.core.unfreeze(grad)
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
  return flax.core.freeze(grad)


def extra_ray_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
  # Get extra rays
  rng, cur_key = random.split(cur_key)
  extra_rays = render_utils.get_outgoing_rays(
      rng,
      rays,
      jax.lax.stop_gradient(rays.viewdirs),
      jax.lax.stop_gradient(
          model_results['shader']['normals_to_use']
      ),
      {},
      random_generator_2d=model.random_generator_2d,
      use_mis=False,
      samplers=model.uniform_importance_samplers,
      num_secondary_samples=1,
  )

  # Render extra rays
  rng, cur_key = random.split(cur_key)
  extra_results = model.apply(
      variables,
      rng,
      extra_rays,
      train_frac=kwargs['train_frac'],
      compute_extras=False,
      mesh=kwargs['mesh'],
      linear_rgb=True,
  )

  rng, cur_key = random.split(cur_key)
  extra_results_nocorr = model.apply(
      variables,
      rng,
      extra_rays,
      train_frac=kwargs['train_frac'],
      compute_extras=False,
      mesh=kwargs['mesh'],
      linear_rgb=True,
  )

  # Create extra batch
  pred_outputs = {
      'rgb': extra_results['material']['integrator']['rgb'],
      'rgb_nocorr': extra_results_nocorr['material']['integrator']['rgb'],
  }

  extra_batch = batch.replace(
      rgb=jax.lax.stop_gradient(extra_results['render']['cache_rgb']),
  )

  data_loss, _ = compute_data_loss(
      extra_batch,
      pred_outputs,
      extra_rays,
      config
  )

  return data_loss


def radiometric_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
  # Get extra rays
  rng, cur_key = random.split(cur_key)
  extra_rays = render_utils.get_outgoing_rays(
      rng,
      rays,
      rays.viewdirs,
      model_results['shader']['normals'],
      {},
      random_generator_2d=model.random_generator_2d,
      use_mis=False,
      samplers=model.uniform_importance_samplers,
      num_secondary_samples=1,
  )

  # Render extra rays
  rng, cur_key = random.split(cur_key)
  extra_results = model.apply(
      variables,
      rng,
      extra_rays,
      train_frac=kwargs['train_frac'],
      compute_extras=kwargs['compute_extras'],
      mesh=kwargs['mesh'],
      linear_rgb=True,
  )

  rng, cur_key = random.split(cur_key)
  extra_results_nocorr = model.apply(
      variables,
      rng,
      extra_rays,
      train_frac=kwargs['train_frac'],
      compute_extras=kwargs['compute_extras'],
      mesh=kwargs['mesh'],
      linear_rgb=True,
  )

  # Create extra batch
  pred_outputs = {
      'rgb': extra_results['render']['material_rgb'],
      'rgb_nocorr': extra_results_nocorr['render']['material_rgb'],
  }

  diff = compute_unbiased_loss(
      jax.lax.stop_gradient(pred_outputs),
      extra_results['render']['cache_rgb'],
      config
  )

  return diff.mean()


def bmild_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
  if model_results['sampler'] is None:
    return 0.0

  weights = jax.lax.stop_gradient(
      model_results['sampler'][-1]['weights']
  )[..., None]
  shader_rays = rays.replace(
      lossmult=(
          rays.lossmult[..., None] * weights
      )
  )
  shader_batch = batch.replace(
      rgb=jnp.repeat(
          batch.rgb[..., None, :], weights.shape[-2], axis=-2
      )
  )
  
  shader_data_loss, _ = compute_data_loss(
      shader_batch,
      model_results['shader'],
      shader_rays,
      config
  )

  return shader_data_loss


def emission_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
  if 'lighting_emission' not in model_results['shader']:
    return 0.0

  # Outputs
  emission = model_results['shader']['lighting_emission']
  irradiance = model_results['shader']['lighting_irradiance']
  irradiance_nocorr = model_results['shader']['lighting_irradiance_nocorr']
  residual_albedo = model_results['shader']['material_residual_albedo']

  # Compute unbiased difference
  material_results = {
      'rgb': residual_albedo * jax.lax.stop_gradient(irradiance),
      'rgb_nocorr': residual_albedo * jax.lax.stop_gradient(irradiance_nocorr),
  }
  diff = compute_unbiased_loss(
      material_results,
      jax.lax.stop_gradient(emission),
      config
  )
  
  # Compute emission regularizers
  zero_loss = (
      math.safe_sqrt(emission + 1e-5)
  ) * config.emission_zero_loss_mult

  difference_loss = (
      jnp.square(
          emission - jax.lax.stop_gradient(
              emission
          )
      )
  ) * config.emission_constant_loss_mult

  # Loss
  weights = jax.lax.stop_gradient(
      model_results['sampler'][-1]['weights']
  )[..., None]

  return (
      (diff * weights).sum(axis=-2).mean()
      + (zero_loss * weights).sum(axis=-2).mean()
      + (difference_loss * weights).sum(axis=-2).mean()
  )


def light_sampling_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
  if (
      'light_sampler' not in model_results
      or model_results['light_sampler'] is None
  ):
    return 0.0

  # Get extra rays
  light_sampler_results = model_results['light_sampler']

  if 'ref_rays' in model_results['shader']:
    extra_rays = model_results['shader']['ref_rays']
    extra_ray_samples = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        model_results['shader']['ref_samples']
    )

    function_vals = jax.lax.stop_gradient(
        jnp.linalg.norm(
            extra_ray_samples['radiance_in'],
            axis=-1
        )
    )
  else:
    # rng, cur_key = random.split(cur_key)
    # inds = jax.random.categorical(
    #     rng,
    #     logits=jnp.repeat(
    #         math.safe_log(
    #             model_results['sampler'][-1]['weights'][..., None, None]
    #         ),
    #         3,
    #         axis=-1
    #     ),
    #     axis=-3,
    #     shape=(
    #         light_sampler_results['vmf_origins'].shape[:-3]
    #         + light_sampler_results['vmf_origins'].shape[-2:]
    #     )
    # )

    # light_sampler_results = jax.tree_util.tree_map(
    #     lambda x: jnp.take_along_axis(
    #         x, inds[..., None, :x.shape[-2], :x.shape[-1]], axis=-3
    #     ),
    #     light_sampler_results,
    # )
    
    extra_origins = light_sampler_results['vmf_origins'][..., 0, :]
    extra_normals = light_sampler_results['vmf_normals'][..., 0, :]

    rng, cur_key = random.split(cur_key)
    extra_rays, extra_ray_samples = render_utils.get_secondary_rays(
        rng,
        rays,
        jax.lax.stop_gradient(extra_origins),
        jax.lax.stop_gradient(rays.viewdirs),
        jax.lax.stop_gradient(extra_normals),
        {},
        random_generator_2d=model.random_generator_2d,
        use_mis=False,
        samplers=model.light_importance_samplers,
        num_secondary_samples=config.num_light_samples,
        light_sampler_results=jax.lax.stop_gradient(light_sampler_results),
    )

    # Render extra rays
    rng, cur_key = random.split(cur_key)
    extra_results = model.apply(
        variables,
        rng,
        extra_rays,
        train_frac=kwargs['train_frac'],
        compute_extras=kwargs['compute_extras'],
        mesh=kwargs['mesh'],
        passes=('cache',),
        zero_backfacing=True,
        linear_rgb=True,
    )

    function_vals = jax.lax.stop_gradient(
        jnp.linalg.norm(
            extra_results['render']['cache_rgb'], axis=-1
        )
    )
    
  # Viewdirs
  extra_viewdirs = jax.lax.stop_gradient(
      extra_rays.viewdirs.reshape(
          function_vals.shape + (3,)
      )
  )
  
  # VMF parameters
  vmf_means = light_sampler_results['vmf_means']
  vmf_kappas = light_sampler_results['vmf_kappas']
  vmf_weights = light_sampler_results['vmf_weights']
  vmf_normals = light_sampler_results['vmf_normals']

  vmf_means = vmf_means.reshape(-1, vmf_means.shape[-2], 3)
  vmf_kappas = vmf_kappas.reshape(-1, vmf_kappas.shape[-2], 1)
  vmf_weights = vmf_weights.reshape(-1, vmf_weights.shape[-2], 1)
  vmf_normals = vmf_normals.reshape(-1, 3)

  return render_utils.vmf_loss_fn(
      (vmf_means, vmf_kappas, vmf_weights),
      vmf_normals,
      extra_viewdirs,
      function_vals,
      extra_ray_samples,
  )


EXTRA_LOSS_DICT = {
    'bmild': bmild_loss,
    'emission': emission_loss,
    'extra_ray': extra_ray_loss,
    'radiometric': radiometric_loss,
    'light_sampling': light_sampling_loss,
}


def create_train_step(
    model: models.Model,
    config: configs.Config,
    dataset: Optional[datasets.Dataset] = None,
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
      state: TrainState, state of the model/optimizer.
      batch: dict, a mini-batch of data for training.
      cameras: module containing camera poses.
      train_frac: float, the fraction of training that is complete.

    Returns:
      A tuple (new_state, stats, rng) with
        new_state: TrainState, new training state.
        stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        rng: jnp.ndarray, updated random number generator.
    """
    rng, key = random.split(rng)
    
    def per_output_loss_fn(
        model,
        variables,
        config: Any,
        cur_key: Any,
        rays: utils.Rays,
        model_results: Any,
        output_key: Any,
        **kwargs,
    ):
      losses = {}

      ## Sampler losses
      if model_results['sampler'] is not None:
        # Gradient scaling
        if (
            config.use_gradient_scaling
        ):
          for i in range(len(model_results['sampler'])):
            gaussians = (
                model_results['sampler'][i]['means'],
                model_results['sampler'][i]['covs'],
            )

            # Equation 6 in https://arxiv.org/abs/2305.02756.
            dsq = jnp.sum(
                (gaussians[0] - rays.origins[..., None, :]) ** 2, axis=-1
            )
            scaling = jnp.clip(
                dsq / config.gradient_scaling_sigma ** 2, 0, 1
            )

            # Scale sampler & shader outputs
            ray_results = {
                k: v for k, v in model_results['sampler'][i].items()
                if k not in ['means', 'covs', 'tdist', 'sdist']
            }
            ray_results_extras = {
                k: v for k, v in model_results['sampler'][i].items()
                if k in ['means', 'covs', 'tdist', 'sdist']
            }
            ray_results = coord.rescale_gradient(
                scaling, ray_results
            )
            model_results['sampler'][i] = dict(
                **ray_results,
                **ray_results_extras,
            )

            if i == len(model_results['sampler']) - 1:
              model_results['shader'] = coord.rescale_gradient(
                  scaling, model_results['shader']
              )

        if isinstance(config.interlevel_loss_mults, tuple) or (
            config.interlevel_loss_mults > 0
            and model_results['sampler'] is not None
        ) and model_results['sampler'] is not None:
          losses['interlevel'] = interlevel_loss(
              model_results['sampler'],
              config.interlevel_loss_mults,
              config.interlevel_loss_blurs,
              config
          )

        if isinstance(config.distortion_loss_mult, tuple) or (
            config.distortion_loss_mult > 0
            and model_results['sampler'] is not None
        ):
          losses['distortion'] = distortion_loss(
              model_results['sampler'],
              config.distortion_loss_mult,
              config
          )

        # Normal weight ease in
        if config.normal_weight_ease_frac > 0 and config.use_normal_weight_ease:
          normal_loss_weight = jnp.clip(
              train_frac / config.normal_weight_ease_frac,
              0.0,
              1.0
          )
          normal_loss_weight = normal_loss_weight * 0.75 + 0.25
        else:
          normal_loss_weight = 1.0

        # Normal weight decay
        if config.use_normal_weight_decay:
          use_decay = train_frac >= config.normal_weight_decay_start_frac

          normal_loss_weight_decay = jnp.clip(
              1 - (
                  train_frac - config.normal_weight_decay_start_frac
              ) / config.normal_weight_decay_rate,
              0.1,
              1.0
          ) * use_decay + (1.0 - use_decay)
        else:
          normal_loss_weight_decay = 1.0

        if (
            config.orientation_loss_mult > 0
            and model_results['sampler'] is not None
        ):
          losses['orientation'] = orientation_loss(
              rays,
              model_results['sampler'],
              config
          ) * normal_loss_weight

        if (
            config.predicted_normal_loss_mult > 0
        ):
          losses['predicted_normals'] = predicted_normal_loss(
              model_results['sampler'],
              config,
          ) * normal_loss_weight_decay

        if (
            config.predicted_normal_reverse_loss_mult > 0
        ):
          losses['predicted_normals_reverse'] = predicted_normal_reverse_loss(
              model_results['sampler'],
              config,
          )

        if config.eikonal_coarse_loss_mult > 0 or config.eikonal_loss_mult > 0:
          losses['eikonal'] = eikonal_loss(
              model_results['sampler'],
              config
          )
      
      ## Integrator losses
      if model_results['integrator'] is not None:
        data_loss, stats = compute_data_loss(
            batch,
            model_results['integrator'],
            rays,
            config
        )
        losses['data'] = data_loss

        # Exposure
        if config.exposure_prediction_loss_mult > 0:
          losses['exposure_prediction'] = exposure_prediction_loss(
              rays,
              model_results['integrator'],
              config,
              dataset
          )
      else:
        stats = {}
        
      ## Extra losses
      for loss_key in config.extra_losses.keys():
        if isinstance(config.extra_losses[loss_key], dict):
          if output_key in config.extra_losses[loss_key]:
            loss_mult = config.extra_losses[loss_key][output_key]
          else:
            loss_mult = 0.0
        else:
          if output_key == 'main':
            loss_mult = config.extra_losses[loss_key]
          else:
            loss_mult = 0.0

        if not loss_mult > 0.0:
          continue

        rng, cur_key = random.split(cur_key)
        cur_loss = EXTRA_LOSS_DICT[loss_key](
            model,
            variables,
            rng,
            rays,
            config,
            batch,
            model_results,
            **kwargs,
        )
        losses[loss_key] = loss_mult * cur_loss
      
      return losses, stats

    def loss_fn(variables: Any, loss_scaler: jmp.LossScale):
      cur_key = key
      losses = {}
      rays = batch.rays
      jax_cameras = None
      transformed_jax_cameras = None
      mutable_camera_params = None

      if config.cast_rays_in_train_step:
        transformed_cameras = cameras[:3]

        if config.optimize_cameras:
          image_sizes = jnp.array(
              [(x.shape[1], x.shape[0]) for x in dataset.images]
          )
          jax_cameras = jax.vmap(dataset.jax_camera_from_tuple_fn)(
              cameras, image_sizes
          )
          # Apply the camera delta and convert back into the original format.
          (
              transformed_jax_cameras,
              mutable_camera_params,
          ) = config.camera_delta_cls(is_training=True).apply(
              variables['camera_params'],
              jax_cameras,
              rngs={'params': key},
              mutable='precondition',
          )
          transformed_cameras = jax.vmap(camera_utils.tuple_from_jax_camera)(
              transformed_jax_cameras
          )

        rng, cur_key = random.split(cur_key)
        rays = camera_utils.cast_ray_batch(
            (*transformed_cameras, *cameras[3:]),
            rays,
            camtype,
            rng=rng,
            jitter=config.jitter_rays,
            xnp=jnp
        )

      # Indicates whether we need to compute output normal or depth maps in 2D
      # or the semantic maps.
      compute_extras = (
          config.compute_disp_metrics
          or config.compute_normal_metrics
          or config.patch_loss_mult > 0.0
          or config.semantic_dir
      )

      rng, cur_key = random.split(cur_key)
      model_results = model.apply(
          variables,
          rng,
          rays,
          train_frac=train_frac,
          compute_extras=compute_extras,
          mesh=dataset.mesh,
      )
      
      # De-biasing
      rng, cur_key = random.split(cur_key)
      rays_nocorr = camera_utils.cast_ray_batch(
          (*cameras[:3], *cameras[3:]),
          rays,
          camtype,
          rng=rng,
          jitter=config.jitter_rays,
          xnp=jnp
      )

      rng, cur_key = random.split(cur_key)
      model_results_nocorr = model.apply(
          variables,
          rng,
          rays_nocorr,
          train_frac=train_frac,
          compute_extras=compute_extras,
          mesh=dataset.mesh,
      )
      
      for result_key in model_results_nocorr.keys():
        if result_key == 'render':
          continue
        
        for field_key in ['integrator', 'shader']:
          for final_key in model_results_nocorr[result_key][field_key].keys():
            model_results[result_key][field_key][f'{final_key}_nocorr'] = (
                model_results_nocorr[result_key][field_key][final_key]
            )
      
      # Per output losses
      stats = {}

      for k in model_results.keys():
        # Pass on render
        if k == 'render':
          continue
        
        # Update loss type
        cur_config = copy.deepcopy(config)
        
        if 'loss_type' in model_results[k]:
          cur_config.data_loss_type = model_results[k]['loss_type']

        if 'convert_srgb' in model_results[k]:
          cur_config.convert_srgb = model_results[k]['convert_srgb']
        else:
          cur_config.convert_srgb = False

        # Get loss
        rng, cur_key = random.split(cur_key)
        per_output_losses, per_output_stats = per_output_loss_fn(
            model,
            variables,
            cur_config,
            rng,
            rays,
            model_results[k],
            output_key=k,
            train_frac=train_frac,
            compute_extras=compute_extras,
            mesh=dataset.mesh,
        )
        
        # Add losses
        for loss_k in per_output_losses.keys():
          if k == 'main':
            output_k = loss_k
          else:
            output_k = f'{k}_{loss_k}'

          if (
              isinstance(per_output_losses[loss_k], list)
              or isinstance(per_output_losses[loss_k], tuple)
          ):
            losses[output_k] = [
                l * model_results[k]['loss_weight']
                for l in per_output_losses[loss_k]
            ]
          else:
            losses[output_k] = (
                per_output_losses[loss_k] * model_results[k]['loss_weight']
            )
        
        if k == 'main':
          stats = per_output_stats
        
      # Regularizers
      if config.param_regularizers:
        losses['regularizer'] = param_regularizer_loss(variables, config)

      # Cameras
      if transformed_jax_cameras is not None:
        if config.optimize_cameras and config.focal_length_var_loss_mult > 0:
          losses['focal_length_var'] = jnp.mean(
              config.focal_length_var_loss_mult
              * jnp.var(jnp.log(transformed_jax_cameras.focal_length))
          )

        if config.optimize_cameras and config.principal_point_var_loss_mult > 0:
          losses['principal_point_var'] = jnp.mean(
              config.principal_point_var_loss_mult
              * jnp.var(transformed_jax_cameras.principal_point, axis=0)
          )

        if config.optimize_cameras and config.principal_point_reg_loss_mult > 0:
          losses['principal_point_reg'] = jnp.mean(
              config.principal_point_reg_loss_mult
              * jnp.square(
                  transformed_jax_cameras.principal_point
                  - jax_cameras.principal_point
              ).sum(axis=-1)
          )

        if (
            config.optimize_cameras
            and config.radial_distortion_var_loss_mult > 0
        ):
          losses['radial_distortion_var'] = jnp.mean(
              config.radial_distortion_var_loss_mult
              * jnp.var(transformed_jax_cameras.radial_distortion, axis=0)
          )

      losses_flat = {}

      for k, v in losses.items():
        if isinstance(v, list) or isinstance(v, tuple):
          for i, vi in enumerate(v):
            losses_flat[k + '_' + str(i)] = vi
        elif isinstance(v, dict):
          for ki, vi in v.items():
            losses_flat[k + '/' + ki] = vi
        else:
          losses_flat[k] = v

      stats['loss'] = jnp.sum(jnp.array(list(losses_flat.values())))
      stats['losses'] = losses_flat

      if config.debug_mode:
        stats['weight_l2s'] = summarize_tree(tree_norm_sq, variables['params'])

        # Log some summary statistics of t/s distances along rays and the size
        # of each t/s ray interval.
        def percentile_fn(x):
          return jnp.percentile(x.flatten(), jnp.linspace(0, 100, 101))

        for ri, rh in enumerate(model_results['sampler']):
          s = rh['sdist']
          t = rh['tdist']
          ds = s[..., 1:] - s[..., :-1]
          dt = t[..., 1:] - t[..., :-1]
          stats[f'ray_normalized_distance{ri}'] = percentile_fn(s)
          stats[f'ray_normalized_distance{ri}_log_delta'] = math.safe_log(
              percentile_fn(ds)
          )
          stats[f'ray_metric_distance{ri}_log'] = math.safe_log(
              percentile_fn(t)
          )
          stats[f'ray_metric_distance{ri}_log_delta'] = math.safe_log(
              percentile_fn(dt)
          )

      final_loss = stats['loss']
      final_loss = loss_scaler.scale(final_loss)

      return final_loss, (stats, mutable_camera_params)

    loss_scaler = jmp.NoOpLossScale()
    if config.enable_loss_scaler:
      loss_scaler = jmp.StaticLossScale(loss_scale=config.loss_scale)

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (stats, mutable_camera_params)), grad = loss_grad_fn(
        state.params, loss_scaler
    )

    # Perform preconditioning before pmean.
    pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
    grad = pmean(grad)
    stats = pmean(stats)
    mutable_camera_params = pmean(mutable_camera_params)

    grad = loss_scaler.unscale(grad)

    if config.debug_mode:
      stats['grad_norms'] = summarize_tree(tree_norm, grad['params'])
      stats['grad_maxes'] = summarize_tree(tree_abs_max, grad['params'])

      for name, g in flax.traverse_util.flatten_dict(grad, sep='/').items():
        # pylint: disable=cell-var-from-loop
        jax.lax.cond(
            jnp.any(~jnp.isfinite(g)),
            lambda: jax.debug.print(f'Warning: {name} has non-finite grads'),
            lambda: None,
        )
        jax.lax.cond(
            jnp.all(g == 0),
            lambda: jax.debug.print(f'Warning: {name} has all-zero grads'),
            lambda: None,
        )
        # pylint: enable=cell-var-from-loop

    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)
    grad = clip_gradients(grad, config)

    new_state = state.apply_gradients(grads=grad)

    camera_delta = config.camera_delta_cls()
    if config.optimize_cameras and camera_delta.precondition_running_estimate:
      new_params = new_state.params.copy(
          {
              'camera_params': new_state.params['camera_params'].copy(
                  mutable_camera_params
              )
          }
      )
      new_state = new_state.replace(params=new_params)

    if config.debug_mode:
      opt_delta = jax.tree_util.tree_map(
          lambda x, y: x - y, new_state, state
      ).params['params']
      stats['opt_update_norms'] = summarize_tree(tree_norm, opt_delta)
      stats['opt_update_maxes'] = summarize_tree(tree_abs_max, opt_delta)

    stats['psnrs'] = jnp.nan_to_num(image.mse_to_psnr(stats['mses']), nan=-1.0)
    stats['psnr'] = stats['psnrs'][-1]
    return new_state, stats, rng

  train_pstep = jax.pmap(
      train_step,
      axis_name='batch',
      in_axes=(0, 0, 0, None, None),
      donate_argnums=(0, 1),
  )
  return train_pstep


def create_optimizer(
    config: configs.Config,
    variables: FrozenVariableDict,
    model: models.Model | None = None,
) -> Tuple[TrainState, Callable[[int], float]]:
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

  def get_lr_fn(lr_init, lr_final, **lr_kwargs):
    return functools.partial(
        math.learning_rate_decay,
        lr_init=lr_init,
        lr_final=lr_final,
        **lr_kwargs,
    )

  lr_fn_main = get_lr_fn(config.lr_init, config.lr_final, **lr_kwargs)
  tx_model = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)
  all_false = jax.tree_util.tree_map(lambda _: False, variables)
  
  def contruct_optimizer(opt_params, prefix, tx_model):
    lr_kwargs = {
        'max_steps': (
            opt_params['max_steps']
            if 'max_steps' in opt_params else config.max_steps
        ),
        'lr_delay_steps': (
            opt_params['lr_delay_steps']
            if 'lr_delay_steps' in opt_params else config.lr_delay_steps
        ),
        'lr_delay_mult': (
            opt_params['lr_delay_mult']
            if 'lr_delay_mult' in opt_params else config.lr_delay_mult
        ),
    }
    adam_kwargs = {
        'b1': (
            opt_params['adam_b1']
            if 'adam_b1' in opt_params else config.adam_beta1
        ),
        'b2': (
            opt_params['adam_b2']
            if 'adam_b2' in opt_params else config.adam_beta2
        ),
        'eps': (
            opt_params['adam_eps']
            if 'adam_eps' in opt_params else config.adam_eps
        ),
    }
    
    # Keep opt for other params the same
    model_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: prefix not in path.split('/')
    )
    model_mask = model_traversal.update(lambda _: True, all_false)
    tx = optax.masked(tx_model, model_mask)
    
    # Opt for current params
    extra_lr_fn = get_lr_fn(
        opt_params['lr_init'] if 'lr_init' in opt_params else config.lr_init,
        opt_params['lr_final'] if 'lr_final' in opt_params else config.lr_final,
        **lr_kwargs
    )
    extra_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: prefix in path.split('/')
    )
    extra_mask = extra_traversal.update(lambda _: True, all_false)
    extra_tx = optax.adam(
        learning_rate=extra_lr_fn, **adam_kwargs
    )
    
    # Return
    return optax.chain(
        tx,
        optax.masked(extra_tx, extra_mask),
    )

  # Extra optimizers
  if config.extra_opt_params is not None:
    for prefix, params in config.extra_opt_params.items():
      tx_model = contruct_optimizer(params, prefix, tx_model)

  # Add the optimizer for the camera parameters if enabled.
  if config.optimize_cameras:
    model_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: 'camera_params' not in path
    )
    model_mask = model_traversal.update(lambda _: True, all_false)
    tx = optax.masked(tx_model, model_mask)

    camera_lr_fn = configs.parse_call_def(config.camera_lr_schedule_def)
    camera_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: 'camera_params' in path
    )
    camera_mask = camera_traversal.update(lambda _: True, all_false)
    tx_camera = config.camera_optimizer(
        learning_rate=camera_lr_fn, **config.camera_optimizer_kwargs
    )
    tx = optax.chain(tx, optax.masked(tx_camera, camera_mask))
  else:
    # Only mask the model optimizer when the camera optimizer is on. This is
    # to preserve backward compatibility with checkpoints that do not have
    # the mask on the model.
    # NOTE: The mask must be considered if any other parameters are added.
    tx = tx_model

  if config.grad_accum_steps > 1:
    tx = optax.MultiSteps(tx, config.grad_accum_steps, use_grad_mean=True)

  return TrainState.create(apply_fn=None, params=variables, tx=tx), lr_fn_main


def create_render_fn(
    model: models.Model,
    dataset: Optional[datasets.Dataset] = None,
    mapping_fn: Any = jax.pmap,
) -> Callable[
    [
        FrozenVariableDict,
        float,
        Optional[Tuple[jnp.ndarray, ...]],
        jnp.array,
        utils.Rays | utils.Pixels,
    ],
    MutableMapping[str, Any],
]:
  """Creates pmap'ed or vmap'ed function for full image rendering."""

  camtype = None
  if dataset is not None:
    camtype = dataset.camtype

  def render_eval_fn(
      variables,
      train_frac: float,
      cameras: Optional[Tuple[jnp.ndarray, ...]],
      _: jnp.ndarray,
      rays: utils.Rays | utils.Pixels,
  ):
    if isinstance(rays, utils.Pixels):
      assert cameras is not None and camtype is not None, (
          'When passing Pixels into render_eval_fn, cameras and camtype needs'
          f' to be not None. Got cameras={cameras} camtype={camtype}.'
      )
      rays = camera_utils.cast_ray_batch(
          cameras,
          rays,
          camtype,
          xnp=jnp
      )

    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            compute_extras=True,
            train_frac=train_frac,
            train=False,
            mesh=dataset.mesh,
        )['integrator'],
        axis_name='batch',
    )

  # call the mapping_fn over only the data input.
  render_eval_mfn = mapping_fn(
      render_eval_fn,
      # Shard variables and rays. Copy train_frac and rng.
      #
      # variables should be replicated manually by calling
      # flax.jax_utils.replicate
      in_axes=(0, None, 0, None, 0),
      axis_name='batch',
  )
  return render_eval_mfn


def setup_model(
    config: configs.Config,
    rng: jnp.array,
    dataset: Optional[datasets.Dataset] = None,
) -> Tuple[
    models.Model,
    TrainState,
    Callable[
        [
            FrozenVariableDict,
            float,
            Optional[Tuple[jnp.ndarray, ...]],
            jnp.array,
            utils.Rays | utils.Pixels,
        ],
        MutableMapping[str, Any],
    ],
    Callable[
        [jnp.array, TrainState, utils.Batch, Optional[Tuple[Any, ...]], float],
        Tuple[TrainState, Dict[str, Any], jnp.array],
    ],
    Callable[[int], float],
]:
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

  dummy_rays = utils.dummy_rays(
      include_exposure_idx=config.rawnerf_mode,
      include_exposure_values=True
  )
  model, variables = models.construct_model(
      rng,
      dummy_rays,
      config,
      dataset=dataset
  )

  if (
      config.optimize_cameras
      and dataset
      and dataset.split == utils.DataSplit.TRAIN
  ):
    rng, key = random.split(rng)
    camera_delta = config.camera_delta_cls()
    camera_params = camera_delta.init(
        {'params': key},
        dataset.get_train_cameras(config, return_jax_cameras=True),
    )
    variables = variables.copy({'camera_params': camera_params})

  state, lr_fn = create_optimizer(config, variables, model=model)
  render_eval_pfn = create_render_fn(model, dataset=dataset)
  train_pstep = create_train_step(model, config, dataset=dataset)

  return model, state, render_eval_pfn, train_pstep, lr_fn


def replace_param_subset(
    state,
    cache_state,
    put_prefix='CacheModel/',
    take_prefix='CacheModel/',
):
  flat_params = flax.traverse_util.flatten_dict(
      state.params['params'], sep='/'
  )
  flat_cache_params = flax.traverse_util.flatten_dict(
      cache_state.params['params'], sep='/'
  )

  for put_key in flat_params:
    if not put_key.startswith(put_prefix):
      continue
    
    cur_key = put_key[len(put_prefix):]
    take_key = take_prefix + cur_key

    if take_key in flat_cache_params:
      flat_params[put_key] = flat_cache_params[take_key].copy()

  params = flax.traverse_util.unflatten_dict(flat_params, sep='/')
  return state.replace(params=flax.core.freeze({'params': params}))


class DummyState(flax.struct.PyTreeNode):
  params: Any


def restore_partial_checkpoint(
    config,
    state,
    prefixes: list[Any] = [],
    replace_dict: Any = None,
):
  if config.ckpt_dir is not None:
    # Filter params
    partial_params = flax.traverse_util.flatten_dict(
        flax.core.unfreeze(state.params['params']), sep='/'
    )
    partial_params = {
        k: v for k, v in partial_params.items()
        if any(k.startswith(prefix) for prefix in prefixes)
    }
    partial_params = flax.traverse_util.unflatten_dict(
        partial_params, sep='/'
    )

    # Create partial state
    partial_state = DummyState(
        params=flax.core.freeze({'params': partial_params})
    )
    partial_state = checkpoints.restore_checkpoint(
        config.ckpt_dir, partial_state
    )

    # Replace params
    for put_prefix, take_prefix in replace_dict.items():
      state = replace_param_subset(
          state,
          partial_state,
          put_prefix=put_prefix + '/',
          take_prefix=take_prefix + '/',
      )
  
  return state
