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
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence, Tuple

import chex
import flax
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import grid_utils
from internal import image_utils
from internal import loss_utils
from internal import math
from internal import models
from internal import ref_utils
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax


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


def compute_data_loss(batch, renderings, rays, config):
  """Computes data loss terms for RGB, normal, and depth outputs."""
  data_losses = []
  stats = collections.defaultdict(lambda: [])

  # lossmult can be used to apply a weight to each ray in the batch.
  # For example: masking out rays, applying the Bayer mosaic mask, upweighting
  # rays from lower resolution images and so on.
  lossmult = 1.0 if rays.lossmult is None else rays.lossmult
  lossmult = jnp.broadcast_to(lossmult, batch.rgb[..., :3].shape)
  if config.disable_multiscale_loss:
    lossmult = jnp.ones_like(lossmult)
  if batch.masks is not None:
    lossmult = lossmult * batch.masks

  for rendering in renderings:
    if rendering['rgb'] is None:
      mse = -1.0
      sub_data_loss = 0
    else:
      resid_sq = (rendering['rgb'] - batch.rgb[..., :3]) ** 2
      mse = (lossmult * resid_sq).sum() / lossmult.sum()
      stats['mses'].append(mse)

      if config.data_loss_type == 'mse':
        # Mean-squared error (L2) loss.
        data_loss = resid_sq
      elif config.data_loss_type == 'geman-mcclure':
        data_loss = math.general_lossfun(
            resid_sq, -2.0, config.robust_loss_scale
        )
      elif config.data_loss_type == 'cauchy':
        data_loss = math.general_lossfun(
            resid_sq, 0.0, config.robust_loss_scale
        )
      elif config.data_loss_type == 'charb':
        # Charbonnier loss.
        data_loss = jnp.sqrt(resid_sq + config.charb_padding**2)
      elif config.data_loss_type == 'rawnerf':
        # Clip raw values against 1 to match sensor overexposure behavior.
        rgb_render_clip = jnp.minimum(1.0, rendering['rgb'])
        resid_sq_clip = (rgb_render_clip - batch.rgb[..., :3]) ** 2
        # Scale by gradient of log tonemapping curve.
        scaling_grad = 1.0 / (1e-3 + jax.lax.stop_gradient(rgb_render_clip))
        # Reweighted L2 loss.
        data_loss = resid_sq_clip * scaling_grad**2
      else:
        assert False
      sub_data_loss = (lossmult * data_loss).mean()

    data_losses.append(sub_data_loss)

    if config.compute_disp_metrics:
      # Using mean to compute disparity, but other distance statistics can
      # be used instead.
      disp = 1 / (1 + rendering['distance_mean'])
      stats['disparity_mses'].append(((disp - batch.disps) ** 2).mean())

    if config.compute_normal_metrics:
      # TODO(barron): Report MAEs for all rendering['normals*'], and only do
      # this for the last scale (then try deleting proposal normals for a
      # speedup).
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


def interlevel_loss(ray_history, config):
  """Computes the interlevel loss defined in mip-NeRF 360."""
  return loss_utils.spline_interlevel_loss(
      ray_history, **config.spline_interlevel_params
  )


def distortion_loss(ray_history, config):
  if config.distortion_loss_curve_fn is None:
    curve_fn = lambda x: x
  else:
    curve_fn, kwargs = config.distortion_loss_curve_fn
    curve_fn = functools.partial(curve_fn, **kwargs)
  return loss_utils.distortion_loss(
      ray_history,
      target=config.distortion_loss_target,
      mult=config.distortion_loss_mult,
      curve_fn=curve_fn,
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
  )


def eikonal_loss(ray_history, config):
  return loss_utils.eikonal_loss(
      ray_history,
      coarse_mult=config.eikonal_coarse_loss_mult,
      mult=config.eikonal_loss_mult,
  )


def param_regularizer_loss(variables, config):
  """Computes regularizer loss(es) over optimized parameters."""
  # Do some bookkeeping to ensure that every regularizer is valid.
  reg_used = {k: False for k in config.param_regularizers}
  params_flat = flax.traverse_util.flatten_dict(variables['params'])
  losses = {k: 0.0 for k in config.param_regularizers.keys()}
  for name_tuple, param in params_flat.items():
    name = '/'.join(name_tuple)
    for prefix in config.param_regularizers:
      if name.startswith(prefix):
        reg_used[prefix] = True
        mult, acc_fn, alpha, scale = config.param_regularizers[prefix]
        if (alpha == 2) and (scale == 1):
          # Special-casing this common setting gives a small speedup and much
          # faster compilation times.
          losses[prefix] += mult * 0.5 * acc_fn(param**2)
        else:
          losses[prefix] += mult * acc_fn(
              math.general_lossfun(param, alpha, scale)
          )
        if not config.disable_pmap_and_jit:
          print(
              'Regularizing '
              + f'{mult}*{acc_fn.__name__}(lossfun{(name, alpha, scale)})'
          )
  # If some regularizer was not used, the gin config is probably wrong.
  for reg, used in reg_used.items():
    if not used:
      raise ValueError(f'Regularizer {reg} not used.')
  return losses


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
  return grad


def create_train_step(
    model: models.Model,
    config: configs.Config,
    lpips_loss_fn: Optional[Any] = None,
    lpips_params: Optional[Dict[str, Any]] = None,
    dataset: Optional[datasets.Dataset] = None,
    devices: Optional[Sequence[jax.Device]] = None,
):
  """Creates the pmap'ed Nerf training function.

  Args:
    model: The linen model.
    config: The configuration.
    lpips_loss_fn: Loss function for lpips loss.
    lpips_params: Model paramers for lpips model.
    dataset: Training dataset.
    devices: If not None, the devices to pmap the train_step function over.

  Returns:
    pmap'ed training function.
  """
  if dataset is None:
    camtype = camera_utils.ProjectionType.PERSPECTIVE
  else:
    camtype = dataset.camtype

  def train_step(rng, state, batch, cameras, train_frac):
    """One optimization step.

    Args:
      rng: jnp.ndarray, random number generator.
      state: TrainState, state of the model/optimizer.
      batch: utils.Batch, a mini-batch of data for training.
      cameras: module containing camera poses.
      train_frac: float, the fraction of training that is complete.

    Returns:
      A tuple (new_state, stats, rng) with
        new_state: TrainState, new training state.
        stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        rng: jnp.ndarray, updated random number generator.
    """
    rng, key = random.split(rng)

    def loss_fn(variables: Any):
      rays = batch.rays
      jax_cameras = None
      transformed_jax_cameras = None
      mutable_camera_params = None
      if config.cast_rays_in_train_step:
        transformed_cameras = cameras[:3]
        if config.optimize_cameras:
          # Convert camera to JAX cameras so we can apply the camera deltas.
          # TODO(keunhong): Consider switching the whole codebase to the camera
          # class once we've implemented Fisheye.
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
        rays = camera_utils.cast_ray_batch(
            (*transformed_cameras, *cameras[3:]),
            rays,
            camtype,
            dataset.scene_bbox,
            xnp=jnp,
        )

      # Indicates whether we need to compute output normal or depth maps in 2D.
      compute_extras = (
          config.compute_disp_metrics
          or config.compute_normal_metrics
      )

      renderings, ray_history = model.apply(
          variables,
          key if config.randomized else None,
          rays,
          train_frac=train_frac,
          compute_extras=compute_extras,
          zero_glo=False,
      )

      losses = {}

      losses['data'], stats = compute_data_loss(batch, renderings, rays, config)

      losses['interlevel'] = interlevel_loss(ray_history, config)

      if config.distortion_loss_mult > 0:
        losses['distortion'] = distortion_loss(ray_history, config)

      if (
          config.orientation_coarse_loss_mult > 0
          or config.orientation_loss_mult > 0
      ):
        losses['orientation'] = orientation_loss(rays, ray_history, config)

      if (
          config.predicted_normal_coarse_loss_mult > 0
          or config.predicted_normal_loss_mult > 0
      ):
        losses['predicted_normals'] = predicted_normal_loss(ray_history, config)

      if config.param_regularizers:
        losses['regularizer'] = param_regularizer_loss(variables, config)

      if config.eikonal_coarse_loss_mult > 0 or config.eikonal_loss_mult > 0:
        losses['eikonal'] = eikonal_loss(ray_history, config)

      if transformed_jax_cameras is not None:
        if config.optimize_cameras and config.focal_length_var_loss_mult > 0:
          log_focal_length = jnp.log(transformed_jax_cameras.focal_length)
          losses['focal_length_var'] = jnp.mean(
              config.focal_length_var_loss_mult
              * (log_focal_length - jnp.mean(log_focal_length)) ** 2
          )

        if config.optimize_cameras and config.principal_point_var_loss_mult > 0:
          losses['principal_point_var'] = jnp.mean(
              config.principal_point_var_loss_mult
              * (
                  transformed_jax_cameras.principal_point
                  - jnp.mean(transformed_jax_cameras.principal_point, axis=0)
              )
              ** 2
          )

        if config.optimize_cameras and config.principal_point_reg_loss_mult > 0:
          losses['principal_point_reg'] = jnp.mean(
              config.principal_point_reg_loss_mult
              * (
                  transformed_jax_cameras.principal_point
                  - jax_cameras.principal_point
              )
              ** 2
          )

        if (
            config.optimize_cameras
            and config.radial_distortion_var_loss_mult > 0
        ):
          losses['radial_distortion_var'] = jnp.mean(
              config.radial_distortion_var_loss_mult
              * abs(
                  transformed_jax_cameras.radial_distortion
                  - jnp.mean(transformed_jax_cameras.radial_distortion, axis=0)
              )
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

        for ri, rh in enumerate(ray_history):
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

      return final_loss, (stats, mutable_camera_params)

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (stats, mutable_camera_params)), grad = loss_grad_fn(state.params)

    # Perform preconditioning before pmean.
    pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
    grad = pmean(grad)
    stats = pmean(stats)
    mutable_camera_params = pmean(mutable_camera_params)

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

    # TODO(barron): Find the source of gradient NaNs.
    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)
    grad = clip_gradients(grad, config)

    new_state = state.apply_gradients(grads=grad)

    camera_delta = config.camera_delta_cls()
    if config.optimize_cameras and camera_delta.precondition_running_estimate:
      new_params = flax.core.copy(
          new_state.params,
          add_or_replace=({
              'camera_params': flax.core.copy(
                  new_state.params['camera_params'],
                  add_or_replace=mutable_camera_params,
              )
          }),
      )
      new_state = new_state.replace(params=new_params)

    if config.debug_mode:
      opt_delta = jax.tree_util.tree_map(
          lambda x, y: x - y, new_state, state
      ).params['params']
      stats['opt_update_norms'] = summarize_tree(tree_norm, opt_delta)
      stats['opt_update_maxes'] = summarize_tree(tree_abs_max, opt_delta)

    stats['psnrs'] = jnp.nan_to_num(
        image_utils.mse_to_psnr(stats['mses']), nan=-1.0
    )
    stats['psnr'] = stats['psnrs'][-1]
    return new_state, stats, rng

  train_pstep = jax.pmap(
      train_step,
      axis_name='batch',
      in_axes=(0, 0, 0, None, None),
      donate_argnums=(0, 1) if config.donate_args_to_train else (),
      devices=devices,
  )
  return train_pstep


def _get_grid_weight_fn(
    grid_size: int,
    scale_supersample: float,
    bbox: chex.Array,
    target_resolution_fn: Callable[..., int],
    method='cosine_sequential',
) -> Callable[..., chex.Array]:
  """Creates a function that returns a weight given a target grid size."""
  bbox_size = max(bbox[1] - bbox[0])

  def cosine_sequential_grid_weight_fn(step: int) -> chex.Array:
    target_grid_size = target_resolution_fn(step)
    log_grid_size = jnp.log2(grid_size)
    # Compute how far off the current target size is from the previous
    # scale. To make the math simple, we assume that the grid sizes are
    # ideal (i.e., they might be fractional).
    scale_diff_to_target = jnp.log2(target_grid_size) - (
        log_grid_size - 1 / scale_supersample
    )
    weight = jnp.clip(scale_supersample * scale_diff_to_target, 0.0, 1.0)
    # Make the window cosiney.
    weight = 0.5 * (1 + jnp.cos(jnp.pi * weight + jnp.pi))
    return weight

  def cosine_fractional_grid_weight_fn(step: int) -> chex.Array:
    target_grid_size = target_resolution_fn(step)
    weight = jnp.clip(target_grid_size / grid_size, 0.0, 1.0)
    weight = 0.5 * (1 + jnp.cos(jnp.pi * weight + jnp.pi))
    return weight

  def gaussian_grid_weight_fn(step: int) -> chex.Array:
    target_res = 1 / target_resolution_fn(step)
    # TODO(barron): Turn this into a function that can be shared.
    weight = math.approx_erf(
        bbox_size / (jnp.sqrt(8) * (target_res * grid_size))
    )
    return weight

  match method:
    case 'cosine_sequential':
      return cosine_sequential_grid_weight_fn
    case 'cosine_fractional':
      return cosine_fractional_grid_weight_fn
    case 'gaussian':
      return gaussian_grid_weight_fn
    case _:
      raise ValueError(f'Unknown method {method!r}')


def scale_grids_by_coarse_to_fine(
    model: models.Model, config: configs.Config, variables: chex.ArrayTree
) -> optax.GradientTransformation:
  """Applies coarse-to-fine by scaling the learning rates of the grids.

  This creates masked gradient transforms for each grid that scale the updates
  based on a scheduled target grid size.

  The benefit of applying coarse-to-fine by scaling gradient updates elegantly
  avoids the issue of adaptive optimizers such as Adam being "shocked" by
  parameters that are suddenly introduces. The shock happens since Adam keeps
  track of an exponential moving average of the parameter updates which is all
  zeros. When the weight of the parameter becomes non-zero, Adam scales the
  updates based on this moving average which causes the updates to be abnormally
  large. This does not occur if we are scaling the Adam updates themselves!

  Args:
    model: The scene model.
    config: The model and training configuration.
    variables: The variables to be trained.

  Returns:
    A gradient transformation object comprised of chained mask transforms that
    apply the coarse-to-fine scaling.
  """
  if (
      model.grid_representation is None
      or model.grid_representation.lower() not in ['ngp', 'hash']
  ):
    raise ValueError('Only HashEncoding supports with coarse to fine.')

  target_resolution_fn = configs.parse_call_def(
      config.grid_c2f_resolution_schedule_def
  )

  all_false = jax.tree_util.tree_map(lambda _: False, variables)
  grid_txs = []
  for level, grid_kwargs in enumerate(model.grid_params_per_level):
    grid = grid_utils.HashEncoding(**grid_kwargs)
    level_name = f'grid_{level}'
    grid_params = variables['params'][level_name]
    for param_name in grid_params:
      grid_size = int(str(param_name).split('_')[-1])
      traversal = flax.traverse_util.ModelParamTraversal(
          # pylint: disable=cell-var-from-loop
          lambda path, _: (level_name in path and param_name in path)
      )
      mask = traversal.update(lambda _: True, all_false)
      grid_weight_fn = _get_grid_weight_fn(
          grid_size=grid_size,
          scale_supersample=grid.scale_supersample,
          bbox=grid.bbox,
          target_resolution_fn=target_resolution_fn,
          method=config.grid_c2f_weight_method,
      )
      grid_tx = optax.masked(optax.scale_by_schedule(grid_weight_fn), mask)
      grid_txs.append(grid_tx)

  return optax.chain(*grid_txs)


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

  def get_lr_fn(lr_init, lr_final):
    return functools.partial(
        math.learning_rate_decay,
        lr_init=lr_init,
        lr_final=lr_final,
        **lr_kwargs,
    )

  lr_fn_main = get_lr_fn(config.lr_init, config.lr_final)
  tx_model = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)

  all_false = jax.tree_util.tree_map(lambda _: False, variables)
  if config.lr_init_grid is not None:
    # Create a second optimizer for only the grid variables (eg. NGP hash maps).
    lr_fn_grid = get_lr_fn(config.lr_init_grid, config.lr_final_grid)
    not_grid_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: 'grid' not in path
    )
    grid_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: 'grid' in path
    )
    not_grid_mask = not_grid_traversal.update(lambda _: True, all_false)
    grid_mask = grid_traversal.update(lambda _: True, all_false)
    tx_grid = optax.adam(learning_rate=lr_fn_grid, **adam_kwargs)
    tx_model = optax.chain(
        optax.masked(tx_model, not_grid_mask),
        optax.masked(tx_grid, grid_mask),
    )

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

  if config.enable_grid_c2f:
    if model is None:
      raise ValueError('The model must be provided for coarse-to-fine.')
    tx = optax.chain(
        tx, scale_grids_by_coarse_to_fine(model, config, variables)
    )

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
        jnp.ndarray,
        utils.Rays,
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
      rays: utils.Rays,
  ):
    if rays.origins is None:
      assert cameras is not None and camtype is not None, (
          'When passing rays.origins = None into render_eval_fn, cameras and '
          f'camtype must NOT be None. Got cameras={cameras} camtype={camtype}.'
      )
      rays = camera_utils.cast_ray_batch(
          cameras, rays, camtype, dataset.scene_bbox, xnp=jnp
      )
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.
            rays,
            train_frac=train_frac,
            compute_extras=True,
            train=False,
        ),
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
    rng: jnp.ndarray,
    dataset: Optional[datasets.Dataset] = None,
) -> Tuple[
    models.Model,
    TrainState,
    Callable[
        [
            FrozenVariableDict,
            float,
            Optional[Tuple[jnp.ndarray, ...]],
            jnp.ndarray,
            utils.Rays,
        ],
        MutableMapping[str, Any],
    ],
    Callable[
        [
            jnp.ndarray,
            TrainState,
            utils.Batch,
            Optional[Tuple[Any, ...]],
            float,
        ],
        Tuple[TrainState, Dict[str, Any], jnp.ndarray],
    ],
    Callable[[int], float],
]:
  """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

  dummy_rays = utils.dummy_rays(
      include_exposure_idx=config.rawnerf_mode, include_exposure_values=True
  )
  model, variables = models.construct_model(
      rng, dummy_rays, config, dataset=dataset
  )

  if (
      config.optimize_cameras
      and dataset
      and dataset.split == utils.DataSplit.TRAIN
  ):
    rng, key = random.split(rng)
    camera_delta = config.camera_delta_cls()
    camera_params = jax.jit(camera_delta.init)(
        {'params': key},
        dataset.get_train_cameras(config, return_jax_cameras=True),
    )
    variables = flax.core.copy(
        variables, add_or_replace={'camera_params': camera_params}
    )

  state, lr_fn = create_optimizer(config, variables, model=model)
  render_eval_pfn = create_render_fn(model, dataset=dataset)
  train_pstep = create_train_step(
      model,
      config,
      dataset=dataset,
  )

  return model, state, render_eval_pfn, train_pstep, lr_fn
