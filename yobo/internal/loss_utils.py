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
"""Loss functions for regularization."""

import jax
import jax.numpy as jnp

from google_research.yobo.internal import ref_utils
from google_research.yobo.internal import stepfun


def semantic_loss(batch, renderings, rays, *, coarse_mult=1.0, mult=1.0):
  """Computes semantic loss terms for semantic outputs."""
  semantic_losses = []

  # lossmult can be used to apply a weight to each ray in the batch.
  # For example: masking out rays, applying the Bayer mosaic mask, upweighting
  # rays from lower resolution images and so on.
  lossmult = rays.lossmult
  lossmult = jnp.broadcast_to(lossmult, batch.semantic.shape)
  denom = lossmult.sum()

  for idx, rendering in enumerate(renderings):
    # We use coarse_mult to indicate if the semantic head is used in PropMLP.
    # And we assume the final NerfMLP always has the semantic head.
    if coarse_mult > 0 or idx == len(renderings) - 1:
      resid_sq = (rendering['semantic_pred'] - batch.semantic) ** 2
      semantic_losses.append((lossmult * resid_sq).sum() / denom)

  semantic_losses = jnp.array(semantic_losses)
  loss = (
      coarse_mult * jnp.sum(semantic_losses[:-1]) + mult * semantic_losses[-1]
  )
  return loss, dict(semantic_mses=semantic_losses)


def interlevel_loss(ray_history, *, mults):
  """Computes the interlevel loss defined in mip-NeRF 360."""
  # Stop the gradient from the interlevel loss onto the NeRF MLP.
  num_rounds = len(ray_history[:-1])  # How many rounds of sampling happen.
  if not isinstance(mults, tuple):
    mults = (mults,) * num_rounds

  if len(mults) < num_rounds:
    print(f'mults = {mults} must have a length of {num_rounds}.')
    return []

  c = jax.lax.stop_gradient(ray_history[-1]['sdist'])
  w = jax.lax.stop_gradient(ray_history[-1]['weights'])
  losses_interlevel = []
  for mult, ray_results in zip(mults, ray_history[:-1]):
    cp = ray_results['sdist']
    wp = ray_results['weights']
    losses = stepfun.lossfun_outer(c, w, cp, wp)
    losses_interlevel.append(mult * jnp.mean(losses))
  return losses_interlevel


def spline_interlevel_loss(ray_history, *, mults, blurs):
  """A spline-based alternative to interlevel_loss that lets us blur stuff."""
  num_rounds = len(ray_history[:-1])  # How many rounds of sampling happen.
  if not isinstance(mults, tuple):
    mults = (mults,) * num_rounds
  if len(mults) < num_rounds:
    print(f'mults = {mults} must have a length of {num_rounds}.')
    return []

  if len(blurs) < num_rounds:
    print(f'blurs = {blurs} must have a length of {num_rounds}.')
    return []

  c = ray_history[-1]['sdist']
  w = ray_history[-1]['weights']
  losses_interlevel = []

  for mult, blur, ray_results in zip(mults, blurs, ray_history[:-1]):
    cp = ray_results['sdist']
    wp = ray_results['weights']

    w_blur = stepfun.blur_and_resample_weights(cp, c, w, blur)

    # Stop the gradient from the interlevel loss onto the NeRF MLP.
    w_blur = jax.lax.stop_gradient(w_blur)

    # A truncated chi-squared loss, similar to the old interlevel loss.
    losses = jnp.maximum(0, w_blur - wp) ** 2 / (
        wp + jnp.finfo(jnp.float32).eps
    )

    losses_interlevel.append(mult * jnp.mean(losses))
  return losses_interlevel


def distortion_loss(
    ray_history,
    *,
    target='sdist',
    mult=1.0,
    curve_fn=lambda x: x,
    normalize=False,
):
  """Computes the distortion loss regularizer defined in mip-NeRF 360."""
  last_ray_results = ray_history[-1]
  c = curve_fn(last_ray_results[target])

  w = last_ray_results['weights']
  # w = w / jnp.sum(w, axis=-1, keepdims=True)

  loss = jnp.mean(stepfun.lossfun_distortion(c, w, normalize))
  return mult * loss


def orientation_loss(
    rays,
    ray_history,
    *,
    target='normals',
    coarse_mult=1.0,
    mult=1.0,
    stop_gradient_weights=False,
):
  """Computes the orientation loss regularizer defined in ref-NeRF."""
  total_loss = 0.0
  for i, ray_results in enumerate(ray_history):
    w = ray_results['weights']
    # w = w / jnp.sum(w, axis=-1, keepdims=True)
    w = jax.lax.stop_gradient(w)

    if stop_gradient_weights:
      w = jax.lax.stop_gradient(w)

    n = ray_results[target]

    if n is None:
      continue

    # Negate viewdirs to represent normalized vectors from point to camera.
    v = -rays.viewdirs
    n_dot_v = (n * v[Ellipsis, None, :]).sum(axis=-1)
    loss = jnp.mean(
        jnp.abs(
            (
                jnp.abs(
                    w * (jnp.minimum(0.0, n_dot_v) ** 2)
                )
            ).sum(axis=-1)
            + 1e-5
        )
    )

    if i < len(ray_history) - 1:
      total_loss += coarse_mult * loss
    else:
      total_loss += mult * loss

  return total_loss


def predicted_normal_loss(
    ray_history, *, coarse_mult=1.0, mult=1.0,
    gt='normals', pred='normals_pred',
):
  """Computes the predicted normal supervision loss defined in ref-NeRF."""
  total_loss = 0.0
  for i, ray_results in enumerate(ray_history):
    w = ray_results['weights']
    # w = w / jnp.sum(w, axis=-1, keepdims=True)
    w = jax.lax.stop_gradient(w)

    n = ray_results[gt]
    n = jax.lax.stop_gradient(n)
    n_pred = ray_results[pred]

    if n is None or n_pred is None:
      continue

    loss = jnp.mean(
        jnp.abs(
            (
                jnp.abs(
                    w * (1.0 - jnp.sum(n * n_pred, axis=-1))
                )
            ).sum(axis=-1)
            + 1e-5
        )
    )

    if i < len(ray_history) - 1:
      total_loss += coarse_mult * loss
    else:
      total_loss += mult * loss

  return total_loss


def patch_loss(
    batch,
    renderings,
    *,
    charb_padding,
    bilateral_strength,
    patch_variance_weighting,
    mult=1.0,
):
  """Computes a smoothing regularizer over output depth patches."""
  rgb_gt = batch.rgb[Ellipsis, :3]
  depths = renderings[-1]['distance_mean']
  # Flatten patches (last two axes) into one axis.
  rgb_gt = jnp.reshape(rgb_gt, rgb_gt.shape[:-3] + (-1, 3))
  depths = jnp.reshape(depths, depths.shape[:-2] + (-1,))
  # Pairwise Charbonnier loss over all pixels in depth patch.
  resid_sq = (depths[Ellipsis, None] - depths[Ellipsis, None, :]) ** 2
  loss = jnp.sqrt(resid_sq + charb_padding**2)  # Charbonnier loss

  rgb_resid_sq = jnp.sum(
      (rgb_gt[Ellipsis, None, :] - rgb_gt[Ellipsis, None, :, :]) ** 2, axis=-1
  )
  bilateral_rgb = jnp.exp(-bilateral_strength * rgb_resid_sq)

  # Compute variance of ground truth RGB patch.
  patch_mean = jnp.mean(rgb_gt, axis=-2, keepdims=True)
  patch_variance = jnp.mean(jnp.square(rgb_gt - patch_mean), axis=(-2, -1))
  patch_weight = jnp.exp(-(patch_variance_weighting**2) * patch_variance)
  # Add back two dimensions to match the loss computed pairwise.
  patch_weight = patch_weight[Ellipsis, None, None]
  patch_weight *= bilateral_rgb

  return mult * jnp.mean(loss * patch_weight)