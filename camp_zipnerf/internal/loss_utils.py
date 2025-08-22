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

"""Loss functions for regularization."""

from internal import math
from internal import ref_utils
from internal import stepfun
import jax
import jax.numpy as jnp


def spline_interlevel_loss(
    ray_history, *, mults, blurs=None, eps=jnp.finfo(jnp.float32).eps
):
  """A spline-based alternative to interlevel_loss that lets us blur stuff."""
  num_rounds = len(ray_history[:-1])  # How many rounds of sampling happen.
  if not isinstance(mults, tuple):
    mults = (mults,) * num_rounds
  if not isinstance(blurs, tuple):
    blurs = (blurs,) * num_rounds

  if len(mults) != num_rounds:
    raise ValueError(f'mults = {mults} must have a length of {num_rounds}.')
  if len(blurs) != num_rounds:
    raise ValueError(f'blurs = {blurs} must have a length of {num_rounds}.')

  c = ray_history[-1]['sdist']
  w = ray_history[-1]['weights']

  # Stop the gradient from the interlevel loss onto the NeRF MLP.
  w = jax.lax.stop_gradient(w)

  losses_interlevel = []
  for mult, blur, ray_results in zip(mults, blurs, ray_history[:-1]):
    if mult == 0:
      continue
    cp = ray_results['sdist']
    wp = ray_results['weights']

    if blur is None:
      # If `blur` is not specified, compute a blur amount *for each individual
      # interval in `c`* automatically. We blur each interval of the NeRF
      # distribution by a box filter whose halfwidth is the weighted geometric
      # mean of the intervals of the proposal distribution that overlap with
      # that NeRF interval.
      blur = math.safe_exp(
          stepfun.resample(c, cp, math.safe_log(jnp.diff(cp)), use_avg=True)
      )

    w_blur = stepfun.blur_and_resample_weights(cp, c, w, blur)

    # A truncated chi-squared loss, similar to the old interlevel loss.
    losses = jnp.maximum(0, w_blur - wp) ** 2 / (wp + eps)

    losses_interlevel.append(mult * jnp.mean(losses))
  return losses_interlevel


def distortion_loss(
    ray_history,
    *,
    target='sdist',
    mult=1.0,
    curve_fn=lambda x: x,
):
  """Computes the distortion loss regularizer defined in mip-NeRF 360."""
  last_ray_results = ray_history[-1]
  c = curve_fn(last_ray_results[target])
  w = last_ray_results['weights']
  loss = jnp.mean(stepfun.lossfun_distortion(c, w))
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
    if i < len(ray_history) - 1:
      curr_mult = coarse_mult
    else:
      curr_mult = mult
    if not curr_mult > 0.0:
      continue
    w = ray_results['weights']
    if stop_gradient_weights:
      w = jax.lax.stop_gradient(w)
    n = ray_results[target]
    if n is None:
      raise ValueError('Normals cannot be None if orientation loss is on.')
    # Negate viewdirs to represent normalized vectors from point to camera.
    v = -rays.viewdirs
    loss = ref_utils.orientation_loss(w, n, v)
    total_loss += curr_mult * loss
  return total_loss


def predicted_normal_loss(ray_history, *, coarse_mult=1.0, mult=1.0):
  """Computes the predicted normal supervision loss defined in ref-NeRF."""
  total_loss = 0.0
  for i, ray_results in enumerate(ray_history):
    if i < len(ray_history) - 1:
      curr_mult = coarse_mult
    else:
      curr_mult = mult
    if not curr_mult > 0.0:
      continue
    w = ray_results['weights']
    n = ray_results['normals']
    n_pred = ray_results['normals_pred']
    if n is None or n_pred is None:
      raise ValueError(
          'Predicted normals and gradient normals cannot be None if '
          'predicted normal loss is on.'
      )
    loss = jnp.mean((w * (1.0 - jnp.sum(n * n_pred, axis=-1))).sum(axis=-1))
    total_loss += curr_mult * loss
  return total_loss


def eikonal_equation(n, eps=jnp.finfo(jnp.float32).tiny):
  """Compute eikonal equation on normals, checking how close norm is to 1."""
  norm = jnp.sqrt(jnp.maximum(jnp.sum(n**2, axis=-1), eps))
  return jnp.mean((norm - 1.0) ** 2.0)


def eikonal_loss(ray_history, *, coarse_mult=1.0, mult=1.0):
  """Computes the eikonal normal regularization loss defined in VolSDF."""
  total_loss = 0.0
  for i, ray_results in enumerate(ray_history):
    if i < len(ray_history) - 1:
      curr_mult = coarse_mult
    else:
      curr_mult = mult
    if not curr_mult > 0.0:
      continue
    n = ray_results['raw_grad_density']
    if n is None:
      raise ValueError('Density gradient cannot be None if eikonal loss is on.')
    loss = eikonal_equation(n)
    total_loss += curr_mult * loss
  return total_loss
