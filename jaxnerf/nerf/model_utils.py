# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Helper functions/classes for model definition."""

import functools
import operator

import flax
from flax import nn
from flax import optim
import jax
from jax import lax
from jax import random
import jax.numpy as jnp


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: optim.Optimizer
  model_state: nn.Collection


class MLP(nn.Module):
  """A simple MLP."""

  def apply(self,
            x,
            condition=None,
            net_depth=8,
            net_width=256,
            net_depth_condition=1,
            net_width_condition=128,
            net_activation=nn.relu,
            skip_layer=4,
            num_rgb_channels=3,
            num_sigma_channels=1):
    """Multi-layer perception for nerf.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.
      condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.
      net_depth: int, the depth of the first part of MLP.
      net_width: int, the width of the first part of MLP.
      net_depth_condition: int, the depth of the second part of MLP.
      net_width_condition: int, the width of the second part of MLP.
      net_activation: function, the activation function used in the MLP.
      skip_layer: int, add a skip connection to the output vector of every
        skip_layer layers.
      num_rgb_channels: int, the number of RGB channels.
      num_sigma_channels: int, the number of density channels.

    Returns:
      raw_rgb: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_rgb_channels].
      raw_sigma: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_sigma_channels].
    """
    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

    inputs = x
    for i in range(net_depth):
      x = dense_layer(x, net_width)
      x = net_activation(x)
      if i % skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    raw_sigma = dense_layer(x, num_sigma_channels).reshape(
        [-1, num_samples, num_sigma_channels])
    if condition is not None:
      # Output of the first part of MLP.
      bottleneck = dense_layer(x, net_width)
      # Broadcast condition from [batch, feature] to
      # [batch, num_samples, feature] since all the samples along the same ray
      # has the same viewdir.
      condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
      # Collapse the [batch, num_samples, feature] tensor to
      # [batch * num_samples, feature] so that it can be feed into nn.Dense.
      condition = condition.reshape([-1, condition.shape[-1]])
      x = jnp.concatenate([bottleneck, condition], axis=-1)
      # Here use 1 extra layer to align with the original nerf model.
      for i in range(net_depth_condition):
        x = dense_layer(x, net_width_condition)
        x = net_activation(x)
    raw_rgb = dense_layer(x, num_rgb_channels).reshape(
        [-1, num_samples, num_rgb_channels])
    return raw_rgb, raw_sigma


def sample_along_rays(key, rays, num_samples, near, far, randomized, lindisp):
  """Stratified sampling along the rays.

  Args:
    key: jnp.ndarray, random generator key.
    rays: jnp.ndarray(float32), [batch_size, 6].
    num_samples: int.
    near: float, near clip.
    far: float, far clip.
    randomized: bool, use randomized stratified sampling.
    lindisp: bool, sampling linearly in disparity rather than depth.

  Returns:
    z_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
    points: jnp.ndarray, [batch_size, num_samples, 3], sampled points.
  """
  origins = rays[Ellipsis, 0:3]
  directions = rays[Ellipsis, 3:6]
  batch_size = origins.shape[0]

  t_vals = jnp.linspace(0., 1., num_samples)
  if not lindisp:
    z_vals = near * (1. - t_vals) + far * t_vals
  else:
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
  if randomized:
    mids = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
    upper = jnp.concatenate([mids, z_vals[Ellipsis, -1:]], -1)
    lower = jnp.concatenate([z_vals[Ellipsis, :1], mids], -1)
    t_rand = random.uniform(key, [batch_size, num_samples])
    z_vals = lower + (upper - lower) * t_rand
  else:
    # Broadcast z_vals to make the returned shape consistent.
    z_vals = jnp.broadcast_to(z_vals[None, Ellipsis], [batch_size, num_samples])

  return (z_vals, (origins[Ellipsis, None, :] +
                   z_vals[Ellipsis, :, None] * directions[Ellipsis, None, :]))


def posenc(x, deg):
  """Concatenate `x` with a positional encoding of `x` with degree `deg`.

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    deg: int the degree of encoding.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if deg == 0:
    return x
  # Instead of computing [sin(x), cos(x)], we use the trig identity
  # cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
  half_pi = jnp.pi / 2
  pos_enc = jnp.sin(
      jnp.concatenate(
          functools.reduce(operator.iconcat,
                           [[2**i * x, 2**i * x + half_pi] for i in range(deg)],
                           []),
          axis=-1))
  encoded = jnp.concatenate([x] + [pos_enc], axis=-1)
  return encoded


def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd):
  """Volumetric Rendering Function.

  Args:
    rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
    sigma: jnp.ndarray(float32), density, [batch_size, num_samples, 1].
    z_vals: jnp.ndarray(float32), [batch_size, num_samples].
    dirs: jnp.ndarray(float32), [batch_size, 3].
    white_bkgd: bool.

  Returns:
    comp_rgb: jnp.ndarray(float32), [batch_size, 3].
    disp: jnp.ndarray(float32), [batch_size].
    acc: jnp.ndarray(float32), [batch_size].
    weights: jnp.ndarray(float32), [batch_size, num_samples]
  """
  eps = 1e-10
  dists = jnp.concatenate([
      z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1],
      jnp.broadcast_to([1e10], z_vals[Ellipsis, :1].shape)
  ], -1)
  dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)
  # Note that we're quietly turning sigma from [..., 0] to [...].
  alpha = 1.0 - jnp.exp(-sigma[Ellipsis, 0] * dists)
  accum_prod = jnp.concatenate([
      jnp.ones_like(alpha[Ellipsis, :1], alpha.dtype),
      jnp.cumprod(1.0 - alpha[Ellipsis, :-1] + eps, axis=-1)
  ],
                               axis=-1)
  weights = alpha * accum_prod

  comp_rgb = (weights[Ellipsis, None] * rgb).sum(axis=-2)
  depth = (weights * z_vals).sum(axis=-1)
  acc = weights.sum(axis=-1)
  # Equivalent to (but slightly more efficient and stable than):
  #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
  inv_eps = 1 / eps
  disp = acc / depth
  disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
  if white_bkgd:
    comp_rgb = comp_rgb + (1. - acc[Ellipsis, None])
  return comp_rgb, disp, acc, weights


def piecewise_constant_pdf(key, bins, weights, num_samples, randomized):
  """Piecewise-Constant PDF sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, num_bins].
    num_samples: int, the number of samples.
    randomized: bool, use randomized samples.

  Returns:
    z_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  eps = 1e-5

  # Get pdf
  weights += eps  # prevent nans
  pdf = weights / weights.sum(axis=-1, keepdims=True)
  cdf = jnp.cumsum(pdf, axis=-1)
  cdf = jnp.concatenate([jnp.zeros(list(cdf.shape[:-1]) + [1]), cdf], axis=-1)

  # Take uniform samples
  if randomized:
    u = random.uniform(key, list(cdf.shape[:-1]) + [num_samples])
  else:
    u = jnp.linspace(0., 1., num_samples)
    u = jnp.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

  # Invert CDF. This takes advantage of the fact that `bins` is sorted.
  mask = (u[Ellipsis, None, :] >= cdf[Ellipsis, :, None])

  def minmax(x):
    x0 = jnp.max(jnp.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), -2)
    x0 = jnp.minimum(x0, x[Ellipsis, -2:-1])
    x1 = jnp.maximum(x1, x[Ellipsis, 1:2])
    return x0, x1

  bins_g0, bins_g1 = minmax(bins)
  cdf_g0, cdf_g1 = minmax(cdf)

  denom = (cdf_g1 - cdf_g0)
  denom = jnp.where(denom < eps, 1., denom)
  t = (u - cdf_g0) / denom
  z_samples = bins_g0 + t * (bins_g1 - bins_g0)

  # Prevent gradient from backprop-ing through samples
  return lax.stop_gradient(z_samples)


def sample_pdf(key, bins, weights, rays, z_vals, num_samples, randomized):
  """Hierarchical sampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, num_bins].
    rays: jnp.ndarray(float32), [batch_size, 6].
    z_vals: jnp.ndarray(float32), [batch_size, num_coarse_samples].
    num_samples: int, the number of samples.
    randomized: bool, use randomized samples.

  Returns:
    z_vals: jnp.ndarray(float32),
      [batch_size, num_coarse_samples + num_fine_samples].
    points: jnp.ndarray(float32),
      [batch_size, num_coarse_samples + num_fine_samples, 3].
  """
  z_samples = piecewise_constant_pdf(key, bins, weights, num_samples,
                                     randomized)
  origins = rays[Ellipsis, 0:3]
  directions = rays[Ellipsis, 3:6]
  # Compute united z_vals and sample points
  z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples], axis=-1), axis=-1)
  return z_vals, (
      origins[Ellipsis, None, :] + z_vals[Ellipsis, None] * directions[Ellipsis, None, :])


def add_gaussian_noise(key, raw, noise_std, randomized):
  """Adds gaussian noise to `raw`. Used to regularize "raw" density.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    raw: jnp.ndarray(float32), arbitrary shape.
    noise_std: float, The standard deviation of the noise to be added.
    randomized: bool, add noise if randomized is True.

  Returns:
    raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
  """
  if (noise_std is not None) and randomized:
    return raw + random.normal(key, raw.shape, dtype=raw.dtype) * noise_std
  else:
    return raw
