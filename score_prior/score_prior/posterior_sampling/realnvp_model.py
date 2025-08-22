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

"""RealNVP model architecture.

This is a JAX/Flax implementation of the original PyTorch version:
https://github.com/HeSunPU/DPI/blob/main/DPItorch/generative_model/realnvpfc_model.py.
"""
# pylint:disable=line-too-long
from typing import Any, Callable, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

PRNGKey = jax.Array
Array = jnp.ndarray
Shape = Tuple[int, Ellipsis]
Dtype = Any


class BatchNorm(nn.Module):
  """Batch normalization layer for RealNVP. Scale and bias are trainable params."""
  use_running_average: Optional[bool] = None
  momentum: float = 0.9
  epsilon: float = 1e-5
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones

  @nn.compact
  def __call__(self, x, logdet=0, reverse=False):
    """Normalizes the input using batch statistics."""
    use_running_average = self.use_running_average
    feature_shape = (x.shape[1],)

    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s, jnp.float32), feature_shape)
    ra_var = self.variable('batch_stats', 'var',
                           lambda s: jnp.ones(s, jnp.float32), feature_shape)

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
    else:
      # Compute batch stats.
      mean = jnp.mean(x, axis=0)
      var = jnp.mean(jnp.square(x- mean), axis=0)

      if not self.is_initializing():
        ra_mean.value = (
            self.momentum * ra_mean.value + (1 - self.momentum) * mean)
        ra_var.value = (
            self.momentum * ra_var.value + (1 - self.momentum) * var)

    # Normalize.
    scale = self.param('scale', self.scale_init, feature_shape)
    bias = self.param('bias', self.bias_init, feature_shape)

    if not reverse:
      out = (x - mean) / jnp.sqrt(var + self.epsilon)
      out = out * scale + bias
      logdet += jnp.sum(jnp.log(scale) - 0.5 * jnp.log(var))
    else:
      out = (x - bias) / scale
      out = out * jnp.sqrt(var + self.epsilon) + mean
      logdet -= jnp.sum(jnp.log(scale) - 0.5 * jnp.log(var))
    return out, logdet


class ZeroFC(nn.Module):
  """A fully-connected layer with all-zeros initialization."""
  features: int

  @nn.compact
  def __call__(self, x, logscale_factor=3.0):
    scale = self.param(
        'fc_scale', lambda k, sh: jnp.zeros(self.features), (self.features,))
    out = nn.Dense(self.features,
                   kernel_init=jax.nn.initializers.zeros,
                   bias_init=jax.nn.initializers.zeros)(x)
    out = out * jnp.exp(3 * scale)
    return out


class AffineCoupling(nn.Module):
  """Coupling layer that splits the input data along the channel dimension."""
  out_dim: int
  seqfrac: int = 4
  train: bool = True
  batch_norm: bool = True
  init_std: float = 0.05

  @nn.compact
  def __call__(self, inputs, logdet=0, reverse=False):
    # Split
    xa, xb = jnp.split(inputs, 2, axis=1)

    # Neural net
    out_dim = self.out_dim
    net = nn.Dense(
        int(out_dim / (2 * self.seqfrac)),
        kernel_init=jax.nn.initializers.normal(self.init_std),
        bias_init=jax.nn.initializers.zeros)(xa)
    net = nn.leaky_relu(net, negative_slope=0.01)
    if self.batch_norm:
      net = nn.BatchNorm(
          use_running_average=not self.train, epsilon=1e-2, momentum=0.9)(net)
    net = nn.Dense(
        int(out_dim / (2 * self.seqfrac)),
        kernel_init=jax.nn.initializers.normal(self.init_std),
        bias_init=jax.nn.initializers.zeros)(net)
    net = nn.leaky_relu(net, negative_slope=0.01)
    if self.batch_norm:
      net = nn.BatchNorm(
          use_running_average=not self.train, epsilon=1e-2, momentum=0.9)(net)
    net = ZeroFC(2 * (out_dim // 2))(net)

    log_s0, t = jnp.split(net, 2, axis=1)
    log_s = jnp.tanh(log_s0)
    s = jnp.exp(log_s)
    if not reverse:
      yb = (xb + t) * s
      logdet += jnp.sum(log_s, axis=1)
    else:
      yb = xb / s - t
      logdet -= jnp.sum(log_s, axis=1)

    y = jnp.concatenate((xa, yb), axis=-1)

    return y, logdet


class ActNorm(nn.Module):
  """Batch normalization with learnable scale and bias parameters.

  Forward pass: `y = x * sigma + mu`
  Reverse pass: `x = (y - mu) / sigma`

  `mu` and `sigma` are trainable variables (contrary to batch normalization).
  We initialize `mu` and `sigma` in a data-dependent manner, such that the first
  batch of data used for initialization is normalized to zero-mean
  and unit-variance. Thus, when calling `model.init`, please initialize with
  a batch of inputs from the actual training data (not dummy inputs).
  """

  @nn.compact
  def __call__(self, inputs, logdet=0, reverse=False):
    # Data dependent initialization. Will use the values of the batch
    # given during `model.init`.
    axes = tuple(i for i in range(len(inputs.shape) - 1))
    def dd_mean_initializer(*_):
      """Data-dependant init for `mu`."""
      nonlocal inputs
      mean = jnp.mean(inputs).reshape((1,))
      if reverse and self.is_initializing():
        mean = jnp.zeros_like(mean)
      return -mean

    def dd_std_initializer(*_):
      """Data-dependant init for `sigma`."""
      nonlocal inputs
      std = jnp.std(inputs).reshape((1,))
      if reverse and self.is_initializing():
        sigma_inv = jnp.zeros_like(std)
      else:
        sigma_inv = jnp.log(std + 1e-6)
      return sigma_inv

    # Mean and variance:
    shape = (1,) * len(axes) + (inputs.shape[-1],)
    loc = self.param('actnorm_loc', dd_mean_initializer, shape)
    log_scale_inv = self.param(
        'actnorm_log_scale_inv', dd_std_initializer, shape)

    scale_inv = jnp.exp(log_scale_inv)
    log_abs = -log_scale_inv

    # Log-det factor:
    in_dim = inputs.shape[1]

    if not reverse:
      y = (1.0 / scale_inv) * (inputs + loc)
      logdet += in_dim * jnp.sum(log_abs)
    else:
      y = inputs * scale_inv - loc
      logdet -= in_dim * jnp.sum(log_abs)

    # Logdet and return
    return y, logdet


class ScaledSoftplus(nn.Module):
  """Scaled softplus layer, where the log-scale value is learned.

  This is used in the MRI problem to apply a positivity constraint to images.
  Reverse pass pseudocode: `softplus(x) * exp(log_scale_factor)`
  """
  init_scale: float = 1.

  def setup(self):
    """Initialize log-scale factor."""
    log_scale = jnp.log(self.init_scale) * jnp.ones(1)
    self.log_scale = self.param('log_scale', lambda k, sh: log_scale, (1,))

  def __call__(self, inputs, logdet=0, reverse=True):
    if not reverse:
      raise NotImplementedError(
          'PositivityConstraint cannot be applied in forward direction.')

    scale_factor = jnp.exp(self.log_scale)
    x = nn.softplus(inputs) * scale_factor

    # Compute log-det of reverse pass.
    det_softplus = jnp.sum(inputs - nn.softplus(inputs), axis=1)
    det_scale = self.log_scale * inputs.shape[1]
    logdet += det_softplus + det_scale
    return x, logdet


class FlowStep(nn.Module):
  """Normalizing flow step with alternating `AffineCoupling` layers."""
  width: int
  seqfrac: int = 4
  batch_norm: bool = True
  init_std: float = 0.05
  train: bool = True

  def setup(self):
    self.actnorm1 = ActNorm()
    self.actnorm2 = ActNorm()
    self.coupling1 = AffineCoupling(
        self.width, self.seqfrac, batch_norm=self.batch_norm,
        init_std=self.init_std, train=self.train)
    self.coupling2 = AffineCoupling(
        self.width, self.seqfrac, batch_norm=self.batch_norm,
        init_std=self.init_std, train=self.train)

  @nn.compact
  def __call__(self, x, logdet=0, reverse=False):
    if not reverse:
      x, logdet = self.actnorm1(x, logdet=logdet, reverse=False)
      x, logdet = self.coupling1(x, logdet=logdet, reverse=False)
      x = x[:, ::-1]
      x, logdet = self.actnorm2(x, logdet=logdet, reverse=False)
      x, logdet = self.coupling2(x, logdet=logdet, reverse=False)
      x = x[:, ::-1]
    else:
      x = x[:, ::-1]
      x, logdet = self.coupling2(x, logdet=logdet, reverse=True)
      x, logdet = self.actnorm2(x, logdet=logdet, reverse=True)
      x = x[:, ::-1]
      x, logdet = self.coupling1(x, logdet=logdet, reverse=True)
      x, logdet = self.actnorm1(x, logdet=logdet, reverse=True)

    return x, logdet


def get_orders(out_dim, n_flow, permute='random'):
  """Get channel ordering and inverse ordering for each flow step."""
  orders, inv_orders = [], []
  for k in range(n_flow):
    if permute == 'random':
      order = np.random.RandomState(k).permutation(out_dim)
      order_list = list(order)
      inv_order = [order_list.index(i) for i in range(out_dim)]
      order = jnp.array(order)
      inv_order = jnp.array(inv_order)
    elif permute == 'reverse':
      order = jnp.arange(out_dim - 1, -1, -1)
      inv_order = jnp.arange(out_dim)
    else:
      # No permutation.
      order = jnp.arange(out_dim)
      inv_order = jnp.arange(out_dim - 1, -1, -1)
    orders.append(order)
    inv_orders.append(inv_order)
  return orders, inv_orders


class RealNVP(nn.Module):
  """RealNVP model."""
  out_dim: int  # dimensionality of output
  n_flow: int
  orders: List[jnp.ndarray]
  reverse_orders: List[jnp.ndarray]
  seqfrac: int = 4
  include_softplus: bool = False
  init_softplus_log_scale: float = 1
  batch_norm: bool = True
  init_std: float = 0.05
  train: bool = True

  @nn.compact
  def __call__(self, x, logdet=0, reverse=False):
    """Apply RealNVP."""
    for k in range(self.n_flow):
      flow_idx = k if not reverse else self.n_flow - k - 1
      # Reverse: apply reordering before flow step.
      if reverse:
        x = jnp.take(x, self.reverse_orders[flow_idx], axis=1)
      x, logdet = FlowStep(
          self.out_dim, self.seqfrac, batch_norm=self.batch_norm,
          init_std=self.init_std, train=self.train,
          name=f'flow_{flow_idx + 1}')(
              x, logdet=logdet, reverse=reverse)
      # Forward: apply reordering after flow step.
      if not reverse:
        x = jnp.take(x, self.orders[flow_idx], axis=1)

    if (reverse or self.is_initializing()) and self.include_softplus:
      x, logdet = ScaledSoftplus(
          init_scale=self.init_softplus_log_scale, name='softplus')(
              x, logdet=logdet)

    return x, logdet
