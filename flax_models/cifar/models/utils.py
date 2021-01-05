# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Shake-shake and shake-drop functions."""

from typing import Optional, Tuple
import flax
from flax import nn
import jax
import jax.numpy as jnp


_BATCHNORM_MOMENTUM = 0.9
_BATCHNORM_EPSILON = 1e-5


def activation(x,
               train,
               apply_relu = True,
               name = ''):
  """Applies BatchNorm and then (optionally) ReLU.

  Args:
    x: Tensor on which the activation should be applied.
    train: If False, will use the moving average for batch norm statistics.
        Else, will use statistics computed on the batch.
    apply_relu: Whether or not ReLU should be applied after batch normalization.
    name: How to name the BatchNorm layer.

  Returns:
    The input tensor where BatchNorm and (optionally) ReLU where applied.
  """
  batch_norm = nn.BatchNorm.partial(
      use_running_average=not train,
      momentum=_BATCHNORM_MOMENTUM,
      epsilon=_BATCHNORM_EPSILON)
  x = batch_norm(x, name=name)
  if apply_relu:
    x = jax.nn.relu(x)
  return x


# Kaiming initialization with fan out mode. Should be used to initialize
# convolutional kernels.
conv_kernel_init_fn = jax.nn.initializers.variance_scaling(
    2.0, 'fan_out', 'normal')


def dense_layer_init_fn(key,
                        shape,
                        dtype = jnp.float32):
  """Initializer for the final dense layer.

  Args:
    key: PRNG key to use to sample the weights.
    shape: Shape of the tensor to initialize.
    dtype: Data type of the tensor to initialize.

  Returns:
    The initialized tensor.
  """
  num_units_out = shape[1]
  unif_init_range = 1.0 / (num_units_out)**(0.5)
  return jax.random.uniform(key, shape, dtype, -1) * unif_init_range


def shake_shake_train(xa,
                      xb,
                      rng = None):
  """Shake-shake regularization in training mode.

  Shake-shake regularization interpolates between inputs A and B
  with *different* random uniform (per-sample) interpolation factors
  for the forward and backward/gradient passes.

  Args:
    xa: Input, branch A.
    xb: Input, branch B.
    rng: PRNG key.

  Returns:
    Mix of input branches.
  """
  if rng is None:
    rng = flax.nn.make_rng()
  gate_forward_key, gate_backward_key = jax.random.split(rng, num=2)
  gate_shape = (len(xa), 1, 1, 1)

  # Draw different interpolation factors (gate) for forward and backward pass.
  gate_forward = jax.random.uniform(
      gate_forward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
  gate_backward = jax.random.uniform(
      gate_backward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
  # Compute interpolated x for forward and backward.
  x_forward = xa * gate_forward + xb * (1.0 - gate_forward)
  x_backward = xa * gate_backward + xb * (1.0 - gate_backward)
  # Combine using stop_gradient.
  return x_backward + jax.lax.stop_gradient(x_forward - x_backward)


def shake_shake_eval(xa, xb):
  """Shake-shake regularization in testing mode.

  Args:
    xa: Input, branch A.
    xb: Input, branch B.

  Returns:
    Mix of input branches.
  """
  # Blend between inputs A and B 50%-50%.
  return (xa + xb) * 0.5


def shake_drop_train(x,
                     mask_prob,
                     alpha_min,
                     alpha_max,
                     beta_min,
                     beta_max,
                     rng = None):
  """ShakeDrop training pass.

  See https://arxiv.org/abs/1802.02375

  Args:
    x: Input to apply ShakeDrop to.
    mask_prob: Mask probability.
    alpha_min: Alpha range lower.
    alpha_max: Alpha range upper.
    beta_min: Beta range lower.
    beta_max: Beta range upper.
    rng: PRNG key (if `None`, uses `flax.nn.make_rng`).

  Returns:
    The regularized tensor.
  """
  if rng is None:
    rng = flax.nn.make_rng()
  bern_key, alpha_key, beta_key = jax.random.split(rng, num=3)
  rnd_shape = (len(x), 1, 1, 1)
  # Bernoulli variable b_l in Eqn 6, https://arxiv.org/abs/1802.02375.
  mask = jax.random.bernoulli(bern_key, mask_prob, rnd_shape)
  mask = mask.astype(jnp.float32)

  alpha_values = jax.random.uniform(
      alpha_key,
      rnd_shape,
      dtype=jnp.float32,
      minval=alpha_min,
      maxval=alpha_max)
  beta_values = jax.random.uniform(
      beta_key, rnd_shape, dtype=jnp.float32, minval=beta_min, maxval=beta_max)
  # See Eqn 6 in https://arxiv.org/abs/1802.02375.
  rand_forward = mask + alpha_values - mask * alpha_values
  rand_backward = mask + beta_values - mask * beta_values
  return x * rand_backward + jax.lax.stop_gradient(
      x * rand_forward - x * rand_backward)


def shake_drop_eval(x,
                    mask_prob,
                    alpha_min,
                    alpha_max):
  """ShakeDrop eval pass.

  See https://arxiv.org/abs/1802.02375

  Args:
    x: Input to apply ShakeDrop to.
    mask_prob: Mask probability.
    alpha_min: Alpha range lower.
    alpha_max: Alpha range upper.

  Returns:
    The regularized tensor.
  """
  expected_alpha = (alpha_max + alpha_min) / 2
  # See Eqn 6 in https://arxiv.org/abs/1802.02375.
  return (mask_prob + expected_alpha - mask_prob * expected_alpha) * x
