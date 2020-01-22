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

"""Shake-shake and ShakeDrop utility functions."""
import flax.nn
import jax
import jax.numpy as jnp


def shake_shake_train(xa, xb, rng=None):
  """Shake-shake regularization: training.

  Shake-shake regularization interpolates between inputs A and B
  with *different* random uniform (per-sample) interpolation factors
  for the forward and backward/gradient passes

  Args:
    xa: input, branch A
    xb: input, branch B
    rng: PRNG key

  Returns:
    Mix of input branches
  """
  if rng is None:
    rng = flax.nn.make_rng()
  gate_forward_key, gate_backward_key = jax.random.split(rng, num=2)
  gate_shape = (len(xa), 1, 1, 1)

  # Draw different interpolation factors (gate) for forward and backward pass
  gate_forward = jax.random.uniform(
      gate_forward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
  gate_backward = jax.random.uniform(
      gate_backward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
  # Compute interpolated x for forward and backward
  x_forward = xa * gate_forward + xb * (1.0 - gate_forward)
  x_backward = xa * gate_backward + xb * (1.0 - gate_backward)
  # Combine using stop_gradient
  x = x_backward + jax.lax.stop_gradient(x_forward - x_backward)
  return x


def shake_shake_eval(xa, xb):
  """Shake-shake regularization: evaluation.

  Args:
    xa: input, branch A
    xb: input, branch B

  Returns:
    Mix of input branches
  """
  # Blend between inputs A and B 50%-50%.
  return (xa + xb) * 0.5


def shake_drop_train(x, mask_prob, alpha_min, alpha_max, beta_min, beta_max,
                     rng=None):
  """ShakeDrop training pass.

  See https://arxiv.org/abs/1802.02375

  Args:
    x: input to apply ShakeDrop to
    mask_prob: mask probability
    alpha_min: alpha range lower
    alpha_max: alpha range upper
    beta_min: beta range lower
    beta_max: beta range upper
    rng: PRNG key (if `None`, uses `flax.nn.make_rng`)

  Returns:
  """
  if rng is None:
    rng = flax.nn.make_rng()
  bern_key, alpha_key, beta_key = jax.random.split(rng, num=3)
  rnd_shape = (len(x), 1, 1, 1)
  # Bernoulli variable b_l in Eqn 6, https://arxiv.org/abs/1802.02375
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
  # See Eqn 6 in https://arxiv.org/abs/1802.02375
  rand_forward = mask + alpha_values - mask * alpha_values
  rand_backward = mask + beta_values - mask * beta_values
  x = x * rand_backward + jax.lax.stop_gradient(x * rand_forward -
                                                x * rand_backward)
  return x


def shake_drop_eval(x, mask_prob, alpha_min, alpha_max):
  """ShakeDrop eval pass.

  See https://arxiv.org/abs/1802.02375

  Args:
    x: input to apply ShakeDrop to
    mask_prob: mask probability
    alpha_min: alpha range lower
    alpha_max: alpha range upper

  Returns:
  """
  expected_alpha = (alpha_max + alpha_min) / 2
  # See Eqn 6 in https://arxiv.org/abs/1802.02375
  x = (mask_prob + expected_alpha - mask_prob * expected_alpha) * x
  return x
