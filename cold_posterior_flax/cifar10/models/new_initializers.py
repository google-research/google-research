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

"""Initializers for Flax."""
import jax.numpy as jnp
import jax.scipy.stats
import numpy as onp


def compute_fans(shape, in_axis=-2, out_axis=-1):
  receptive_field_size = onp.prod(shape) / shape[in_axis] / shape[out_axis]
  fan_in = shape[in_axis] * receptive_field_size
  fan_out = shape[out_axis] * receptive_field_size
  return fan_in, fan_out


def lecun_normed():
  """Norm-standardized lecun_normal initialization."""

  def init(key, shape, dtype=jnp.float32):
    fan_in, _ = compute_fans(shape, -2, -1)
    denominator = fan_in
    scale = 1
    variance = jnp.array(scale / denominator, dtype=dtype)

    # constant is stddev of standard normal truncated to (-2, 2)
    stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
    k = jax.random.truncated_normal(key, -2, 2, shape, dtype) * stddev
    k /= (jnp.linalg.norm(k, axis=0, keepdims=True, ord=2) + 1e-8)
    return k

  return init


inv_softplus = lambda x: onp.log(onp.exp(x) - 1)


def init_softplus_ones(_, shape, dtype=jnp.float32):
  return jnp.ones(shape, dtype) * inv_softplus(1.0)


def init_softplus_eps(_, shape, dtype=jnp.float32):
  return jnp.ones(shape, dtype) * inv_softplus(1e-5)
