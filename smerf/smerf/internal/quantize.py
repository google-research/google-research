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

"""Simulate quantization during training and for quantizing after training."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from smerf.internal import math


MAX_BYTE = 2.0**8 - 1.0


def quantize_float_to_byte(x):
  """Converts float32 to uint8."""
  return np.minimum(MAX_BYTE, np.maximum(0.0, np.round(MAX_BYTE * x))).astype(
      np.uint8
  )


def dequantize_byte_to_float(x, xnp):
  """Converts uint8 to float32."""
  return x.astype(xnp.float32) / MAX_BYTE


def differentiable_byte_quantize(x):
  """Implements rounding with a straight-through-estimator."""
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(
      jnp.round(jnp.clip(x, 0.0, 1.0) * MAX_BYTE) / MAX_BYTE
  )


def simulate_quantization(x, v_min, v_max):
  """Simulates quant. during training: [-inf, inf] -> [v_min, v_max]."""
  x = jax.nn.sigmoid(x)  # Bounded to [0, 1].
  x = differentiable_byte_quantize(x)  # quantize and dequantize.
  return math.denormalize(x, v_min, v_max)  # Bounded to [v_min, v_max].


def dequantize_and_interpolate(x_grid, data, v_min, v_max):
  """Dequantizes and denormalizes and then linearly interpolates grid values."""
  x_floor = jnp.floor(x_grid).astype(jnp.int32)
  x_ceil = jnp.ceil(x_grid).astype(jnp.int32)
  local_coordinates = x_grid - x_floor
  res = jnp.zeros(x_grid.shape[:-1] + (data.shape[-1],))

  # If any of data's spatial dims have length=0, then the following gather
  # operations will fail. This will occur if alive_voxels was empty during
  # baking. Return a default value instead.
  if any(dim == 0 for dim in data.shape[:-1]):
    return res

  corner_coords = [[False, True] for _ in range(local_coordinates.shape[-1])]
  for z in itertools.product(*corner_coords):
    w = jnp.ones(local_coordinates.shape[:-1])
    l = []
    for i, b in enumerate(z):
      w = w * (
          local_coordinates[Ellipsis, i] if b else (1 - local_coordinates[Ellipsis, i])
      )
      l.append(x_ceil[Ellipsis, i] if b else x_floor[Ellipsis, i])
    gathered_data = data[tuple(l)]
    gathered_data = dequantize_byte_to_float(gathered_data, jnp)
    gathered_data = math.denormalize(gathered_data, v_min, v_max)
    res = res + w[Ellipsis, None] * gathered_data.reshape(res.shape)
  return res


def map_quantize(*l):
  """For quantization after training."""

  def sigmoid_and_quantize_float_to_byte(x):
    if x is None:
      return None
    cpu = jax.devices('cpu')[0]  # Prevents JAX from moving array to GPU.
    x = jax.device_put(x, cpu)
    x = jax.nn.sigmoid(x)
    return quantize_float_to_byte(x)

  return jax.tree_util.tree_map(sigmoid_and_quantize_float_to_byte, l)
