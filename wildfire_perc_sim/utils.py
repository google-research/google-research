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

"""Shared utilities for wildfire simulator."""

import enum
import functools
from typing import Any, Dict, Sequence, Tuple, Union

import jax
from jax import jit
from jax import lax
from jax import numpy as jnp
from jax import random
import numpy as np

EPS = 1e-5
INF = 1e2

TensorLike = Union[jnp.ndarray, np.ndarray]


class BoundaryCondition(enum.Enum):
  """Permissible Boundary Conditions for wildfire_simulator!"""
  INFINITE = 0
  PERIODIC = 1
  LATERAL_PERIODIC = 2


@functools.partial(jit, static_argnums=0)
def radius_tensor(kernel_dim):
  """Compute radius r tensor (constant everywhere)."""

  kernel_radius = float((kernel_dim - 1) // 2)

  # Note: Cannot use jnp.arange
  # (See https://github.com/jax-ml/jax/issues/5186#issuecomment-757355059)
  r_x = np.arange(kernel_radius, kernel_radius - kernel_dim,
                  -1).reshape(1, kernel_dim)
  r_x = jnp.concatenate([r_x] * kernel_dim, axis=0)[:, :, None]

  r_y = np.arange(kernel_radius, kernel_radius - kernel_dim,
                  -1).reshape(kernel_dim, 1)
  r_y = jnp.concatenate([r_y] * kernel_dim, axis=1)[:, :, None]

  return jnp.concatenate([r_x, r_y], axis=2).transpose(2, 1, 0)


@jit
def normalize(x):
  """Normalize the given tensor along its last axis."""
  return x / (jnp.linalg.norm(x, axis=0, keepdims=True) + EPS)


@functools.partial(jit, static_argnums=(1, 2))
def pad_tensor_3d(
    tensor,
    kernel_shape,
    boundary_condition = BoundaryCondition.INFINITE
):
  """Pad convolution image tensor according to boundary condition.

  Args:
    tensor: 3D tensor to be padded
    kernel_shape: 2-Tuple for the shape of the convolution kernel
      (kernel_shape[0] == kernel_shape[1] must be satisfied currently)
    boundary_condition: Should be one of the values in
      :class:`~utils.BoundaryCondition`

  Returns:
    Padded 3D tensor.
  """
  assert tensor.ndim == 3

  tensor_shape = tensor.shape
  padding_layers = int((kernel_shape[0] - 1) // 2)

  if boundary_condition == BoundaryCondition.INFINITE:
    zero_tensor = jnp.zeros(
        (padding_layers, tensor_shape[1], tensor_shape[2])) + EPS
    tensor_padded = jnp.concatenate((zero_tensor, tensor, zero_tensor), 0)

    zero_tensor = jnp.zeros((tensor_shape[0] + 2 * padding_layers,
                             padding_layers, tensor_shape[2])) + EPS
    tensor_padded = jnp.concatenate((zero_tensor, tensor_padded, zero_tensor),
                                    1)

  elif boundary_condition == BoundaryCondition.PERIODIC:
    tensor_padded = jnp.concatenate((tensor[-padding_layers:, :, :], tensor,
                                     tensor[0:padding_layers, :, :]), 0)
    tensor_padded = jnp.concatenate(
        (tensor_padded[:, -padding_layers:, :], tensor_padded,
         tensor_padded[:, 0:padding_layers, :]), 1)

  elif boundary_condition == BoundaryCondition.LATERAL_PERIODIC:
    zero_tensor = jnp.zeros((tensor_shape[0], padding_layers, tensor_shape[2]))
    tensor_padded = jnp.concatenate((zero_tensor, tensor, zero_tensor), 1)
    tensor_padded = jnp.concatenate(
        (tensor_padded[-padding_layers:, :, :], tensor_padded,
         tensor_padded[0:padding_layers, :, :]), 0)

  else:
    raise ValueError(
        '`boundary_condition` must be one of `infinite`, `periodic`, or'
        '`lateral_periodic`')

  return tensor_padded


@functools.partial(jit, static_argnums=(2,))
def set_border(field, val, layers = 1):
  """Set borders of field to given value."""
  field_out = field
  field_out = field_out.at[Ellipsis, :layers, :].set(val)
  field_out = field_out.at[Ellipsis, -layers:, :].set(val)
  field_out = field_out.at[Ellipsis, :, :layers].set(val)
  field_out = field_out.at[Ellipsis, :, -layers:].set(val)
  return field_out


@jit
def gradient_o1(field, h):
  """Compute first-order accurate 2D gradient."""
  # First order central difference in interior
  # Forward / backward difference at edges
  assert field.ndim == 2

  grad = jnp.zeros((field.shape[0], field.shape[1], 2))

  grad = grad.at[1:-2, :, 0].set((field[2:-1, :] - field[0:-3, :]) / (2 * h))
  grad = grad.at[0, :, 0].set((field[1, :] - field[0, :]) / h)
  grad = grad.at[-1, :, 0].set((field[-1, :] - field[-2, :]) / h)

  grad = grad.at[:, 1:-2, 1].set((field[:, 2:-1] - field[:, 0:-3]) / (2 * h))
  grad = grad.at[:, 0, 1].set((field[:, 1] - field[:, 0]) / h)
  grad = grad.at[:, -1, 1].set((field[:, -1] - field[:, -2]) / h)

  return grad


def termial(n):
  """Return the sum of i from i=1 to n."""
  return n * (n + 1) // 2  # pytype: disable=bad-return-type  # jax-ndarray


def _get_xy_stencil(neighborhood_size):
  """Compute x and y stencils for neighbor computation.

  Constructs the indices of the lower triangular matrix of a matrix of size
  `neighborhood_size x neighborhood_size`.

  Args:
    neighborhood_size: size of the neighborhood

  Returns:
    A 2-tuple of x-indices and y-indices.
  """
  return jnp.nonzero(jnp.tril(jnp.ones((neighborhood_size, neighborhood_size))))


def get_stencil(neighborhood_size):
  """Generate an array of indices of neighbors for a given neighborhood size."""

  x_arr, y_arr = _get_xy_stencil(neighborhood_size)
  xy_norm = jnp.linalg.norm(jnp.stack([x_arr, y_arr]), axis=0)
  dist_array = jnp.column_stack([x_arr, y_arr, xy_norm])

  # Sort by distance to origin
  dist_array_sorted = dist_array[dist_array[:, 2].argsort()]

  # Select desired closest number of neighbors
  stencil = dist_array_sorted[:neighborhood_size, :2]

  # Convert to integers
  stencil = stencil.astype('int32')

  # Add reflection points
  stencil_refxy = jnp.column_stack((stencil[:, 1], stencil[:, 0]))
  stencil = jnp.concatenate((stencil, stencil_refxy))

  stencil_refx = jnp.column_stack((stencil[:, 0], -stencil[:, 1]))
  stencil = jnp.concatenate((stencil, stencil_refx))

  stencil_refy = jnp.column_stack((-stencil[:, 0], stencil[:, 1]))
  stencil = jnp.concatenate((stencil, stencil_refy))

  # Remove duplicates created by reflection
  stencil = jnp.unique(stencil, axis=0)

  return stencil


@jit
def reparameterize(prng, mean,
                   logvar):
  """Reparameterization Trick to sample from a Normal Distribution."""
  prng, key = random.split(prng)
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(key, logvar.shape)
  return mean + eps * std


def apply_percolation_convolution(kernel,
                                  state):
  """Propagate Percolation Model by a single step.

  Args:
    kernel: Convolution kernel of 5 or 6 dimensions. If the kernel is of 6
      dimensions, a multiply reduction is performed along the first dimension.
      The final kernel must have the dimensions [batch, window_height,
      window_width, in_state_channels, out_state_channels].
    state: Current 4D internal state of the dynamical system.

  Returns:
    Updated 4D internal state of the dynamical system.
  """
  assert kernel.ndim in (5, 6)
  if kernel.ndim == 6:
    kernel = jnp.prod(kernel, axis=0, keepdims=False)
  conv = functools.partial(
      lax.conv_general_dilated,
      window_strides=(1, 1),
      padding='SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
  # Note: The kernels are unique for each state in the batch so we
  #       need to manually add a batch dimension of 1
  state = jnp.expand_dims(state, 1)
  return jnp.squeeze(jax.vmap(conv)(state, kernel), 1)


def restructure_sequence_data(
    seq,
    make_finite = True):
  """Restructure TFDS sequence data.

  TFDS returns batched sequence as [L, ...] x B. Restructure that sequence
  to [B, ...] x L.

  Args:
    seq: Sequence Feature from TFDS.
    make_finite: If True, remove any inf/nan with 0s

  Returns:
    Restructured Sequence.
  """
  useq = restructure_distributed_sequence_data(
      [jnp.expand_dims(s, 0) for s in seq], make_finite)
  return [s[:, 0] for s in useq]


def restructure_distributed_sequence_data(
    seq,
    make_finite = True):
  """Restructure TFDS sequence data.

  TFDS returns batched sequence as [N, L, ...] x B. Restructure that sequence
  to [B, N, ...] x L.

  Args:
    seq: Sequence Feature from TFDS.
    make_finite: If True, remove any inf/nan with 0s

  Returns:
    Restructured Sequence.
  """
  updated_seq = [[] for _ in range(seq[0].shape[1])]
  for s in seq:
    for i in range(s.shape[1]):
      updated_seq[i].append(s[:, i:(i + 1)])
  res = [
      jnp.asarray(np.concatenate(u, axis=1)).transpose(
          (1, 0, *list(range(2, seq[0].ndim)))) for u in updated_seq
  ]
  if make_finite:
    return [jnp.nan_to_num(x, nan=0, posinf=INF, neginf=-INF) for x in res]
  return res


def sigmoid(x, c):
  """Computes 1 / (1 + exp(-cx)) in a numerically stable manner."""
  t = jnp.exp(-c * jnp.abs(x))
  return jnp.where(x >= 0, 1 / (1 + t), t / (1 + t))


def denormalize_video(video):
  """De-normalize video to [0; 255] range and cast it to uint8."""
  dtype = np.uint8 if isinstance(video, np.ndarray) else jnp.uint8
  return (((video - video.min()) / (video.max() - video.min())) *
          255).astype(dtype)


def prepend_dict_keys(d, k):
  """Prepends `k` to all the keys of `d`."""
  return {k + key: d[key] for key, val in d.items()}
