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

# pyformat: mode=yapf
"""Math utility functions."""

from typing import Optional, Union

from internal import math
import jax
from jax import numpy as jnp
import optax


def matmul(a, b):
  """jnp.matmul defaults to bfloat16 on TPU, but this doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def safe_sqrt(x,
              *,
              eps = jnp.finfo(jnp.float32).eps,
              value_at_zero = 0.0):
  """A safe version of jnp.sqrt that avoid evaluating at zero.

  Note: sqrt(x) = sqrt(eps) = 3e-4 when x < eps = 1.19e-7.

  Args:
    x: The operand.
    eps: A small number to prevent NaNs.
    value_at_zero: The value to clamp x to near zero. The return value will be
      sqrt(value_at_zero)

  Returns:
    The sqrt(x), or sqrt(value_at_zero) near zero.
  """
  safe_x = jnp.where(x > eps, x, jnp.full_like(x, value_at_zero))
  return jnp.sqrt(safe_x)


def safe_acos(t,
              eps = jnp.finfo(jnp.float32).eps):
  """A safe version of arccos which avoids evaluating at -1 or 1."""
  return jnp.arccos(jnp.clip(t, -1.0 + eps, 1.0 - eps))


def safe_log(x,
             *,
             eps = jnp.finfo(jnp.float32).eps,
             value_at_zero = jnp.finfo(jnp.float32).eps):
  """Computes a safe log that avoids evaluating at zero.

  Args:
    x: Input array.
    eps: A small number to prevent NaNs.
    value_at_zero: The value to clamp x to near zero. The return value will be
      sqrt(value_at_zero)

  Returns:
    log(x) or log(value_at_zero) near zero.
  """
  safe_x = jnp.where(x > eps, x, jnp.full_like(x, value_at_zero))
  return jnp.log(safe_x)


def normalize(
    x,
    axis = -1,
    # pylint: disable=redefined-builtin
    ord = None,
    eps = jnp.finfo(jnp.float32).eps,
):
  """Normalize a vector."""
  return x / optax.safe_norm(x, axis=axis, ord=ord, min_norm=eps, keepdims=True)


def inv_sqrtm(
    matrix,
    normalize_eigvals = False,
):
  """Takes the inverse matrix square root of a PSD matrix.

  Forked from `coord.sqrtm`.

  Args:
    matrix: (..., d, d) A positive semi-definite matrix.
    normalize_eigvals: If True, normalize the eigenvalues by the geometric mean.

  Returns:
    The inverse square root of the matrix, and (eigvec, eigval) if return_eigs
    is True.
  """
  eigvec, eigval = jax.lax.linalg.eigh(
      matrix, symmetrize_input=False, sort_eigenvalues=False)

  if normalize_eigvals:
    # Equivalent to dividing by geometric mean, but numerically stabler.
    log_eigval = jnp.log(eigval)
    eigval = jnp.exp(log_eigval - jnp.mean(log_eigval, axis=-1, keepdims=True))

  scaling = math.safe_div(1, math.safe_sqrt(eigval))
  scaling = scaling[Ellipsis, None, :]
  sqrtm_mat = matmul(eigvec * scaling, jnp.moveaxis(eigvec, -2, -1))

  return sqrtm_mat, (eigvec, eigval)


def to_homogeneous(v):
  """Converts a vector to a homogeneous representation.

  Args:
    v: (*, C) A non-homogeneous vector.

  Returns:
    (*, C+1) A homogeneous version of v.
  """
  return jnp.concatenate([v, jnp.ones_like(v[Ellipsis, :1])], axis=-1)


def from_homogeneous(v):
  """Converts a homogeneous vector to a non-homogeneous vector.

  Args:
    v: (*, C+1) A homogeneous vector.

  Returns:
    (*, C) The non-homogeneous version of v.
  """
  return v[Ellipsis, :-1] / v[Ellipsis, -1:]


def apply_homogeneous_transform(transform,
                                vectors):
  """Apply a homogeneous transformation to a collection of vectors.

  Args:
    transform: (C+1,C+1) A homogeneous transformation matrix.
    vectors: (*,C) An array containing 3D points.

  Returns:
    (*,C) The points transformed by the array.
  """
  vectors_h = to_homogeneous(vectors.reshape((-1, vectors.shape[-1])))
  transformed = from_homogeneous(matmul(transform, vectors_h.T).T)
  return transformed.reshape(vectors.shape)


def generalized_bias_and_gain(x, slope,
                              threshold):
  """Maps the input according to the generalized bias and gain function.

  References:
    https://arxiv.org/abs/2010.09714

  Args:
    x: The inputs array with values in [0, 1] to map.
    slope: The slope parameter of the curve which controls the slope of the
      curve at the threshold.
    threshold: The value at which `x` reverses its shape, and the point at which
      the output is guaranteed to be equal to the input.

  Returns:
    The output of the curve at each input point `x`.
  """
  eps = jnp.finfo(jnp.float32).tiny
  left_curve = (threshold * x) / (x + slope * (threshold - x) + eps)
  right_curve = ((1 - threshold) * (x - 1) / (1 - x - slope *
                                              (threshold - x) + eps) + 1)
  return jnp.where(x < threshold, left_curve, right_curve)
