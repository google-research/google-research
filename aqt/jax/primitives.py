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

"""Primitives for neural network quantization implemented in jax.
"""
from typing import Any, Iterable, Optional, TypeVar

import jax
from jax import lax
from jax.interpreters.xla import DeviceArray
import jax.numpy as jnp

# Custom type for bounds for type hinting
BoundsT = TypeVar('BoundsT', float, jnp.ndarray)

jnp_dtype = Any  # type of jax dtype; b/164524367 use more specific type.

# Global bool to control the use of epsilon in the denominator of the scaling
# methods signed_int_scale and unsigned_int_scale. Epsilon is added to avoid
# division by 0. For testing, one may choose to disable the epsilon by setting
# this global to True.
# As this is a global variable, please modify it only before calling any
# functions that use it.
DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = False


def clip_to_signed_int(x, *, prec,
                       dtype):
  """Clip x to range [-2**(prec - 1) + 1, 2**(prec - 1) - 1].

  Args:
    x: Argument to be clipped.
    prec: Precision of two's complement number.
    dtype: Desired type of the result.

  Returns:
    Result.
  """
  lbound = -2**(prec - 1) + 1
  ubound = 2**(prec - 1) - 1
  return jnp.clip(x, a_min=lbound, a_max=ubound).astype(dtype)


def clip_to_unsigned_int(x, *, prec,
                         dtype):
  """Clip x to range of unsigned int interval with the given precision.

  Args:
    x: Argument to be clipped.
    prec: The target precision for quantization.
    dtype: Desired type of the result.

  Returns:
    Result.
  """
  lbound = 0
  ubound = 2**prec - 1
  return jnp.clip(x, a_min=lbound, a_max=ubound).astype(dtype)


def add_straight_through_estimator(jax_function):
  """Defines the gradient of a function to be the straight-through-estimator.

  Specifically, the Jacobian-vector product associated with the function is
  defined to be the identity.

  This causes Jax to effectively ignore this function in the backwards pass.

  Args:
    jax_function: A Jax function that has been decorated with @jax.custom_vjp.
      It is expected to take in and return one positional argument.
  """
  # See
  # https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  def ste(primals, tangents):
    return jax_function(primals[0]), tangents[0]

  jax_function.defjvp(ste)


@jax.custom_jvp
def floor_with_gradient(x):
  """Floor with Straight-Through-Estimator gradient."""
  return jnp.floor(x)


add_straight_through_estimator(floor_with_gradient)


@jax.custom_jvp
def round_with_gradient(x):
  """Round with Straight-Through-Estimator gradient."""
  return jnp.floor(x + jnp.array(0.5))


add_straight_through_estimator(round_with_gradient)


def round_and_clip_to_signed_int(x, *, prec,
                                 dtype):
  """Round and clip x to range of signed type with precision 'prec'.

  Requires prec <= 24.

  Args:
    x: The argument to be quantized.
    prec: The target precision for quantization.
    dtype: Desired type of the result.

  Returns:
    Result.
  """
  return clip_to_signed_int(round_with_gradient(x), prec=prec, dtype=dtype)


def floor_and_clip_to_unsigned_int(x, *, prec,
                                   dtype):
  """Floor-and-clip x to range of unsigned type with precision 'prec'.

  Requires prec <= 24.

  Args:
    x: The argument to be quantized.
    prec: The target precision for quantization.
    dtype: Desired type of the result.

  Returns:
    Result.
  """
  return clip_to_unsigned_int(floor_with_gradient(x), prec=prec, dtype=dtype)


def signed_int_bound(prec):
  """Computes the bound value for scaling signed input with precision 'prec'."""
  if prec == 1:
    # since after scaling we have rounding [-0.25, 0.25] interval guarantees
    # all x will be rounded to zero and avoids division by zero
    return 0.25
  elif prec > 1:
    # TODO(lew): run an experiment with 2bit rounding with 2**(prec-1) - 0.5
    # bound
    return 2**(prec - 1.0) - 1.0
  else:  # prec < 1
    raise ValueError('prec value should be >= 1.')


def max_abs_weights(x,
                    *,
                    axis = None):
  """Computes the maximum of the absolute value of weights along the axis."""
  abs_max_x = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
  return abs_max_x


def signed_int_scale(x,
                     *,
                     prec,
                     axis = None):
  """Computes a scale s, s.t.

  -2**(prec - 1) + 1 <= s * x <= 2**(prec - 1) - 1.

  Does not propagate gradients.

  Args:
    x: The input to be scaled.
    prec: Signed int precision of the scaled result.
    axis: Dimensions of input to consider for scaling.

  Returns:
    The scaling value.
  """
  abs_max_x = max_abs_weights(x, axis=axis)
  if not DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING:
    abs_max_x += jnp.finfo(jnp.float32).eps  # to avoid div by 0
  scale = signed_int_bound(prec) / abs_max_x
  scale = lax.stop_gradient(scale)
  return scale


def unsigned_int_bound(prec):
  """Computes bound value for scaling unsigned input with precision 'prec'."""
  # NOTE: it's computing 2**prec, which is above the largest unsigned
  # value for that prec. This factor is used for scaling input to range
  # [0, 2**prec].
  if prec < 0:  # TODO(shivaniagrawal) or allow only some values: 4, 8 etc.
    raise ValueError('prec value should be >= 0.')

  return 2**prec


def unsigned_int_scale(x,
                       *,
                       prec,
                       axis = None):
  """Computes a scale s, s.t.

  0 <= s * x <= 2**prec, where min(x) >= 0.
  Does not propagate gradients.

  Args:
    x: The input to be scaled.
    prec: Unsigned int precision of the scaled result.
    axis: Dimensions of input to consider for scaling.

  Returns:
    The scaling value.
  """
  max_x = jnp.max(x, axis=axis, keepdims=True)
  if not DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING:
    max_x += jnp.finfo(jnp.float32).eps  # to avoid div by 0

  scale = unsigned_int_bound(prec) / max_x
  scale = lax.stop_gradient(scale)
  return scale
