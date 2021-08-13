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

"""Primitives for neural network quantization implemented in jax."""
from typing import Any, Iterable, Optional, TypeVar

import jax
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


def round_and_clip_to_signed_int(x,
                                 *,
                                 prec,
                                 dtype,
                                 half_shift):
  """Round and clip x to range of signed type with precision 'prec'.

  Requires prec <= 24.

  Args:
    x: The argument to be quantized.
    prec: The target precision for quantization.
    dtype: Desired type of the result.
    half_shift: Uses all available values in a given integer type and makes them
      symmetric by adding 0.5. E.g. instead of to [-1, 0, 1] we will round to
      [-1.5, -0.5, 0.5, 1.5].

  Returns:
    Result.
  """

  # epsilon has to be big enough so that its subtraction in bound computation is
  # not rounded down to zero. It has to be small enough so it has no ML effect.
  # epsilon is necessary for half_shift when prec=1 so that values get floored
  # to -1/0 (not -1/0/1) after clipping to bound.
  epsilon = 2**(-7)
  bound = signed_int_bound(prec=prec, half_shift=half_shift)
  if half_shift:
    bound -= epsilon
    x = jnp.clip(x, a_min=-bound, a_max=bound).astype(dtype)
    x = floor_with_gradient(x) + 0.5
  else:
    # TODO(lew): Use the formula for better gradients. Needs a sweep though.
    # bound = 2**(prec - 1) - 0.5 - epsilon
    x = jnp.clip(x, a_min=-bound, a_max=bound).astype(dtype)
    x = round_with_gradient(x)
  return x


def floor_and_clip_to_unsigned_int(x,
                                   *,
                                   prec,
                                   dtype,
                                   half_shift):
  """Floor-and-clip x to range of unsigned type with precision 'prec'.

  Requires prec <= 24.

  Args:
    x: The argument to be quantized.
    prec: The target precision for quantization.
    dtype: Desired type of the result.
    half_shift: Needs to be false.

  Returns:
    Result.
  """
  assert not half_shift
  x = floor_with_gradient(x)
  # TODO(lew): should be (a_max=2**prec - epsilon) for a better gradient.
  x = jnp.clip(x, a_min=0, a_max=2**prec - 1).astype(dtype)
  return x


def signed_int_bound(prec, half_shift):
  """Computes the bound value for scaling signed input with precision 'prec'."""
  if prec >= 1:
    # bound
    if half_shift:
      return 2**(prec - 1.0)
    else:
      # TODO(lew): run an experiment with 2bit rounding with 2**(prec-1) - 0.5
      # When prec=1 and half_shift is turned off, a zero bound will make the
      # scaling factor zero. This causes division by zero problem when dividing
      # the scaling factor back. So we need to return a nonzero value that
      # makes inputs rounded to zero.
      return 2**(prec - 1.0) - 1.0 if prec > 1 else 0.25
  else:  # prec < 1
    raise ValueError('prec value should be >= 1.')


def max_abs_weights(x,
                    *,
                    axis = None):
  """Computes the maximum of the absolute value of weights along the axis."""
  abs_max_x = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
  return abs_max_x


def unsigned_int_bound(prec):
  """Computes bound value for scaling unsigned input with precision 'prec'."""
  # NOTE: it's computing 2**prec, which is above the largest unsigned
  # value for that prec. This factor is used for scaling input to range
  # [0, 2**prec].
  if prec < 0:  # TODO(shivaniagrawal) or allow only some values: 4, 8 etc.
    raise ValueError('prec value should be >= 0.')

  return 2**prec
