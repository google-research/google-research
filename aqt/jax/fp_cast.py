# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Emulate downcasting float32/bfloat16 to custom floating-point formats."""

import functools
from typing import Tuple

import dataclasses
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class FloatingPointBounds:
  """Dataclass representing the bounds for a floating-point type.

  The type is presumed to have 'flush to zero' semantics.

  Attributes:
    flush_to_zero_bound: The magnitude of the smallest representable value. If a
      logical value with an absolute value less than this is cast to this type,
      it is flushed to zero.
    saturation_bound: The magnitude of the largest representable value. If a
      logical value with an absolute value greater than this is cast to this
      type, it is clipped to this value.
  """

  flush_to_zero_bound: float
  saturation_bound: float


def get_bounds(exp_min, exp_max,
               sig_bits):
  """Returns the clipping bounds for a giving floating-point specification.

  Args:
    exp_min: The denormal exponent of the target format.
    exp_max: Maximum exponent of the target format (no support for infs & nans)
    sig_bits: The number of significant bits in the target format (excluding the
      hidden bit).

  Returns:
    A FloatingPointBounds dataclass.
  """
  return FloatingPointBounds(
      flush_to_zero_bound=2**exp_min,
      saturation_bound=2**exp_max * (2 - 2**-sig_bits))


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def downcast_sat_ftz(
    x,
    exp_min,
    exp_max,
    sig_bits,
):
  """Downcast bfloat16, or float32 to a lower precision floating-point.

  - The downcast returns an argument of the *same* type as x but with
    numerical characteristics of the target fp-format.
  - The implementation (sat)urates to the largest representable number in the
    range instead of rounding to infinity.
  - Rounding mode is round-nearest-even for TPU (or configured rounding).
  - Support for special values:
    - The target fp is assumed to *not* support special values, as exp_max is
      used for the numerical range. For the sake of emulation special values are
      propagated as values on the input type, such that
      downcast(nan/inf) -> nan/inf is preserved.
  - Denormals: the target format doesn't support denormals. The minimum
    representable positive value is 2^exp_min. Denormals flush to zero (ftz).

  Args:
    x: The argument to be converted.
    exp_min: The denormal exponent of the target format.
    exp_max: Maximum exponent of the target format (no support for infs & nans)
    sig_bits: The number of significant bits in the target format (excluding
      the hidden bit).

  Returns:
   Cast of x to a lower precision float, emulating degraded precision, but of
   the same type as x.
  """
  if x.dtype not in (jnp.float32, jnp.bfloat16):
    raise ValueError('Argument is expected to be of type jnp.float32 '
                     'or jnp.bfloat16')
  # Mask for exponent bits in fp32 representation.
  exp_mask = 0x7f800000
  # NaNs / Infs have representation-value >= specials_bound.
  specials_bound = 0x7f800000
  # Binary representation of +1.0.
  one = 0x3f800000
  # Mask for mantissa bits (lower 23-bits) of fp32 representation.
  mant_mask = 0x007fffff
  xf = x.astype(jnp.float32)
  xi = xf.view(jnp.int32)
  exp = xi & exp_mask
  # Scale the argument to the unit binade.
  xi_one = (xi & mant_mask) | one
  offset = 2**(23 - sig_bits)
  # Addition of offset creates alignment to shift-off and round trailing bits.
  # Subtraction brings the rounded result back to the unit binade.
  xf_one_rnd = (xi_one.view(jnp.float32) + offset) - offset
  # Scale back to the original binade.
  xf_rnd = xf_one_rnd * exp.view(jnp.float32)
  bounds = get_bounds(exp_min=exp_min, exp_max=exp_max, sig_bits=sig_bits)
  xf_rnd_sat = jnp.minimum(xf_rnd, bounds.saturation_bound)
  # Flush denormals to zero and recover sign.
  xf_rnd_sat_ftz = jnp.sign(xf) * xf_rnd_sat * (
      xf_rnd_sat >= bounds.flush_to_zero_bound)
  xf_rnd_sat_ftz = jnp.where(exp >= specials_bound, xf, xf_rnd_sat_ftz)
  return xf_rnd_sat_ftz.astype(x.dtype)


@downcast_sat_ftz.defjvp
def _downcast_sat_ftz_jvp(
    exp_min, exp_max, sig_bits, primals,
    tangents):
  """Computes the straight-through-estimator gradient of downcast_sat_ftz.

  This defines the approximate gradient of the `downcast_sat_ftz` function.
  Because that function is discontinuous, this defines the gradient of an
  approximate continuous version of that function. Specifically, it defines the
  gradient of:

  x = jnp.clip(x, -saturation_bound, saturation_bound)

  This ignores the rounding operations in the significand that occur in
  `downcast_sat_ftz`, and thus implements the straight-through estimator.

  Args:
    exp_min: The denormal exponent of the target format.
    exp_max: Maximum exponent of the target format (no support for infs & nans)
    sig_bits: The number of significant bits in the target format (excluding the
      hidden bit).
    primals: Primal value of the downcast_sat_ftz input. It should be a tuple
      with a single value.
    tangents: Tangent value of downcast_sat_ftz inputs evaluated at 'primals'.
      It should be a tuple with a single value.

  Returns:
    A tuple consisting of the primal output of downcast_sat_ftz and the
    corresponding tangent tensor.

  Raises:
    ValueError: `primals` or `tangents` is the wrong length.
  """
  if len(primals) != 1 or len(tangents) != 1:
    raise ValueError(
        'The primal and tangent tuples should only have one element each. '
        'This element corresponds to the one differentiable input to '
        'downcast_sat_ftz.')
  (x,), (x_dot,) = primals, tangents
  y = downcast_sat_ftz(
      x, exp_min=exp_min, exp_max=exp_max, sig_bits=sig_bits)
  # Differentiable approximation to downcast_sat_ftz whose gradient is used as a
  # straight-through-estimator for the downcasting operation.
  def differentiable_downcast(x):
    bounds = get_bounds(exp_min, exp_max, sig_bits)
    return jnp.clip(x, -bounds.saturation_bound, bounds.saturation_bound)
  _, y_tangent = jax.jvp(differentiable_downcast, (x,), (x_dot,))
  return y, y_tangent
