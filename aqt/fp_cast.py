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

"""Emulate downcasting float32/bfloat16 to custom floating-point formats."""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp


@jax.custom_gradient
def downcast_sat_ftz_with_gradient(
    x,
    exp_min,
    exp_max,
    sig_bits,
):
  """Downcast with Straight-Through-Estimator gradient (see downcast_sat_ftz).

  Args:
    x: The argument to be converted.
    exp_min: The denormal exponent of the target format.
    exp_max: Maximum exponent of the target format (no support for infs & nans)
    sig_bits: The number of significant bits in the target format (excluding
      the hidden bit).

  Returns:
    Nested tuple (f, (df/dx, df/d{exp_min}, df/d{exp_max}, df/d{sig_bits}),
    where f is the downcast_sat_ftz operation.
  """
  return (downcast_sat_ftz(x, exp_min, exp_max, sig_bits),
          lambda dx: (dx * jnp.ones_like(x), 0., 0., 0.))


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
  ftz_bound = 2**exp_min
  sat_bound = 2**exp_max * (2 - 2**-sig_bits)
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
  xf_rnd_sat = jnp.minimum(xf_rnd, sat_bound)
  # Flush denormals to zero and recover sign.
  xf_rnd_sat_ftz = jnp.sign(xf) * xf_rnd_sat * (xf_rnd_sat >= ftz_bound)
  xf_rnd_sat_ftz = jnp.where(exp >= specials_bound, xf, xf_rnd_sat_ftz)
  return xf_rnd_sat_ftz.astype(x.dtype)
