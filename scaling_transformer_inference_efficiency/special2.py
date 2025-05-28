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

"""Log-base-2 versions of special functions.

Most special functions are implemented in terms of base-e exponentials and
logarithms. However, most hardware is implemented in terms of base-2.0
exponentials and logarithms, and converting between bases requires a
multiplication. By providing base-2.0 versions of these functions, we can save
some multiplies, requiring the caller to do the compensating multiplies. Often
the caller can do so more efficiently, e.g. by folding them into nearby matrix
multiplies or rescales.
"""

from typing import Optional

from jax import lax
import jax.numpy as jnp

LOG2_E = 1.44269504089  # = log2(e)
LN_2 = 0.69314718056  # = ln(2) = 1.0 / LOG2_E


def exp2(x):
  """Computes 2.0^x.

  This is slightly more efficient than jnp.exp(x), because the hardware natively
  supports base-2.0 exponentials. The following equivalence holds:

    jnp.exp(x) == exp2(x * LOG2_E)

  Args:
    x: Input array.

  Returns:
    2.0^x.
  """
  # NOTE(reinerp): This version generates more efficient TPU code than jnp.exp2.
  two = jnp.float32(2.0)
  return lax.pow(two.astype(x.dtype), x)


def softmax2(x, axis = -1):
  """Like jax.nn.softmax, but uses base-2 exponential rather than base-e.

  Since the hardware natively provides base-2 exponential, this is slightly more
  efficient. The following equivalence holds:

    jax.nn.softmax(x, axis) = softmax2(x * LOG2_E, axis)

  Args:
    x: Input array.
    axis: Which axis to reduces over.

  Returns:
    exp2(x) / sum(exp2(x))
  """
  x_max = jnp.max(x, axis=axis, keepdims=True)
  unnormalized = exp2(x - lax.stop_gradient(x_max))
  return unnormalized / jnp.sum(unnormalized, axis=axis, keepdims=True)


def logsumexp2(x, axis = -1):
  """Like jax.scipy.special.logsumexp, but uses base-2 rather than base-e.

  Since the hardware natively provides base-2 exponential, this is slightly more
  efficient. The following equivalence holds:

    jax.scipy.special.logsumexp(x, axis) = logsumexp2(x * LOG2_E, axis) * LN_2

  Args:
    x: Input array.
    axis: Which axis to reduce over.

  Returns:
    log2(sum(exp2(x))).
  """
  x_max = lax.stop_gradient(jnp.max(x, axis=axis))
  x_sum = jnp.sum(
      exp2(x - lax.expand_dims(x_max, dimensions=(axis,))), axis=axis)
  # TODO(reinerp): This jnp.log2 isn't being pattern-matched correctly by XLA.
  # Make small repro and file a bug.
  return jnp.log2(x_sum) + x_max


def swish2(x):
  """Faster alternative to jax.nn.swish.

  Satisfies:

    jax.nn.swish(x) = swish2(x * 0.5)

  Args:
    x: Input array.

  Returns:
    swish(x * 2.0)
  """
  return x * (lax.tanh(x) + 1)
