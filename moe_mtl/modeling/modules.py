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

"""Implemented customized modules."""

import flax.linen as nn
from flax.linen.dtypes import promote_dtype
import jax
import jax.numpy as jnp


def _threshold(x, threshold = 0.1):
  return (x > threshold).astype("float")


def straight_through_threshold(x, threshold = 0.1):
  # Create an exactly-zero expression with Sterbenz lemma that has
  # an exactly-one gradient.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(_threshold(x, threshold))


class TAPSDense(nn.Dense):
  """TAPS."""

  @nn.compact
  def __call__(self, inputs, first=True):
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
      first: Whether the use the first mask (currently only two tasks), so we
             have only two masks.
    Returns:
      The transformed input.

    """
    kernel = self.param("kernel",
                        self.kernel_init,
                        (jnp.shape(inputs)[-1], self.features),
                        self.param_dtype)
    if self.use_bias:
      bias = self.param("bias", self.bias_init, (self.features,),
                        self.param_dtype)
    else:
      bias = None

    mask_1 = self.param("mask_1",
                        nn.initializers.constant(1, self.param_dtype),
                        (1,),
                        self.param_dtype)
    mask_2 = self.param("mask_2",
                        nn.initializers.constant(1, self.param_dtype),
                        (1,),
                        self.param_dtype)
    residual_1 = self.param("residual_1",
                            nn.initializers.constant(0.0, self.param_dtype),
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)
    residual_2 = self.param("residual_2",
                            nn.initializers.constant(0.0, self.param_dtype),
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    if first:
      weight = jax.lax.stop_gradient(
          kernel) + residual_1 * straight_through_threshold(mask_1, 0.05)
    else:
      weight = jax.lax.stop_gradient(
          kernel) + residual_2 * straight_through_threshold(mask_2, 0.05)
    y = jax.lax.dot_general(inputs, weight,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    if first:
      return y, mask_1
    else:
      return y, mask_2
