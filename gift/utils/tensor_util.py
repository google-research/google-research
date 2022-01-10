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

"""Utility functions to perform various ops on tensors."""

import jax
import jax.numpy as jnp


def convex_interpolate(x, y, lmbdas):
  """Interpolate with convex combination.

  Args:
    x: float jnp array; `[bs, ...]`
    y: float jnp array; `[bs, ...]`
    lmbdas: float array; `[num_of_interpolations, bs]`

  Returns:
    z (interolated states) with shape `[bs x num_of_interpolations, ...]`
  """
  # TODO(samiraabnar): Make sure this method is not redefined else where and is
  # just reused.
  assert x.shape == y.shape, f'x.shape != y.shape, {x.shape} != {y.shape}'

  if (not jnp.isscalar(lmbdas)) and len(x.shape) > (len(lmbdas.shape) - 1):
    # If lambdas.shape
    lmbdas = jax.lax.broadcast_in_dim(
        lmbdas, shape=lmbdas.shape + x.shape[1:], broadcast_dimensions=(0, 1))

  z = x[None, Ellipsis] * (1 - lmbdas) + y[None, Ellipsis] * (lmbdas)
  z = z.reshape((-1,) + z.shape[2:])
  return z


def constant_initializer(unused_key, shape, fill_value):
  return jnp.full(shape, fill_value)
