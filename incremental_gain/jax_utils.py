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

"""Basic utilities for jax."""

import jax
import jax.numpy as jnp


# https://github.com/deepmind/optax/blob/3408e90869a45453c7f40dc7a19902fce94675a5/optax/_src/transform.py
def safe_norm(x, min_norm, *args, **kwargs):
  """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.

    The gradients of jnp.maximum(jnp.linalg.norm(x), min_norm) at 0.0 is NaN,
    because jax will evaluate both branches of the jnp.maximum.
    The version in this function will return the correct gradient of 0.0 in this
    situation.
    Args:
    x: jax array.
    min_norm: lower bound for the returned norm.
  """
  norm = jnp.linalg.norm(x, *args, **kwargs)
  x = jnp.where(norm < min_norm, jnp.ones_like(x), x)
  return jnp.where(norm < min_norm, min_norm,
                   jnp.linalg.norm(x, *args, **kwargs))


def pytree_allclose(lhs, rhs, *args, **kwargs):
  bools, _ = jax.flatten_util.ravel_pytree(
      jax.tree_util.tree_multimap(
          lambda x, y: jnp.allclose(x, y, *args, **kwargs), lhs, rhs))
  return bools.all()


def pytree_zeros_like(pytree):
  return jax.tree_util.tree_map(jnp.zeros_like, pytree)
