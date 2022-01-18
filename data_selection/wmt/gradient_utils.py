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

"""Utils for gradient alignment code."""

import jax
import jax.numpy as jnp


def tree_dot(tree_x, tree_y):
  a = jax.tree_util.tree_multimap(
      lambda x, y: jax.lax.dot(x.ravel(), y.ravel()), tree_x, (tree_y))
  return jnp.sum(jnp.array(jax.tree_util.tree_flatten(a)[0]))


def tree_mult(tree_x, val_y):
  return jax.tree_util.tree_map(
      lambda x: x.ravel() * val_y, tree_x)


def tree_div(tree_x, val_y):
  return jax.tree_util.tree_map(
      lambda x: x.ravel() / val_y, tree_x)


def tree_diff(tree_x, tree_y):
  return jax.tree_util.tree_multimap(
      lambda x, y: x.ravel() - y.ravel(), tree_x, (tree_y))


def tree_norm(tree_x):
  a = jax.tree_util.tree_map(
      lambda x: jnp.sum(jnp.square(x)), tree_x)
  b = jnp.sum(jnp.array(jax.tree_util.tree_flatten(a)[0]))
  return jnp.sqrt(b)
