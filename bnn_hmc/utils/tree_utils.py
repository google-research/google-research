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
"""Utility functions for jax pytree manipulations."""

import jax
from jax import numpy as jnp


def tree_get_types(tree):
  return [p.dtype for p in jax.tree.flatten(tree)[0]]


def tree_add(a, b):
  return jax.tree.map(lambda e1, e2: e1+e2, a, b)


def tree_diff(a, b):
  return jax.tree.map(lambda p_a, p_b: p_a - p_b, a, b)


def tree_dot(a, b):
  return sum([jnp.sum(e1 * e2) for e1, e2 in
              zip(jax.tree.leaves(a), jax.tree.leaves(b))])


def tree_dist(a, b):
  dist_sq = sum([jnp.sum((e1 - e2)**2) for e1, e2 in
                 zip(jax.tree.leaves(a), jax.tree.leaves(b))])
  return jnp.sqrt(dist_sq)


def tree_scalarmul(a, s):
  return jax.tree.map(lambda e: e*s, a)


def get_first_elem_in_sharded_tree(tree):
  return jax.tree.map(lambda p: p[0], tree)


def tree_norm(a):
  return float(jnp.sqrt(sum([jnp.sum(p_a**2) for p_a in jax.tree.leaves(a)])))


def normal_like_tree(a, key):
  treedef = jax.tree.structure(a)
  num_vars = len(jax.tree.leaves(a))
  all_keys = jax.random.split(key, num=(num_vars + 1))
  noise = jax.tree.map(lambda p, k: jax.random.normal(k, shape=p.shape), a,
                            jax.tree.unflatten(treedef, all_keys[1:]))
  return noise, all_keys[0]
