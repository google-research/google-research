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

"""Tree-extended utility functions."""

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from imp.max.utils import linalg
from imp.max.utils import typing


def tree_aggregate_array(
    tree,
    aggregate_fn = jnp.sum):
  """Aggregates all leaves of a given tree by 'aggregate_fn'."""
  tree_with_aggregated_leaves = jax.tree.map(aggregate_fn, tree)
  aggregated_leaves, _ = jax.tree.flatten(tree_with_aggregated_leaves)
  return aggregate_fn(jnp.array(aggregated_leaves))


def tree_convert_jax_array_to_numpy(array_tree):
  return jax.tree.map(np.asarray, array_tree)


def tree_convert_jax_float_to_float(array_tree):
  return jax.tree.map(float, array_tree)


def tree_low_rank_projector(
    array_tree,
    rank,
    method = 'svd',
):
  """Tree-maps projection methods in utils.linalg."""
  if method == 'svd':
    projection_fn = functools.partial(
        linalg.svd_projector,
        rank=rank,
    )
  else:
    raise NotImplementedError(method)

  def _is_jax_array_leaf(tree_node):
    return isinstance(tree_node, jax.Array)

  return jax.tree.map(projection_fn, array_tree, is_leaf=_is_jax_array_leaf)


def tree_project_array(
    array_tree,
    projection_state_tree,
    back_projection = False,
    precision = None,
):
  """Tree-maps `project_array` function in utils.linalg."""
  projection_fn = functools.partial(
      linalg.project_array,
      back_projection=back_projection,
      precision=precision,
  )
  return jax.tree.map(projection_fn, array_tree, projection_state_tree)
