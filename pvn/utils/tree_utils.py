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

"""Tree Utils.

"""

import dataclasses
import re
from typing import Mapping, Tuple, Union, Any

from absl import logging
import chex
import flax.serialization
import jax
import numpy as np
import optax


def _traverse_with_names(tree):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree.map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key]):
        yield (key + "/" + path).rstrip("/"), v
  elif isinstance(tree, (list, tuple)):
    for idx, subtree in enumerate(tree):
      for path, v in _traverse_with_names(subtree):
        yield (str(idx) + "/" + path).rstrip("/"), v
  else:
    yield "", tree


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names is provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree_util.tree_flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(f, tree, *rest):
  """Like jax.tree.map but with a filter on the leaf path name.

  Args:
    f: A function with first parameter `name` (path-like "a/b/c") and remaining
      parameters values of `tree` and `*rest` corresponding to the given `name`
      Should return a new value for parameter `name`.
    tree: The tree of parameters `f` should be applied to.
    *rest: more trees of the exact same structure.

  Returns:
    A tree identical in structure to `tree` and `*rest` but with the leaves the
    result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
  """
  names_and_vals, tree_def = tree_flatten_with_names(tree)
  names, vals = zip(*names_and_vals)
  rest_vals = [list(zip(*tree_flatten_with_names(t)[0]))[1] for t in rest]
  vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
  return tree_def.unflatten(vals)


def tree_map_with_regex(f, tree, regex_rules, not_f=lambda x: x, name=None):
  """Apply jax-style tree_map based on regex rules.

  Args:
    f: a function that is being applied to every variable.
    tree: jax tree of arrays.
    regex_rules: a list of tuples `(pattern, args)`, where `pattern` is a regex
      which used for variable matching and `args` are positional arguments
      passed to `f`. If some variable is not matched, we apply `not_f` transform
      which is id by default. If multiple patterns match, then only the first
      rule is applied.
    not_f: optional function which is applied to variables that do not match any
      pattern.
    name: a name of transform for logging purposes.

  Returns:
    a tree, transformed by `f` according to the given rules.
  """

  def _f(vname, v):
    for pattern, *args in regex_rules:
      if re.fullmatch(pattern, vname):
        if name and jax.process_index() == 0:
          logging.info(
              "Applying %s to %s with %s due to `%s`",
              name,
              vname,
              args,
              pattern,
          )
        return f(v, *args)
    return not_f(v)

  return tree_map_with_names(_f, tree)


def filter_empty_nodes(
    mask_tree, *other_trees
):
  """Process TrainState to filter out empty optax.EmptyState.

  Every tree passed is expected to have the same structure as mask_tree. All
  keys in mask_tree which have optax.EmptyState as a value will be filtered
  out of all *other_trees and returned. The resulting trees will be missing
  these leaves, but will still have the same structure relative to one another.

  This processing is necessary because orbax.checkpoint doesn't support these
  "null" valued leaves.

  Args:
    mask_tree: A TrainState containing some values as EmptyState.
    *other_trees: A list of TrainStates matching mask_tree.

  Returns:
    A single dict (or tuple of dict) matching provided
    *other_trees, but serialized as state_dicts.
  """

  def _filter(mask_tree, *other_trees):
    """Filters optax.EmptyState out of PyTree."""
    if not isinstance(mask_tree, dict):
      return other_trees
    result_trees = [{} for _ in other_trees]
    for k, v in mask_tree.items():
      if v is not None and not isinstance(
          v,
          (
              optax.EmptyState,
              optax.MaskedNode,
          ),
      ):
        values = _filter_helper(v, *(t[k] for t in other_trees))
        for i, v1 in enumerate(values):
          if isinstance(v1, dict):
            if v1:
              result_trees[i][k] = v1
          else:
            result_trees[i][k] = v1
    return tuple(result_trees)

  def _filter_helper(mask_tree, *other_trees):
    return jax.tree_util.tree_map(
        _filter, mask_tree, *other_trees, is_leaf=lambda x: isinstance(x, dict)
    )

  mask_tree = flax.serialization.to_state_dict(mask_tree)
  other_trees = (flax.serialization.to_state_dict(t) for t in other_trees)
  r = _filter_helper(mask_tree, *other_trees)
  return r[0] if len(r) == 1 else r
