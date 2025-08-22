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

"""Extensions to Jax/Flax core functions for mixture of experts training."""

import dataclasses
import re

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np

# Type Stubs
ParamTree = Any
PyTreeDef = jax.tree_util.PyTreeDef


def scatter_nd(indices, updates,
               shape):
  """JAX implementation of tf.scatter_nd.

  See https://www.tensorflow.org/api_docs/python/tf/scatter_nd, and
  https://github.com/jax-ml/jax/discussions/3658.

  Notes:
  - If multiple indices point to the same position, the output value at this
    position is accumulated.
  - Indices falling outside of the created array are quietly ignored.

  Args:
    indices: [NUM_ITEMS, N_DIMS] array of indices to update.
    updates: [NUM_ITEMS, ...] array of new data points.
    shape: Dimensions of the output array.

  Returns:
    An array of shape `shape` with updated values at given indices.
  """
  zeros = jnp.zeros(shape, updates.dtype)
  key = tuple(jnp.moveaxis(indices, -1, 0))
  return zeros.at[key].add(updates)


def match_fn(prefix):
  """Creates a function returning true iff a string matches the prefix.

  Args:
    prefix: Regex prefix to match. If none, then return match function will not
      match any strings.

  Returns:
    Prefix match function.
  """
  if not prefix:
    return lambda name: False
  params_regex = re.compile(f"^{prefix}")
  return lambda name: params_regex.match(name) is not None


def tree_flatten_with_names(
    tree):
  """Like jax.tree.flatten but also fetches leaf names.

  Specialized to parameter trees of the form {"key0": {"subkey0": Any}, ...}.

  Args:
    tree: Tree of parameters to flatten.

  Returns:
    - A list of leaf name and value pairs: [(name, value), ...].
    - A tree definition object representing the structure of the flattened tree.
  """
  # PyTrees don't treat None values as leaves, so we explicitly declare them as
  # such.
  vals, tree_def = jax.tree.flatten(tree, is_leaf=lambda x: x is None)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traversal should visit the same number of leaves.
  if len(val_names) != len(vals):
    raise ValueError(f"Pytree traversal detected {len(val_names)} names, "
                     f"but {len(vals)} leafs.\nTreeDef is:\n{tree_def}")

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(
    f,
    param_tree,
    filter_fn = lambda name: True):
  """Like jax.tree.map but with filter on leaf path names.

  Specialized to parameter trees of the form {"key0": {"subkey0": Any}, ...}.

  Args:
    f: Function to be applied to each parameter in `param_tree`.
    param_tree: Tree of parameters that `f` should be applied to.
    filter_fn: Filter function called on each tree leave's path name (of the
      form "a/b/c"), to determine whether `f` should be applied to a given leaf
      or the leaf should be kept as-is.

  Returns:
    A tree identical in structure to `param_tree` but with the map function `f`
    applied to those leaves satisfying the `filter_fn` filter.
  """
  names_and_values, tree_def = tree_flatten_with_names(param_tree)
  vals = [f(v) if filter_fn(name) else v for name, v in names_and_values]
  return tree_def.unflatten(vals)


def tree_replicate_by_name(
    param_tree,
    filter_fn,
    devices = None):
  """Replicates leaf arrays whose name is matched by a filter.

  Args:
    param_tree: Tree of parameters.
    filter_fn: Leaf node filter function.
    devices: XLA devices.

  Returns:
    A tree identical in structure to `param_tree` except that those leaves which
    satisfy `filter_fn` are replicated across devices.
  """
  devices = devices or jax.local_devices()
  return tree_map_with_names(lambda x: jax.device_put_replicated(x, devices),
                             param_tree, filter_fn)


def tree_shard_by_name(
    param_tree,
    filter_fn,
    devices = None):
  """Shards arrays whose name is matched by a filter.

  Args:
    param_tree: Tree of parameters.
    filter_fn: Leaf node filter function.
    devices: XLA devices.

  Returns:
    A tree identical in structure to `param_tree` except that those leaves which
    satisfy `filter_fn` are sharded across devices.
  """
  devices = devices or jax.local_devices()
  return tree_map_with_names(lambda x: _array_shard(x, devices), param_tree,
                             filter_fn)


def tree_unreplicate_by_name(param_tree,
                             filter_fn):
  """Unreplicates arrays whose name is matched by a filter.

  Args:
    param_tree: Tree of parameters.
    filter_fn: Leaf node filter function.

  Returns:
    A tree identical in structure to `param_tree` except that those leaves which
    satisfy `filter_fn` are unreplicated.
  """
  return tree_map_with_names(lambda x: x[0], param_tree, filter_fn)


def _traverse_with_names(
    param_tree):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(param_tree):
    param_tree = flax.serialization.to_state_dict(param_tree)
  if isinstance(param_tree, (dict, flax.core.FrozenDict)):
    keys = sorted(param_tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(param_tree[key]):
        yield (key + "/" + path).rstrip("/"), v
  else:
    yield "", param_tree


def _array_shard(
    x,
    devices = None
):
  """Shards a single array over the first axis across multiple local devices."""
  devices = devices or jax.local_devices()
  x = jnp.asarray(x)
  assert x.shape[0] == len(devices)
  return jax.device_put_sharded(list(x), devices)
