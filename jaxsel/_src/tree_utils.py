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

"""Utility functions for handling pyTrees."""

from typing import Sequence, Any

from absl import app
import jax
from jax import numpy as jnp

PyTree = Any  # typing for humans


def global_norm(tree):
  """Takes a PyTree and returns its global L2 norm."""
  leaf_norms2 = jax.tree_map(lambda x: jnp.linalg.norm(x)**2, tree)
  return jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + y, leaf_norms2, 0))


def tree_stack(trees):
  """Takes a list of trees and stacks every corresponding leaf.

  For example, given two trees ((a, b), c) and ((a', b'), c'), returns
  ((stack(a, a'), stack(b, b')), stack(c, c')).
  Useful for turning a list of objects into something you can feed to a
  vmapped function.

  Args:
    trees: trees iterable to stack

  Returns:
    tree: pyTree with extra dimension,
      representing the batch of pytrees
  """
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = jax.tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)

  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.stack(l) for l in grouped_leaves]
  tree = treedef_list[0].unflatten(result_leaves)
  return tree


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
