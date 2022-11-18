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

"""Utilities to reshape to and from the format for per_device xmap."""
import math
from typing import Dict, Tuple, Union

import jax
from jax.experimental.pjit import PartitionSpec as P

from scaling_transformer_inference_efficiency import partitioning


class Leaf(Tuple):
  """Stops pytree map recursion at this leaf."""


def sharding_factor(
    axis,  # pylint: disable=g-bare-generic
    mesh_axis_sizes):
  """Takes in how many axis a dimension is sharded along, returns the # shardings.
  """
  if axis is None:
    return 1
  if isinstance(axis, str):
    return mesh_axis_sizes[axis]
  else:
    return math.prod([mesh_axis_sizes[a] for a in axis])


def get_sharded_axis_shapes(
    axis,  # pylint: disable=g-bare-generic
    mesh_axis_sizes):
  """Takes in how many axis a dimension is sharded along, returns to append to a list.
  """
  if axis is None:
    return [None]
  if isinstance(axis, str):
    return [mesh_axis_sizes[axis]]
  else:
    return [mesh_axis_sizes[a] for a in axis]


def flatten(sharding):
  """Flatten, but include Nones unlike jax.tree_flatten."""
  flat = []
  for i in sharding:
    if isinstance(i, Tuple):
      flat += list(i) + [
          None
      ]  # add a None for the dimension which is visible on device
    elif isinstance(i, str):
      flat.append(i)
      flat.append(
          None)  # add a None for the dimension which is visible on device
    else:
      flat.append(None)
  return flat


def logical_to_layout(sharding):
  return flatten(
      partitioning.logical_to_physical(sharding)
  )  # TODO(sholto): consider refactoring to be clearer e.g. Layout Class?


def fold_out(mesh, param, sharding):
  """Converts a tensor to hard xmap form by folding out the dimensions it is sharded along.
  """
  # x (8, 32, 32) PartitionSpec('batch.Z', 'time', 'embed.XY')
  #                                       -> [Z, 8//Z, 32, X, Y, 32//XY]
  sharding_pattern = logical_to_layout(sharding)
  physical_sharding = partitioning.logical_to_physical(sharding)
  mesh_axis_sizes = dict(zip(mesh.axis_names,
                             mesh.devices.shape))  # {X: 2, Y: 2, Z: 2}

  new_shape = []
  # We want to construct two lists
  # One is the size of each axis [Z, 8//Z, 32, X, Y, 32//XY] (an example)
  # One is how they map to dimensions ['z', None, None, 'x', 'y', None]
  for dim, axis in zip(param.shape, physical_sharding):
    # get the mapping
    if axis is None:
      new_shape.append(dim)
    elif isinstance(axis, Tuple):
      new_shape += get_sharded_axis_shapes(axis, mesh_axis_sizes)
      new_shape += [dim // sharding_factor(axis, mesh_axis_sizes)]
    else:  # just a str
      new_shape += get_sharded_axis_shapes(axis, mesh_axis_sizes)
      new_shape += [dim // sharding_factor(axis, mesh_axis_sizes)]

  # TODO(sholto): Do we need to reorder ('x', 'y', 'z', ...)?
  #       Will this impact perf if used within pjit?
  return param.reshape(new_shape), Leaf(sharding_pattern)


def unzip_tree(params, folded_out):
  """Unzips a pytree to get separate shapes and layout trees matching param's structure.
  """
  outer_treedef = jax.tree_util.tree_structure(params)
  inner_treedef = jax.tree_util.tree_structure((P(), P()))
  # E.g. goes from Params( tensor, prefx) ->
  #       tensor_tree_like(Params.structure), layout_tree_like(Params.structure)
  return jax.tree_util.tree_transpose(outer_treedef, inner_treedef, folded_out)


def fold_in(param, sharding):
  """Takes a known logical sharding, reforms."""
  device_layout_sharding = logical_to_layout(sharding)
  new_shape = []
  accum = 1
  for axis, dim in zip(device_layout_sharding, param.shape):
    if axis is None:
      new_shape.append(dim * accum)
      accum = 1
    else:
      accum *= dim

  return param.reshape(new_shape)
