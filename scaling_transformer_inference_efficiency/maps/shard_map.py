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

import functools
import math
from typing import Any, Callable, Dict, Tuple, Union

import jax
from jax.experimental.maps import Mesh
from jax.experimental.maps import xmap
from jax.experimental.pjit import PartitionSpec as P

from scaling_transformer_inference_efficiency import partitioning


# pylint: disable = g-bare-generic


class Layout(Tuple):
  """Stops pytree map recursion at this leaf. Simply a wrapped tuple."""


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


def layout_sharding(
    physical_sharding):  # pylint: disable = g-bare-generic
  """Flatten, but include Nones unlike jax.tree_flatten."""
  flat = []
  for i in physical_sharding:
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
  return tuple(flat)


def logical_to_layout(
    logical_sharding):  # pylint: disable = g-bare-generic
  return layout_sharding(partitioning.logical_to_physical(logical_sharding))


def fold_out(mesh, param,
             sharding):
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
  return param.reshape(new_shape), Layout(sharding_pattern)


def unzip_tree(params, folded_out):
  """Unzips a pytree to get separate shapes and layout trees matching param's structure.
  """
  outer_treedef = jax.tree_util.tree_structure(params)
  inner_treedef = jax.tree_util.tree_structure((P(), P()))
  # E.g. goes from Params( tensor, prefx) ->
  #       tensor_tree_like(Params.structure), layout_tree_like(Params.structure)
  return jax.tree_util.tree_transpose(outer_treedef, inner_treedef, folded_out)


def fold_out_tree(mesh, pytree, logical_axes):
  fold_out_for_mesh = functools.partial(fold_out, mesh)
  folded_out = jax.tree_map(fold_out_for_mesh, pytree, logical_axes)
  pytree_xmap, layout = unzip_tree(pytree, folded_out)
  return pytree_xmap, layout


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


###############################################################################
######## Eliminate non-phsyical versions above in refactor ####################
###############################################################################


def fold_out_from_physical(mesh, param,
                           physical_sharding):
  """Converts a tensor to hard xmap form by folding out the dimensions it is sharded along.
  """
  # x (8, 32, 32) PartitionSpec('batch.Z', 'time', 'embed.XY')
  #                                       -> [Z, 8//Z, 32, X, Y, 32//XY]
  sharding_pattern = layout_sharding(physical_sharding)
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

  return param.reshape(new_shape), Layout(sharding_pattern)


def fold_in_from_physical(param, physical_sharding):
  """Takes a known logical sharding, reforms."""
  device_layout_sharding = layout_sharding(physical_sharding)
  new_shape = []
  accum = 1
  for axis, dim in zip(device_layout_sharding, param.shape):
    if axis is None:
      new_shape.append(dim * accum)
      accum = 1
    else:
      accum *= dim

  return param.reshape(new_shape)


def shard_map(
    fn,  # pylint: disable=g-bare-generic
    mesh,
    in_specs,
    out_specs,
    donate_argnums=(),
):  # pylint: disable=g-bare-generic
  """Replicates shard map style functionality - the user never sees reshaped shapes.
  """

  fold_out_for_mesh = functools.partial(fold_out_from_physical, mesh)

  def wrap_fn_unwrap(*args):

    assert jax.tree_util.tree_structure(args) == jax.tree_util.tree_structure(
        in_specs)

    folded_out = jax.tree_map(fold_out_for_mesh, args, in_specs)
    pytree_xmap, in_layout = unzip_tree(args, folded_out)
    out_layout = jax.tree_map(layout_sharding, out_specs)
    assert jax.tree_util.tree_structure(pytree_xmap) == jax.tree_structure(
        in_layout)
    in_layout = jax.tree_map(tuple, in_layout)
    # when using shardmap, resources are always an identity function
    axis_resources = {v: v for v in mesh.axis_names}
    result = xmap(
        fn,
        in_axes=in_layout,
        out_axes=out_layout,
        axis_resources=axis_resources,
        axis_sizes=mesh.shape,
        donate_argnums=donate_argnums)(*pytree_xmap)

    assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(
        out_specs)
    folded_in = jax.tree_map(fold_in_from_physical, result, out_specs)
    return folded_in

  return wrap_fn_unwrap


