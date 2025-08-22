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

"""Utils for sharding and sharding annotation."""

import functools
from typing import Callable, Any, Sequence

from absl import logging
from flax.core import meta
import jax
from jax.experimental import mesh_utils
from jax.interpreters import pxla
import numpy as np

from imp.max.utils import typing


def global_mesh():
  """Returns global xmap/pjit mesh resource environment."""
  return pxla.thread_resources.env.physical_mesh


def global_mesh_defined():
  """Checks if global xmap/pjit mesh resource environment is defined."""
  return global_mesh().devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def tree_pspecs_to_named_shardings(pspecs_tree,
                                   mesh):
  def pspec_to_sharding(spec):
    return jax.sharding.NamedSharding(mesh, spec)
  return jax.tree.map(pspec_to_sharding, pspecs_tree)


def replication_named_shardings(mesh):
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def shard_array(array,
                shardings,
                mesh = None,
                enforce = False,
                match_ranks = True):
  """Shards an array according to the sharding axes.

  Args:
    array: A jax array to be sharded or passed as is.
    shardings: The optional sharding annotations in string format (string or a
      sequence of strings).
    mesh: The mesh to use for the partitioning. If None, the global mesh
      resource is used if available.
    enforce: Whether to enfore sharding if mesh is defined but sharding
      annotation is empty.
    match_ranks: Whether to match the rank of the inputs array with the
      sharding annotations.

  Returns:
    Potentially sharded jax array.
  """
  # In case no mesh is defined, just skip
  if mesh is None and not global_mesh_defined():
    logging.debug(
        'No mesh defined. Skipping sharding for array with shape/dtype: %s/%s',
        array.shape,
        array.dtype,
    )
    return array

  # In case sharding is empty and replication of an already-sharded array is
  # not intended, just skip
  if (shardings is None or not shardings) and not enforce:
    logging.debug(
        'No sharding defined for array with shape/dtype: %s/%s',
        array.shape,
        array.dtype,
    )
    return array

  # Prepare axis resource for sharding
  shardings = shardings or ()
  axis_resource = jax.sharding.PartitionSpec(*shardings)

  # Verify ranks
  if match_ranks and axis_resource not in (jax.sharding.PartitionSpec(),
                                           jax.sharding.PartitionSpec(None)):
    if array.ndim != len(shardings):
      raise ValueError(
          'The sharding axes and array rank must match. Instead, received '
          f'`{shardings=}` for an array with shape=`{array.shape}`.'
      )

  # If mesh is provided, define the NamedSharding setting and shard array
  if mesh is not None:
    named_sharding = jax.sharding.NamedSharding(mesh, axis_resource)
    sharded_array = jax.lax.with_sharding_constraint(array, named_sharding)

  # Otherwise, refer to the global mesh defined in the parent context
  else:
    sharded_array = jax.lax.with_sharding_constraint(array, axis_resource)

  logging.debug(
      'Array with shape/dtype %s/%s was sharded with %s',
      array.shape,
      array.dtype,
      shardings,
  )
  return sharded_array


def shard_arrays_tree(arrays_tree,
                      shardings_tree,
                      mesh = None,
                      enforce = False,
                      match_ranks = True):
  """Shards an array according to the sharding axes.

  Note that leaves in `shardings_tree` must be one of the following:
  {None, jax.sharding.PartitionSpec, Sequence[str | None | Sequence[str]]}
  If an empty tuple is passed, it won't be considered as a "replication". If you
  intend to replicate a leafe, pass None. Also, if any leaf in `arrays_tree` is
  not an instance of jax.Array, it will be passes as-is and won't be sharded.

  Args:
    arrays_tree: A tree of jax arrays to be sharded or passed as is.
    shardings_tree: The optional sharding annotations in with the same structure
      as `arrays_tree` and strings or a sequence of strings as leaves.
    mesh: The mesh to use for the partitioning. If None, the global mesh
      resource is used if available.
    enforce: Whether to enfore sharding if mesh is defined but sharding
      annotation is empty.
    match_ranks: Whether to match the rank of the inputs array with the
      sharding annotations.

  Returns:
    Potentially sharded jax array tree.
  """
  flattened_arrays, arrays_treedef = jax.tree.flatten(arrays_tree)
  def _is_sharding_leaf(xs):
    if isinstance(xs, Sequence):
      # Check if this sequence has valid members
      instances = []
      for x in xs:
        if isinstance(x, (str, type(None))):
          instances.append(True)
        elif isinstance(x, Sequence):
          # Check if a member of the sequence is a valid sharding super-axis.
          # An example could be (('expert', 'data'), 'model') where the first
          # entry is a valid sharding super-axis itself.
          subinstances = [isinstance(i, str) for i in x]
          if not subinstances:
            # Empty tuple/list is not a valid entry of a sharding annotation
            # For example, this is not acceptable: ((), 'model')
            instances.append(False)
          else:
            instances.append(all(subinstances))
        else:
          instances.append(False)
      if not instances:
        # Empty tuple/list is not a valid leaf
        return False
      else:
        return all(instances)
    else:
      return isinstance(xs, (jax.sharding.PartitionSpec, type(None)))
  flattened_shardings, _ = jax.tree.flatten(
      shardings_tree, is_leaf=_is_sharding_leaf)
  if (not flattened_shardings
      or (len(flattened_shardings) == 1 and flattened_shardings[0] is None)):
    flattened_shardings = [None] * len(flattened_arrays)
  elif len(flattened_shardings) != len(flattened_arrays):
    raise ValueError('The shardings tree and arrays must match.')
  sharded_flattened_arrays = []
  for array, shardings in zip(flattened_arrays, flattened_shardings):
    if isinstance(array, jax.Array):
      sharded_flattened_arrays.append(
          shard_array(array=array,
                      shardings=shardings,
                      mesh=mesh,
                      enforce=enforce,
                      match_ranks=match_ranks))
    else:
      sharded_flattened_arrays.append(array)
  return jax.tree.unflatten(arrays_treedef, sharded_flattened_arrays)


def modulate_param_init(
    param_init_fn,
    shardings,
    mesh = None,
):
  """Wraps a function's return value with Partitioned and shards value too.

  Example::

    >>> import flax.linen as nn
    >>> kernel_init = modulate_param_init(
    ...     nn.initializers.lecun_normal(), (None, "data"))
    >>> partitioned_dense = nn.Dense(features=3, kernel_init=kernel_init)

  Args:
    param_init_fn: The param initializer function to be wrapped.
    shardings: The sharding axes passed to ``Partitioned``.
    mesh: The mesh to use for the partitioning. If None, the global mesh
      resource is used if available.

  Returns:
    A function wrapping ``fn`` that will return an instance of ``Partitioned``.
  """

  def _fetch_names(shardings):
    if shardings is None or not shardings:
      return ()
    else:
      names = ()
      for axis in shardings:
        if not isinstance(axis, (str, type(None))):
          raise ValueError(
              f'Invalid param sharding axis: {axis}. Only `str` and `None` are '
              'accepted by `flax.core.meta.Partitioned`.')
        names += (axis,)
      return names

  @functools.wraps(param_init_fn)
  def wrapper(*args, **kwargs):
    value = param_init_fn(*args, **kwargs)
    value = shard_array(value, shardings, enforce=True, match_ranks=True)
    names = _fetch_names(shardings)
    return meta.Partitioned(value=value, names=names, mesh=mesh)

  return wrapper


def create_tpu_device_mesh(
    ici_mesh_shape,
    dcn_mesh_shape,
    contiguous_submeshes = False,
):
  """Creates a single- or multi-slice device mesh from mesh shapes.

  Args:
    ici_mesh_shape: The mesh shape for a single slice, or for each slice in a
      multi-slice setting.
    dcn_mesh_shape: The mesh shape to use for between-slice parallelism. If
      None, creates a single-slice mesh.
    contiguous_submeshes: If True, the mesh_utils.create_device_mesh() call will
      attempt to create a mesh where each process's local devices form a
      contiguous submesh. This is unused when `dcn_mesh_shape` is not None.

  Returns:
    An ndarray of JAX devices.
  """
  contiguous_submeshes = bool(contiguous_submeshes)
  if any(s > 1 for s in dcn_mesh_shape):
    devices = jax.devices()
    device_kind = devices[-1].device_kind
    if device_kind == 'cpu':
      target_shape = np.array(ici_mesh_shape) * np.array(dcn_mesh_shape)
      device_mesh = np.array(devices).reshape(target_shape)
    else:
      try:
        device_mesh = mesh_utils.create_hybrid_device_mesh(
            ici_mesh_shape, dcn_mesh_shape, devices=devices
        )
      except AssertionError as e:
        raise ValueError(
            'Setting a nontrivial dcn_mesh_shape requires multiple slices. '
            f'[{ici_mesh_shape=}, {dcn_mesh_shape=}, {devices=}]'
        ) from e
  else:
    device_mesh = mesh_utils.create_device_mesh(
        ici_mesh_shape, contiguous_submeshes=contiguous_submeshes
    )
  logging.info('device_mesh: %s', device_mesh)
  return device_mesh
