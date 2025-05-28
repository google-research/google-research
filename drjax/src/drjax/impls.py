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

"""Implementations of MapReduce primitives in JAX."""

from collections.abc import Mapping
from typing import Any

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental.shard_alike import shard_alike
from jax.interpreters import pxla
from jax.sharding import PartitionSpec as P


# The type aliases below encode a federated programming model, inpsired by
# the programming model of TensorFlow Federated (TFF). Notable differences
# are that placements are static, and can only appear on arrays. There are no
# nested placements. Placement dimension comes first, so there is no need to
# worry about a total order on placements or figuring out what representation a
# nested-placed thing should have.

UnplacedArray = jnp.ndarray
PlacedArray = jnp.ndarray

# Special alias for 'arrays with an extra sequence dimension in the first
# element'.
SequenceArray = jnp.ndarray
PyTreeUnplaced = Any
PyTreePlaced = Any
PyTreeSequence = Any


def call_jaxpr(fn, arg):
  # Handles multi-element arguments.
  if isinstance(arg, tuple):
    return fn(*arg)
  else:
    return fn(arg)


def read_sequence_batch(
    array_sequence, index
):
  sequence_at_idx = jax.tree_util.tree_map(
      lambda x: x[index, Ellipsis], array_sequence
  )
  return sequence_at_idx


def _global_mesh_defined():
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = pxla.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def _mesh_axis_names():
  mesh = pxla.thread_resources.env.physical_mesh
  return mesh.axis_names


def _placement_axis_in_mesh(placement):
  """Checks if a placements axis is present in the global mesh."""
  if not _global_mesh_defined():
    return False
  placement_is_in_mesh = placement in _mesh_axis_names()
  if not placement_is_in_mesh:
    logging.log_first_n(
        logging.WARNING,
        'No mesh axis named "%s" found in the current mesh, which had names'
        ' %s. DrJax will not inject sharding constraints.',
        10,
        placement,
        _mesh_axis_names(),
    )
  return placement_is_in_mesh


def _constrain_if_mesh(x, pspec):
  if not _global_mesh_defined():
    return x
  return jax.lax.with_sharding_constraint(x, pspec)


class PlacedComputations:
  """Concrete implementations of federated primitives in JAX."""

  def __init__(
      self,
      placements_to_n_elements,
      use_spmd_axis_name = True,
  ):
    self._placements_to_n_elements = placements_to_n_elements
    self._use_spmd_axis_name = use_spmd_axis_name

  def broadcast_to_placement(
      self,
      arg,
      placement,
  ):
    """Broadcasts (replicates) to the specified placement.

    That is, given an `arg` of shape `[a, ... b]`, and a `placement` with `n`
    elements, the result of this function should be an array of shape
    `[n, a, ... b]`, each of whose slices on the zeroth axis is identical to
    `arg`.

    This function shards the resulting replicated array along a mesh axis
    corresponding to 'placement' if such a mesh is defined at compilation time.
    Otherwise no sharding is applied and the result is fully replicated.

    This function must also direct the GSPMD compiler to shard the
    zeroth-axis slices of this replicated array in a similar manner to the
    argument.

    Args:
      arg: An array to be broadcast.
      placement: String representing the placement to which to broadcast `arg`.

    Returns:
      A logically replicated array along the zeroth axis, as described above.
    """
    arg = jnp.array(arg)
    n_elements = self._placements_to_n_elements[placement]

    # Note that this pspec will only result in a sharding constraint defined if
    # a mesh is installed at tracing time.
    if _placement_axis_in_mesh(placement):
      pspec = P(placement, *([P.UNCONSTRAINED] * len(arg.shape)))
    else:
      # Without a placements axis in the mesh, we simply explicitly tell the
      # compiler that there are no constraints on this tensor. This will leave
      # the choices in the hands of the compiler.
      pspec = P(*([P.UNCONSTRAINED] * (len(arg.shape) + 1)))

    def single_arg_broadcast(x):
      replicated_tensor = jnp.tile(x, reps=[n_elements] + [1] * len(x.shape))
      if not _global_mesh_defined():
        # No sharding expected, don't worry about it.
        return replicated_tensor
      else:

        def _shard_slice_like_arg(s):
          s_sharded, _ = shard_alike(s, x)
          return s_sharded

        original_dims_constrained = jax.vmap(_shard_slice_like_arg, in_axes=0)(
            replicated_tensor
        )
        fully_constrained = _constrain_if_mesh(original_dims_constrained, pspec)
        return fully_constrained

    return jax.jit(single_arg_broadcast)(arg)

  def normalized_broadcast_to_placement(
      self,
      arg,
      placement,
  ):
    # This broadcasts arg / placement_size. This is intended only for reverse-
    # mode differentiation of mean-based aggregations.
    n_elements = self._placements_to_n_elements[placement]
    unnormalized_broadcast = self.broadcast_to_placement(arg, placement)
    return jnp.divide(unnormalized_broadcast, n_elements)

  def mean_from_placement(self, arg):
    placement_idx = 0
    return jnp.mean(arg, axis=[placement_idx])

  def weighted_mean_from_placement(
      self, arg, weight
  ):
    placement_idx = 0
    return jnp.average(arg, axis=[placement_idx], weights=weight)

  def sum_from_placement(self, arg):
    placement_idx = 0
    return jnp.sum(arg, axis=[placement_idx])

  def map_to_placement(self, fn, arg, placement):
    """Maps a function to the specified placement.

    Suppose the user has a mesh with three axes, [placement, 'y', z']. Suppose
    we are asked to map a function f, of signature ([a], [b]) -> [c]. Suppose we
    have an array e, of shape [d, a, b], layed out along the mesh's three axes.

    Now, our mapping implementation is required to be able to: map f across the
    axis of size d, producing an array of shape [d, c], whose d-sized axis is
    layed out along the placement axis of the mesh, with layout of c-sized axis
    inherited from the operation of f. e.g., if f is jit-compiled with no
    sharding specifications, the layout of c will be determined by JAX's
    sharding propagation.

    In the case that the user has a mesh which does _not_ have placement as an
    axis name, the resulting array of shape [d, c], where the axis of size d is
    replicated across devices, and the axis of size c may be similarly sharded
    as above (depending on annotations internal to f, and the way the function
    constructed here is jit-compiled).

    The implementation here is intended to satisfy the sketch above (and its
    generalizations, including to pytrees, etc).

    Args:
      fn: Function to be mapped.
      arg: PyTree of arrays for which to map leading axis. Each array in the
        structure is assumed to have a leading axis of the same size, the number
        of elements at `placement`.
      placement: String representing the placement of input and output arrays to
        this map.

    Returns:
      The result of mapping `fn` over the leading axis, satisfying the sharding
      requirements specified above.
    """

    def _constrain_at_placement_with_slices_like(x, y):
      pspec = P(placement, *([P.UNCONSTRAINED] * (len(x.shape) - 1)))
      placement_constrained = _constrain_if_mesh(x, pspec)

      def _shard_slice(s):
        s_sharded, _ = shard_alike(s, y[0])
        return s_sharded

      return jax.vmap(_shard_slice, in_axes=0)(placement_constrained)

    # `spmd_axis_name`` causes any internal with_sharding_constraints or
    # shard_map calls inside the `vmapped` function to respect the
    # sharding along this axis name. But it doesn't enrich annotations on
    # input / output tensors. Since we have a very limited mapping semantic
    # here, adding these annotations is always safe for us, as long as
    # `placement` is in the current mesh.
    if _placement_axis_in_mesh(placement):
      arg = jax.tree_util.tree_map(
          _constrain_at_placement_with_slices_like, arg, arg
      )
      if self._use_spmd_axis_name:
        mapped_fn = jax.vmap(
            # We must not have an `axis_name` argument here in order to work
            # with any potential `shard_map` inside of `fn`.
            fn,
            in_axes=0,
            out_axes=0,
            spmd_axis_name=placement,
        )
      else:
        mapped_fn = jax.vmap(
            fn,
            in_axes=0,
            out_axes=0,
        )

      return call_jaxpr(mapped_fn, arg)
    else:
      # Users should be free to use whatever mesh their model needs without
      # _necessarily_ registering a mesh-dimension for every placement with
      # which they are programming.
      mapped_fn = jax.vmap(
          fn,
          axis_name=placement,
          in_axes=0,
      )
      return call_jaxpr(mapped_fn, arg)
