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

"""Tests for the sharding behavior of the implementation of DrJAX primitives."""

import functools
import math
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as PSpec
import numpy as np

from . import impls

# We program drjax to expect a mesh axis named _CLIENTS_AXIS. Consider defining
# and exporting a constant to help make mesh construction easier and guarantees
# cleaner. Other constants defined here are intended to help make the operations
# of the assertions in the tests below more transparent.
_CLIENTS_AXIS = 'clients'
_CLIENTS_AXIS_SIZE = 2
_NUM_CLIENTS = 100

_DATA_AXIS = 'data'
_DATA_AXIS_SIZE = 2
_DATA_SIZE = 10


# Inline a helper function for creating and manipulating test meshes from
# JAX's internals.
def create_global_mesh(mesh_shape, axis_names):
  size = math.prod(mesh_shape)
  if len(jax.devices()) < size:
    raise unittest.SkipTest(f'Test requires {size} global devices.')
  devices = sorted(jax.devices(), key=lambda d: d.id)
  mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
  global_mesh = jax.sharding.Mesh(mesh_devices, axis_names)
  return global_mesh


class BroadcastShardingBehaviorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._placements = {_CLIENTS_AXIS: _NUM_CLIENTS}
    self._comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
    )

  def test_broadcast_with_1x1_fully_replicates(self):
    arg = jnp.zeros(shape=[_DATA_SIZE])
    with create_global_mesh([1, 1], [_CLIENTS_AXIS, _DATA_AXIS]):
      result = self._comp_factory.broadcast_to_placement(arg, _CLIENTS_AXIS)
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    sharding = result.sharding
    # There is only one chip we talk to, so this sharding 'looks' fully
    # replicated.
    self.assertTrue(sharding.is_fully_replicated)

  def test_broadcast_clients_shards_along_clients(self):
    arg = jnp.zeros(shape=[_DATA_SIZE])
    with create_global_mesh(
        [_CLIENTS_AXIS_SIZE, _DATA_AXIS_SIZE], [_CLIENTS_AXIS, _DATA_AXIS]
    ):
      result = self._comp_factory.broadcast_to_placement(arg, _CLIENTS_AXIS)
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    sharding = result.sharding
    # If this sharding were fully replicated, we would be *replicating* the data
    # on each chip, rather than putting half of the clients' broadcasted arrays
    # on one set of client chips and half on the other.
    self.assertFalse(sharding.is_fully_replicated)
    # Each shard should host half the clients, but the arg's original dimension
    # should be replicated.
    self.assertEqual(
        sharding.shard_shape(result.shape),
        (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE),
    )

  def test_broadcast_preserves_sharding_with_no_clients_mesh(self):
    arg = jnp.zeros(shape=[_DATA_SIZE])
    # Replicating a situation in which the caller's mesh has no clients axis; in
    # this case, we should preserve the sharding of any broadcast tensors, but
    # not shard along the (nonexistent) clients axis.
    no_mesh_comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
    )
    with create_global_mesh([_DATA_AXIS_SIZE], [_DATA_AXIS]) as mesh:
      arg_spec = PSpec(_DATA_AXIS)
      sharded_arg = jax.device_put(
          arg, device=jax.sharding.NamedSharding(mesh, arg_spec)
      )
      result = no_mesh_comp_factory.broadcast_to_placement(
          sharded_arg, _CLIENTS_AXIS
      )
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    sharding = result.sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    # The data should be partitioned across chips.
    self.assertFalse(sharding.is_fully_replicated)
    # The resulting broadcast array should have the same sharding as its input
    # for the non-injected dimensions, and replication on the clients dimension.
    self.assertEqual(sharding.spec, PSpec(None, _DATA_AXIS))
    # Here, the clients axis should be replicated on each set of chips;
    # however,the data making up the broadcasted array should be split along
    # the 'data' dimension; thus only the second dimension of the tensor should
    # be split.
    self.assertEqual(
        sharding.shard_shape(result.shape),
        (_NUM_CLIENTS, _DATA_SIZE // _DATA_AXIS_SIZE),
    )

  def test_broadcast_preserves_arg_sharding_with_clients_mesh(self):
    arg = jnp.zeros(shape=[_DATA_SIZE])
    with create_global_mesh(
        [_CLIENTS_AXIS_SIZE, _DATA_AXIS_SIZE], [_CLIENTS_AXIS, _DATA_AXIS]
    ) as mesh:
      arg_spec = PSpec(_DATA_AXIS)
      sharded_arg = jax.device_put(
          arg, device=jax.sharding.NamedSharding(mesh, arg_spec)
      )
      result = self._comp_factory.broadcast_to_placement(
          sharded_arg, _CLIENTS_AXIS
      )
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    sharding = result.sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    # The data should be partitioned across chips.
    self.assertFalse(sharding.is_fully_replicated)
    # The resulting broadcast array should have the same sharding as its input
    # for the non-injected dimensions, and replication on the clients dimension.
    self.assertEqual(sharding.spec, PSpec(_CLIENTS_AXIS, _DATA_AXIS))
    # Similarly to the above, the clients dimension should be 'split' across the
    # two sets of chips making up the clients axis of the mesh. However, in this
    # case the data making up the broadcasted array should *also* be split along
    # the 'data' dimension; thus each dimension of the global array's shape
    # should be cut in half, with sub-arrays of shape (_NUM_CLIENTS //
    # _CLIENTS_AXIS_SIZE,
    # _DATA_SIZE // _DATA_AXIS_SIZE) living on each of
    # the 4 chips.
    self.assertEqual(
        sharding.shard_shape(result.shape),
        (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE // _DATA_AXIS_SIZE),
    )


@jax.jit
def add(x, y):
  # A computation that can be sharded out. Here we rely on the
  # 'computation-follows-data' behavior of jax.jit.
  return x + y


def _place_args_at_clients(*args, comp_factory):
  return tuple(
      comp_factory.broadcast_to_placement(x, _CLIENTS_AXIS) for x in args
  )


class MapFnShardingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._placements = {_CLIENTS_AXIS: _NUM_CLIENTS}
    self._sequence_length = 10
    self._comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
    )

  def test_map_respects_clients_sharding(self):
    with create_global_mesh(
        [_CLIENTS_AXIS_SIZE, _DATA_AXIS_SIZE], [_CLIENTS_AXIS, _DATA_AXIS]
    ):
      arg1_at_c, arg2_at_c = _place_args_at_clients(
          jnp.zeros(shape=[_DATA_SIZE]),
          jnp.ones(shape=[_DATA_SIZE]),
          comp_factory=self._comp_factory,
      )
      result = self._comp_factory.map_to_placement(
          add, (arg1_at_c, arg2_at_c), _CLIENTS_AXIS
      )
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    sharding = result.sharding
    # The data should be partitioned across chips.
    self.assertFalse(sharding.is_fully_replicated)
    # Each shard should host half the clients, but the arg's original dimension
    # should be replicated, since the arguments were replicated.
    self.assertEqual(
        sharding.shard_shape(result.shape),
        (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE),
    )

  def test_map_respects_non_clients_sharding(self):
    arg_spec = PSpec(_DATA_AXIS)
    with create_global_mesh(
        [_CLIENTS_AXIS_SIZE, _DATA_AXIS_SIZE], [_CLIENTS_AXIS, _DATA_AXIS]
    ) as mesh:
      sharded_arg1 = jax.device_put(
          jnp.zeros(shape=[_DATA_SIZE]),
          device=jax.sharding.NamedSharding(mesh, arg_spec),
      )
      sharded_arg2 = jax.device_put(
          jnp.ones(shape=[_DATA_SIZE]),
          device=jax.sharding.NamedSharding(mesh, arg_spec),
      )
      arg1_at_c, arg2_at_c = _place_args_at_clients(
          sharded_arg1,
          sharded_arg2,
          comp_factory=self._comp_factory,
      )
      result = self._comp_factory.map_to_placement(
          add, (arg1_at_c, arg2_at_c), _CLIENTS_AXIS
      )
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    # The data should be partitioned across chips.
    sharding = result.sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertFalse(sharding.is_fully_replicated)
    # The resulting array here should be fully sharded, just like the argument,
    # by computation-follows-data semantics.
    self.assertEqual(sharding.spec, PSpec(_CLIENTS_AXIS, _DATA_AXIS))
    # Since the argument was fully split across the data and clients axes, the
    # result should be too: each of the 4 chips hosts a sub-array slice of data,
    # of shape (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE //
    # _DATA_AXIS_SIZE), so that the entire (global) shape is (_NUM_CLIENTS,
    # _DATA_SIZE).
    self.assertEqual(
        sharding.shard_shape(result.shape),
        (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE // _DATA_AXIS_SIZE),
    )

  def test_map_forces_clients_sharding_with_model_parallelism(self):
    arg_spec = PSpec(_DATA_AXIS)
    with create_global_mesh(
        [_CLIENTS_AXIS_SIZE, _DATA_AXIS_SIZE], [_CLIENTS_AXIS, _DATA_AXIS]
    ) as mesh:
      sharded_arg1 = jax.device_put(
          jnp.zeros(shape=[_DATA_SIZE]),
          device=jax.sharding.NamedSharding(mesh, arg_spec),
      )
      sharded_arg2 = jax.device_put(
          jnp.ones(shape=[_DATA_SIZE]),
          device=jax.sharding.NamedSharding(mesh, arg_spec),
      )
      sharded_arg1 = jnp.tile(sharded_arg1, reps=[_NUM_CLIENTS, 1])
      sharded_arg2 = jnp.tile(sharded_arg2, reps=[_NUM_CLIENTS, 1])
      result = self._comp_factory.map_to_placement(
          add, (sharded_arg1, sharded_arg2), _CLIENTS_AXIS
      )

    # Our arguments should _not_ be sharded across the clients axis.
    self.assertEqual(
        sharded_arg1.sharding.shard_shape(sharded_arg1.shape),
        (_NUM_CLIENTS, _DATA_SIZE // _DATA_AXIS_SIZE),
    )
    self.assertEqual(
        sharded_arg2.sharding.shard_shape(sharded_arg2.shape),
        (_NUM_CLIENTS, _DATA_SIZE // _DATA_AXIS_SIZE),
    )
    # But the result should be fully partitioned across chips.
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    sharding = result.sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertFalse(sharding.is_fully_replicated)
    # The resulting array here should be fully sharded, _even though the
    # argument was not_, because our vmap impl inserts sharding constraints on
    # the placement dimension on its arguments.
    self.assertEqual(sharding.spec, PSpec(_CLIENTS_AXIS, _DATA_AXIS))
    # Since the argument was fully split across the data and clients axes, the
    # result should be too: each of the 4 chips hosts a sub-array slice of data,
    # of shape (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE //
    # _DATA_AXIS_SIZE), so that the entire (global) shape is (_NUM_CLIENTS,
    # _DATA_SIZE).
    self.assertEqual(
        sharding.shard_shape(result.shape),
        (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE // _DATA_AXIS_SIZE),
    )

  def test_map_of_shard_map_fully_shards_result(self):
    arg_spec = PSpec(_DATA_AXIS)

    with create_global_mesh(
        [_CLIENTS_AXIS_SIZE, _DATA_AXIS_SIZE], [_CLIENTS_AXIS, _DATA_AXIS]
    ) as mesh:

      @functools.partial(
          shard_map,
          mesh=mesh,
          in_specs=(arg_spec, arg_spec),
          out_specs=arg_spec,
      )
      def shard_map_add(x, y):
        return x + y

      sharded_arg1 = jax.device_put(
          jnp.zeros(shape=[_DATA_SIZE]),
          device=jax.sharding.NamedSharding(mesh, arg_spec),
      )
      sharded_arg2 = jax.device_put(
          jnp.ones(shape=[_DATA_SIZE]),
          device=jax.sharding.NamedSharding(mesh, arg_spec),
      )
      sharded_arg1 = jnp.tile(sharded_arg1, reps=[_NUM_CLIENTS, 1])
      sharded_arg2 = jnp.tile(sharded_arg2, reps=[_NUM_CLIENTS, 1])
      result = self._comp_factory.map_to_placement(
          shard_map_add, (sharded_arg1, sharded_arg2), _CLIENTS_AXIS
      )

    # The result should be fully partitioned across chips, regardless of input
    # sharding.
    self.assertEqual(result.shape, (_NUM_CLIENTS, _DATA_SIZE))
    sharding = result.sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertFalse(sharding.is_fully_replicated)
    # The resulting array here should be fully sharded, since the fed_map
    # implementation respects the _CLIENTS_AXIS sharding and the shard_map
    # respects the 'data' sharding.
    self.assertEqual(sharding.spec, PSpec(_CLIENTS_AXIS, _DATA_AXIS))
    # Since the argument was fully split across the data and clients axes, the
    # result should be too: each of the 4 chips hosts a sub-array slice of data,
    # of shape (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE //
    # _DATA_AXIS_SIZE), so that the entire (global) shape is (_NUM_CLIENTS,
    # _DATA_SIZE).
    self.assertEqual(
        sharding.shard_shape(result.shape),
        (_NUM_CLIENTS // _CLIENTS_AXIS_SIZE, _DATA_SIZE // _DATA_AXIS_SIZE),
    )


if __name__ == '__main__':
  absltest.main()
