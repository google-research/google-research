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

"""Sharding test."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from imp.max.utils import sharding


class ShardingTest(absltest.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._mesh = jax.sharding.Mesh(
        sharding.create_tpu_device_mesh(ici_mesh_shape=(1, 1),
                                        dcn_mesh_shape=(1, 1)),
        ['data', 'model'],
    )

  def test_create_tpu_device_mesh(self):
    self.assertEqual(self._mesh.device_ids.shape, (1, 1))
    self.assertEqual(self._mesh.axis_names, ('data', 'model'))
    self.assertEqual(self._mesh.size, 1)

  def test_global_mesh(self):
    with self._mesh:
      # We should be able to fetch the mesh inside the context
      self.assertTrue(sharding.global_mesh_defined())
      self.assertEqual(sharding.global_mesh(), self._mesh)

    # We shouldn't be able to fetch the mesh outside of context
    self.assertFalse(sharding.global_mesh_defined())

  def test_shard(self):
    inputs = jnp.ones((3, 3))
    with self._mesh:
      sharded_inputs = sharding.shard_array(inputs, ('data', 'model'))
      self.assertEqual(sharded_inputs.shape, (3, 3))
      self.assertEqual(sharded_inputs.addressable_shards[0].data.shape, (3, 3))
      self.assertTrue(sharded_inputs.is_fully_addressable)
      self.assertIsInstance(
          sharded_inputs.sharding, jax.sharding.SingleDeviceSharding
      )


if __name__ == '__main__':
  absltest.main()
