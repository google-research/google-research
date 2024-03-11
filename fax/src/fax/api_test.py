# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

import functools

import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from . import api


@functools.wraps(api.fax_program)
def fax_program(*, placements):
  return api.fax_program(placements=placements, self_module=api)


class ApiTest(tf.test.TestCase):

  def test_sharded_broadcast(self):
    @fax_program(placements={"clients": 100})
    def broadcast_val(val):
      return api.federated_broadcast(val)

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("some_axis",))
    arg_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("some_axis")
    )
    with mesh:
      result = broadcast_val(
          jax.device_put(jnp.ones(shape=[8, 8]), arg_sharding)
      )

    self.assertAllClose(result, jnp.ones(shape=[100, 8, 8]))

    self.skipTest("Following assertion fails with an empty PartitionSpec.")
    # No clients dimension in the mesh, we don't lay out the clients along that
    # nonexistent dimension, but rather replicate them. Notice that we don't
    # need to specify the sharding to FAX; it should be inferred by GSPMD.
    expected_result_pspec = jax.sharding.PartitionSpec(None, "some_axis")
    self.assertEqual(
        result.sharding, jax.sharding.NamedSharding(mesh, expected_result_pspec)
    )

  def test_temp_sens_example(self):
    def one_if_over(threshold, value):
      return jax.lax.cond(value > threshold, lambda: 1.0, lambda: 0.0)

    @fax_program(placements={"clients": 100})
    def temp_sens_example(threshold, values):
      threshold_at_clients = api.federated_broadcast(threshold)
      values_over = api.federated_map_clients(
          one_if_over, (threshold_at_clients, values)
      )
      return api.federated_mean(values_over)

    key = jax.random.PRNGKey(2)
    random_measurements = jax.random.uniform(key, shape=[100])

    self.assertEqual(
        temp_sens_example(jnp.array(0.5), random_measurements), 0.53
    )

  def test_temp_sens_example_multiple_clients(self):
    def one_if_over(threshold, value):
      return jax.lax.cond(value > threshold, lambda: 1.0, lambda: 0.0)

    @fax_program(placements={"clients": 100})
    def temp_sens_example_100_clients(threshold, values):
      threshold_at_clients = api.federated_broadcast(threshold)
      values_over = api.federated_map_clients(
          one_if_over, (threshold_at_clients, values)
      )

      return api.federated_mean(values_over)

    @fax_program(placements={"clients": 10})
    def temp_sens_example_10_clients(threshold, values):
      threshold_at_clients = api.federated_broadcast(threshold)
      values_over = api.federated_map_clients(
          one_if_over, (threshold_at_clients, values)
      )
      return api.federated_mean(values_over)

    key = jax.random.PRNGKey(2)
    random_measurements_100 = jax.random.uniform(key, shape=[100])
    random_measurements_10 = jax.random.uniform(key, shape=[10])

    self.assertEqual(
        temp_sens_example_100_clients(jnp.array(0.5), random_measurements_100),
        0.53,
    )
    self.assertEqual(
        temp_sens_example_10_clients(jnp.array(0.5), random_measurements_10),
        0.4,
    )
    # We should be able to recover the original result flipping back to the
    # original function.
    self.assertEqual(
        temp_sens_example_100_clients(jnp.array(0.5), random_measurements_100),
        0.53,
    )

  def test_raises_outside_program_context(self):
    with self.assertRaises(api.OperatorUndefinedError):
      api.federated_broadcast(jnp.array(0.5))

    num_clients = 10

    @fax_program(placements={"clients": num_clients})
    def test(values):
      return api.federated_mean(values)

    # Should not raise, inside a fax context.
    test(jax.random.uniform(jax.random.PRNGKey(42), shape=[num_clients]))

    # Should raise again now that we've left the context.
    with self.assertRaises(api.OperatorUndefinedError):
      api.federated_broadcast(jnp.array(0.5))


if __name__ == "__main__":
  tf.test.main()
