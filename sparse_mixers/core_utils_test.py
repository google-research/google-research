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

"""Tests for core_utils."""

import os
# Emulate 2 devices on CPU. Import before JAX.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax
from jax import numpy as jnp
import numpy as np

from sparse_mixers import core_utils


class ScatterNdTest(absltest.TestCase):

  def test_scatter_nd_simple(self):
    indices = jnp.array([[0, 1]])
    updates = jnp.array([[1, -2, 3]])

    actual_result = core_utils.scatter_nd(indices, updates, shape=(1, 2, 3))
    expected_result = jnp.array([[[0, 0, 0], [1, -2, 3]]])
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_ignore_outside_indices(self):
    indices = jnp.array([[0, 0], [1, 2], [2, 0]])
    updates = jnp.array([1., 2., 3.])

    actual_result = core_utils.scatter_nd(indices, updates, shape=(3, 2))
    expected_result = jnp.array([[1., 0.], [0., 0], [3., 0.]])
    np.testing.assert_allclose(actual_result, expected_result)

  def test_scatter_nd_cumulative_updates(self):
    indices = jnp.array([[1, 1], [1, 1], [1, 1]])
    updates = jnp.array([1., 2., 3.])

    actual_result = core_utils.scatter_nd(indices, updates, shape=(3, 2))
    expected_result = jnp.array([[0., 0.], [0., 6.], [0., 0.]])
    np.testing.assert_allclose(actual_result, expected_result)


class MatchFnTest(absltest.TestCase):

  def test_regex_prefix(self):
    match_fn = core_utils.match_fn(r".*test.*")
    self.assertTrue(match_fn("/test/something"))
    self.assertTrue(match_fn("to/test/or/not/"))
    self.assertFalse(match_fn("no/match"))

  def test_empty_prefix(self):
    match_fn = core_utils.match_fn(None)
    self.assertFalse(match_fn("/test/something"))
    self.assertFalse(match_fn("to/test/or/not/"))


class TreeTest(absltest.TestCase):

  def test_tree_flatten_with_names(self):
    tree = {"ff_0": {"kernel": 0, "bias": 1}, "ff_1": {"kernel": 2, "bias": 3}}
    names_and_values, _ = core_utils.tree_flatten_with_names(tree)

    expected_names_and_values = [("ff_0/bias", 1), ("ff_0/kernel", 0),
                                 ("ff_1/bias", 3), ("ff_1/kernel", 2)]
    self.assertEqual(names_and_values, expected_names_and_values)

    # Check that values match regular JAX tree_flatten.
    self.assertEqual([x for _, x in names_and_values],
                     jax.tree.flatten(tree)[0])

  def test_tree_map_with_names(self):
    tree = {"a": 1, "b": 2}
    mapped_tree = core_utils.tree_map_with_names(
        f=lambda x: -x, param_tree=tree, filter_fn=lambda name: name == "b")

    self.assertEqual(mapped_tree, {"a": 1, "b": -2})

  def test_tree_replicate_by_name(self):
    n = jax.local_device_count()
    tree = dict(a=jnp.ones((4,)), b=25)
    replicated_tree = core_utils.tree_replicate_by_name(
        tree, filter_fn=lambda x: x == "a")

    self.assertIsInstance(replicated_tree["a"], jax.Array)
    self.assertIsInstance(replicated_tree["b"], int)
    self.assertEqual(replicated_tree["a"].shape, (n, 4))
    self.assertEqual(replicated_tree["a"].sharding,
                     jax.sharding.PmapSharding.default((n, 4), 0))

  def test_tree_shard_by_name(self):
    n = jax.local_device_count()
    tree = dict(a=jnp.ones((n, 4)), b=25)
    sharded_tree = core_utils.tree_shard_by_name(tree, lambda n: n == "a")

    self.assertIsInstance(sharded_tree["a"], jax.Array)
    self.assertIsInstance(sharded_tree["b"], int)

    self.assertEqual(sharded_tree["a"].shape, (n, 4))
    self.assertEqual(sharded_tree["a"].sharding,
                     jax.sharding.PmapSharding.default((n, 4), 0))

  def test_tree_unreplicate_by_name(self):
    tree = dict(a=jnp.ones((8, 4)), b=25)
    unreplicated_tree = core_utils.tree_unreplicate_by_name(
        tree, lambda n: n == "a")

    self.assertIsInstance(unreplicated_tree["a"], jnp.ndarray)
    self.assertIsInstance(unreplicated_tree["b"], int)
    self.assertEqual(unreplicated_tree["a"].shape, (4,))


if __name__ == "__main__":
  absltest.main()
