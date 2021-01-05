# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Tests for gfsa.training.train_maze_lib."""

import functools
from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
from gfsa.datasets import graph_bundle
from gfsa.datasets.mazes import maze_schema
from gfsa.datasets.mazes import maze_task
from gfsa.training import train_maze_lib


class TrainMazeLibTest(absltest.TestCase):

  def test_iterative_fixed_point(self):

    def foo(x):

      def halfway_to_2x(x, y):
        return x + 0.5 * y

      return train_maze_lib.iterative_fixed_point(
          halfway_to_2x, x, jnp.zeros_like(x), iterations=100)

    a = np.array([1., 2., 3.])
    fixed_point = foo(a)
    np.testing.assert_allclose(fixed_point, 2 * a)

    foo_jacobian = jax.jacobian(foo)(a)
    np.testing.assert_allclose(foo_jacobian, 2 * np.eye(3))

  def test_soft_maze_values(self):
    maze = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]).astype(bool)

    # Convert the maze into an adjacency matrix.
    maze_graph, coords = maze_schema.encode_maze(maze)
    primitive_edges = maze_task.maze_primitive_edges(maze_graph)
    example = graph_bundle.convert_graph_with_edges(maze_graph, primitive_edges,
                                                    maze_task.BUILDER)
    edge_primitives = example.edges.apply_add(
        in_array=jnp.eye(4),
        out_array=jnp.zeros([len(maze_graph),
                             len(maze_graph), 4]),
        in_dims=(0,),
        out_dims=(0, 1))

    # Compute values for getting to a particular square, under a low temperature
    # (so that it's approximately shortest paths)
    values, q_vals, policy = train_maze_lib.soft_maze_values(
        edge_primitives,
        target_state_index=coords.index((0, 1)),
        temperature=1e-7)

    expected_values_at_coords = np.array([
        [-1, 0, -1, -2, -3],
        [-2, np.nan, -2, -3, -4],
        [-3, -4, -3, -4, -5],
    ])
    expected_values = [expected_values_at_coords[c] for c in coords]
    np.testing.assert_allclose(values, expected_values, atol=1e-6)

    # Check Q vals and policy at top left corner.
    np.testing.assert_allclose(q_vals[0], [-2, -1, -2, -3])
    np.testing.assert_allclose(policy[0], [0, 1, 0, 0])

    # Check reverse-mode gradient under a higher temperature.
    fun = functools.partial(
        train_maze_lib.soft_maze_values,
        target_state_index=coords.index((0, 1)),
        temperature=1)
    jax.test_util.check_grads(fun, (edge_primitives,), 1, "rev")

    # Check gradient under batching.
    jax.test_util.check_grads(
        jax.vmap(fun), (edge_primitives[None, Ellipsis],), 1, "rev")

  def test_loss_fn(self):
    maze = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]).astype(bool)

    # Convert the maze into an adjacency matrix.
    maze_graph, _ = maze_schema.encode_maze(maze)
    primitive_edges = maze_task.maze_primitive_edges(maze_graph)
    example = graph_bundle.convert_graph_with_edges(maze_graph, primitive_edges,
                                                    maze_task.BUILDER)

    def mock_model(example, dynamic_metadata):
      del example, dynamic_metadata
      return jnp.full([3, 14, 14], 1 / 14)

    loss, metrics = train_maze_lib.loss_fn(
        mock_model, (example, 1), num_goals=4)
    self.assertGreater(loss, 0)
    for metric in metrics.values():
      self.assertEqual(metric.shape, ())


if __name__ == "__main__":
  absltest.main()
