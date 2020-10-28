# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for gfsa.datasets.mazes.maze_task."""

from absl.testing import absltest
import numpy as np
from gfsa.datasets.mazes import maze_schema
from gfsa.datasets.mazes import maze_task


class MazeTaskTest(absltest.TestCase):

  def test_primitive_edges(self):
    maze = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]).T.astype(bool)

    maze_graph, _ = maze_schema.encode_maze(maze)
    primitive_edges = maze_task.maze_primitive_edges(maze_graph)

    subset_of_expected_edges = [
        ("cell_0_0", "cell_0_0", 0),
        ("cell_0_0", "cell_0_1", 1),
        ("cell_0_0", "cell_0_0", 2),
        ("cell_0_0", "cell_1_0", 3),
    ]
    for expected in subset_of_expected_edges:
      self.assertIn(expected, primitive_edges)


if __name__ == "__main__":
  absltest.main()
