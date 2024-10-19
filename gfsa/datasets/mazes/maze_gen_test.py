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

"""Tests for gfsa.datasets.mazes.maze_gen."""

from absl.testing import absltest
import numpy as np
from gfsa.datasets.mazes import maze_gen


class MazeGenTest(absltest.TestCase):

  def test_generate_small_labmaze(self):
    m1 = maze_gen.generate_small_labmaze(2**31 - 1)
    m2 = maze_gen.generate_small_labmaze(2**31 - 1)
    np.testing.assert_array_equal(m1, m2)


if __name__ == "__main__":
  absltest.main()
