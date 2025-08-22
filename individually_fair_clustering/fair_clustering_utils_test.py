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

"""Test file for the fair clustering utility functions."""

from absl.testing import absltest
import numpy as np
from individually_fair_clustering import fair_clustering_utils


class FairClusteringUtilsTest(absltest.TestCase):
  """"Test class for the fair clustering utility functions."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dataset_size = 10
    self.dataset = np.array([[i] for i in range(self.dataset_size)])
    self.error = 0.001

  def test_top_two_closest_to_centers(self):
    t2cc = fair_clustering_utils.TopTwoClosestToCenters(
        dataset=self.dataset, centers_ids=[0, 1]
    )

    self.assertBetween(
        t2cc.SampleWithD2Distribution(), 2, self.dataset_size - 1
    )

    new_cost = 1 + np.sum([(i-2)**2 for i in range(3, self.dataset_size)])

    self.assertBetween(
        t2cc.CostAfterSwap(1, 2), new_cost - self.error, new_cost + self.error
    )

    t2cc.SwapCenters(pos_center_to_remove=1, pos_center_to_add=2)

    self.assertIn(
        t2cc.SampleWithD2Distribution(),
        set(range(self.dataset_size)) - set([0, 2]),
    )


if __name__ == "__main__":
  absltest.main()
