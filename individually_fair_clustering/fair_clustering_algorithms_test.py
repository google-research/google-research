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

"""Test file for the fair clustering algorithms."""

from absl.testing import absltest
import numpy as np
from individually_fair_clustering import fair_clustering_algorithms


class FairClusteringAlgorithmsTest(absltest.TestCase):
  """Test class for the fair clustering algorithms."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dataset_size = 100
    self.dataset = np.array([[i] for i in range(self.dataset_size)])
    self.k = 5
    self.distance_threshold_vec = np.array(
        [float(self.dataset_size) / self.k] * self.dataset_size
    )

  def test_local_search_plus_plus(self):
    sol = fair_clustering_algorithms.LocalSearchPlusPlus(
        self.dataset, k=self.k, dist_threshold_vec=self.distance_threshold_vec
    )

    self.assertLessEqual(len(sol), self.k)

  def test_icml20(self):
    sol = fair_clustering_algorithms.LocalSearchICML2020(
        self.dataset, k=self.k, dist_threshold_vec=self.distance_threshold_vec
    )

    self.assertLessEqual(len(sol), self.k)


if __name__ == "__main__":
  absltest.main()
