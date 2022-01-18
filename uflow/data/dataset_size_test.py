# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests that all datasets have expected length."""

from absl.testing import absltest

from uflow.data import generic_flow_dataset
from uflow.data import kitti
from uflow.data import sintel
from uflow.data.dataset_locations import dataset_locations


DATASETS_AND_SIZE = {
    # Note that sintel train has 1064 images, but only 1041 ground truth flows.
    # Sintel train provides 23 video snippets, and for each video snippet there
    # is one fewer flow than there are images (e.g., for a video of 2 frames,
    # you would only have 1 flow image).
    'sintel-test-clean': 552,
    'sintel-test-final': 552,
    'sintel-train-clean': 1041,
    'sintel-train-final': 1041,
    'kitti15-train-pairs': 200,
    'kitti15-test-pairs': 200,
    'chairs-all': 22872,
}


class DatasetSizeTest(absltest.TestCase):

  def _check_size(self, dataset, expected_size):
    count = 0
    for _ in dataset:
      count += 1
    self.assertEqual(count, expected_size)

  def test_sintel(self):
    for dataset in ['sintel-test-clean', 'sintel-test-final',
                    'sintel-train-clean', 'sintel-train-final']:
      size = DATASETS_AND_SIZE[dataset]
      path = dataset_locations[dataset]
      ds = sintel.make_dataset(path, mode='test')
      self._check_size(ds, size)

  def test_kitti(self):
    for dataset in ['kitti15-train-pairs', 'kitti15-test-pairs']:
      size = DATASETS_AND_SIZE[dataset]
      path = dataset_locations[dataset]
      ds = kitti.make_dataset(path, mode='eval')
      self._check_size(ds, size)

  def test_chairs(self):
    for dataset in ['chairs-all']:
      size = DATASETS_AND_SIZE[dataset]
      path = dataset_locations[dataset]
      ds = generic_flow_dataset.make_dataset(path, mode='test')
      self._check_size(ds, size)

if __name__ == '__main__':
  absltest.main()
