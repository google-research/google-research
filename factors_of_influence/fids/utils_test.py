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

"""Tests for utils."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from factors_of_influence.fids import utils


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.segmentation = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  @parameterized.named_parameters(
      {
          'testcase_name': 'Old 0',
          'old_background_label': 0,
          'expected': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      },
      {
          'testcase_name': 'Old 254',
          'old_background_label': 254,
          'expected': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
      },
      {
          'testcase_name': 'Old 9',
          'old_background_label': 9,
          'expected': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
      },
      {
          'testcase_name': 'Old 5',
          'old_background_label': 5,
          'expected': np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9]),
      },
  )
  def testSegmentationNewBackgroundLabel(self, old_background_label, expected):
    new_segmentation_map = utils.segmentation_set_background_label_to_zero(
        self.segmentation,
        old_background_label=old_background_label)
    self.assertAllEqual(new_segmentation_map, expected)
    self.assertAllEqual(new_segmentation_map.shape, expected.shape)
    self.assertEqual(new_segmentation_map.dtype, np.uint16)


if __name__ == '__main__':
  tf.test.main()
