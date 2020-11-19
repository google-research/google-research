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

"""Tests for ...box_utils.np_box_list."""

import numpy as np
import tensorflow as tf

from tf3d.object_detection.box_utils import np_box_list


class BoxList3dTest(tf.test.TestCase):

  def test_invalid_box_data(self):
    with self.assertRaises(ValueError):
      np_box_list.BoxList3d(length=np.array([0]),
                            height=np.array([0]),
                            width=np.array([0]),
                            center_x=np.array([1]),
                            center_y=np.array([1]),
                            center_z=np.array([1]),
                            rotation_z_radians=np.array([0]))
    with self.assertRaises(ValueError):
      np_box_list.BoxList3d(length=np.array([1, -1]),
                            height=np.array([1, 1]),
                            width=np.array([1, 1]),
                            center_x=np.array([1, 1]),
                            center_y=np.array([1, 1]),
                            center_z=np.array([1, 1]),
                            rotation_z_radians=np.array([0, 0]))
    with self.assertRaises(ValueError):
      np_box_list.BoxList3d(length=np.array([1, 1]),
                            height=np.array([1, 1]),
                            width=np.array([1, 1]),
                            center_x=np.array([1]),
                            center_y=np.array([1, 1]),
                            center_z=np.array([1, 1]),
                            rotation_z_radians=np.array([0, 0]))


if __name__ == '__main__':
  tf.test.main()
