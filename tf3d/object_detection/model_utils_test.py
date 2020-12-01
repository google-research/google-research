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

"""Tests for ...tf3d.object_detection.model_utils."""

import math
import numpy as np
import tensorflow as tf
from tf3d import standard_fields
from tf3d.object_detection import model_utils


class ModelUtilsTest(tf.test.TestCase):

  def test_normalize_cos_sin_rotation(self):
    outputs = {
        'cos': tf.constant([[3.0], [6.0]], dtype=tf.float32),
        'sin': tf.constant([[4.0], [8.0]], dtype=tf.float32)
    }
    model_utils.normalize_cos_sin_rotation(
        outputs=outputs, cos_key='cos', sin_key='sin')
    self.assertAllClose(outputs['cos'].numpy(), np.array([[0.6], [0.6]]))
    self.assertAllClose(outputs['sin'].numpy(), np.array([[0.8], [0.8]]))

  def test_make_boxes_positive(self):
    outputs = {
        'length': tf.constant([[-1.0], [2.0], [-2.0]], dtype=tf.float32),
        'height': tf.constant([[-2.0], [3.0], [-4.0]], dtype=tf.float32),
        'width': tf.constant([[-3.0], [4.0], [-1.0]], dtype=tf.float32),
    }
    model_utils.make_box_sizes_positive(
        outputs=outputs,
        length_key='length',
        height_key='height',
        width_key='width')
    self.assertAllClose(outputs['length'].numpy(),
                        np.array([[1.0], [2.0], [2.0]]))
    self.assertAllClose(outputs['height'].numpy(),
                        np.array([[2.0], [3.0], [4.0]]))
    self.assertAllClose(outputs['width'].numpy(), np.array([[3.0], [4.0],
                                                            [1.0]]))

  def test_rectify_outputs(self):
    lengths = tf.constant([[[1.2], [-0.9], [2.5], [1.3], [1.7], [21.0]]])
    heights = tf.constant([[[-3.2], [0.4], [-2.8], [1.4], [-1.1], [11.0]]])
    widths = tf.constant([[[5.1], [0.7], [2.3], [3.4], [3.1], [-31.0]]])
    rot_x_sin = tf.constant([[[0.0], [1.0], [-0.5], [2.0], [3.0], [4.0]]])
    rot_x_cos = tf.constant([[[1.0], [0.0], [-0.5], [2.0], [3.0], [4.0]]])
    rot_y_sin = tf.constant([[[0.0], [1.0], [-0.5], [2.0], [3.0], [4.0]]])
    rot_y_cos = tf.constant([[[1.0], [0.0], [-0.5], [2.0], [3.0], [4.0]]])
    rot_z_sin = tf.constant([[[0.0], [1.0], [-0.5], [2.0], [3.0], [4.0]]])
    rot_z_cos = tf.constant([[[1.0], [0.0], [-0.5], [2.0], [3.0], [4.0]]])
    expected_lengths = tf.constant([[[1.2], [0.9], [2.5], [1.3], [1.7],
                                     [21.0]]])
    expected_heights = tf.constant([[[3.2], [0.4], [2.8], [1.4], [1.1],
                                     [11.0]]])
    expected_widths = tf.constant([[[5.1], [0.7], [2.3], [3.4], [3.1], [31.0]]])
    expected_rot_x_sin = tf.constant([[[0.0], [1.0], [-math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2], [math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2]]])
    expected_rot_x_cos = tf.constant([[[1.0], [0.0], [-math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2], [math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2]]])
    expected_rot_y_sin = tf.constant([[[0.0], [1.0], [-math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2], [math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2]]])
    expected_rot_y_cos = tf.constant([[[1.0], [0.0], [-math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2], [math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2]]])
    expected_rot_z_sin = tf.constant([[[0.0], [1.0], [-math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2], [math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2]]])
    expected_rot_z_cos = tf.constant([[[1.0], [0.0], [-math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2], [math.sqrt(2) / 2],
                                       [math.sqrt(2) / 2]]])
    outputs = {
        standard_fields.DetectionResultFields.object_length_voxels:
            lengths,
        standard_fields.DetectionResultFields.object_height_voxels:
            heights,
        standard_fields.DetectionResultFields.object_width_voxels:
            widths,
        standard_fields.DetectionResultFields.object_rotation_x_cos_voxels:
            rot_x_cos,
        standard_fields.DetectionResultFields.object_rotation_x_sin_voxels:
            rot_x_sin,
        standard_fields.DetectionResultFields.object_rotation_y_cos_voxels:
            rot_y_cos,
        standard_fields.DetectionResultFields.object_rotation_y_sin_voxels:
            rot_y_sin,
        standard_fields.DetectionResultFields.object_rotation_z_cos_voxels:
            rot_z_cos,
        standard_fields.DetectionResultFields.object_rotation_z_sin_voxels:
            rot_z_sin,
    }
    model_utils.rectify_outputs(outputs=outputs)
    expected_outputs = {
        standard_fields.DetectionResultFields.object_length_voxels:
            expected_lengths,
        standard_fields.DetectionResultFields.object_height_voxels:
            expected_heights,
        standard_fields.DetectionResultFields.object_width_voxels:
            expected_widths,
        standard_fields.DetectionResultFields.object_rotation_x_cos_voxels:
            expected_rot_x_cos,
        standard_fields.DetectionResultFields.object_rotation_x_sin_voxels:
            expected_rot_x_sin,
        standard_fields.DetectionResultFields.object_rotation_y_cos_voxels:
            expected_rot_y_cos,
        standard_fields.DetectionResultFields.object_rotation_y_sin_voxels:
            expected_rot_y_sin,
        standard_fields.DetectionResultFields.object_rotation_z_cos_voxels:
            expected_rot_z_cos,
        standard_fields.DetectionResultFields.object_rotation_z_sin_voxels:
            expected_rot_z_sin,
    }
    for key in outputs:
      if key in expected_outputs:
        self.assertAllClose(outputs[key].numpy(), expected_outputs[key].numpy())


if __name__ == '__main__':
  tf.test.main()
