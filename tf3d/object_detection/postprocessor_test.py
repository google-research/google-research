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

"""Tests for ...object_detection.postprocessor."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.object_detection import postprocessor


class PostprocessorTest(tf.test.TestCase):

  def test_postprocess(self):
    num_classes = 10
    n = 1000

    outputs = {
        standard_fields.DetectionResultFields.objects_score:
            tf.random.uniform((n, num_classes),
                              minval=-2.0,
                              maxval=2.0,
                              dtype=tf.float32),
        standard_fields.DetectionResultFields.objects_rotation_matrix:
            tf.random.uniform((n, 3, 3), minval=-1.0, maxval=1.0,
                              dtype=tf.float32),
        standard_fields.DetectionResultFields.objects_center:
            tf.random.uniform((n, 3),
                              minval=10.0,
                              maxval=20.0,
                              dtype=tf.float32),
        standard_fields.DetectionResultFields.objects_length:
            tf.random.uniform((n, 1), minval=0.1, maxval=3.0, dtype=tf.float32),
        standard_fields.DetectionResultFields.objects_height:
            tf.random.uniform((n, 1), minval=0.1, maxval=3.0, dtype=tf.float32),
        standard_fields.DetectionResultFields.objects_width:
            tf.random.uniform((n, 1), minval=0.1, maxval=3.0, dtype=tf.float32),
    }

    postprocessor.postprocess(
        outputs=outputs,
        score_thresh=0.1,
        iou_thresh=0.5,
        max_output_size=10)

    for key in [
        standard_fields.DetectionResultFields.objects_length,
        standard_fields.DetectionResultFields.objects_height,
        standard_fields.DetectionResultFields.objects_width,
        standard_fields.DetectionResultFields.objects_center,
        standard_fields.DetectionResultFields.objects_class,
        standard_fields.DetectionResultFields.objects_score
    ]:
      self.assertEqual(len(outputs[key].shape), 2)
    self.assertEqual(
        len(outputs[standard_fields.DetectionResultFields
                    .objects_rotation_matrix].shape), 3)


if __name__ == '__main__':
  tf.test.main()
