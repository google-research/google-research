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

"""Tests for ...tf3d.semantic_segmentation.metric."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.semantic_segmentation import metric


class MetricTest(tf.test.TestCase):

  def test_semantic_segmentation_metric(self):
    label_map = {1: 'car', 2: 'bus', 3: 'sign', 9: 'pedestrian', 12: 'cyclist'}
    num_classes = 20
    num_frames = 5
    num_points = 100
    inputs = []
    outputs = []
    for _ in range(num_frames):
      inputs_i = {
          standard_fields.InputDataFields.object_class_points:
              tf.random.uniform([num_points, 1],
                                minval=0,
                                maxval=num_classes,
                                dtype=tf.int32),
          standard_fields.InputDataFields.point_loss_weights:
              tf.random.uniform([num_points, 1],
                                minval=0.0,
                                maxval=1.0,
                                dtype=tf.float32),
          standard_fields.InputDataFields.num_valid_points:
              tf.random.uniform([], minval=1, maxval=num_points,
                                dtype=tf.int32),
      }
      inputs.append(inputs_i)
      outputs_i = {
          standard_fields.DetectionResultFields.object_semantic_points:
              tf.random.uniform([num_points, num_classes],
                                minval=-2.0,
                                maxval=2.0,
                                dtype=tf.float32)
      }
      outputs.append(outputs_i)
    m = metric.SemanticSegmentationMetric(
        multi_label=False,
        num_classes=num_classes,
        label_map=label_map,
        eval_prefix='eval')
    for i in range(num_frames):
      m.update_state(inputs[i], outputs[i])
    metrics_dict = m.get_metric_dictionary()
    for object_name in ['car', 'bus', 'sign', 'pedestrian', 'cyclist']:
      self.assertIn('eval_recall/{}'.format(object_name), metrics_dict)
      self.assertIn('eval_precision/{}'.format(object_name), metrics_dict)
      self.assertIn('eval_iou/{}'.format(object_name), metrics_dict)
    self.assertIn('eval_avg/mean_pixel_accuracy', metrics_dict)
    self.assertIn('eval_avg/mean_iou', metrics_dict)


if __name__ == '__main__':
  tf.test.main()
