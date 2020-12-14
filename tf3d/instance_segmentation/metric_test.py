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

"""Tests for ...tf3d.instance_segmentation.metric."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.instance_segmentation import metric


class MetricTest(tf.test.TestCase):

  def test_instance_segmentation_metric(self):
    label_map = {1: 'car', 2: 'bus', 3: 'sign', 9: 'pedestrian', 12: 'cyclist'}
    max_num_gt_objects = 10
    max_num_predicted_objects = 20
    max_num_voxels = 1000
    num_classes = 20
    num_frames = 5
    inputs = []
    outputs = []
    for _ in range(num_frames):
      num_voxels = tf.random.uniform([],
                                     minval=1,
                                     maxval=max_num_voxels,
                                     dtype=tf.int32)
      num_gt_objects = tf.random.uniform([],
                                         minval=1,
                                         maxval=max_num_gt_objects,
                                         dtype=tf.int32)
      num_predicted_objects = tf.random.uniform(
          [], minval=1, maxval=max_num_predicted_objects, dtype=tf.int32)
      inputs_i = {
          standard_fields.InputDataFields.objects_class:
              tf.random.uniform([num_gt_objects, 1],
                                minval=1,
                                maxval=num_classes,
                                dtype=tf.int32),
          standard_fields.InputDataFields.object_instance_id_voxels:
              tf.random.uniform([num_voxels, 1],
                                minval=0,
                                maxval=num_gt_objects,
                                dtype=tf.int32),
      }
      inputs.append(inputs_i)
      outputs_i = {
          standard_fields.DetectionResultFields.objects_score:
              tf.random.uniform([num_predicted_objects, 1],
                                minval=0.0,
                                maxval=1.0,
                                dtype=tf.float32),
          standard_fields.DetectionResultFields.objects_class:
              tf.random.uniform([num_predicted_objects, 1],
                                minval=1,
                                maxval=num_classes,
                                dtype=tf.int32),
          standard_fields.DetectionResultFields.instance_segments_voxel_mask:
              tf.random.uniform([num_predicted_objects, num_voxels],
                                minval=0.0,
                                maxval=1.0,
                                dtype=tf.float32),
      }
      outputs.append(outputs_i)
    iou_threshold = 0.5
    m = metric.InstanceSegmentationMetric(
        iou_threshold=iou_threshold,
        num_classes=num_classes,
        label_map=label_map,
        eval_prefix='eval')
    for i in range(num_frames):
      m.update_state(inputs[i], outputs[i])
    metrics_dict = m.get_metric_dictionary()
    for object_name in ['car', 'bus', 'sign', 'pedestrian', 'cyclist']:
      self.assertIn('eval_IOU{}_AP/{}'.format(iou_threshold, object_name),
                    metrics_dict)
    self.assertIn('eval_avg/mean_AP_IOU{}'.format(iou_threshold), metrics_dict)


if __name__ == '__main__':
  tf.test.main()
