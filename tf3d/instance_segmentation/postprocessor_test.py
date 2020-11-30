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

"""Tests for ...tf3d.instance_segmentation.postprocessor."""

import numpy as np
import tensorflow as tf
from tf3d import standard_fields
from tf3d.instance_segmentation import postprocessor


class PostprocessorTest(tf.test.TestCase):

  def test_postprocess_without_nms(self):
    num_voxels = 10000
    outputs = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.random.uniform([num_voxels, 10],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32),
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            tf.random.uniform([num_voxels, 64],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32)
    }
    postprocessor.postprocess(
        outputs=outputs,
        num_furthest_voxel_samples=200,
        sampler_score_vs_distance_coef=0.5,
        embedding_similarity_strategy='distance',
        apply_nms=False,
        nms_score_threshold=0.1)
    self.assertAllEqual(
        outputs[standard_fields
                .DetectionResultFields.instance_segments_voxel_mask].shape,
        np.array([200, num_voxels]))
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.objects_class].shape,
        np.array([200, 1]))
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.objects_score].shape,
        np.array([200, 1]))

  def test_postprocess_with_nms(self):
    num_voxels = 10000
    outputs = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.random.uniform([num_voxels, 10],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32),
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            tf.random.uniform([num_voxels, 64],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32)
    }
    postprocessor.postprocess(
        outputs=outputs,
        num_furthest_voxel_samples=200,
        sampler_score_vs_distance_coef=0.5,
        embedding_similarity_strategy='distance',
        apply_nms=True,
        nms_score_threshold=0.1)
    num_instances = outputs[standard_fields.DetectionResultFields
                            .instance_segments_voxel_mask].shape[0]
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields
                .instance_segments_voxel_mask].shape,
        np.array([num_instances, num_voxels]))
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.objects_class].shape,
        np.array([num_instances, 1]))
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.objects_score].shape,
        np.array([num_instances, 1]))


if __name__ == '__main__':
  tf.test.main()
