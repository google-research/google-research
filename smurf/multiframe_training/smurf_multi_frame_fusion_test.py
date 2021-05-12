# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for multiframe_training."""

from typing import Dict, Tuple, Union

from absl.testing import absltest
import absl.testing.parameterized as parameterized
import tensorflow as tf  # tf

from smurf.multiframe_training import smurf_multi_frame_fusion

_IMAGE_HEIGHT = 10
_IMAGE_WIDTH = 20


class SmurfMultiFrameFusionTest(tf.test.TestCase, parameterized.TestCase):

  def _deserialize(self, raw_data, dtype, height, width, channels):
    return tf.reshape(
        tf.io.decode_raw(raw_data, dtype), [height, width, channels])

  def _deserialize_png(self, raw_data):
    image_uint = tf.image.decode_png(raw_data)
    return tf.image.convert_image_dtype(image_uint, tf.float32)

  def _create_image_triplet(self):
    return tf.ones((3, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3), dtype=tf.float32)

  def _create_flows_and_masks(
      self):
    flow = tf.ones((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 2), dtype=tf.float32)
    mask = tf.ones((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 1), dtype=tf.float32)
    return flow, mask

  def _decode_sequence_example(
      self,
      sequence_example,
      visualization = True):
    output = {}
    context_features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'flow_uv': tf.io.FixedLenFeature([], tf.string),
        'flow_valid': tf.io.FixedLenFeature([], tf.string),
    }
    sequence_features = {
        'images': tf.io.FixedLenSequenceFeature([], tf.string),
    }
    if visualization:
      context_features['flow_viz'] = tf.io.FixedLenFeature([], tf.string)

    parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
        sequence_example.SerializeToString(), context_features,
        sequence_features)

    output['height'] = parsed_context['height']
    output['width'] = parsed_context['width']
    output['flow'] = self._deserialize(parsed_context['flow_uv'], tf.float32,
                                       _IMAGE_HEIGHT, _IMAGE_WIDTH, 2)
    output['mask'] = self._deserialize(parsed_context['flow_valid'], tf.float32,
                                       _IMAGE_HEIGHT, _IMAGE_WIDTH, 1)
    output['images'] = tf.map_fn(
        self._deserialize_png, parsed_sequence['images'], dtype=tf.float32)
    if visualization:
      output['viz'] = self._deserialize_png(parsed_context['flow_viz'])

    return output

  def test_get_flow_inference_function(self):
    images = self._create_image_triplet()
    # Non-existing checkpoint should give us a model with random initialization.
    flow_infer = smurf_multi_frame_fusion.get_flow_inference_function(
        checkpoint='', height=128, width=128)
    flow = flow_infer(images[0], images[1])
    flow_height, flow_width = flow.shape.as_list()[-3:-1]
    self.assertEqual(flow_height, _IMAGE_HEIGHT)
    self.assertEqual(flow_width, _IMAGE_WIDTH)

  def test_get_occlusion_inference_function(self):
    flow, _ = self._create_flows_and_masks()
    occlusion_infer = (
        smurf_multi_frame_fusion.get_occlusion_inference_function())
    mask = occlusion_infer(flow, flow)
    mask_height, mask_width = mask.shape.as_list()[-3:-1]
    self.assertEqual(mask_height, _IMAGE_HEIGHT)
    self.assertEqual(mask_width, _IMAGE_WIDTH)

  def test_run_multiframe_fusion(self):
    images = self._create_image_triplet()

    def dummy_flow_infer(image1, image2):
      del image2
      return tf.zeros_like(image1[Ellipsis, :2], tf.float32)

    def dummy_occlusion_infer(flow1, flow2):
      del flow2
      return tf.zeros_like(flow1[Ellipsis, :1], tf.float32)

    flow, mask = smurf_multi_frame_fusion.run_multiframe_fusion(
        images, dummy_flow_infer, dummy_occlusion_infer)

    flow_height, flow_width = flow.shape.as_list()[-3:-1]
    self.assertEqual(flow_height, _IMAGE_HEIGHT)
    self.assertEqual(flow_width, _IMAGE_WIDTH)

    mask_height, mask_width = mask.shape.as_list()[-3:-1]
    self.assertEqual(mask_height, _IMAGE_HEIGHT)
    self.assertEqual(mask_width, _IMAGE_WIDTH)

  @parameterized.named_parameters(
      ('with_visualization', False),
      ('without_visualization', True),
  )
  def test_create_output_sequence_example(
      self, add_visualization):
    images = self._create_image_triplet()
    flow, mask = self._create_flows_and_masks()

    sequence_example = smurf_multi_frame_fusion.create_output_sequence_example(
        images, flow, mask, add_visualization=add_visualization)

    parsed_exampled = self._decode_sequence_example(sequence_example,
                                                    add_visualization)

    # Check if everything matches the input.
    self.assertEqual(parsed_exampled['height'], _IMAGE_HEIGHT)
    self.assertEqual(parsed_exampled['width'], _IMAGE_WIDTH)
    self.assertAllEqual(parsed_exampled['flow'], flow[0])
    self.assertAllEqual(parsed_exampled['mask'], mask[0])
    if add_visualization:
      self.assertAllEqual(
          tf.shape(parsed_exampled['viz']), (_IMAGE_HEIGHT, _IMAGE_WIDTH, 3))
    # Frame t-1 is dropped.
    self.assertAllEqual(parsed_exampled['images'], images[1:])


if __name__ == '__main__':
  absltest.main()
