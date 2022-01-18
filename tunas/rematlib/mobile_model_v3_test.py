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

# Lint as: python2, python3
"""Tests for mobile_model_v3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from tunas import mobile_search_space_v3
from tunas import search_space_utils
from tunas import test_utils
from tunas.rematlib import mobile_model_v3


class MobileModelV3Test(test_utils.ModelTest, parameterized.TestCase):

  def test_output_shapes(self):
    model_spec = mobile_search_space_v3.mobilenet_v3_large()
    model = mobile_model_v3.get_model(model_spec, num_classes=50)

    features = tf.ones([8, 224, 224, 3])
    model.build(features.shape)
    logits, endpoints = model.apply(features, training=True)

    self.assertEqual(logits.shape, tf.TensorShape([8, 50]))
    self.assertEqual([x.shape for x in endpoints], [
        [8, 112, 112, 16],
        [8, 56, 56, 24],
        [8, 28, 28, 40],
        [8, 14, 14, 112],
        [8, 7, 7, 160],
    ])

  def test_output_shapes_with_variable_filter_sizes(self):
    filter_multipliers = (0.5, 1.0, 2.0)

    model_spec = mobile_search_space_v3.mobilenet_v3_large()
    model_spec = search_space_utils.scale_conv_tower_spec(
        model_spec, multipliers=filter_multipliers)
    model_spec = test_utils.with_random_masks(model_spec)
    model = mobile_model_v3.get_model(model_spec, num_classes=50)

    features = tf.ones([8, 224, 224, 3])
    model.build(features.shape)
    logits, endpoints = model.apply(features, training=True)

    self.assertEqual(logits.shape, tf.TensorShape([8, 50]))
    self.assertEqual([x.shape for x in endpoints], [
        [8, 112, 112, int(16 * max(filter_multipliers))],
        [8, 56, 56, int(24 * max(filter_multipliers))],
        [8, 28, 28, int(40 * max(filter_multipliers))],
        [8, 14, 14, int(112 * max(filter_multipliers))],
        [8, 7, 7, int(160 * max(filter_multipliers))],
    ])

  def test_output_shapes_with_variable_kernel_sizes(self):
    model_spec = mobile_search_space_v3.mobilenet_v3_like_search()
    model_spec = test_utils.with_random_masks(model_spec)
    model = mobile_model_v3.get_model(
        model_spec, num_classes=50, force_stateless_batch_norm=True)

    features = tf.ones([8, 224, 224, 3])
    model.build(features.shape)
    logits, endpoints = model.apply(features, training=True)

    self.assertEqual(logits.shape, tf.TensorShape([8, 50]))
    self.assertLen(endpoints, 5)
    endpoints[0].shape.assert_is_compatible_with([8, 112, 112, None])
    endpoints[1].shape.assert_is_compatible_with([8, 56, 56, None])
    endpoints[2].shape.assert_is_compatible_with([8, 28, 28, None])
    endpoints[3].shape.assert_is_compatible_with([8, 14, 14, None])
    endpoints[4].shape.assert_is_compatible_with([8, 7, 7, None])

  @parameterized.parameters(list(mobile_search_space_v3.ALL_SSDS))
  def test_search_space_construction(self, ssd):
    model_spec = mobile_search_space_v3.get_search_space_spec(ssd)
    model_spec = test_utils.with_random_masks(model_spec)
    model = mobile_model_v3.get_model(
        model_spec, num_classes=1001, force_stateless_batch_norm=True)

    inputs = tf.random_normal(shape=[128, 224, 224, 3])
    model.build(inputs.shape)

    output, unused_endpoints = model.apply(inputs, training=True)
    self.assertEqual(output.shape, [128, 1001])

  @parameterized.parameters(list(mobile_search_space_v3.ALL_SSDS))
  def test_dynamic_input_size(self, ssd):
    model_spec = mobile_search_space_v3.get_search_space_spec(ssd)
    model_spec = test_utils.with_random_pruning(model_spec)
    model = mobile_model_v3.get_model(
        model_spec, num_classes=1001, force_stateless_batch_norm=False)

    # Input height/width are not known at graph construction time.
    inputs = tf.placeholder(shape=[128, None, None, 3], dtype=tf.float32)
    model.build(inputs.shape)

    output, unused_endpoints = model.apply(inputs, training=True)
    self.assertEqual(output.shape, [128, 1001])


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
