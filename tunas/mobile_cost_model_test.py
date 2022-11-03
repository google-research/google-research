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

"""Tests for mobile_cost_model.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import mobile_cost_model
from tunas import mobile_search_space_v3
from tunas import schema
from tunas import test_utils


def _assign_random_mask(oneof):
  # Assign random one-hot masks for both ops and filters.
  mask = test_utils.random_one_hot(len(oneof.choices))
  return schema.OneOf(oneof.choices, oneof.tag, mask)


def _assign_random_mask_to_ops_only(oneof):
  # Similar to _assign_random_mask(), but only assigns masks to ops
  # (and not to filters).
  if oneof.tag == basic_specs.OP_TAG:
    return _assign_random_mask(oneof)
  else:
    return oneof


def _make_single_layer_model(layer):
  return basic_specs.ConvTowerSpec(
      blocks=[
          basic_specs.Block(layers=[layer], filters=32),
      ],
      filters_base=8)


class MobileCostModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(list(mobile_search_space_v3.ALL_SSDS))
  def test_coupled_tf_features_with_mobile_model_v3(self, ssd):
    model_spec = mobile_search_space_v3.get_search_space_spec(ssd)
    model_spec = schema.map_oneofs(_assign_random_mask, model_spec)
    features = mobile_cost_model.coupled_tf_features(model_spec)
    self.assertEqual(features.dtype, tf.float32)
    self.assertEqual(features.shape.rank, 1)

  def test_v3_single_choice_zero_only(self):
    layer = schema.OneOf(
        choices=[basic_specs.ZeroSpec()],
        tag=basic_specs.OP_TAG,
        mask=tf.constant([1.0]))
    features = mobile_cost_model.coupled_tf_features(
        _make_single_layer_model(layer))
    self.assertAllClose(self.evaluate(features), [1.0])

  def test_v3_zero_or_conv(self):
    mask = tf.placeholder(shape=[2], dtype=tf.float32)
    layer = schema.OneOf(
        choices=[
            basic_specs.ZeroSpec(),
            mobile_search_space_v3.ConvSpec(kernel_size=5, strides=1),
        ],
        tag=basic_specs.OP_TAG,
        mask=mask)
    features = mobile_cost_model.coupled_tf_features(
        _make_single_layer_model(layer))

    with self.session() as sess:
      self.assertAllClose([1.0, 0.0], sess.run(features, {mask: [1.0, 0.0]}))
      self.assertAllClose([0.0, 1.0], sess.run(features, {mask: [0.0, 1.0]}))

  def test_v3_zero_or_zero(self):
    mask = tf.placeholder(shape=[2], dtype=tf.float32)
    layer = schema.OneOf(
        choices=[
            basic_specs.ZeroSpec(),
            basic_specs.ZeroSpec(),
        ],
        tag=basic_specs.OP_TAG,
        mask=mask)
    features = mobile_cost_model.coupled_tf_features(
        _make_single_layer_model(layer))

    with self.session() as sess:
      self.assertAllClose([1.0, 0.0], sess.run(features, {mask: [1.0, 0.0]}))
      self.assertAllClose([0.0, 1.0], sess.run(features, {mask: [0.0, 1.0]}))

  def test_v3_zero_or_conv_with_child(self):
    kernel_size_mask = tf.placeholder(shape=[3], dtype=tf.float32)
    kernel_size = schema.OneOf([3, 5, 7], basic_specs.OP_TAG, kernel_size_mask)

    layer_mask = tf.placeholder(shape=[2], dtype=tf.float32)
    layer = schema.OneOf(
        choices=[
            basic_specs.ZeroSpec(),
            mobile_search_space_v3.ConvSpec(kernel_size=kernel_size, strides=1),
        ],
        tag=basic_specs.OP_TAG,
        mask=layer_mask)

    features = mobile_cost_model.coupled_tf_features(
        _make_single_layer_model(layer))

    with self.session() as sess:
      self.assertAllClose(
          [1.0, 0.0, 0.0, 0.0],
          sess.run(features, {layer_mask: [1, 0], kernel_size_mask: [1, 0, 0]}))
      self.assertAllClose(
          [1.0, 0.0, 0.0, 0.0],
          sess.run(features, {layer_mask: [1, 0], kernel_size_mask: [0, 1, 0]}))
      self.assertAllClose(
          [1.0, 0.0, 0.0, 0.0],
          sess.run(features, {layer_mask: [1, 0], kernel_size_mask: [0, 0, 1]}))
      self.assertAllClose(
          [0.0, 1.0, 0.0, 0.0],
          sess.run(features, {layer_mask: [0, 1], kernel_size_mask: [1, 0, 0]}))
      self.assertAllClose(
          [0.0, 0.0, 1.0, 0.0],
          sess.run(features, {layer_mask: [0, 1], kernel_size_mask: [0, 1, 0]}))
      self.assertAllClose(
          [0.0, 0.0, 0.0, 1.0],
          sess.run(features, {layer_mask: [0, 1], kernel_size_mask: [0, 0, 1]}))

  def test_v3_two_children(self):
    kernel_size1_mask = tf.placeholder(shape=[3], dtype=tf.float32)
    kernel_size1 = schema.OneOf(
        [3, 5, 7], basic_specs.OP_TAG, kernel_size1_mask)

    kernel_size2_mask = tf.placeholder(shape=[2], dtype=tf.float32)
    kernel_size2 = schema.OneOf(
        [3, 5], basic_specs.OP_TAG, kernel_size2_mask)

    layer_mask = tf.placeholder(shape=[2], dtype=tf.float32)
    layer = schema.OneOf(
        choices=[
            mobile_search_space_v3.SeparableConvSpec(
                kernel_size=kernel_size1, strides=1),
            mobile_search_space_v3.ConvSpec(
                kernel_size=kernel_size2, strides=1),
        ],
        tag=basic_specs.OP_TAG,
        mask=layer_mask)

    features = mobile_cost_model.coupled_tf_features(
        _make_single_layer_model(layer))

    with self.session() as sess:
      self.assertAllClose(
          [1.0, 0.0, 0.0, 0.0, 0.0],
          sess.run(features, {
              layer_mask: [1, 0],  # select the first mask
              kernel_size1_mask: [1, 0, 0],
              kernel_size2_mask: [1, 0]  # should be ignored
          }))
      self.assertAllClose(
          [0.0, 1.0, 0.0, 0.0, 0.0],
          sess.run(features, {
              layer_mask: [1, 0],  # select the first mask
              kernel_size1_mask: [0, 1, 0],
              kernel_size2_mask: [1, 0]  # should be ignored
          }))
      self.assertAllClose(
          [0.0, 0.0, 1.0, 0.0, 0.0],
          sess.run(features, {
              layer_mask: [1, 0],  # select the first mask
              kernel_size1_mask: [0, 0, 1],
              kernel_size2_mask: [1, 0]  # should be ignored
          }))
      self.assertAllClose(
          [0.0, 0.0, 0.0, 1.0, 0.0],  # select the second mask
          sess.run(features, {
              layer_mask: [0, 1],
              kernel_size1_mask: [1, 0, 0],  # should be ignored
              kernel_size2_mask: [1, 0]
          }))
      self.assertAllClose(
          [0.0, 0.0, 0.0, 0.0, 1.0],  # select the second mask
          sess.run(features, {
              layer_mask: [0, 1],
              kernel_size1_mask: [1, 0, 0],  # should be ignored
              kernel_size2_mask: [0, 1]
          }))

  def test_estimate_cost_integration_test(self):
    indices = [
        0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 2, 1, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0,
        0
    ]
    cost = mobile_cost_model.estimate_cost(indices, 'proxylessnas_search')
    self.assertNear(cost, 84.0, err=1.0)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
