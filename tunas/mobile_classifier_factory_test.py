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

"""Tests for mobile_classifier_factory.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import mobile_classifier_factory
from tunas import mobile_search_space_v3
from tunas import schema


class MobileClassifierFactoryTest(tf.test.TestCase, parameterized.TestCase):

  def test_get_model_spec_with_single_model(self):
    model_spec = mobile_classifier_factory.get_model_spec(
        mobile_search_space_v3.MOBILENET_V3_LARGE)
    self.assertIsInstance(model_spec, basic_specs.ConvTowerSpec)

    def validate(oneof):
      self.assertLen(oneof.choices, 1)
    schema.map_oneofs(validate, model_spec)

  def test_get_model_spec_for_search_space(self):
    model_spec = mobile_classifier_factory.get_model_spec(
        mobile_search_space_v3.MOBILENET_V3_LIKE_SEARCH)
    self.assertIsInstance(model_spec, basic_specs.ConvTowerSpec)

  @parameterized.parameters(0, 0.5)
  def test_get_model_for_search_with_v3_model(self, dropout_rate):
    model_spec = mobile_search_space_v3.get_search_space_spec(
        mobile_search_space_v3.MOBILENET_V3_LARGE)
    model = mobile_classifier_factory.get_model_for_search(
        model_spec, dropout_rate=dropout_rate)

    train_inputs = tf.ones([1, 224, 224, 3])
    model.build(train_inputs.shape)

    train_output, _ = model.apply(train_inputs, training=True)
    train_regularization_loss = model.regularization_loss()
    trainable_variables = model.trainable_variables()

    self.assertEqual(train_output.shape.as_list(), [1, 1001])
    self.assertEqual(train_regularization_loss.shape.as_list(), [])
    self.assertEqual(train_regularization_loss.dtype, train_inputs.dtype)

    valid_inputs = tf.ones([2, 224, 224, 3])
    valid_output, _ = model.apply(valid_inputs, training=False)
    valid_regularization_loss = model.regularization_loss()

    self.assertEqual(valid_output.shape.as_list(), [2, 1001])
    self.assertEqual(valid_regularization_loss.shape.as_list(), [])
    self.assertEqual(valid_regularization_loss.dtype, valid_inputs.dtype)

    # No trainable variables should be created during the validation pass.
    self.assertEqual(
        trainable_variables, model.trainable_variables())

  def test_get_standalone_model_v3(self):
    ssd = mobile_search_space_v3.MOBILENET_V3_LARGE
    model_spec = mobile_search_space_v3.get_search_space_spec(ssd)
    model = mobile_classifier_factory.get_standalone_model(model_spec)

    inputs = tf.ones([2, 224, 224, 3])
    model.build(inputs.shape)

    outputs, _ = model.apply(inputs, training=False)
    self.assertEqual(outputs.shape, [2, 1001])


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
