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

"""Tests for kws_streaming.layers.zero_mean_constraint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from kws_streaming.layers import non_scaling_dropout
tf.disable_eager_execution()


class NonScalingDropoutTest(tf.test.TestCase):

  def setUp(self):
    super(NonScalingDropoutTest, self).setUp()
    tf.compat.v1.reset_default_graph()
    self.sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(self.sess)
    self.test_inputs = _get_random_data()
    self.seed = 1337

  def tearDown(self):
    tf.compat.v1.keras.backend.clear_session()
    self.sess.close()
    super(NonScalingDropoutTest, self).tearDown()

  def test_non_scaling_dropout_training(self):
    training = True
    keep_prob = 0.75
    input_shape = self.test_inputs.shape
    noise_shape = [input_shape[0], 1, input_shape[2]]

    # Keras implementation
    layer = non_scaling_dropout.NonScalingDropout(
        rate=(1 - keep_prob),
        noise_shape=noise_shape,
        seed=self.seed,
        training=training)
    actual_output = layer(self.test_inputs)

    # TF implementation
    expected_output = _original_non_scaling_dropout(
        self.test_inputs,
        keep_prob=keep_prob,
        noise_shape=noise_shape,
        seed=self.seed)
    # Test that the layer works in the same way as the original in the base-case
    self.assertAllClose(expected_output, actual_output)

  def test_non_scaling_dropout_inference(self):
    training = False
    keep_prob = 0.75
    input_shape = self.test_inputs.shape
    noise_shape = [input_shape[0], 1, input_shape[2]]

    # Keras implementation
    layer = non_scaling_dropout.NonScalingDropout(
        rate=(1 - keep_prob),
        noise_shape=noise_shape,
        seed=self.seed,
        training=training)
    output = layer(self.test_inputs)

    # When applied during inference, should not do anything
    self.assertAllClose(self.test_inputs, output)

  def test_non_scaling_dropout_keep_prob_is_one(self):
    training = True
    keep_prob = 1
    input_shape = self.test_inputs.shape
    noise_shape = [input_shape[0], 1, input_shape[2]]

    # Keras implementation
    layer = non_scaling_dropout.NonScalingDropout(
        rate=(1 - keep_prob),
        noise_shape=noise_shape,
        seed=self.seed,
        training=training)
    output = layer(self.test_inputs)

    # since keep_prob is 1, shouldn't affect the result
    self.assertAllClose(self.test_inputs, output)

  def test_non_scaling_dropout_noise_shape_inference(self):
    training = True
    keep_prob = 0.5
    layer = non_scaling_dropout.NonScalingDropout(
        rate=(1 - keep_prob), seed=self.seed, training=training)
    layer(self.test_inputs)

    # When given no input shape, check that the shape is inferred from the input
    self.assertAllEqual(self.test_inputs.shape,
                        self.sess.run(layer.noise_shape))

  def test_non_scaling_dropout_noise_broadcasting(self):
    training = True
    keep_prob = 0.5
    input_shape = [3, 5]

    # Set all the inputs to 1 so we can monitor the output
    inputs = np.ones(input_shape, dtype="float32")
    layer = non_scaling_dropout.NonScalingDropout(
        rate=(1 - keep_prob),
        noise_shape=[1, input_shape[1]],
        seed=self.seed,
        training=training)
    output = self.sess.run(layer(inputs))

    self.assertAllInSet(output, [0, 1])

    # Check that the row is broadcasted correctly
    row = output[0]
    for i in range(output.shape[0]):
      self.assertAllEqual(row, output[i])


def _original_non_scaling_dropout(inputs,
                                  keep_prob,
                                  noise_shape=None,
                                  seed=None,
                                  name=None):
  with tf.compat.v1.name_scope(name, "NonScalingDropout", [inputs]) as name:
    if keep_prob == 1:
      return inputs
    noise_mask = tf.compat.v1.random_uniform(noise_shape, seed=seed) < keep_prob
    return inputs * tf.compat.v1.cast(noise_mask, tf.float32)


def _get_random_data():
  np.random.seed(1337)
  values = np.random.rand(
      200,
      34,
      16,
  )
  return np.float32(values)


if __name__ == "__main__":
  tf.test.main()
