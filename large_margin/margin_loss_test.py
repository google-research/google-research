# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for margin_loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from large_margin import margin_loss


class MarginLossTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(margin_loss)

  @parameterized.named_parameters(
      ("norm_1", 1),
      ("norm_2", 2),
      ("norm_inf", np.inf),
  )
  def test_loss(self, dist_norm):
    image_shape = (28, 28, 1)
    num_classes = 10
    batch_size = 5
    images = tf.convert_to_tensor(
        np.random.rand(*((batch_size,) + image_shape)), dtype=tf.float32)
    labels = tf.convert_to_tensor(
        np.random.randint(0, high=num_classes, size=batch_size), dtype=tf.int32)
    # Toy model.
    endpoints = {}
    endpoints["input_layer"] = images
    # Convolution layer.
    net = tf.layers.conv2d(
        images,
        filters=3,
        kernel_size=16,
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    endpoints["conv_layer"] = net
    # Global average pooling layer.
    net = tf.reduce_mean(net, axis=[1, 2])
    # Output layer.
    logits = tf.layers.dense(net, num_classes)
    loss = margin_loss.large_margin(
        logits=logits,
        one_hot_labels=tf.one_hot(labels, num_classes),
        layers_list=[endpoints["input_layer"], endpoints["conv_layer"]],
        gamma=10,
        alpha_factor=4,
        top_k=1,
        dist_norm=dist_norm
    )
    var_list = tf.global_variables()
    init = tf.global_variables_initializer()

    # Test gradients are not None.
    gs = tf.gradients(loss, var_list)
    for g in gs:
      self.assertIsNotNone(g)

    # Test loss shape.
    with self.test_session() as sess:
      sess.run(init)
      self.assertEqual(sess.run(loss).shape, ())


if __name__ == "__main__":
  tf.test.main()
