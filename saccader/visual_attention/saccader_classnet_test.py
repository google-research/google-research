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

"""Unit test for Saccader-Classification Network model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from saccader.visual_attention import saccader_classnet
from saccader.visual_attention import saccader_classnet_config


def _get_test_cases():
  """Provides test cases."""
  is_training = [True, False]
  policy = ["learned", "random", "ordered_logits", "sobel_mean", "sobel_var"]
  classnet_type = ["resnet_v2_50", "nasnet"]
  return itertools.product(
      policy,
      is_training,
      classnet_type
      )


class SaccaderClassNetTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(saccader_classnet)

  @parameterized.parameters(
      _get_test_cases()
      )
  def test_build(self, policy, is_training, classnet_type):
    config = saccader_classnet_config.get_config()
    num_times = 2
    image_shape = (100, 100, 3)
    num_classes = 10
    config.num_classes = num_classes
    config.num_times = num_times
    config.classnet_type = classnet_type
    batch_size = 3
    images = np.random.rand(*((batch_size,) + image_shape))
    images = tf.placeholder_with_default(images, shape=(None,) + image_shape)
    images = tf.cast(images, dtype=tf.float32)
    model = saccader_classnet.SaccaderClassNet(config)
    logits = model(images, images,
                   num_times=num_times,
                   is_training_saccader=is_training,
                   is_training_classnet=is_training,
                   policy=policy)[0]
    init_op = model.init_op
    self.evaluate(init_op)
    self.assertEqual((batch_size, num_classes),
                     self.evaluate(logits).shape)

  @parameterized.parameters(
      _get_test_cases()
      )
  def test_reuse_model(self, policy, is_training, classnet_type):
    config = saccader_classnet_config.get_config()
    num_times = 2
    image_shape = (100, 100, 3)
    num_classes = 10
    config.num_classes = num_classes
    config.num_times = num_times
    config.classnet_type = classnet_type
    batch_size = 3
    images = tf.constant(
        np.random.rand(*((batch_size,) + image_shape)), dtype=tf.float32)
    model = saccader_classnet.SaccaderClassNet(config)
    logits1 = model(images, images, num_times=num_times,
                    is_training_saccader=is_training,
                    is_training_classnet=is_training,
                    policy=policy)[0]
    num_params = len(tf.all_variables())
    l2_loss1 = tf.losses.get_regularization_loss()
    # Build twice with different num_times.
    logits2 = model(images, images, num_times=num_times+1,
                    is_training_saccader=is_training,
                    is_training_classnet=is_training,
                    policy=policy)[0]
    l2_loss2 = tf.losses.get_regularization_loss()

    # Ensure variables are reused.
    self.assertLen(tf.all_variables(), num_params)
    init = tf.global_variables_initializer()
    self.evaluate(init)
    logits1, logits2 = self.evaluate((logits1, logits2))
    l2_loss1, l2_loss2 = self.evaluate((l2_loss1, l2_loss2))
    np.testing.assert_almost_equal(l2_loss1, l2_loss2, decimal=5)


if __name__ == "__main__":
  tf.test.main()
