# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Unit test for Saccader model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from saccader.visual_attention import saccader
from saccader.visual_attention import saccader_config


def _get_test_cases():
  """Provides test cases."""
  is_training = [True, False]
  policy = ["learned", "random", "ordered_logits", "sobel_mean", "sobel_var"]
  i = 0
  cases = []
  for p in policy:
    for t in is_training:
      cases.append(("case_%d" % i, p, t))
      i += 1
  return tuple(cases)


class SaccaderTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(saccader)

  @parameterized.named_parameters(
      *_get_test_cases()
      )
  def test_build(self, policy, is_training):
    config = saccader_config.get_config()
    num_times = 2
    image_shape = (224, 224, 3)
    num_classes = 10
    config.num_classes = num_classes
    config.num_times = num_times
    batch_size = 3
    images = tf.constant(
        np.random.rand(*((batch_size,) + image_shape)), dtype=tf.float32)
    model = saccader.Saccader(config)
    logits = model(images, num_times=num_times, is_training=is_training,
                   policy=policy)[0]
    init_op = model.init_op
    self.evaluate(init_op)
    self.assertEqual((batch_size, num_classes),
                     self.evaluate(logits).shape)

  @parameterized.named_parameters(
      *_get_test_cases()
      )
  def test_locations(self, policy, is_training):
    config = saccader_config.get_config()
    num_times = 2
    image_shape = (224, 224, 3)
    num_classes = 10
    config.num_classes = num_classes
    config.num_times = num_times
    batch_size = 4
    images = tf.constant(
        np.random.rand(*((batch_size,) + image_shape)), dtype=tf.float32)
    model = saccader.Saccader(config)
    _, locations_t, _, _ = model(
        images, num_times=num_times, is_training=is_training,
        policy=policy)
    init_op = model.init_op
    self.evaluate(init_op)
    locations_t_ = self.evaluate(locations_t)
    # Locations should be different across time.
    print("HERE")
    print(np.abs(locations_t_[0] - locations_t_[1]).mean())
    print(locations_t_[0])
    print(locations_t_[1])
    print(policy)
    print(is_training)
    self.assertNotAlmostEqual(
        np.abs(locations_t_[0] - locations_t_[1]).mean(), 0.)

  @parameterized.named_parameters(
      *_get_test_cases()
  )
  def test_reuse_model(self, policy, is_training):
    config = saccader_config.get_config()
    num_times = 2
    image_shape = (224, 224, 3)
    num_classes = 10
    config.num_classes = num_classes
    config.num_times = num_times
    batch_size = 3
    images1 = tf.constant(
        np.random.rand(*((batch_size,) + image_shape)), dtype=tf.float32)
    model = saccader.Saccader(config)
    logits1 = model(images1, num_times=num_times, is_training=is_training,
                    policy=policy)[0]
    num_params = len(tf.all_variables())
    l2_loss1 = tf.losses.get_regularization_loss()
    # Build twice with different num_times and different batch size.
    images2 = tf.constant(
        np.random.rand(*((batch_size - 1,) + image_shape)), dtype=tf.float32)
    logits2 = model(images2, num_times=num_times+1, is_training=is_training,
                    policy=policy)[0]
    l2_loss2 = tf.losses.get_regularization_loss()

    # Ensure variables are reused.
    self.assertLen(tf.all_variables(), num_params)
    init = tf.global_variables_initializer()
    self.evaluate(init)
    logits1, logits2 = self.evaluate((logits1, logits2))
    l2_loss1, l2_loss2 = self.evaluate((l2_loss1, l2_loss2))
    np.testing.assert_almost_equal(l2_loss1, l2_loss2, decimal=9)

if __name__ == "__main__":
  tf.test.main()
