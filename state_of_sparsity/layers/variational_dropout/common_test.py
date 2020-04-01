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

"""Tests for common variational dropout utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.testing.parameterized as parameterized
import numpy as np
import tensorflow.compat.v1 as tf

import state_of_sparsity.layers.variational_dropout as vd


# Parameter sets to test the helper functions on. Size of the first dimension
# of the parameters, size of the second dimension of the parameters, minimum
# value the parameters should take, maximum value the parameters should take.
HELPER_TEST = [(32, 80, -10, 10)]


@parameterized.parameters(HELPER_TEST)
class HelperTest(parameterized.TestCase):

  def setUp(self):  # pylint: disable=g-missing-super-call
    tf.reset_default_graph()

  def _get_weights(self, d, k, min_val, max_val):
    theta = tf.random_uniform(
        [d, k],
        min_val,
        max_val,
        dtype=tf.float32)
    log_sigma2 = tf.random_uniform(
        [d, k],
        min_val,
        max_val,
        dtype=tf.float32)
    return (theta, log_sigma2)

  def testHelper_ComputeLogAlpha(self, d, k, min_val, max_val):
    # Fix the random seed
    tf.set_random_seed(15)

    theta, log_sigma2 = self._get_weights(d, k, min_val, max_val)

    # Compute the log alpha values
    log_alpha = vd.common.compute_log_alpha(log_sigma2, theta, value_limit=None)

    sess = tf.Session()
    log_sigma2, log_alpha, theta = sess.run(
        [log_sigma2, log_alpha, theta])

    # Verify the output shapes
    self.assertEqual(log_sigma2.shape, (d, k))
    self.assertEqual(log_alpha.shape, (d, k))
    self.assertEqual(theta.shape, (d, k))

    # Verify the calculated values
    expected_log_alpha = log_sigma2 - np.log(np.power(theta, 2) + 1e-8)
    self.assertTrue(np.all(np.isclose(expected_log_alpha, log_alpha)))

  def testHelper_ComputeLogSigma2(self, d, k, min_val, max_val):
    # Fix the random seed
    tf.set_random_seed(15)

    theta, log_alpha = self._get_weights(d, k, min_val, max_val)

    # Compute the log \sigma^2 values
    log_sigma2 = vd.common.compute_log_sigma2(log_alpha, theta)

    sess = tf.Session()
    log_sigma2, log_alpha, theta = sess.run(
        [log_sigma2, log_alpha, theta])

    # Verify the output shapes
    self.assertEqual(log_sigma2.shape, (d, k))
    self.assertEqual(log_alpha.shape, (d, k))
    self.assertEqual(theta.shape, (d, k))

    # Verify the calculated values
    expected_log_sigma2 = log_alpha + np.log(np.power(theta, 2) + 1e-8)
    self.assertTrue(np.all(np.isclose(expected_log_sigma2, log_sigma2)))

  def testHelper_ComputeLogAlphaAndBack(self, d, k, min_val, max_val):
    theta, true_log_sigma2 = self._get_weights(d, k, min_val, max_val)

    # Compute the log alpha values
    log_alpha = vd.common.compute_log_alpha(
        true_log_sigma2, theta, value_limit=None)

    # Compute the log \sigma^2 values
    log_sigma2 = vd.common.compute_log_sigma2(log_alpha, theta)

    sess = tf.Session()
    true_log_sigma2, log_alpha, log_sigma2 = sess.run(
        [true_log_sigma2, log_alpha, log_sigma2])

    # Verify the output shapes
    self.assertEqual(true_log_sigma2.shape, (d, k))
    self.assertEqual(log_sigma2.shape, (d, k))
    self.assertEqual(log_alpha.shape, (d, k))

    # The calculated log \sigma^2 values should be the same as the
    # ones that we calculate through the log \alpha values
    for is_close in np.isclose(true_log_sigma2, log_sigma2).flatten():
      self.assertTrue(is_close)

  def testHelper_ThresholdLogAlphas(self, d, k, min_val, max_val):
    theta, log_sigma2 = self._get_weights(d, k, min_val, max_val)

    # Compute the log alpha values
    value_limit = 8.
    log_alpha = vd.common.compute_log_alpha(
        log_sigma2, theta, value_limit=value_limit)

    sess = tf.Session()
    log_alpha = sess.run(log_alpha)

    # Verify the output shapes
    self.assertEqual(log_alpha.shape, (d, k))

    # Verify that all log alpha values are within the valid range
    for value in log_alpha.flatten():
      self.assertLessEqual(value, value_limit)
      self.assertGreaterEqual(value, -value_limit)


if __name__ == "__main__":
  tf.test.main()
