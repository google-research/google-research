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

# Lint as: python2, python3
"""Tests for learning.brain.research.deep_calibration.v2.calibration_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from uq_benchmark_2019 import calibration_lib
tf.enable_v2_behavior()


class CalibrationLibTest(tf.test.TestCase):

  def test_find_scaling_temperature(self):
    nsamples, nclasses = 10**4, 5
    logits = tf.random.uniform([nsamples, nclasses], 0, 300)
    target_temperature = 100
    scaled_logits = logits / target_temperature
    labels = tfp.distributions.Categorical(logits=scaled_logits).sample()
    temperature = calibration_lib.find_scaling_temperature(labels, logits)
    logging.info('temperature=%0.3f, target=%0.3f',
                 temperature, target_temperature)

    rel_error = (temperature - target_temperature) / target_temperature
    self.assertAlmostEqual(0, rel_error, places=1)

  def test_find_scaling_temperature_invalid_input(self):
    nsamples, nclasses = 10**4, 5
    logits = np.ones([nsamples, nclasses])
    labels = np.ones([nsamples])
    with self.assertRaises(ValueError):
      calibration_lib.find_scaling_temperature(labels[None], logits)
    with self.assertRaises(ValueError):
      calibration_lib.find_scaling_temperature(labels, logits[None])
    with self.assertRaises(ValueError):
      calibration_lib.find_scaling_temperature(labels, logits.T)

  def test_apply_scaling_temperature(self):
    batch_shape = [7, 11]
    nclasses = 5
    logits = tf.random.uniform(batch_shape + [nclasses], 0, 10)
    temperature = 3

    probs = tf.math.softmax(logits, axis=-1)
    expected = tf.math.softmax(logits / temperature, axis=-1)
    actual = calibration_lib.apply_temperature_scaling(temperature, probs)
    self.assertAllEqual(actual.shape, logits.shape)
    self.assertAllClose(expected, actual)


if __name__ == '__main__':
  tf.test.main()
