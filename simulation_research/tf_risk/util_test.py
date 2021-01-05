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

"""Tests for util.py (e.g. analytical option pricing formula)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf  # tf

from simulation_research.tf_risk import util


class UtilTest(tf.test.TestCase):

  def assertAllAlmostEqual(self, a, b, delta=1e-7):
    self.assertEqual(a.shape, b.shape)
    a = a.flatten()
    b = b.flatten()
    self.assertLessEqual(np.max(np.abs(a - b)), delta)

  def test_std_estimate_works_with_tensors(self):
    num_dims = 16
    mean_est = tf.ones(num_dims)
    mean_sq_est = tf.ones(num_dims)
    stddev_est = util.stddev_est(mean_est, mean_sq_est)

    with self.test_session() as session:
      stddev_est_eval = session.run(stddev_est)

    self.assertAllEqual(stddev_est_eval, np.zeros(num_dims))

  def test_std_estimate_works_with_arrays(self):
    num_dims = 16
    mean_est = np.ones(num_dims)
    mean_sq_est = np.ones(num_dims)
    stddev_est = util.stddev_est(mean_est, mean_sq_est)

    self.assertAllEqual(stddev_est, np.zeros(num_dims))

  def test_half_clt_conf_interval_is_correct(self):
    confidence_level = 0.95
    num_samples = 400

    # Test scaler float value.
    stddev = 2.0
    correct_value = stddev / np.sqrt(num_samples) * 1.959963984540
    conf_interval_half_width = util.half_clt_conf_interval(
        confidence_level, num_samples, stddev)

    self.assertAlmostEqual(correct_value, conf_interval_half_width)

    # Test array float values.
    stddev = np.array([2.0, 1.0])
    correct_value = stddev / np.sqrt(num_samples) * 1.959963984540
    conf_interval_half_width = util.half_clt_conf_interval(
        confidence_level, num_samples, stddev)

    self.assertAllAlmostEqual(correct_value, conf_interval_half_width)

  def test_half_clt_conf_interval_with_zero_stdev(self):
    confidence_level = 0.95
    num_samples = 400

    # Test zero scaler.
    stddev = 0.0
    correct_value = stddev / np.sqrt(num_samples) * 1.959963984540
    conf_interval_half_width = util.half_clt_conf_interval(
        confidence_level, num_samples, stddev)

    self.assertAlmostEqual(correct_value, conf_interval_half_width)

    # Test array with zero elements.
    stddev = np.array([2.0, 0.0])
    correct_value = stddev / np.sqrt(num_samples) * 1.959963984540
    conf_interval_half_width = util.half_clt_conf_interval(
        confidence_level, num_samples, stddev)

    self.assertAllAlmostEqual(correct_value, conf_interval_half_width)

  def test_half_clt_conf_interval_unsupported_type(self):
    confidence_level = 0.95
    num_samples = 400
    # Integer stddev not supported.
    stddev = 2

    with self.assertRaises(TypeError):
      util.half_clt_conf_interval(confidence_level, num_samples, stddev)

  def test_running_mean_estimate_is_correct(self):
    np.random.seed(0)
    num_dims = 8
    mean_est = np.random.normal(size=[num_dims])
    batch_est = np.random.normal(size=[num_dims])

    num_samples = 128
    batch_size = 16

    updated_mean_est = util.running_mean_estimate(
        mean_est, batch_est, num_samples, batch_size)

    mean_est_fraction = float(num_samples) / float(num_samples + batch_size)
    batch_est_fraction = float(batch_size) / float(num_samples + batch_size)

    self.assertAllAlmostEqual(
        updated_mean_est,
        mean_est_fraction * mean_est +  batch_est_fraction * batch_est)

  def test_call_put_parity(self):
    current_price = 100.0
    interest_rate = 0.05
    vol = 0.2
    strike = 120.0
    maturity = 1.0

    call_price = util.black_scholes_call_price(current_price, interest_rate,
                                               vol, strike, maturity)
    put_price = util.black_scholes_put_price(current_price, interest_rate, vol,
                                             strike, maturity)
    total_price = current_price - strike * tf.exp(- interest_rate * maturity)

    with self.test_session() as session:

      call_price_eval = session.run(call_price)
      put_price_eval = session.run(put_price)
      total_price_eval = session.run(total_price)

    self.assertGreater(call_price_eval, 0.0)
    self.assertGreater(put_price_eval, 0.0)
    self.assertAlmostEqual(call_price_eval - put_price_eval, total_price_eval,
                           delta=1e-5)

  def test_barrier_parity(self):
    current_price = 100.0
    interest_rate = 0.05
    vol = 0.2
    strike = 120.0
    barrier = 150.0
    maturity = 1.0

    put_up_in_price = util.black_scholes_up_in_put_price(current_price,
                                                         interest_rate,
                                                         vol, strike, barrier,
                                                         maturity)
    put_up_out_price = util.black_scholes_up_out_put_price(current_price,
                                                           interest_rate,
                                                           vol, strike, barrier,
                                                           maturity)
    put_price = util.black_scholes_put_price(current_price, interest_rate, vol,
                                             strike, maturity)

    with self.test_session() as session:

      put_up_in_price_eval = session.run(put_up_in_price)
      put_up_out_price_eval = session.run(put_up_out_price)
      put_price_eval = session.run(put_price)

    self.assertGreater(put_up_in_price_eval, 0.0)
    self.assertGreater(put_up_out_price_eval, 0.0)
    self.assertAlmostEqual(put_up_in_price_eval + put_up_out_price_eval,
                           put_price_eval, delta=1e-5)

if __name__ == "__main__":
  tf.test.main()
