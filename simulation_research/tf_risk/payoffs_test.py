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

"""Tests for payoff functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # tf

from simulation_research.tf_risk import payoffs


class PayoffsTest(tf.test.TestCase):

  def test_call_payoff_is_correct(self):
    np.random.seed(0)
    num_samples = 8
    price = 100.0
    strike = 100.0

    price_samples = price * np.exp(
        np.random.normal(size=[num_samples]).astype(np.float32))

    payoff_samples = payoffs.call_payoff(price_samples, strike)

    with self.session() as session:
      payoff_samples_eval = session.run(payoff_samples)

    for i in range(num_samples):
      self.assertEqual(payoff_samples_eval[i],
                       (price_samples[i] -
                        strike) if price_samples[i] > strike else 0.0)

  def test_call_max_payoff_is_correct(self):
    np.random.seed(0)
    num_samples = 8
    prices = np.asarray([100.0, 120.0])
    num_dims = len(prices)
    strike = 117.0

    prices_samples = np.expand_dims(prices, 0) * np.exp(
        np.random.normal(size=[num_samples, num_dims]).astype(dtype=np.float32))

    payoff_samples = payoffs.call_max_payoff(prices_samples, strike)

    with self.session() as session:
      payoff_samples_eval = session.run(payoff_samples)

    for i in range(num_samples):
      offset = np.max(prices_samples[i]) - strike
      self.assertEqual(payoff_samples_eval[i], offset * (offset >= 0))

  def test_put_payoff_is_correct(self):
    np.random.seed(0)
    num_samples = 8
    price = 100.0
    strike = 100.0

    price_samples = price * np.exp(
        np.random.normal(size=[num_samples]).astype(np.float32))

    payoff_samples = payoffs.put_payoff(price_samples, strike)

    with self.session() as session:
      payoff_samples_eval = session.run(payoff_samples)

    for i in range(num_samples):
      self.assertEqual(payoff_samples_eval[i],
                       (strike -
                        price_samples[i]) if strike > price_samples[i] else 0.0)

  def test_put_up_in_payoff_is_correct(self):
    np.random.seed(0)
    num_samples = 8
    price = 100.0
    strike = 120.0
    barrier = 130.0

    price_samples = price * np.exp(
        np.random.normal(size=[num_samples]).astype(np.float32))
    running_max_samples = np.maximum(
        price_samples,
        price_samples *
        (1.0 + np.random.uniform(size=[num_samples]).astype(np.float32)))

    payoff_samples = payoffs.put_up_in_payoff(
        (price_samples, running_max_samples), strike, barrier)

    with self.session() as session:
      payoff_samples_eval = session.run(payoff_samples)

    for i in range(num_samples):
      self.assertEqual(payoff_samples_eval[i], (strike - price_samples[i]) if
                       (price_samples[i] < strike) and
                       (running_max_samples[i] >= barrier) else 0.0)

  def test_default_weighted_avg_price_is_correct(self):
    np.random.seed(0)
    num_samples = 16
    num_underlyings = 7
    prices = np.random.uniform(size=[num_samples, num_underlyings]).astype(
        np.float32)

    weighted_avg_prices = payoffs.weighted_avg_price(prices)

    with self.session() as session:
      weighted_avg_prices_eval = session.run(weighted_avg_prices)

    self.assertAllClose(weighted_avg_prices_eval, np.mean(prices, axis=-1))

  def test_weighted_avg_price_is_correct_with_same_weights(self):
    np.random.seed(0)
    num_samples = 16
    num_underlyings = 7
    prices = np.random.uniform(size=[num_samples, num_underlyings]).astype(
        np.float32)

    weighted_avg_prices = payoffs.weighted_avg_price(prices)

    with self.session() as session:
      weighted_avg_prices_eval = session.run(weighted_avg_prices)

    self.assertAllClose(weighted_avg_prices_eval, np.mean(prices, axis=-1))

  def test_weighted_avg_price_is_correct_with_specified_weights(self):
    np.random.seed(0)
    num_samples = 16
    num_underlyings = 7
    prices = np.random.uniform(size=[num_samples, num_underlyings]).astype(
        np.float32)
    weights = np.random.uniform(size=[num_underlyings]).astype(np.float32)

    weighted_avg_prices = payoffs.weighted_avg_price(prices, weights)

    with self.session() as session:
      weighted_avg_prices_eval = session.run(weighted_avg_prices)

    normalized_weights = np.expand_dims(weights / np.sum(weights), 0)
    self.assertAllClose(weighted_avg_prices_eval,
                        np.sum(prices * normalized_weights, axis=-1))

  def test_basket_call_payoff_is_correct(self):
    np.random.seed(0)
    num_samples = 8
    prices = np.asarray([100.0, 110.0, 105.0], dtype=np.float32)
    num_dims = len(prices)
    strike = 100.0
    weights = np.asarray([1.1, 0.8, 0.6], dtype=np.float32)

    prices_samples = np.expand_dims(
        prices, axis=0) * np.exp(
            np.random.normal(size=[num_samples, num_dims]).astype(np.float32))

    payoff_samples = payoffs.basket_call_payoff(prices_samples, strike, weights)

    with self.session() as session:
      payoff_samples_eval = session.run(payoff_samples)

    for i in range(num_samples):
      normalized_weights = weights / np.sum(weights)
      weighted_avg_price = np.sum(prices_samples[i] * normalized_weights)
      self.assertEqual(payoff_samples_eval[i],
                       (weighted_avg_price -
                        strike) if weighted_avg_price > strike else 0.0)

  def test_basket_put_payoff_is_correct(self):
    np.random.seed(0)
    num_samples = 8
    prices = np.asarray([100.0, 110.0, 105.0], dtype=np.float32)
    num_dims = len(prices)
    strike = 120.0
    weights = np.asarray([1.1, 0.8, 0.6], dtype=np.float32)

    prices_samples = np.expand_dims(
        prices, axis=0) * np.exp(
            np.random.normal(size=[num_samples, num_dims]).astype(np.float32))

    payoff_samples = payoffs.basket_put_payoff(prices_samples, strike, weights)

    with self.session() as session:
      payoff_samples_eval = session.run(payoff_samples)

    for i in range(num_samples):
      normalized_weights = weights / np.sum(weights)
      weighted_avg_price = np.sum(prices_samples[i] * normalized_weights)
      self.assertEqual(
          payoff_samples_eval[i],
          (strike - weighted_avg_price) if strike > weighted_avg_price else 0.0)


if __name__ == "__main__":
  tf.test.main()
