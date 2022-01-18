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

"""Library of payoffs for some financial derivatives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from typing import Tuple, Optional, Union

TensorOrFloat = Union[tf.Tensor, float]
TensorOrNumpyArray = Union[tf.Tensor, np.ndarray]


def _positive_part(x):
  """Return the element-wise positive part of a tensor."""
  return tf.nn.relu(x)


def call_payoff(price, strike):
  """Element-wise call payoffs for a price scalar tensor and a given strike.

  If price and strike do not have the same shape, then the shapes should be
  compatible with a broadcast between price a
  Returns:
    A [num_samples] tensor containinnd strike (typically broadcast of
  strike onto the price tensor).
  The payoff is positive_part(price - strike).

  Args:
    price: typically a [num_samples] scalar tensor.
    strike: typically a scalar. g the element-wise call payoffs.
  """
  return _positive_part(price - tf.convert_to_tensor(strike, dtype=price.dtype))


def call_max_payoff(prices, strike):
  """Element-wise call of max payoffs for a prices tensor and a given strike.

  Price is typically a [num_samples, num_underlyings] tensor and strike
  a scalar.
  The payoff function is positive_part(max(prices, axis=-1) - strike).

  Args:
    prices: typically a [num_samples, num_underlyings] scalar tensor.
    strike: typically a scalar.

  Returns:
    A [num_samples] tensor containing the element-wise call of max payoffs.
  """
  return call_payoff(tf.reduce_max(prices, axis=-1, keep_dims=True), strike)


def put_payoff(price, strike):
  """Element-wise put payoffs for a price scalar tensor and a given strike.

  If price and strike do not have the same shape, then the shapes should be
  compatible with a broadcast between price and strike (typically broadcast of
  strike onto the price tensor).
  The payoff is positive_part(strike - price).

  Args:
    price: typically a [num_samples] scalar tensor.
    strike: typically a scalar.

  Returns:
    A [num_samples] tensor containing the element-wise put payoffs.
  """
  return _positive_part(tf.convert_to_tensor(strike, dtype=price.dtype) - price)


def put_up_in_payoff(price_and_running_max,
                     strike,
                     barrier):
  price, running_max = price_and_running_max
  barrier_crossed = tf.cast(running_max >= barrier, dtype=price.dtype)
  return barrier_crossed * put_payoff(price, strike)


def weighted_avg_price(prices,
                       weights = None
                      ):
  """Compute a weighted average of a vector of prices, defaults to price mean.

  Args:
    prices: a [..., num_underlyings] asset price tensor (at least 1d).
    weights: a [num_underlyings] weighting scheme tensor, if none the weights
      are uniform. Weights will be normalized by their sum.

  Returns:
    The weighted average of prices by weights.
  """
  prices = tf.convert_to_tensor(prices)

  if weights is None:
    weights = tf.ones(shape=tf.shape(prices)[-1], dtype=prices.dtype)
  else:
    weights = tf.convert_to_tensor(weights)
    price_shape = prices.get_shape()
    weight_shape = weights.get_shape()

    if price_shape.rank < 1:
      raise ValueError("prices' rank should be >= 1 but is %d." %
                       price_shape.rank)
    if weight_shape.rank != 1:
      raise ValueError("If specified, price weights should be of rank 1.")

    num_underlyings = price_shape.as_list()[-1]
    num_weights = weight_shape.as_list()[-1]

    if num_weights != num_underlyings:
      raise ValueError("%d price weights are given but prices' last dimension"
                       "is %d." % (num_weights, num_underlyings))

  weights /= tf.reduce_sum(weights)
  weights = tf.expand_dims(weights, 0)
  return tf.reduce_sum(prices * weights, axis=-1)


def basket_call_payoff(prices,
                       strike,
                       weights = None
                      ):
  """Apply the call payoff to a weighted average of the prices.

  Args:
    prices: a [..., num_underlyings] asset price tensor (at least 1d).
    strike: a scalar.
    weights: a [num_underlyings] weighting scheme tensor, if none the weights
      are uniform. Weights will be normalized by their sum.

  Returns:
    A tf.shape(prices)[:-1] tensor entailing the payoffs.
  """
  return call_payoff(weighted_avg_price(prices, weights), strike)


def basket_put_payoff(prices,
                      strike,
                      weights = None
                     ):
  """Apply the put payoff to a weighted average of the prices.

  Args:
    prices: a [..., num_underlyings] asset price tensor (at least 1d).
    strike: a scalar.
    weights: a [num_underlyings] weighting scheme tensor, if none the weights
      are uniform. Weights will be normalized by their sum.

  Returns:
    A tf.shape(prices)[:-1] tensor entailing the payoffs.
  """
  return put_payoff(weighted_avg_price(prices, weights), strike)
