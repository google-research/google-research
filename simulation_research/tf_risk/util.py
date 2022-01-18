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

"""Pricing functions and other useful functions.

Tensorflow version of numerically useful functions for derivative pricing:
cumulative density function of Gaussian distribution, Black-Scholes formulas
for vanilla options and extensions to first generation exotics.

For reference see http://www.cmap.polytechnique.fr/~touzi/Poly-MAP552.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf

from typing import Union, TypeVar

NumpyArrayOrFloat = Union[np.ndarray, float]
NumpyArrayOrFloatTypeVar = TypeVar("NumpyArrayOrFloatTypeVar", np.ndarray,
                                   float)
TensorOrFloat = Union[tf.Tensor, float]
TensorOrNumpyArray = Union[tf.Tensor, np.ndarray]


def gaussian_cdf(x):
  """Gaussian cumulative density function."""
  half = tf.constant(0.5, dtype=x.dtype)
  one = tf.constant(1.0, dtype=x.dtype)
  two = tf.constant(2.0, dtype=x.dtype)
  return half * (one + tf.math.erf(x / tf.sqrt(two)))


def stddev_est(mean_est,
               mean_sq_est):
  """Standard deviation estimate from estimates of mean and mean of square."""
  if isinstance(mean_est, tf.Tensor):
    return tf.sqrt(mean_sq_est - mean_est**2)
  return np.sqrt(mean_sq_est - mean_est**2)


def half_clt_conf_interval(confidence_level, num_samples,
                           stddev
                          ):
  """Half-width of Central Limit Theorem confidence interval."""
  target_quantile = (1.0 - float(confidence_level)) * 0.5
  scale = stddev / np.sqrt(float(num_samples))
  conf_interval = scipy.stats.norm.isf(target_quantile, scale=scale)
  if isinstance(stddev, np.ndarray):
    conf_interval[scale == 0.0] = 0.0
  elif isinstance(stddev, float):
    conf_interval = conf_interval if stddev != 0.0 else 0.0
  else:
    raise TypeError("Unsupported type for stddev. Expected numpy.ndarray or "
                    "float, received %s" % type(stddev))
  return conf_interval


def running_mean_estimate(estimate,
                          batch_estimate,
                          num_samples,
                          batch_size):
  """Update a running mean estimate with a mini-batch estimate."""
  tot_num_samples = float(num_samples + batch_size)
  estimate_fraction = float(num_samples) / tot_num_samples
  batch_fraction = float(batch_size) / tot_num_samples
  return estimate * estimate_fraction + batch_estimate * batch_fraction


def black_scholes_call_price(current_price,
                             interest_rate,
                             vol,
                             strike,
                             maturity):
  """Analytical price of a European call under the Black-Scholes model."""
  d_1 = (
      tf.log(current_price / strike) +
      (interest_rate + vol * vol * 0.5) * maturity)
  d_1 /= vol * tf.sqrt(maturity)
  d_2 = d_1 - vol * tf.sqrt(maturity)
  return (current_price * gaussian_cdf(d_1) -
          strike * tf.exp(-interest_rate * maturity) * gaussian_cdf(d_2))


def black_scholes_put_price(current_price,
                            interest_rate,
                            vol, strike,
                            maturity):
  """Analytical price of a European put under the Black-Scholes model."""
  call_price = black_scholes_call_price(current_price, interest_rate, vol,
                                        strike, maturity)
  return (call_price - current_price +
          strike * tf.exp(-interest_rate * maturity))


def black_scholes_up_in_put_price(current_price,
                                  interest_rate,
                                  vol,
                                  strike,
                                  barrier,
                                  maturity):
  """Analytical price of a European up in put under the Black-Scholes model."""
  gamma = 1 - 2.0 * interest_rate / (vol**2)
  corrected_strike = strike * (current_price**2) / (barrier**2)
  return (current_price / barrier)**(gamma - 2) * black_scholes_put_price(
      current_price, interest_rate, vol, corrected_strike, maturity)


def black_scholes_up_out_put_price(current_price,
                                   interest_rate,
                                   vol,
                                   strike,
                                   barrier,
                                   maturity):
  """Analytical price of a European up out put under the Black-Scholes model."""
  up_in_put_price = black_scholes_up_in_put_price(current_price, interest_rate,
                                                  vol, strike, barrier,
                                                  maturity)
  put_price = black_scholes_put_price(current_price, interest_rate, vol, strike,
                                      maturity)
  return put_price - up_in_put_price
