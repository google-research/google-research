# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utility functions related to privacy (taken from pCVR model code)."""

import math
from typing import Callable, Tuple

from absl import logging
import dp_accounting
import numpy as np


def compute_noise(
    target_epsilon,
    num_examples,
    epochs,
    steps_per_epoch,
    batch_size,
    delta=None,
):

  delta = delta or 1.0 / num_examples

  def _noise_to_epsilon_func(noise):
    return _compute_epsilon_dpsgd(
        batch_size, noise, epochs * steps_per_epoch, num_examples, delta
    )

  noise, epsilon = _epsilon_to_noise_func(
      target_epsilon, _noise_to_epsilon_func, precision=4
  )
  logging.info(
      (
          'DPSGD: set noise=%.4f, which achieves epsilon=%.2f '
          'target epsilon=%.2f) delta=%.4g'
      ),
      noise,
      epsilon,
      target_epsilon,
      delta,
  )
  return noise


def _compute_epsilon_dpsgd(
    batch_size,
    noise,
    steps,
    num_examples,
    delta,
    orders=list(np.linspace(1.1, 10.9, 99)) + list(range(11, 64)),
):
  """Computes DP epsilon.

  Args:
    batch_size: batch size of DPSGD.
    noise: the noise multiplier.
    steps: number of steps in DPSGD.
    num_examples: number of examples in the dataset.
    delta: the DP delta.
    orders: the Renyi orders to use.

  Returns:
    The DP epsilon.
  """
  if steps == 0:
    return 0.0
  q = batch_size / num_examples  # sampling ratio
  accountant = dp_accounting.rdp.RdpAccountant(orders)
  event = dp_accounting.SelfComposedDpEvent(
      dp_accounting.PoissonSampledDpEvent(
          q, dp_accounting.GaussianDpEvent(noise)
      ),
      steps,
  )
  accountant.compose(event)
  eps, best_order = accountant.get_epsilon_and_optimal_order(delta)

  if best_order == orders[0]:
    print('Consider decreasing the range of order.')
  elif best_order == orders[-1]:
    print('Consider increasing the range of order.')
  return eps


def _epsilon_to_noise_func(
    target_epsilon,
    noise_to_epsilon_func,
    precision = 2,
):
  """Binary search for a noise value to achieve the target epsilon.

  Args:
    target_epsilon: the target epsilon.
    noise_to_epsilon_func: the function that maps noise to epsilon. Should be a
      decreasing function.
    precision: the precision for the noise.

  Returns:
    The noise that approximately achieves the target epsilon, and the epsilon
    for the noise.
  """
  # If even smallest noise is enough to guarantee the target epsilon.
  smallest_noise = 1 / 10**precision
  largest_epsilon = noise_to_epsilon_func(smallest_noise)
  if largest_epsilon <= target_epsilon:
    return smallest_noise, largest_epsilon
  # Find the upper bound of the range, i.e. a noise value with epsilon smaller
  # than the target epsilon.
  largest_noise = 10
  while noise_to_epsilon_func(largest_noise) > target_epsilon:
    largest_noise *= 2
  return _epsilon_to_noise_func_helper(
      target_epsilon,
      noise_to_epsilon_func,
      1,
      largest_noise * 10**precision,
      precision,
  )


def _epsilon_to_noise_func_helper(
    target_epsilon, noise_to_epsilon_func, low_noise, high_noise, precision
):
  """A helper for epsilon_to_noise_func.

  Args:
    target_epsilon: the target epsilon.
    noise_to_epsilon_func: the function that maps noise to epsilon.
    low_noise: (low_noise / 10**precision) is the lower bound of the range for
      noise.
    high_noise: (high_noise / 10**precision) is the upper bound of the range for
      noise.
    precision: the precision for the noise.

  Returns:
    The noise that approximately achieves the target epsilon, and the epsilon
    for the noise.
  """
  if high_noise - low_noise <= 1:
    return high_noise / 10**precision, noise_to_epsilon_func(
        high_noise / 10**precision
    )
  mid = int(math.floor((high_noise + low_noise) / 2))
  mid_epsilon = noise_to_epsilon_func(mid / 10**precision)
  if mid_epsilon < target_epsilon:
    return _epsilon_to_noise_func_helper(
        target_epsilon, noise_to_epsilon_func, low_noise, mid, precision
    )
  elif mid_epsilon > target_epsilon:
    return _epsilon_to_noise_func_helper(
        target_epsilon, noise_to_epsilon_func, mid, high_noise, precision
    )
  return mid / 10**precision, mid_epsilon
