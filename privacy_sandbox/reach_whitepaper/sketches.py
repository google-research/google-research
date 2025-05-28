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

"""Functions for computing samples of cardinality estimates via sketches."""

from typing import Callable

import numpy as np


_BATCH_SIZE = 10
_pre_computed_estimates = {}


def _invert_monotonic(
    f, lower = 0, precision = 0.001
):
  """Inverts monotonic function f."""
  f0 = f(lower)

  def inversion(y):
    """Inverted f."""
    if y < f0:
      raise ValueError(
          "Positive domain inversion error."
          f"f({lower}) = {f0}, but {y} was requested."
      )
    left = lower
    probe = 1
    while f(probe) < y:
      left = probe
      probe *= 2
    right = probe
    mid = (right + left) / 2
    while right - left > precision:
      f_mid = f(mid)
      if f_mid > y:
        right = mid
      else:
        left = mid
      mid = (right + left) / 2
    return mid

  return inversion


def estimated_cardinality(
    count, register_probabilities
):
  """Estimate cardinality given true cardinality and probability distribution over the registers."""

  def estimate(n):
    """This function estimates the expected number of non-empty registers.

    The formula used is sum_{i=1}^n 1 - (1 - p_i)^n where p_i is the
    probability of the $i$th register to be chosen.

    Args:
      n: The number of users stored in the sketch.

    Returns:
      The expected number of non-empty registers.
    """
    return np.sum(-np.expm1(n * np.log1p(-np.asarray(register_probabilities))))

  return _invert_monotonic(estimate, lower=0, precision=1e-7)(count)


def sample_n_non_empty_registers(
    true_cardinality,
    window_size,
    n_repetitions,
    register_probabilities,
    noiser,
):
  """Compute the number of non-empty registers.

  Args:
    true_cardinality: The true number of users.
    window_size: The number of days in the window.
    n_repetitions: The number of samples to generate.
    register_probabilities: The probability distribution over the registers.
    noiser: The noiser function.

  Returns:
    The list of the number of non-empty registers for each repetition.
  """
  true_cardinality_per_day = true_cardinality // window_size
  true_cardinality_reminder = true_cardinality % window_size

  # registers is an array of shape (n_repetitions, window_size,
  # len(register_probabilities)), where registers[i, j, k] is 1 iff
  # the k-th register is non-empty on the j-th day for the $i$th
  # repetition.
  registers = np.random.multinomial(
      true_cardinality_per_day,
      register_probabilities,
      size=(n_repetitions, window_size),
  )
  registers_additions = np.random.multinomial(
      true_cardinality_reminder, register_probabilities, size=n_repetitions
  )
  # We add all not distributed per day samples to the first day of the segment.
  registers[:, 0, :] += registers_additions

  registers = registers + noiser(
      (n_repetitions, window_size), len(register_probabilities)
  )
  # non_empty_registers[i, j, k] is True if the k-th register was set to 1
  # before noise addition.
  denoised_registers = registers >= 0.5
  # merged_non_empty_registers is an array of shape (n_repetitions,
  # len(register_probabilities)), where merged_non_empty_registers[i, j] is the
  # sketch obtained by merging all registers for the i-th repetition.
  merged_non_empty_registers = denoised_registers.sum(axis=1)
  # non_empty_registers[i, j] is True if the register j is non-empty for the
  # i-th repetition.
  non_empty_registers = merged_non_empty_registers >= 1
  # number of non-empty registers per repetition.
  return non_empty_registers.sum(axis=1)


def sample_estimated_cardinalities(
    true_cardinality,
    window_size,
    n_repetitions,
    register_probabilities,
    noiser,
):
  """Get samples of estimated cardinalities given the true cardinality."""
  n_non_empty_registers = sample_n_non_empty_registers(
      true_cardinality,
      window_size,
      n_repetitions,
      register_probabilities,
      noiser,
  )
  result = []
  for count in n_non_empty_registers:
    result.append(estimated_cardinality(count, register_probabilities))
  return result


def sample_estimated_cardinality(
    count,
    window_size,
    noiser,
    register_probabilities,
):
  """Return a sample from the estimates using sketch for given count.

  Note that every time it is called it attempts to get a precomputed value,
  but if all precomputed values are exhausted it generate another batch.

  Args:
    count: The true number of users.
    window_size: The number of days in the window.
    noiser: The noiser function.
    register_probabilities: The probability distribution over the registers.
  """
  key = count, window_size, id(noiser), id(register_probabilities)
  if not _pre_computed_estimates.get(key, []):
    _pre_computed_estimates[key] = sample_estimated_cardinalities(
        count, window_size, _BATCH_SIZE, register_probabilities, noiser
    )
  return _pre_computed_estimates[key].pop()
