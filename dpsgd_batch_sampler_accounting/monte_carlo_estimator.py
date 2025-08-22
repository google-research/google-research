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

"""Library for Monte-Carlo Estimation."""

import dataclasses
import math
from typing import Callable, Sequence

from dp_accounting import pld
import numpy as np


def get_batch_splits(
    sample_size, max_batch_size = None
):
  """Returns a list of batch sizes to use for Monte Carlo sampling.

  Args:
    sample_size: The total number of samples to generate.
    max_batch_size: An upper bound on all batch sizes. If None, all samples are
      assigned to a single batch.
  Returns:
    A list of batch sizes to use for sampling. If there is more than one batch,
    then all batches except the last one will have size `max_batch_size`, and
    the last batch will have size `sample_size % max_batch_size`. For example,
    if `sample_size = 70` and `max_batch_size = 30`, then the returned list will
    be [30, 30, 10].
  """
  if max_batch_size is None:
    return [sample_size]

  # Split the sample size into values each being at most max_batch_size.
  sample_size_sequences = [max_batch_size] * (sample_size // max_batch_size)
  if sample_size % max_batch_size > 0:
    sample_size_sequences.append(sample_size % max_batch_size)
  return sample_size_sequences


def bernoulli_kl_divergence(q, p):
  """Returns the KL divergence between two Bernoulli distributions.

  It is assumed that q and p are probabilities, that is, they are in [0, 1].
  This is not enforced for efficiency.

  Args:
    q: The probability of success of the first Bernoulli distribution.
    p: The probability of success of the second Bernoulli distribution.
  Returns:
    The KL divergence D(Ber(q) || Ber(p)).
  """
  if q == 0:
    if p == 1:
      return math.inf
    return -math.log1p(-p)  # equivalent to - log(1 - p)
  if q == 1:
    if p == 0:
      return math.inf
    return - math.log(p)
  if p == 0 or p == 1:
    return math.inf
  return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def find_p_above_kl_bound(q, kl_bound):
  """Returns the smallest p s.t. KL(q || p) >= kl_bound or 1 if none exists."""
  # We want to search for a `p` in the interval [q, 1].
  search_params = pld.common.BinarySearchParameters(q, 1)
  # inverse_monotone_function finds a value of p such that f(p) <= value
  # that is, - KL(q || p) <= - kl_bound, that is, KL(q || p) >= kl_bound.
  f = lambda p: - bernoulli_kl_divergence(q, p)
  optimal_p = pld.common.inverse_monotone_function(
      f, - kl_bound, search_params, increasing=False)
  return 1.0 if optimal_p is None else optimal_p


@dataclasses.dataclass
class Estimate:
  """Monte Carlo estimate from samples.

  Attributes:
    mean: The mean of the estimate.
    std: The standard deviation of the estimate.
    sample_size: The number of samples used to estimate the mean and std.
    scale: A worst case upper bound on the quantity whose mean is estimated.
      A smaller scale will result in tighter upper confidence bound.
  """
  mean: float
  std: float
  sample_size: int
  scale: float = 1.0

  @classmethod
  def from_values_and_scale(cls, values, scale = 1.0):
    """Returns an Estimate from values and scale."""
    if values.ndim != 1:
      raise ValueError(f'{values.shape=}; expected (sample_size,).')
    sample_size = values.shape[0]
    mean = np.mean(values)
    std = (
        np.sqrt(np.sum((values - mean)**2) / (sample_size * (sample_size - 1)))
    )
    return Estimate(mean, std, sample_size, scale)

  @classmethod
  def from_combining_independent_estimates(
      cls, estimates
  ):
    """Returns an Estimate from combining independent estimates.

    Given means m_1, ... , m_k and standard deviations s_1, ... , s_k of
    independent estimates with sample sizes n_1, ... , n_k, the mean and
    standard deviation of the combined estimate are given by:
    mean = sum(m_i * n_i) / sum(n_i)
    std = sqrt(sum(s_i**2 * n_i * (n_i - 1)) / (sum(n_i) * (sum(n_i) - 1)))

    Args:
      estimates: A sequence of estimates to combine.
    Returns:
      An Estimate from combining the given independent estimates.
    Raises:
      ValueError: If estimates is an empty list or have different scales.
    """
    if not estimates:
      raise ValueError('estimates must be non-empty.')

    scale = estimates[0].scale
    if any(est.scale != scale for est in estimates):
      raise ValueError(f'Estimates have different scales: {estimates=}')

    means = np.array([est.mean for est in estimates])
    stds = np.array([est.std for est in estimates])
    sample_sizes = np.array([est.sample_size for est in estimates])
    total_sample_size = np.sum(sample_sizes)

    mean = np.dot(means, sample_sizes / total_sample_size)
    std = np.linalg.norm(
        stds * np.sqrt(sample_sizes * (sample_sizes - 1))
        / math.sqrt(total_sample_size * (total_sample_size - 1))
    )
    return Estimate(mean, std, total_sample_size, scale)

  def get_upper_confidence_bound(self, error_prob):
    """Returns an upper confidence bound for the estimate.

    For `n` = sample_size, let `X_1, ... , X_n` be drawn from some distribution
    over [0, C]. And let q := (X_1 + ... + X_n) / n.

    Then for `p` such that KL(q/C || p/C) * n >= log(1 / error_prob), we have
    Pr[E[X] <= p] >= 1 - error_prob, where the probability is over the random
    draw of X_1, ... , X_n.

    Proof: For simplicity, assume that C = 1. Suppose the true mean of the
    distribution is `m`. By Chernoff's bound, we have that for any q
    Pr[(X_1 + ... + X_n) / n <= q] <= exp(-KL(q || m) n).
    For the returned value `p` to be smaller than `m`, the realized `q` must be
    such that KL(q || m) > KL(q || p) >= log(1 / error_prob) / n.
    Hence, probability of realizing such a `q` is at most error_prob.

    Args:
      error_prob: The desired probability of error that the true mean is not
      smaller than the returned value.
    Returns:
      An upper confidence bound for the estimate.
    """
    if self.scale == 0:
      return 0.0
    kl_bound = - np.log(error_prob) / self.sample_size
    return self.scale * find_p_above_kl_bound(self.mean / self.scale, kl_bound)


def get_monte_carlo_estimate_with_scaling(
    sampler,
    f,
    scaling_factor,
    sample_size,
    max_batch_size = None,
):
  """Monte-Carlo estimate of expected value of a function with scaling.

  Args:
    sampler: A method that returns a sample of specified size as numpy array.
      The sample can be multi-dimensional, but should have the first dimension
      length equal to specified sample size.
    f: A function that maps samples to corresponding function values. The return
      value of `f` must be 1-dimensional with shape (sample_size,).
    scaling_factor: A float value to multiply the `mean` and `std` by.
    sample_size: The number of times to repeat the sampling.
    max_batch_size: The maximum size to use in a single call to `sampler`.
      If None, then all samples are obtained in a single call to `sampler`.
  Returns:
    Estimate of E[f(x) * scaling] for x ~ sampler.
  Raises:
    RuntimeError: If the return value of `f` is not a one-dimensional vector
      of length equal to the length of the first dimension of its input.
  """
  if scaling_factor == 0:
    # No need to actually sample if scaling factor is 0.
    return Estimate(mean=0, std=0, sample_size=sample_size,
                    scale=scaling_factor)

  sample_size_sequences = get_batch_splits(sample_size, max_batch_size)

  f_values = []
  for size in sample_size_sequences:
    samples = sampler(size)
    f_values.append(f(samples))
    del samples  # Explicitly delete samples to free up memory.
  f_values = np.concatenate(f_values)

  if f_values.shape != (sample_size,):
    raise RuntimeError(f'{f_values.shape=}; expected ({sample_size},).')

  return Estimate.from_values_and_scale(f_values * scaling_factor,
                                        scaling_factor)
