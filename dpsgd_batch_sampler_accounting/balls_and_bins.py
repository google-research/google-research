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

"""Monte-Carlo estimate based accounting for Ball-and-Bins Sampler.

The source in this file is based on the following paper:

Title: Balls-and-Bins Sampling for DP-SGD
Authors: Lynn Chua, Badih Ghazi, Charlie Harrison, Ethan Leeman, Pritish Kamath,
         Ravi Kumar, Pasin Manurangsi, Amer Sinha, Chiyuan Zhang
Link: https://arxiv.org/abs/2412.16802

For the Balls and Bins batch sampler, the analysis is performed using Monte
Carlo sampling to get an estimate and an upper confidence bound on the hockey
stick divergence. Additionally a lower bound is obtained using ideas similar to
that for the Shuffle batch sampler in `dpsgd_bounds.py`.
"""

import enum
import math
from typing import Sequence

import numpy as np
from scipy import special
from scipy import stats

from dpsgd_batch_sampler_accounting import dpsgd_bounds
from dpsgd_batch_sampler_accounting import monte_carlo_estimator as mce


class MaxOfGaussians:
  """Class for max of i.i.d. Gaussian random variables.

  M_{sigma, dim} is the random variable sampled as max(X_1, ... , X_dim)
  where each X_i is drawn from N(0, sigma^2), that is Gaussian with standard
  deviation sigma. We use X_{sigma} to denote the random variable drawn from
  N(0, sigma^2).

  Attributes:
    sigma: The standard deviation of each Gaussian random variable.
    dim: The number of Gaussian random variables.
  """

  def __init__(self, sigma, dim):
    if sigma <= 0:
      raise ValueError(f'sigma must be positive. Found {sigma=}')
    if dim < 1:
      raise ValueError(f'dim must be a positive integer. Found {dim=}')
    self.sigma = sigma
    self.dim = dim

  def logcdf(self, x):
    """Returns log of the cumulative density function of M_{sigma, dim}.

    Pr[M_{sigma, dim} <= x] = Pr[X_{sigma} <= x]^dim

    Args:
      x: The input value at which to compute the CDF.
    Returns:
      The log of the probability that M_{sigma, dim} <= x.
    """
    return self.dim * stats.norm.logcdf(x, scale=self.sigma)

  def sf(self, x):
    """Returns the survival function of M_{sigma, dim}.

    Pr[M_{sigma, dim} > x] = 1 - (1 - Pr[X_{sigma} > x])^dim

    Args:
      x: The input value at which to compute the SF.
    Returns:
      The probability that M_{sigma, dim} > x.
    """
    return -np.expm1(self.dim * np.log1p(-stats.norm.sf(x, scale=self.sigma)))

  def logsf(self, x):
    """Returns log of the SF of M_{sigma, dim}.

    Pr[M_{sigma, dim} > x] = 1 - (1 - Pr[X_{sigma} > x])^dim

    Args:
      x: The value of the return value.
    Returns:
      The log of the probability that M_{sigma, dim} > x.
    """
    return np.log(self.sf(x))

  def cdf(self, x):
    """Returns CDF of max of Gaussians."""
    return np.exp(self.logcdf(x))

  def ppf(self, q):
    """Returns inverse of the CDF of M_{sigma, dim}.

    Recall: Pr[M_{sigma, dim} <= x] = Pr[X_{sigma} <= x]^dim
    And hence, Pr[X_{sigma} <= x] = Pr[M_{sigma, dim} > x]^{1/dim}

    Args:
      q: The CDF value(s) of the return value.
    Returns:
      Values x such that Pr[M_{sigma, dim} <= x] = q.
    """
    cumulative_probs_1d = np.exp(np.log(q) / self.dim)
    return stats.norm.ppf(cumulative_probs_1d, scale=self.sigma)

  def isf(self, q):
    """Returns inverse SF of max of Gaussians.

    Recall: Pr[M_{sigma, dim} > x] = 1 - (1 - Pr[X_{sigma} > x])^dim
    And hence, Pr[X_{sigma} > x] = 1 - (1 - Pr[M_{sigma, dim} > x])^{1/dim}

    Args:
      q: The SF value(s) of the return value.
    Returns:
      Values x such that Pr[M_{sigma, dim} > x] = q.
    """
    survival_probs_1d = - np.expm1(np.log1p(-q) / self.dim)
    return stats.norm.isf(survival_probs_1d, scale=self.sigma)

  def rvs(
      self, size, min_value = -np.inf
  ):
    """Returns samples from M_{sigma, dim} conditioned on being >= min_value.

    Args:
      size: The shape of the return value. If scalar, then an array of that
        size is returned.
      min_value: The minimum value for sampling from conditional distribution.
    Returns:
      Samples from M_{sigma, dim} of shape `size`, each drawn from
      M_{sigma, dim} conditioned on being at least `min_value`.
    """
    size = (size,) if isinstance(size, int) else size

    # We use a heuristic to decide when to use `ppf` vs `isf` for sampling
    # depending on `min_value`. Namely, if the CDF of the single Gaussian
    # corresponding to `min_value` is less than 0.5, we use `ppf` else `isf`.
    if self.logcdf(min_value) / self.dim < math.log(0.5):
      min_value_cdf = self.cdf(min_value)
      # cdf_values below are uniformly random in (min_value_cdf, 1).
      cdf_values = (1 - min_value_cdf) * np.random.rand(*size) + min_value_cdf
      return self.ppf(cdf_values)
    else:
      min_value_sf = self.sf(min_value)
      # sf_values below are uniformly random in (0, min_value_sf).
      sf_values = min_value_sf * np.random.rand(*size)
      return self.isf(sf_values)


def sample_gaussians_conditioned_on_max_value(
    sigma, dim, max_values
):
  """Samples from high dimensional Gaussian conditioned on max value.

  Args:
    sigma: The standard deviation of the Gaussian.
    dim: The dimension of the last axis of returned samples.
    max_values: The max value of the Gaussian.
  Returns:
    If max_values has shape (a_0, ..., a_t), the return value will have shape
    (a_0, ..., a_t, dim), where the value in coordinate
    (i_0, ..., i_t, i_{t+1}) is drawn from Gaussian with scale sigma,
    conditioned on the value being at most max_values[i_0, ..., i_t].
  """
  # max_values_cdf has the CDF corresponding to max_values.
  max_values_cdf = stats.norm.cdf(max_values, scale=sigma)
  # random_seed_values has shape (a_0, ..., a_t, dim), and the value at
  # coordinate (i_0, ..., i_t, i_{t+1}) is uniformly random in the interval
  # (0, CDF(max_values[i_0, ..., i_t])).
  random_seed_values = max_values_cdf[Ellipsis, np.newaxis] * np.random.rand(
      *(max_values.shape + (dim,))
  )
  return stats.norm.ppf(random_seed_values, scale=sigma)


def sample_gaussians_conditioned_on_min_max_value(
    sigma, min_max_value, dim, sample_size
):
  """Samples from high-dimensional Gaussian conditioned on being "out of box".

  Args:
    sigma: The standard deviation of each coordinate of the Gaussian
      distribution.
    min_max_value: The value such that the max value on each row of the returned
      output is at least this value.
    dim: The dimension of the Gaussian distribution.
    sample_size: The number of samples to sample.
  Returns:
    Samples of shape (sample_size, dim), where each row is sampled from the
    Gaussian distribution N(0, sigma^2 I_{dim}) conditioned on the event that
    the max value within the row is at least `min_max_value`.
  """
  # 1. Sample the max value for each sample; shape = (sample_size,)
  max_values = MaxOfGaussians(sigma, dim).rvs(
      size=sample_size, min_value=min_max_value
  )

  # 2. Sample coordinates with values at most the corresponding max value,
  # with shape = (sample_size, dim)
  samples = sample_gaussians_conditioned_on_max_value(sigma, dim, max_values)

  # 3. Insert the max values in a random coordinate within each row.
  row_indices = np.arange(sample_size)
  random_coords = np.random.randint(0, dim, size=(sample_size,))
  samples[row_indices, random_coords] = max_values

  return samples


def sample_order_statistics_from_uniform(
    dim, orders, size
):
  """Samples from order statistics of `dim` i.i.d. samples from U[0, 1].

  We use that the k-th top ranked element of a draw from U[0, 1]^n is
  distributed as Beta(n + 1 - k, k).
  The following process samples a draw from the joint distribution of
  the [k_1, k_2, ..., k_R] ranked elements from U[0, 1]^d.

  * Sample Z_1 ~ Beta(d - k_1 + 1, k_1). Set Y_1 = Z_1.
  * Sample Z_2 ~ Beta(d - k_2 + 1, k_2 - k_1). Set Y_2 = Z_2 * Y_1,
  * Sample Z_3 ~ Beta(d - k_3 + 1, k_3 - k_2). Set Y_3 = Z_3 * Y_2,
  * ... and so on ...
  * Sample Z_R ~ Beta(d - k_R + 1, k_R - k_{R-1}). Set Y_R = Z_R * Y_{R-1}.

  This is equivalent to: Y_k = prod_{i=1}^k Z_i, a "cumulative product".

  Args:
    dim: The total number of i.i.d. U[0, 1] random variables.
    orders: The sequence of order statistics to sample. It is assumed that
      these are sorted in increasing order, and all order values are integers
      in [1, dim]. For efficiency, this condition is not checked.
    size: The outer shape desired. E.g. if size = (a_0, a_1), then the return
      value will have shape (a_0, a_1, len(orders)). If size is scalar, then the
      return value will have shape (size, len(orders)).
  Returns:
    Samples of shape (size, len(orders)) [or size + (len(orders),)], where the
    last axis is sampled from the joint distribution of the [k_1, k_2, ..., k_R]
    ranked elements in a random draw from U[0, 1]^{dim}.
  """
  if isinstance(size, int):
    size = (size, len(orders))
  else:
    size = size + (len(orders),)
  # Sample Z_1, ..., Z_R
  orders = np.asarray(orders)
  random_seed = stats.beta.rvs(dim - orders + 1,
                               np.diff(np.insert(orders, 0, 0)),
                               size=size)
  # We use a numerically stable alternative to np.cumprod(random_seed, axis=-1).
  return np.exp(np.cumsum(np.log(random_seed), axis=-1))


def hockey_stick_divergence_from_privacy_loss(
    epsilon,
    privacy_losses,
):
  """Returns hockey stick divergence from privacy loss."""
  return np.maximum(0, - np.expm1(epsilon - privacy_losses))


def get_order_stats_seq_from_encoding(
    order_stats_encoding,
    num_steps_per_epoch
):
  """Returns the sequence of order statistics from the encoding.

  Encoding for order statistics must be None or a list of size that is a
  multiple of 3; the list is partitioned into tuples of size 3 and for each
  3-tuple of numbers (a, b, c), the orders of np.arange(a, b, c) are
  included. If `order_stats_encoding` is None or an empty list, then None is
  returned.

  Args:
    order_stats_encoding: The encoding of the order statistics.
    num_steps_per_epoch: The number of steps in a single epoch.
  """
  if not order_stats_encoding:
    # Handles the case of both `None` and empty list.
    return None

  if len(order_stats_encoding) % 3 != 0:
    raise ValueError(
        'order_stats_encoding must be a non-empty list of size that is a '
        f'multiple of 3. Found {order_stats_encoding}.'
    )
  order_stats_seq = np.concatenate([
      np.arange(a, b, c, dtype=int)
      for a, b, c in zip(
          order_stats_encoding[::3],
          order_stats_encoding[1::3],
          order_stats_encoding[2::3],
      )
  ])

  if (np.any(np.diff(order_stats_seq) < 1) or order_stats_seq[0] != 1
      or order_stats_seq[-1] > num_steps_per_epoch - 1):
    raise ValueError(
        '`order_stats_seq` must be sorted in increasing order, the first '
        'element should be 1 at last element should be at most '
        f'num_steps_per_epoch - 1. Found {order_stats_seq=}.'
    )
  return order_stats_seq


class AdjacencyType(enum.Enum):
  """Designates the type of adjacency for computing privacy loss distributions.

  ADD: the 'add' adjacency type specifies that the privacy loss distribution
    for a mechanism M is to be computed with mu_upper = M(D) and mu_lower =
    M(D'), where D' contains one more datapoint than D.
  REMOVE: the 'remove' adjacency type specifies that the privacy loss
    distribution for a mechanism M is to be computed with mu_upper = M(D) and
    mu_lower = M(D'), where D' contains one less datapoint than D.

  Note: We abuse notation and use 'ADD' and 'REMOVE' also to indicate the
  direction of adjacency in "Zeroing-Out" neighboring relation. Namely,
  'REMOVE' corresponds to replacing a real datapoint with the special symbol,
  and 'ADD' corresponds to the reverse.
  """
  ADD = 'ADD'
  REMOVE = 'REMOVE'


class BnBAccountant:
  """Privacy accountant for ABLQ with Balls and Bins Sampler.

  For REMOVE adjacency, consider the following upper and lower probability
  measures:
    upper_prob_measure P = sum_{t=1}^T N(e_t, sigma^2 * I) / T
    lower_prob_measure Q = N(0, sigma^2 * I)

    The privacy loss function L(x) is given as log(P(x) / Q(x)) is:
    L(x) = log(sum_{t=1}^T exp(x_t / sigma^2)) - log(T) - 1 / (2 * sigma^2)

  For ADD adjacency, the order of P and Q is reversed, and the loss function
  is negative of the above.

  The hockey stick divergence D_{e^eps}(P || Q) is given by
    E_{x ~ P} max{0, 1 - exp(epsilon - L(x))}.
  For P_t = N(e_t, sigma^2 * I) being the t-th component of the mixture P,
  the above expectation is same as E_{x ~ P_t} max{0, 1 - exp(epsilon - L(x))},
  by symmetry of all the P_t's. In particular, we can take t = 1.

  Furthermore, for any set E such that L(x) <= epsilon for all x not in E,
  the above expectation is same as
    P(E) * E_{P|E}  max{0, 1 - exp(epsilon - L(x))},
  where P|E is the distribution of P conditioned on the sample being in E.
  In particular,
  * for ADD adjacency, we use the set
    E_C := {x : max_t x_t <= C} for the smallest possible C.
  * for REMOVE adjacency, we use the set
    E_C := {x : max{x_1 - 1, max_{t > 1} x_t} >= C} for the largest possible C.

  Attributes:
    max_memory_limit: The maximum number of floats that can be sampled at once
      in memory. The Monte Carlo estimator breaks down the sample size into
      batches, each requiring at most `max_memory_limit` floats in memory.
    lower_bound_accountant: The accountant similar to the ShuffleAccountant for
      obtaining a lower bound on the hockey stick divergence.
  """

  def __init__(self, max_memory_limit = int(1e8)):
    self.max_memory_limit = max_memory_limit
    self.lower_bound_accountant = dpsgd_bounds.ShuffleAccountant(
        mean_upper=1.0, mean_lower=0.0,
    )

  def privacy_loss(
      self,
      sigma,
      samples,  # shape=(size, [num_epochs], num_steps_per_epoch)
      adjacency_type,
  ):
    """Returns privacy loss for samples.

    For REMOVE adjacency, the privacy loss for each x along the last axis of
    `samples` is given as:
    log(sum_{t=1}^T exp(x_t / sigma^2)) - log(T) - 1 / (2 * sigma^2)

    For ADD adjacency, the privacy loss is the negative of the above.

    Args:
      sigma: The scale of Gaussian noise.
      samples: The samples of shape (sample_size, num_steps_per_epoch) or
        (sample_size, num_epochs, num_steps_per_epoch).
      adjacency_type: The type of adjacency to use in computing privacy loss.
    """
    num_steps_per_epoch = samples.shape[-1]
    privacy_loss_per_epoch = (
        special.logsumexp(samples / sigma**2, axis=-1)
        - np.log(num_steps_per_epoch) - 1 / (2 * sigma**2)
    )
    if privacy_loss_per_epoch.ndim == 2:
      privacy_loss_per_epoch = np.sum(privacy_loss_per_epoch, axis=1)
    if adjacency_type == AdjacencyType.ADD:
      return - privacy_loss_per_epoch
    return privacy_loss_per_epoch

  def order_stats_privacy_loss(
      self,
      sigma,
      num_steps_per_epoch,
      samples,  # shape=(sample_size, [num_epochs], R)
      order_stats_weights,  # shape=(R,)
  ):
    """Returns order statistics upper bound on privacy loss for samples.

    The privacy loss for each x along the last axis of `samples` is given as:
    log(sum_{i=1}^{R} w_i * exp(x_i / sigma^2)) - log(T) - 1 / (2 * sigma^2)
    where w_1, ..., w_R are weights provided in `order_stats_weights`.

    This method does not take in the adjacency type, since that is encoded in
    the choice of `order_stats_weights`.

    Args:
      sigma: The scale of Gaussian noise.
      num_steps_per_epoch: The number of batches sampled in a single epoch.
      samples: The samples of shape (sample_size, R) or
        (sample_size, num_epochs, R).
      order_stats_weights: The weights associated with the order statistics, of
        shape (R,).
    Returns:
      The privacy loss of shape (sample_size,).
    """
    privacy_loss_per_epoch = (
        special.logsumexp(samples / sigma**2, axis=-1, b=order_stats_weights)
        - np.log(num_steps_per_epoch) - 1 / (2 * sigma**2)
    )
    if privacy_loss_per_epoch.ndim == 2:
      privacy_loss_per_epoch = np.sum(privacy_loss_per_epoch, axis=1)
    return privacy_loss_per_epoch

  def get_importance_threshold_value(
      self, sigma, epsilon, num_steps,
      adjacency_type = AdjacencyType.REMOVE,
  ):
    """Returns value of C such that L(x) <= eps for all x not in E_C.

    For ADD adjacency, the set E_C is given as:
    E_C := {x : max_t x_t <= C}
    Thus, we can choose C to be the smallest value such that:
    log(T) + 1/(2 * sigma^2) - log(exp(C / sigma^2)) <= epsilon.
    This is equivalent to:
    C >= 0.5 + (log(T) - epsilon) * sigma^2

    For REMOVE adjacency, the set E_C is given as:
    E_C := {x : max{x_1 - 1, max_{t > 1} x_t} >= C}
    Thus, we can choose C to be the largest value such that:
    log(exp((C+1)/sigma^2) + (T-1) * exp(C/sigma^2))
    <= log(T) + epsilon + 1/(2 * sigma^2).
    This is equivalent to:
    C / sigma^2 + log(exp(1/sigma^2) + T-1)
    <= log(T) + epsilon + 1/(2 * sigma^2).
    That is:
    C <= 0.5 + sigma^2 * (epsilon - log(1 + (exp(1/sigma^2) - 1) / T))
       = 0.5 + sigma^2 * (epsilon - log1p(expm1(1/sigma^2) / T))

    Args:
      sigma: The scale of Gaussian noise.
      epsilon: The epsilon value for which to compute the min max value.
      num_steps: The number of batches sampled.
      adjacency_type: The type of adjacency to use in computing threshold.
    Returns:
      The value of C such that L(x) <= epsilon for all x not in E_C.
    """
    if adjacency_type == AdjacencyType.ADD:
      return 0.5 + sigma**2 * (math.log(num_steps) - epsilon)
    else:  # Case: adjacency_type == AdjacencyType.REMOVE
      return (0.5 + sigma**2 * (
          epsilon - math.log1p(math.expm1(1 / sigma**2) / num_steps)))

  def sample_privacy_loss(
      self,
      sample_size,
      sigma,
      num_steps_per_epoch,
      num_epochs = 1,
      importance_threshold = None,
      adjacency_type = AdjacencyType.REMOVE,
  ):
    """Returns samples of privacy loss.

    Args:
      sample_size: The number of samples to return.
      sigma: The scale of Gaussian noise.
      num_steps_per_epoch: The number of batches sampled in a single epoch.
      num_epochs: The number of epochs.
      importance_threshold: The threshold value for importance sampling.
        For ADD adjacency, this is the value such that
        max_t x_t on each row of the returned output is at most
        `importance_threshold`.
        For REMOVE adjacency, this is the value such that
        max{ x_1 - 1, max_{t > 1} x_t} on each row of the returned output is at
        least `importance_threshold`.
        This value must be None when `num_epochs` is greater than 1.
      adjacency_type: The type of adjacency to use in computing privacy loss.
    Returns:
      Samples of shape (sample_size,).
    Raises:
      ValueError: If `min_max_value` is not None and `num_epochs` is not 1.
    """
    if importance_threshold is not None and num_epochs != 1:
      raise ValueError(
          'num_epochs must be 1 if importance_threshold is provided.'
          f'Found {importance_threshold=} and {num_epochs=}.'
      )
    if adjacency_type == AdjacencyType.ADD:
      if importance_threshold is None:
        samples = stats.norm.rvs(
            scale=sigma, size=(sample_size, num_epochs, num_steps_per_epoch)
        )
      else:
        # Guaranteed to be in the single epoch case at this point.
        max_values = importance_threshold * np.ones(sample_size)
        samples = sample_gaussians_conditioned_on_max_value(
            sigma, num_steps_per_epoch, max_values
        )
    else:  # Case: adjacency_type == AdjacencyType.REMOVE
      first_basis_vector = np.zeros(num_steps_per_epoch)
      first_basis_vector[0] = 1.0

      if importance_threshold is None:
        samples = first_basis_vector + stats.norm.rvs(
            scale=sigma, size=(sample_size, num_epochs, num_steps_per_epoch)
        )
      else:
        # Guaranteed to be in the single epoch case at this point.
        samples = sample_gaussians_conditioned_on_min_max_value(
            sigma, importance_threshold, num_steps_per_epoch, sample_size
        ) + first_basis_vector

    losses = self.privacy_loss(sigma, samples, adjacency_type)
    del samples  # Explicitly delete samples to free up memory.
    return losses

  def sample_order_stats_privacy_loss(
      self,
      sample_size,
      sigma,
      num_steps_per_epoch,
      order_stats_seq,
      num_epochs = 1,
      adjacency_type = AdjacencyType.REMOVE,
  ):
    """Returns samples of order statistics upper bounds on privacy loss.

    This method uses more efficient sampling of privacy loss, but yields
    pessimistic estimates. It relies on the following approach for sampling from
    a distribution that dominates the log-sum-exp of i.i.d. Gaussians, for
    order statistics [k_1, ... , k_R]:

    For REMOVE adjacency:
      Sample X_1 ~ N(1, sigma^2).
      Sample Y_1, Y_2, ... , Y_R ~ [k_1, ... , k_R] ranked elements from (T-1)
        samples from N(0, sigma^2).

      Use the following loss function to estimate hockey stick divergence:
      L(Y) = log(
          exp(X_1 / sigma^2)
          + sum_{t=1}^{R-1} (k_{t+1} - k_t) * exp(Y_t / sigma^2)
          + (T - k_R) * exp(Y_R / sigma^2)
      ) - log(T) - 1 / (2 * sigma^2)

    For ADD adjacency:
      Sample Y_1, Y_2, ... , Y_R ~ [k_1, ... , k_R] ranked elements from T
        samples from N(0, sigma^2).
      Use the following loss function to estimate hockey stick divergence:
      L(Y) = - log(
          k_1 * exp(Y_1 / sigma^2)
          + sum_{t=2}^{R} (k_t - k_{t-1}) * exp(Y_t / sigma^2)
      ) + log(T) + 1 / (2 * sigma^2)

    For multiple epochs, the following approach is used:
    For a dominating pair (P, Q), the hockey stick divergence corresponding to
    e-fold composition of a mechanism is given as:
      E_{x_1, ... , x_e ~ P} max{0, 1 - exp(epsilon - L(x_1) - ... - L(x_e))},
    where L(x) is the privacy loss at x corresponding to a single epoch.

    Args:
      sample_size: The number of samples to return.
      sigma: The scale of Gaussian noise.
      num_steps_per_epoch: The number of batches sampled in a single epoch.
      order_stats_seq: The sequence of orders to use for sampling order
        statistics, and computing upper bounds on the privacy loss. It is
        assumed that the orders are sorted in increasing order, the first value
        is 1 and the last value is at most num_steps_per_epoch - 1. This
        condition is not checked for efficiency reasons.
      num_epochs: The number of epochs.
      adjacency_type: The type of adjacency to use in computing privacy loss.
    Returns:
      Samples of privacy loss upper bounds of shape (sample_size,).
    """
    if adjacency_type == AdjacencyType.REMOVE:
      order_stats_weights = np.diff(np.append(order_stats_seq,
                                              num_steps_per_epoch))
      order_stats_weights = np.insert(order_stats_weights, 0, 1)
      first_coordinate_samples = stats.norm.rvs(
          loc=1, scale=sigma, size=(sample_size, num_epochs, 1))
      other_coordinates_samples = stats.norm.ppf(
          sample_order_statistics_from_uniform(num_steps_per_epoch - 1,
                                               order_stats_seq,
                                               (sample_size, num_epochs)),
          scale=sigma,
      )
      samples = np.concatenate(
          (first_coordinate_samples, other_coordinates_samples), axis=-1
      )
      loss_sign = 1.0
    else:  # Case: adjacency_type == AdjacencyType.ADD
      order_stats_weights = np.diff(np.insert(order_stats_seq, 0, 0))
      samples = stats.norm.ppf(
          sample_order_statistics_from_uniform(num_steps_per_epoch,
                                               order_stats_seq,
                                               (sample_size, num_epochs)),
          scale=sigma,
      )
      loss_sign = -1.0

    return loss_sign * self.order_stats_privacy_loss(
        sigma, num_steps_per_epoch, samples, order_stats_weights)

  def estimate_deltas(
      self,
      sigma,
      epsilons,
      num_steps_per_epoch,
      sample_size,
      num_epochs = 1,
      adjacency_type = AdjacencyType.REMOVE,
      use_importance_sampling = True,
  ):
    """Returns estimates of hockey stick divergence at various epsilons.

    Args:
      sigma: The scale of Gaussian noise.
      epsilons: A list of epsilon values for estimating hockey stick divergence.
      num_steps_per_epoch: The number of batches sampled in a single epoch.
      sample_size: The sample size to use for estimation.
      num_epochs: The number of epochs. When set to 1 (default), importance
        sampling is used to estimate the hockey stick divergence. But when set
        to a value greater than 1, importance sampling is not used, and the
        hockey stick divergence is estimated using naive sampling.
      adjacency_type: The type of adjacency to use in computing privacy loss
        distribution.
      use_importance_sampling: If True, then importance sampling is used to
        estimate the hockey stick divergence. Otherwise, naive sampling is used.
        This is only applicable when num_epochs is 1.
    Returns:
      A list of hockey stick divergence estimates corresponding to epsilons.
    """
    max_batch_size = int(self.max_memory_limit / num_steps_per_epoch)

    hsd_estimates = []
    for epsilon in epsilons:
      if num_epochs == 1 and use_importance_sampling:
        importance_threshold = self.get_importance_threshold_value(
            sigma, epsilon, num_steps_per_epoch, adjacency_type)
        if adjacency_type == AdjacencyType.ADD:
          # Use importance sampling for a single epoch, by conditioning
          # on the maximum value.
          scaling_factor = MaxOfGaussians(
              sigma, num_steps_per_epoch).cdf(importance_threshold)
        else:  # Case: adjacency_type == AdjacencyType.REMOVE
          # Use importance sampling for a single epoch, by conditioning
          # on a lower bound on the maximum value.
          scaling_factor = MaxOfGaussians(
              sigma, num_steps_per_epoch).sf(importance_threshold)

        sampler = lambda sample_size: self.sample_privacy_loss(
            sample_size,
            sigma,
            num_steps_per_epoch,
            num_epochs=1,
            importance_threshold=importance_threshold,  # pylint: disable=cell-var-from-loop
            adjacency_type=adjacency_type,
        )
      else:
        # No importance sampling is used for multiple epochs or when
        # specifically disabled by setting `use_importance_sampling = False`.
        sampler = lambda sample_size: self.sample_privacy_loss(
            sample_size,
            sigma,
            num_steps_per_epoch,
            num_epochs=num_epochs,
            adjacency_type=adjacency_type,
        )
        scaling_factor = 1.0

      f = lambda privacy_losses: hockey_stick_divergence_from_privacy_loss(
          epsilon, privacy_losses)  # pylint: disable=cell-var-from-loop

      hsd_estimates.append(mce.get_monte_carlo_estimate_with_scaling(
          sampler, f, scaling_factor, sample_size, max_batch_size
      ))
    return hsd_estimates

  def estimate_order_stats_deltas(
      self,
      sigma,
      epsilons,
      num_steps_per_epoch,
      sample_size,
      order_stats_seq = None,
      num_epochs = 1,
      adjacency_type = AdjacencyType.REMOVE,
  ):
    """Returns pessimistic estimates of HS divergence for multiple epochs.

    The pessimistic estimate is obtained by sampling from the order statistics
    upper bound on the privacy loss. See docstring for
    `sample_order_stats_privacy_loss` for more details.

    Args:
      sigma: The scale of Gaussian noise.
      epsilons: A list of epsilon values for estimating hockey stick divergence.
      num_steps_per_epoch: The number of batches sampled in a single epoch.
      sample_size: The sample size to use for estimation.
      order_stats_seq: The sequence of order statistics to use for upper bound.
        If an integer, then the orders of [1, 2, ..., order_stats_seq] are used.
        If None, then the orders of [1, 2, ..., num_steps_per_epoch - 1] are
        used.
      num_epochs: The number of epochs.
      adjacency_type: The type of adjacency to use in computing privacy loss
        distribution.
    Returns:
      A list of hockey stick divergence estimates corresponding to epsilons.
    Raises:
      ValueError: If `order_stats_seq` is not sorted in increasing order or if
        the first value is not 1 or if any of the values are not in
        [1, num_steps - 1].
    """
    if sigma <= 0:
      raise ValueError(f'sigma must be positive. Found {sigma=}')
    if sample_size < 1:
      raise ValueError(f'sample_size must be positive. Found {sample_size=}')

    # Interpret `order_stats_seq` as a numpy array and check for any errors.
    if order_stats_seq is None:
      if adjacency_type == AdjacencyType.ADD:
        order_stats_seq = np.arange(1, num_steps_per_epoch + 1, dtype=int)
      else:  # Case: adjacency_type = AdjacencyType.REMOVE
        order_stats_seq = np.arange(1, num_steps_per_epoch, dtype=int)
    elif isinstance(order_stats_seq, int):
      # For ADD adjacency, it is also okay for order_stats_seq to be
      # num_steps_per_epoch. But since this is not an interesting setting we do
      # not handle this case.
      if order_stats_seq < 1 or order_stats_seq > num_steps_per_epoch - 1:
        raise ValueError(
            'If an integer, order_stats_seq must be in '
            f'[1, num_steps_per_epoch - 1]. Found {order_stats_seq=}.'
        )
      order_stats_seq = np.arange(1, order_stats_seq + 1, dtype=int)
    else:
      order_stats_seq = np.asarray(order_stats_seq)
      if np.any(np.diff(order_stats_seq) < 1):
        raise ValueError(
            'If a sequence, `order_stats_seq` must be sorted in increasing '
            f'order. Found {order_stats_seq=}'
        )
      if adjacency_type == AdjacencyType.ADD:
        if (order_stats_seq[0] < 1 or
            order_stats_seq[-1] > num_steps_per_epoch):
          raise ValueError(
              'Under ADD adjacency, all orders must be in '
              f'[1, num_steps_per_epoch]. Found {order_stats_seq=}'
          )
      else:  # Case: adjacency_type = AdjacencyType.REMOVE
        if (order_stats_seq[0] != 1 or
            order_stats_seq[-1] > num_steps_per_epoch - 1):
          raise ValueError(
              'Under REMOVE adjacency the first order should be 1 and the '
              'last order should be at most num_steps_per_epoch - 1. Found '
              f'{order_stats_seq=}.'
          )

    # Determine maximum batch size for Monte-Carlo estimation.
    if num_epochs * (len(order_stats_seq) + 1) > self.max_memory_limit:
      raise ValueError(
          'The number of epochs and the number of order statistics are too '
          'large for the given memory limit.'
      )
    max_batch_size = int(
        self.max_memory_limit / (num_epochs * (order_stats_seq.shape[0] + 1))
    )

    scaling_factor = 1.0  # Since no importance sampling is used.

    sampler = lambda sample_size: self.sample_order_stats_privacy_loss(
        sample_size, sigma, num_steps_per_epoch, order_stats_seq, num_epochs,
        adjacency_type)

    hsd_estimates = []
    for epsilon in epsilons:
      f = lambda privacy_losses: hockey_stick_divergence_from_privacy_loss(
          epsilon, privacy_losses)  # pylint: disable=cell-var-from-loop
      hsd_estimates.append(mce.get_monte_carlo_estimate_with_scaling(
          sampler, f, scaling_factor, sample_size, max_batch_size
      ))
    return hsd_estimates

  def get_deltas_lower_bound(
      self,
      sigma,
      epsilons,
      num_steps_per_epoch,
      num_epochs = 1,
  ):
    """Returns lower bounds on delta values for corresponding epsilons."""
    return self.lower_bound_accountant.get_deltas(
        sigma, epsilons, num_steps_per_epoch, num_epochs
    )
