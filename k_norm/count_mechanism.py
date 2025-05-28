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

"""Induced norm count mechanism."""

import numpy as np
from scipy import special
from scipy import stats

from k_norm import sum_mechanism


def sample_orthant_num_positive(eulerian_numbers, k):
  """Samples an orthant class proportional to its total volume.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer l_0 bound.

  Returns:
    An integer j for the number of positive entries in the orthant.
  """
  d = len(eulerian_numbers) - 1
  eulerian_row_sums = np.sum(eulerian_numbers[:, :k], axis=1)
  factorials = special.factorial(np.arange(d + 1))
  weights = np.divide(
      np.multiply(eulerian_row_sums, eulerian_row_sums[::-1]),
      np.multiply(factorials, factorials[::-1]),
  )
  normalized_weights = weights / np.sum(weights)
  return np.random.choice(
      d + 1, p=np.array(normalized_weights, dtype='float64')
  )


def sample_from_orthant(eulerian_numbers, num_positive, k):
  """Samples a count ball orthant with num_positive positive entries.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    num_positive: Integer number of positive entries in orthant.
    k: Integer l_0 bound.

  Returns:
    A uniform random sample from an orthant of the count ball with num_positive
    positive entries.
  """
  d = len(eulerian_numbers) - 1
  if num_positive == 0:
    sample = -np.abs(sum_mechanism.sample_sum_ball(eulerian_numbers, k))
    return sample
  if num_positive == d:
    sample = np.abs(sum_mechanism.sample_sum_ball(eulerian_numbers, k))
    return sample
  cross_section_index = stats.beta.rvs(a=num_positive, b=d - num_positive + 1)
  sample = np.zeros(d)
  j_minus_sample = -np.abs(
      sum_mechanism.sample_sum_ball(
          eulerian_numbers[: (d - num_positive) + 1], k
      )
  )
  sample[num_positive:] = (1 - cross_section_index) * j_minus_sample
  if num_positive == 1:
    sample[0] = cross_section_index
  else:
    w_weights = np.zeros(num_positive + 1)
    factorial_term = np.power(
        cross_section_index, num_positive - 1
    ) / special.factorial(num_positive - 1)
    w_weights[:num_positive] = factorial_term * np.sum(
        eulerian_numbers[num_positive - 1, : k - 1]
    )
    # if num_positive > k, we may sample from the l_1 = k hyperplane
    if num_positive > k:
      w_weights[num_positive] = (
          factorial_term * k * eulerian_numbers[num_positive - 1, k - 1]
      )
    normalized_w_weights = w_weights / np.sum(w_weights)
    cross_section_face_idx = (
        np.random.choice(
            num_positive + 1, p=np.array(normalized_w_weights, dtype='float64')
        )
        + 1
    )
    if cross_section_face_idx > num_positive:
      sigma = sum_mechanism.sample_permutation_with_ascents(
          eulerian_numbers[:num_positive], k - 1
      )
      fundamental_simplex_sample = sum_mechanism.sample_fundamental_simplex(
          num_positive - 1
      )
      permuted_fundamental_simplex_sample = fundamental_simplex_sample[
          sigma - 1
      ]
      j_plus_sample = np.abs(
          sum_mechanism.phi(permuted_fundamental_simplex_sample)
      )
      sample[num_positive - 1] = cross_section_index * (
          k - np.sum(j_plus_sample)
      )
    else:
      j_plus_sample = np.abs(
          sum_mechanism.sample_sum_ball(eulerian_numbers[:num_positive], k - 1)
      )
      sample[num_positive - 1] = cross_section_index
    sample[: num_positive - 1] = cross_section_index * j_plus_sample
  return sample[np.random.permutation(d)]


def sample_count_ball(eulerian_numbers, k):
  """Returns a uniform sample from the induced norm ball for count.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer l_0 bound.
  """
  num_positive = sample_orthant_num_positive(eulerian_numbers, k)
  return sample_from_orthant(eulerian_numbers, num_positive, k)


def count_mechanism(vector, eulerian_numbers, k, epsilon, num_samples):
  """Returns a sample from the epsilon-DP induced norm mechanism for sum_mechanism.

  Args:
    vector: The output will be a noisy version of Numpy array vector.
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: l_0 bound.
    epsilon: The output will be epsilon-DP for float epsilon. Assumes vector is
      1-sensitive with respect to the sum ball.
    num_samples: Number of samples to return.

  Returns:
    A sample from the K-norm mechanism, as described in Section 4 of
    https://arxiv.org/abs/0907.3754, instantiated with the norm induced for
    Count.
  """
  d = len(vector)
  radii = np.random.gamma(shape=d + 1, scale=1 / epsilon, size=num_samples)
  samples = np.asarray([sample_count_ball(eulerian_numbers,
                                          k) for _ in range(num_samples)])
  noises = radii[:, np.newaxis] * samples
  return vector + noises
