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

"""IndExp method for computing differentially private quantiles.

Algorithm 2 from the paper "Privacy-preserving Statistical Estimation with
Optimal Convergence Rates" by Smith (STOC 2011,
http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf) describes the
subroutine used to compute a single quantile. Theorem 3 from the paper ``Optimal
Differential Privacy Composition for Exponential Mechanisms and the Cost of
Adaptivity'' by Dong, Durfee, and Rogers (ICML 2020,
https://arxiv.org/pdf/1909.13830.pdf) describes the composition used for the
approximate DP variant of IndExp.
"""

import numpy as np
import scipy


def racing_sample(log_terms):
  """Numerically stable method for sampling from an exponential distribution.

  Args:
    log_terms: Array of terms of form log(coefficient) - (exponent term).

  Returns:
    A sample from the exponential distribution determined by terms. See
    Algorithm 1 from the paper "Duff: A Dataset-Distance-Based
    Utility Function Family for the Exponential Mechanism"
    (https://arxiv.org/pdf/2010.04235.pdf) for details; each element of terms is
    analogous to a single log(lambda(A_k)) - (eps * k/2) in their algorithm.
  """
  return np.argmin(
      np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)


def opt_comp_p(eps, t):
  """Returns p_{eps, t} for opt_comp_calculator.

  Args:
    eps: Privacy parameter epsilon.
    t: Exponent t.
  """
  return (np.exp(-t) - np.exp(-eps)) / (1.0 - np.exp(-eps))


def opt_comp_calculator(overall_eps, overall_delta, num_comps):
  """Returns the optimal per-composition eps for overall approx DP guarantee.

  Args:
    overall_eps: Desired overall privacy parameter epsilon.
    overall_delta: Desired overall privacy parameter delta.
    num_comps: Total number of compositions.

  Returns:
    eps_0 such that num_compositions eps_0-DP applications of the exponential
    mechanism will overall be (overall_eps, overall_delta)-DP, using the
    expression given in Theorem 3 of DDR20. This assumes that the composition is
    non-adaptive.
  """
  eps_i_range = np.arange(overall_eps / num_comps - 0.01, overall_eps, 0.01)
  num_eps_i = len(eps_i_range)
  max_eps = 0
  for eps_idx in range(num_eps_i):
    eps = eps_i_range[eps_idx]
    max_sum = 0
    for ell in range(num_comps + 1):
      t_ell_star = np.clip((overall_eps + (ell + 1) * eps) / (num_comps + 1),
                           0.0, eps)
      p_t_ell_star = opt_comp_p(eps, t_ell_star)
      term_sum = 0
      for i in range(num_comps + 1):
        term_sum += scipy.special.binom(num_comps, i) * np.power(
            p_t_ell_star, num_comps - i) * np.power(1 - p_t_ell_star, i) * max(
                np.exp(num_comps * t_ell_star -
                       (i * eps)) - np.exp(overall_eps), 0)
      if term_sum > max_sum:
        max_sum = term_sum
    if max_sum > overall_delta:
      return max_eps
    else:
      max_eps = eps
  return max_eps


def ind_exp(sorted_data, data_low, data_high, qs, divided_eps, swap):
  """Returns eps-differentially private collection of quantile estimates for qs.

  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower limit for any differentially private quantile output value.
    data_high: Upper limit for any differentially private quantile output value.
    qs: Increasing array of quantiles in [0,1].
    divided_eps: Privacy parameter epsilon for each estimated quantile. Assumes
      that divided_eps has been computed to ensure the desired overall privacy
      guarantee.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.
  """
  num_quantiles = len(qs)
  outputs = np.empty(num_quantiles)
  sorted_data = np.clip(sorted_data, data_low, data_high)
  data_size = len(sorted_data)
  sorted_data = np.concatenate(([data_low], sorted_data, [data_high]))
  data_gaps = sorted_data[1:] - sorted_data[:-1]
  for q_idx in range(num_quantiles):
    quantile = qs[q_idx]
    if swap:
      sensitivity = 1.0
    else:
      sensitivity = max(quantile, 1 - quantile)
    idx_left = racing_sample(
        np.log(data_gaps) +
        ((divided_eps / (-2.0 * sensitivity)) *
         np.abs(np.arange(0, data_size + 1) - (quantile * data_size))))
    outputs[q_idx] = np.random.uniform(sorted_data[idx_left],
                                       sorted_data[idx_left + 1])
  # Note that the outputs are already clipped to [data_low, data_high], so no
  # further clipping of outputs is necessary.
  return np.sort(outputs)
