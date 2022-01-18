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

"""CDP smooth sensitivity method for computing differentially private quantiles.

The smooth sensitivity method is described in
"Smooth Sensitivity and Sampling in Private Data Analysis" by Nissim,
Raskhodnikova, and Smith
(https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf). Details for
the CDP noise distribution appear in Section 3.1 of "Average-Case Averages:
Private Algorithms for Smooth Sensitivity and Mean Estimation" by Bun and
Steinke (NeurIPS 2019). Details for optimizing t, s, and sigma appear in
Section 3.1.1 of the same paper.
"""

import numpy as np

from dp_multiq import base
from dp_multiq import smooth_utils


def compute_triples(eps, ts):
  """Returns triples of form (t, log(s), sigma) for hyperparameter optimization.

  Args:
    eps: Privacy parameter epsilon.
    ts: Array of possible smooth sensitivity parameters.
  """
  triples = np.empty([len(ts), 3])
  for t_idx in range(len(ts)):
    t = ts[t_idx]
    triples[t_idx, 0] = t
    sigma = opt_sigma(eps, t)
    triples[t_idx, 2] = sigma
    triples[t_idx, 1] = -1.5 * (sigma**2) + np.log(eps - (t / sigma))
  return triples


def opt_sigma(eps, t):
  """Returns optimal sigma as detailed in Section 3.1.1 of Bun and Steinke.

  Args:
    eps: Privacy parameter epsilon.
    t: Smooth sensitivity parameter.
  """
  return np.real(np.roots([5 * eps / t, -5, 0, -1])[0])


def lln(sigma):
  """Returns a sample from the Laplace Log-Normal distribution.

  Args:
    sigma: Sigma parameter for the Laplace Log-Normal distribution.
  """
  return np.random.laplace() * np.exp(sigma * np.random.normal())


def csmooth(sorted_data, data_low, data_high, qs, divided_eps, ts):
  """Returns eps^2/2-CDP quantile estimates for qs.

  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower limit for any differentially private quantile output value.
    data_high: Upper limit for any differentially private quantile output value.
    qs: Increasing array of quantiles in [0,1].
    divided_eps: Privacy parameter epsilon. Assumes eps has already been divided
      so that the overall desired privacy guarantee is achieved.
    ts: Array of smooth sensitivity parameters, one for each q in qs.
  """
  sorted_data = np.clip(sorted_data, data_low, data_high)
  o = np.empty(len(qs))
  triples = compute_triples(divided_eps, ts)
  for i in range(len(qs)):
    t, log_s, sigma = triples[i]
    true_quantile_idx = base.quantile_index(len(sorted_data), qs[i])
    true_quantile_value = sorted_data[true_quantile_idx]
    laplace_log_normal_noise = lln(sigma)
    log_sensitivity = smooth_utils.compute_log_smooth_sensitivity(
        sorted_data, data_low, data_high, true_quantile_idx, t)
    noise = np.sign(laplace_log_normal_noise) * np.exp(
        log_sensitivity + np.log(np.abs(laplace_log_normal_noise)) - log_s)
    o[i] = true_quantile_value + noise
  o = np.clip(o, data_low, data_high)
  return np.sort(o)


def log_choose_triple_idx(triples, eps, log_sensitivities):
  """Returns triple (t, log_s, sigma) that minimizes noisy statistic variance.

  Args:
    triples: Array with entries of form (t, log_s, sigma).
    eps: Privacy parameter epsilon.
    log_sensitivities: Log(t smooth sensitivity) for each t in triples.
  """
  variances = np.empty(len(triples))
  for triple_idx in range(len(triples)):
    numerator = 2 * (np.exp(2 * log_sensitivities[triple_idx]))
    denominator = np.exp(-5 * (triples[triple_idx][2]**2)) * (
        (eps - (triples[triple_idx][0] / triples[triple_idx][2]))**2)
    variances[triple_idx] = numerator / denominator
  return np.argmin(variances)


def csmooth_tune_and_return_ts(sorted_data, data_low, data_high, qs,
                               divided_eps, log_t_low, log_t_high, num_t):
  """Returns ts minimizing variance for data and each q under ~eps^2/2-CDP.

  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower limit for any differentially private quantile output value.
    data_high: Upper limit for any differentially private quantile output value.
    qs: Increasing array of quantiles in [0,1].
    divided_eps: Privacy parameter epsilon. Assumes eps has already been divided
      so that the overall desired privacy guarantee is achieved.
    log_t_low: Tuning range for t has lower bound 10^(log_t_low).
    log_t_high: Tuning range for t has upper bound 10^(log_t_high).
    num_t: Number of logarithmically spaced t used to populate tuning range.
  """
  sorted_data = np.clip(sorted_data, data_low, data_high)
  triples = compute_triples(divided_eps,
                            np.logspace(log_t_low, log_t_high, num_t))
  num_qs = len(qs)
  ts = np.empty(num_qs)
  for i in range(num_qs):
    true_quantile_idx = base.quantile_index(len(sorted_data), qs[i])
    log_sensitivities = np.zeros(len(triples))
    for triple_idx in range(len(triples)):
      t = triples[triple_idx, 0]
      log_sensitivities[
          triple_idx] = smooth_utils.compute_log_smooth_sensitivity(
              sorted_data, data_low, data_high, true_quantile_idx, t)
    ts[i] = triples[log_choose_triple_idx(triples, divided_eps,
                                          log_sensitivities)][0]
  return ts


def csmooth_tune_t_experiment(eps, num_samples, num_trials, num_quantiles_range,
                              data_low, data_high, log_t_low, log_t_high,
                              num_t):
  """Returns 2-D array of ts, tuned for each (num_quantiles, quantile) pair.

  Args:
    eps: Privacy parameter epsilon.
    num_samples: Number of standard Gaussian samples to draw for each trial.
    num_trials: Number of trials to average.
    num_quantiles_range: Array of number of quantiles to estimate.
    data_low: Lower bound for data, used by CSmooth.
    data_high: Upper bound for data, used by CSmooth.
    log_t_low: Tuning range for t has lower bound 10^(log_t_low).
    log_t_high: Tuning range for t has upper bound 10^(log_t_high).
    num_t: Number of logarithmically spaced t used to populate tuning range.
  """
  ts = [np.zeros(num_quantiles) for num_quantiles in num_quantiles_range]
  num_quantiles_idx = 0
  for num_quantiles_idx in range(len(num_quantiles_range)):
    num_quantiles = num_quantiles_range[num_quantiles_idx]
    divided_eps = eps / np.sqrt(num_quantiles)
    for _ in range(num_trials):
      sorted_data = base.gen_gaussian(num_samples, 0, 1)
      qs = np.linspace(0, 1, num_quantiles + 2)[1:-1]
      ts[num_quantiles_idx] += csmooth_tune_and_return_ts(
          sorted_data, data_low, data_high, qs, divided_eps, log_t_low,
          log_t_high, num_t) / num_trials
    print("Finished num_quantiles: {}".format(num_quantiles))
  return ts
