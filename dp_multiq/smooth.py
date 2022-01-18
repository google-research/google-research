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

"""Smooth sensitivity method for computing differentially private quantiles.

Lemmas 2.6 and 2.9 from "Smooth Sensitivity and Sampling in Private Data
Analysis" by Nissim, Radkhodnikova, and Smith
(https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf) describe the
noise scaled to the smooth sensitivity.
"""

import numpy as np

from dp_multiq import base
from dp_multiq import smooth_utils


def smooth(sorted_data, data_low, data_high, qs, divided_eps, divided_delta):
  """Returns (eps, delta)-differentially private quantile estimates for qs.

  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower limit for any differentially private quantile output value.
    data_high: Upper limit for any differentially private quantile output value.
    qs: Increasing array of quantiles in [0,1].
    divided_eps: Privacy parameter epsilon, assumed to be already divided for
      the desired overall eps.
    divided_delta: Privacy parameter delta, assumed to be already divided for
      the desired overall delta.
  """
  sorted_data = np.clip(sorted_data, data_low, data_high)
  o = np.empty(len(qs))
  n = len(sorted_data)
  alpha = divided_eps / 2.0
  beta = divided_eps / (2 * np.log(2 / divided_delta))
  for i in range(len(qs)):
    true_quantile_idx = base.quantile_index(n, qs[i])
    true_quantile_value = sorted_data[true_quantile_idx]
    log_sensitivity = smooth_utils.compute_log_smooth_sensitivity(
        sorted_data, data_low, data_high, true_quantile_idx, beta)
    noise = np.exp(log_sensitivity) * np.random.laplace() / alpha
    o[i] = true_quantile_value + noise
  o = np.clip(o, data_low, data_high)
  return np.sort(o)
