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

"""Functions for Laplace mechanism."""

import numpy as np


def get_laplace_sigma(d, eps, delta):
  """Returns the minimum Laplace mechanism sigma satisfying (eps, delta)-DP.

  We assume that the statistic being computed has l_2 sensitivity 1. See Section
  4.2 in the paper for details.

  Args:
    d: Integer dimension.
    eps: Float privacy parameter epsilon.
    delta: Float privacy parameter delta.
  """
  return np.sqrt(d) / (eps + delta)


def get_laplace_samples(d, sigma, num_samples):
  """Returns samples of shape (num_samples, d) from the Laplace mechanism.

  Args:
    d: Integer dimension.
    sigma: Float noise scale.
    num_samples: Integer number of samples to generate.
  """
  return np.random.laplace(0, sigma, size=(num_samples, d))


def get_laplace_mean_squared_l2_error(d, sigma):
  """Returns the mean squared l_2 error of the specified Laplace mechanism.

  See Corollary 4.2 in the paper for details.

  Args:
    d: Integer dimension.
    sigma: Float noise scale.
  """
  return 2 * d * sigma ** 2
