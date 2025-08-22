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

"""Private lp mechanism."""

import numpy as np
from scipy import stats


def random_signs(vector):
  """Returns vector with each coordinate's sign randomly set.

  Args:
    vector: Numpy array whose signs should be randomly set.
  """
  signs = -np.random.randint(2, size=len(vector)) * 2 + 1
  return signs * vector


def sample_lp_ball(d, p):
  """Returns a uniform random sample from the d-dimensional unit lp ball.

  Args:
    d: Integer dimension of the ball.
    p: Choice of lp norm. This can be any float p >= 1 or p = np.inf.

  Returns:
    A uniform random sample from the desired lp ball using Theorem 1 in
    https://arxiv.org/abs/math/0503650.
  """
  if p == np.inf:
    return np.random.uniform(-1, 1, size=d)
  else:
    # With the given parameters, gamma_samples are sampled from a distribution
    # with density proportional to x^(ca-1) * exp(-|x|^c) = exp(-|x|^p).
    gamma_samples = random_signs(stats.gengamma.rvs(a=1 / p, c=p, size=d))
  exponential_sample = np.random.exponential(scale=1)
  return gamma_samples / (
      np.sum(np.power(np.abs(gamma_samples), p)) + exponential_sample
  ) ** (1 / p)


def lp_mechanism(vector, p, lp_sensitivity, epsilon):
  """Returns a sample from the lp norm mechanism.

  Args:
    vector: The output will be a noisy version of Numpy array vector.
    p: Choice of lp norm. This can be any float p >= 1 or p = np.inf.
    lp_sensitivity: The statistic sensitivity, as a float, with respect to the
      chosen lp norm.
    epsilon: The output will be epsilon-DP for float epsilon.

  Returns:
    A sample from the K-norm mechanism instantiated with the specified lp norm,
    as described in Section 4 of https://arxiv.org/abs/0907.3754. Translated to
    our setting, we use vector = Fx and K = B_p^d. Note that their proof uses
    sensitivity 1.
  """
  d = len(vector)
  radius = np.random.gamma(shape=d + 1, scale=1 / epsilon)
  sample = lp_sensitivity * sample_lp_ball(d, p)
  noise = radius * sample
  return vector + noise
