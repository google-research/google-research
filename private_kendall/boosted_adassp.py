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

"""Boosted AdaSSP differentially private linear regression.

This file implements the Boosted AdaSSP mechanism
(https://arxiv.org/abs/2303.03451) for (epsilon, delta)-differentially private
linear regression.
"""


import numpy as np
from scipy import stats


def gaussian_mechanism(raw_statistic, l2_sensitivity, mu):
  """Returns a noisy version of raw_statistic to ensure mu-GDP.

  Args:
    raw_statistic: Raw statistic, without added noise.
    l2_sensitivity: L2 sensitivity of the statistic.
    mu: The output will satisfy mu-GDP.
  Returns: A noisy, mu-GDP (Gaussian DP) version of the raw statistic. See,
    e.g., Theorem 2.3 in https://arxiv.org/pdf/2303.03451.pdf.
  """
  return raw_statistic + np.random.normal(
      scale=l2_sensitivity / mu, size=np.asarray(raw_statistic).shape
  )


def dp_to_gdp(epsilon, delta):
  """Find mu such that mu-GDP satisfies (epsilon,delta)-DP.

  Args:
    epsilon: The DP algorithm is (epsilon, delta)-DP.
    delta: The DP algorithm is (epsilon, delta)-DP.

  Returns:
    A value mu such that an algorithm that is (epsilon, delta)-DP is also
    mu-GDP. See e.g. Corollary 2.5 in
    https://arxiv.org/pdf/2303.03451.pdf for details.
  """
  # Starting mu is taken from Theorem A.1 in
  # https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf.
  mu = epsilon**2 / (2 * np.log(1.25 / delta))
  while True:
    if delta < stats.norm.cdf(-epsilon / mu + mu / 2) - np.exp(
        epsilon
    ) * stats.norm.cdf(-epsilon / mu - mu / 2):
      mus_1 = np.linspace(mu / 2, mu, 1000)
      for mu_1_idx in range(len(mus_1)):
        mu_1 = mus_1[mu_1_idx]
        if delta < stats.norm.cdf(-epsilon / mu_1 + mu_1 / 2) - np.exp(
            epsilon
        ) * stats.norm.cdf(-epsilon / mu_1 - mu_1 / 2):
          return mu_1
    mu = mu * 2


def clip(matrix, clip_norm):
  """Returns a clipped copy of matrix.

  Each row of the returned matrix will be clipped to have l_2 norm
      min(||row||_2, clip_norm).

  Args:
    matrix: The matrix to clip.
    clip_norm: The l_2 norm for clipping each row of matrix.
  """
  scale = np.minimum(1, clip_norm / np.linalg.norm(matrix, axis=1))
  return np.multiply(matrix, scale[:, np.newaxis])


def boosted_adassp(
    features, labels, num_rounds, feature_clip_norm, gradient_clip_norm,
    epsilon, delta
):
  """Computes an (epsilon, delta)-DP regression model using BoostedAdaSSP.

  Args:
    features: Matrix of feature vectors where each row is an example. Assumed to
      have intercept feature in the last column.
    labels: Vector of labels.
    num_rounds: Number of rounds of boosting.
    feature_clip_norm: Clipping norm for features.
    gradient_clip_norm: Clipping norm for gradients.
    epsilon: The output will satisfy (epsilon, delta)-DP.
    delta: The output will satisfy (epsilon, delta)-DP.

  Returns:
    An array of coefficients for a regression model computed using
    BoostedAdaSSP. See Algorithm 1 of https://arxiv.org/pdf/2303.03451.pdf for
    details.
  """
  _, d = features.shape
  mu = dp_to_gdp(epsilon, delta)
  # Composition via, e.g., Corollary 2.4 in https://arxiv.org/pdf/2303.03451.pdf
  split_mu = mu / np.sqrt(3)
  clipped_features = clip(features, feature_clip_norm)
  clipped_xtx = np.matmul(clipped_features.T, clipped_features)
  private_xtx = gaussian_mechanism(clipped_xtx, 1, split_mu)
  min_eigenvalue = max(0, np.amin(np.linalg.eigh(clipped_xtx)[0]))
  private_lambda = gaussian_mechanism(min_eigenvalue, 1, split_mu)
  private_xtx[np.diag_indices_from(private_xtx)] += private_lambda
  gamma = np.linalg.pinv(private_xtx)
  theta = np.zeros((d, 1))
  round_mu = split_mu / np.sqrt(num_rounds)
  for _ in range(int(num_rounds)):
    y_hat = np.matmul(clipped_features, theta)
    g = labels[:, np.newaxis] - y_hat
    # https://arxiv.org/pdf/2303.03451.pdf uses a clip norm of 1 throughout
    g = clip(g, gradient_clip_norm)
    xtg = np.matmul(clipped_features.T, g)
    private_xtg = gaussian_mechanism(xtg, 1, round_mu)
    theta_t = np.matmul(gamma, private_xtg)
    theta = theta + theta_t
  return theta
