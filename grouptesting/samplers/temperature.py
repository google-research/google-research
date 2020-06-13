# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Compute step length / temperatures in SMC resampling procedures."""

import gin
import jax.numpy as np
from jax.scipy import special


def effective_sample_size(alpha,
                          log_posterior):
  """Quantifies diversity of weights."""
  lognumerator = 2 * special.logsumexp(
      alpha * log_posterior, axis=-1)
  logdenominator = (
      np.log(log_posterior.shape[0]) +
      special.logsumexp(2 * alpha * log_posterior, axis=-1))
  return np.exp(lognumerator - logdenominator)


@gin.configurable
def find_step_length(rho,
                     log_posterior,
                     tolerance=0.02,
                     effective_sample_size_target=0.9):
  """Ensures diversity in the weights induced by log-posterior."""
  low = 0
  up = 1.05 - rho
  alpha = 0.05
  while (np.abs(up - low) > tolerance) and (low < 1.0 - rho):
    if (effective_sample_size(alpha, log_posterior)
        < effective_sample_size_target):
      up = alpha
      alpha = (alpha + low) * 0.5
    else:
      low = alpha
      alpha = (alpha + up) * 0.5
  alpha = np.minimum(alpha, 1.0 - rho)
  return alpha, alpha * log_posterior


def importance_weights(log_unnormalized_probabilities):
  """Normalizes log-weights."""
  return np.exp(log_unnormalized_probabilities -
                special.logsumexp(log_unnormalized_probabilities))
