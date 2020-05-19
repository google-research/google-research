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
"""Sample exhaustively all the possible states."""

import gin
import jax.numpy as np

from grouptesting import bayes
from grouptesting.samplers import sampler
from grouptesting.samplers import temperature


def all_binary_vectors(num_samples, upper_bound = 1.0):
  """Generates all 2^n binary vectors."""
  # TODO(cuturi) use itertools ?
  results = np.array([[0], [1]], dtype=bool)
  bound_per_line = np.ceil(upper_bound * num_samples) + 1
  for _ in range(num_samples - 1):
    num_rows = results.shape[0]
    twice_binary_vectors = np.concatenate((results, results), axis=0)
    zo = np.concatenate((np.zeros((num_rows, 1), dtype=bool),
                         np.ones((num_rows, 1), dtype=bool)),
                        axis=0)
    results = np.concatenate((twice_binary_vectors, zo), axis=1)
    if 0 < upper_bound < 1.0:
      count = np.sum(results, axis=-1)
      results = results[(count < bound_per_line), :]
  return results


@gin.configurable
class ExhaustiveSampler(sampler.Sampler):
  """Produces all possible binary vectors (with up to upper bound positives)."""

  NAME = 'Exh'

  def __init__(self, upper_bound = 0.5):
    super().__init__()
    self.upper_bound = upper_bound

  def produce_sample(self, rng, state):
    self.particles = all_binary_vectors(state.num_patients, self.upper_bound)
    log_posteriors = bayes.log_posterior(
        self.particles, state.past_test_results,
        state.past_groups, state.log_prior_specificity,
        state.log_prior_1msensitivity,
        state.prior_infection_rate)
    self.particle_weights = temperature.importance_weights(log_posteriors)
