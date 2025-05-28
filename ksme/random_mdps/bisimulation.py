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

"""Implementation of π-bisimulation metric."""

from absl import logging
import numpy as np
import ot
from ksme.random_mdps import metric


class Bisimulation(metric.Metric):
  """Implementation of π-bisimulation metric.

  See Castro, 2020: "Scalable methods for computing state similarity in
  deterministic Markov Decision Processes"
  https://arxiv.org/abs/1911.09291
  """

  def _compute(self, tolerance=0.001, verbose=False):
    """Compute exact/online bisimulation metric up to the specified tolerance.

    Args:
      tolerance: float, maximum difference in metric estimate between successive
        iterations. Once this threshold is past, computation stops.
      verbose: bool, whether to print verbose messages.
    """
    # Initial metric is all zeros.
    curr_metric = np.zeros((self.num_states, self.num_states))
    metric_difference = tolerance * 2.
    i = 1
    exact_metric_differences = []
    while metric_difference > tolerance:
      new_metric = np.zeros((self.num_states, self.num_states))
      for x in range(self.num_states):
        for y in range(self.num_states):
          next_state_distrib_1 = self.env.policy_transition_probs[x, :]
          next_state_distrib_2 = self.env.policy_transition_probs[y, :]
          rew_diff = abs(
              self.env.policy_rewards[x] - self.env.policy_rewards[y])
          emd = ot.emd2(next_state_distrib_1, next_state_distrib_2, curr_metric)
          new_metric[x, y] = rew_diff + self.env.gamma * emd

      metric_difference = np.max(abs(new_metric - curr_metric))
      exact_metric_differences.append(metric_difference)
      if verbose:
        logging.info('Iteration %d: %f', i, metric_difference)
      curr_metric = np.copy(new_metric)
      i += 1
    self.metric = curr_metric
