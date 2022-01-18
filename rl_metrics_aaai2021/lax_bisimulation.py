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

"""Implementation of Lax-Bisimulation metrics."""

import time
from absl import logging
import numpy as np
import ot
from rl_metrics_aaai2021 import metric


class LaxBisimulation(metric.Metric):
  """Implementation of LaxBisimulation metrics.

  See Taylor et al., 2008: "Bounding Performance Loss in Approximate MDP
                            Homomorphisms"
  """

  def _compute(self, tolerance, verbose=False):
    """Compute exact/online lax-bisimulation metric up to specified tolerance.

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
    start_time = time.time()
    while metric_difference > tolerance:
      new_metric = np.zeros((self.num_states, self.num_states))
      state_action_metric = np.zeros((self.num_states, self.num_actions,
                                      self.num_states, self.num_actions))
      for s in range(self.num_states):
        for t in range(self.num_states):
          for a in range(self.num_actions):
            for b in range(self.num_actions):
              next_state_distrib_1 = self.env.transition_probs[s, a, :]
              next_state_distrib_2 = self.env.transition_probs[t, b, :]
              rew1 = self.env.rewards[s, a]
              rew2 = self.env.rewards[t, b]
              emd = ot.emd2(
                  next_state_distrib_1, next_state_distrib_2, curr_metric)
              state_action_metric[s, a, t, b] = (
                  abs(rew1 - rew2) + self.gamma * emd)
      # Now that we've updated the state-action metric, we compute the Hausdorff
      # metric.
      for s in range(self.num_states):
        for t in range(s + 1, self.num_states):
          # First we find \sup_x\inf_y d(x, y) from Definition 5 in paper.
          max_a = None
          for a in range(self.num_actions):
            min_b = np.min(state_action_metric[s, a, t, :])
            if max_a is None or min_b > max_a:
              max_a = min_b
          # Next we find \sup_y\inf_x d(x, y) from Definition 5 in paper.
          max_b = None
          for b in range(self.num_actions):
            min_a = np.min(state_action_metric[s, :, t, b])
            if max_b is None or min_a > max_b:
              max_b = min_a
          new_metric[s, t] = max(max_a, max_b)
          new_metric[t, s] = new_metric[s, t]
      metric_difference = np.max(abs(new_metric - curr_metric))
      exact_metric_differences.append(metric_difference)
      if verbose:
        logging.info('Iteration %d: %f', i, metric_difference)
      curr_metric = np.copy(new_metric)
      i += 1
    total_time = time.time() - start_time
    self.metric = curr_metric
    self.statistics = metric.Statistics(
        tolerance, total_time, i, exact_metric_differences)
