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

"""Implementation of Bisimulation metrics."""

import time
from absl import logging
import numpy as np
import ot
from rl_metrics_aaai2021 import metric


class Bisimulation(metric.Metric):
  """Implementation of Bisimulation metrics.

  See Ferns et al., 2004: "Metrics for Finite Markov Decision Processes"
  """

  def _compute(self, tolerance, verbose=False):
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
    start_time = time.time()
    while metric_difference > tolerance:
      new_metric = np.zeros((self.num_states, self.num_states))
      for s in range(self.num_states):
        for t in range(self.num_states):
          for a in range(self.num_actions):
            next_state_distrib_1 = self.env.transition_probs[s, a, :]
            next_state_distrib_2 = self.env.transition_probs[t, a, :]
            rew1 = self.env.rewards[s, a]
            rew2 = self.env.rewards[t, a]
            emd = ot.emd2(
                next_state_distrib_1, next_state_distrib_2, curr_metric)
            act_distance = abs(rew1 - rew2) + self.gamma * emd
            if act_distance > new_metric[s, t]:
              new_metric[s, t] = act_distance
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
