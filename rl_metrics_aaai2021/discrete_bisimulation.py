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

"""Implementation of discrete Bisimulation metrics."""

import copy
import time
import numpy as np
import tensorflow.compat.v1 as tf
from rl_metrics_aaai2021 import metric


class DiscreteBisimulation(metric.Metric):
  """Implementation of discrete Bisimulation metrics.

  See Ferns et al., 2004: "Metrics for Finite Markov Decision Processes"
  """

  def _state_matches_class(self, s1, c):
    """Determines whether state s1 belongs in equivalence class c.

    Args:
      s1: int, index of state in question.
      c: list, equivalence class to check.

    Returns:
      bool, whether the state matches its equivalence class.
    """
    if len(c) == 1 and s1 == c[0]:
      # If it's already a singleton, we know it's ok.
      return True
    for s2 in c:
      if s1 == s2:
        continue
      for a in range(self.num_actions):
        # First check disagreement on rewards.
        if self.env.rewards[s1, a] != self.env.rewards[s2, a]:
          return False
        # Check disagreement on transitions. Stochastic case.
        next_state_distrib_1 = self.env.transition_probs[s1, a, :]
        next_state_distrib_2 = self.env.transition_probs[s2, a, :]
        if next_state_distrib_1[c].sum() != next_state_distrib_2[c].sum():
          return False
    # If we've made it this far, s1 is still ok in c.
    return True

  def _compute(self, tolerance, verbose=False):
    """Compute the bisimulation relation and convert it to a discrete metric.

    Args:
      tolerance: float, unused.
      verbose: bool, whether to print verbose messages.

    Returns:
      Statistics object containing statistics of computation.
    """
    del tolerance
    equivalence_classes_changing = True
    iteration = 0
    start_time = time.time()
    # All states start in the same equivalence class.
    equivalence_classes = [list(range(self.num_states))]
    state_to_class = [0] * self.num_states
    while equivalence_classes_changing:
      equivalence_classes_changing = False
      class_removed = False
      iteration += 1
      new_equivalence_classes = copy.deepcopy(equivalence_classes)
      new_state_to_class = copy.deepcopy(state_to_class)
      for s1 in range(self.num_states):
        if self._state_matches_class(
            s1, equivalence_classes[state_to_class[s1]]):
          continue
        # We must find a new class for s1.
        equivalence_classes_changing = True
        previous_class = new_state_to_class[s1]
        new_state_to_class[s1] = -1
        # Checking if there are still any elements in s1's old class.
        potential_new_class = [
            x for x in new_equivalence_classes[previous_class] if x != s1]
        if potential_new_class:
          new_equivalence_classes[previous_class] = potential_new_class
        else:
          # remove s1's old class from the list of new_equivalence_classes.
          new_equivalence_classes.pop(previous_class)
          class_removed = True
          # Re-index the classes.
          for i, c in enumerate(new_state_to_class):
            if c > previous_class:
              new_state_to_class[i] = c - 1
        for i, c in enumerate(new_equivalence_classes):
          if not class_removed and i == previous_class:
            continue
          if self._state_matches_class(s1, c):
            new_state_to_class[s1] = i
            new_equivalence_classes[i] += [s1]
            break
        if new_state_to_class[s1] < 0:
          # If we haven't found a matching equivalence class, we create a new
          # one.
          new_equivalence_classes.append([s1])
          new_state_to_class[s1] = len(new_equivalence_classes) - 1
      equivalence_classes = copy.deepcopy(new_equivalence_classes)
      state_to_class = copy.deepcopy(new_state_to_class)
      if iteration % 1000 == 0 and verbose:
        tf.logging.info('Iteration {}'.format(iteration))
    #  Now that we have the equivalence classes, we create the metric.
    self.metric = np.ones((self.num_states, self.num_states))
    for c in equivalence_classes:
      for s1 in c:
        for s2 in c:
          self.metric[s1, s2] = 0.
    total_time = time.time() - start_time
    self.statistics = metric.Statistics(-1., total_time, iteration, 0.0)
