# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

  def _check_equivalence(self, s1, equivalence_classes, state_to_class):
    """Determines whether state s1 belongs in equivalence class c.

    Args:
      s1: int, index of state in question.
      c: list of lists, equivalence classes to check.
      state_to_class: list of ints, mapping from state to equivalence class.

    Returns:
      bool, whether the state has been moved to a new equivalence class.
    """

    for c in equivalence_classes:
      update = True
      if not c:
        continue
      if s1 in c:
        continue
      s2 = c[0] #we only need to check against one state in the equivalence class
      for a in range(self.env.num_actions):
        if self.env.rewards[s1, a] != self.env.rewards[s2, a]:
          update = False
          break
        next_state_distrib_1 = self.env.transition_probs[s1, a, :]
        next_state_distrib_2 = self.env.transition_probs[s2, a, :]
        for next_c in equivalence_classes:
          if next_state_distrib_1[next_c].sum() != next_state_distrib_2[next_c].sum():
            update = False
            break
        if not update:
          break
      # If we've made it this far, s1 should be moved to c
      if update:
        self._update_classes(s1, s2, equivalence_classes, state_to_class)
        return True
      else:
        continue

    # if no class was updated, return false
    return False

  def _update_classes(self, s1, s2, equivalence_classes, state_to_class):
    """Moves state s1 to equivalence class of s2.

    Args:
      s1: int, index of state to move.
      s2: int, index of state belonging to the equivalence class to move to.
      equivalence_classes: list of lists, equivalence classes.
      state_to_class: list of ints, mapping from state to equivalence class.

    """
    equivalence_classes[state_to_class[s1]].remove(s1)
    if not equivalence_classes[state_to_class[s1]]:
      equivalence_classes.pop(state_to_class[s1])
      for i, c in enumerate(state_to_class):
        if c > state_to_class[s1]:
          state_to_class[i] = c - 1
    equivalence_classes[state_to_class[s2]].append(s1)
    state_to_class[s1] = state_to_class[s2]

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
    equivalence_classes = [[i] for i in range(self.num_states)]
    state_to_class = [i for i in range(self.num_states)]
    while equivalence_classes_changing:
      equivalence_classes_changing = True
      iteration += 1
      new_equivalence_classes = copy.deepcopy(equivalence_classes)
      new_state_to_class = copy.deepcopy(state_to_class)
      for s1 in range(self.num_states):
        if not self._check_equivalence(s1, new_equivalence_classes, new_state_to_class):
          equivalence_classes_changing = False
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
