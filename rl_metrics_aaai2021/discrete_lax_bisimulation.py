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

"""Implementation of discrete Lax-Bisimulation metrics."""

from rl_metrics_aaai2021 import discrete_bisimulation


class DiscreteLaxBisimulation(discrete_bisimulation.DiscreteBisimulation):
  """Implementation of discrete Lax-Bisimulation metrics.

  This works for both regular bisimulation and the on-policy variant.

  See Taylor et al., 2008: "Bounding Performance Loss in Approximate MDP
                            Homomorphisms"
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
      if not c:
        continue
      if s1 in c:
        continue
      s2 = c[0] #we only need to check against one state in the equivalence class
      matching_action_exists = True
      for a1 in range(self.env.num_actions):
        # if we went through all possible pairs for an action and none of them matched, we can stop
        if not matching_action_exists:
          break
        matching_action_exists = False
        for a2 in range(self.env.num_actions):
          if self.env.rewards[s1, a1] != self.env.rewards[s2, a2]:
            continue
          next_state_distrib_1 = self.env.transition_probs[s1, a1, :]
          next_state_distrib_2 = self.env.transition_probs[s2, a2, :]
          matching_action_exists = True
          for next_c in equivalence_classes:
            if next_state_distrib_1[next_c].sum() != next_state_distrib_2[next_c].sum():
              matching_action_exists = False
              break
          # if we've made it this far, we've found a matching action pair
          if matching_action_exists:
            break
      # If we've made it this far, we've found a pair for all actions and s1 is moved to c
      if matching_action_exists:
        self._update_classes(s1, s2, equivalence_classes, state_to_class)
        return True
      else:
        continue

    # if no class was updated, return false
    return False
