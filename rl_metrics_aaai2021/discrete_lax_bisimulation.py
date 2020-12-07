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

"""Implementation of discrete Lax-Bisimulation metrics."""

from rl_metrics_aaai2021 import discrete_bisimulation


class DiscreteLaxBisimulation(discrete_bisimulation.DiscreteBisimulation):
  """Implementation of discrete Lax-Bisimulation metrics.

  This works for both regular bisimulation and the on-policy variant.

  See Taylor et al., 2008: "Bounding Performance Loss in Approximate MDP
                            Homomorphisms"
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
      for a1 in range(self.num_actions):
        matching_action_exists = False
        for a2 in range(self.num_actions):
          # First check disagreement on rewards.
          if self.env.rewards[s1, a1] != self.env.rewards[s2, a2]:
            continue
          # Check disagreement on transitions. Stochastic setting.
          next_state_distrib_1 = self.env.transition_probs[s1, a1, :]
          next_state_distrib_2 = self.env.transition_probs[s2, a2, :]
          if next_state_distrib_1[c].sum() != next_state_distrib_2[c].sum():
            continue
          # If we've made it this far, action2 simulates action1.
          matching_action_exists = True
          break
        if not matching_action_exists:
          # If there is no matching action, s1 doesn't belong here.
          return False
    # If we've made it this far, s1 is still ok in c.
    return True
