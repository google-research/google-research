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

"""Wrapper to provide access to the MDP objects (namely, `transition_probs` and `rewards`).

This extends the TabularWrapper, which converts a MiniGridEnv to a tabular MDP.

This iterates through the states (assuming the number of states is given by
width * height) and constructs the transition probabilities. It assumes, like
in the stochasticity setting of our local copy of MiniGridEnv, that the
stochasticity probability mass is distributed evenly across the three actions
different from the one chosen, and staying in-place.
"""

import numpy as np
from minigrid_basics.custom_wrappers import tabular_wrapper


def get_next_state_and_reward(pos, env):
  """Return the next state and reward.

  Args:
    pos: pair of ints, current agent position.
    env: MiniGrid environment.

  Returns:
    The next agent position and the reward.
  """
  next_pos = pos
  fwd_pos = env.front_pos
  cell = env.grid.get(*fwd_pos)
  current_cell = env.grid.get(*pos)
  reward = 0
  if cell is None or cell.can_overlap():
    next_pos = fwd_pos
  if cell is not None and cell.type == 'goal':
    env.agent_pos = fwd_pos
    reward = env._reward()  # pylint: disable=protected-access
    env.agent_pos = pos
  # If we are at the goal and the env. is episodic, we remain in the same state
  if current_cell is not None and current_cell.type == 'goal' and env.episodic:
    next_pos = pos
  return next_pos, reward


class MDPWrapper(tabular_wrapper.TabularWrapper):
  """Wrapper to provide access to the MDP objects (namely, `transition_probs` and `rewards`).
  """

  def __init__(self, env, tile_size=8, get_rgb=False):
    super().__init__(env, tile_size=tile_size, get_rgb=get_rgb)
    self.num_actions = len(env.actions)
    self.transition_probs = np.zeros((self.num_states, self.num_actions,
                                      self.num_states))
    self.rewards = np.zeros((self.num_states, self.num_actions))
    env = self.unwrapped
    for y in range(self.height):
      for x in range(self.width):
        s1 = self.pos_to_state[x + y * env.width]
        if s1 < 0:  # Invalid position.
          continue
        env.agent_pos = np.array([x, y])
        for a in range(self.num_actions):
          env.agent_dir = a
          next_pos, r = get_next_state_and_reward([x, y], env)
          s2 = self.pos_to_state[next_pos[0] + next_pos[1] * env.width]
          self.transition_probs[s1, a, s2] = 1. - env.stochasticity
          self.rewards[s1, a] = (1. - env.stochasticity) * r
          # "Slippage" probability mass is distributed across other directions
          # and staying in-place.
          slippage_probability = env.stochasticity / 4.
          self.transition_probs[s1, a, s1] += slippage_probability
          for a2 in range(self.num_actions):
            if a2 == a:
              continue
            self.agent_dir = a2
            next_pos, r = get_next_state_and_reward([x, y], env)
            s2 = self.pos_to_state[next_pos[0] + next_pos[1] * env.width]
            self.transition_probs[s1, a, s2] += slippage_probability
            self.rewards[s1, a] += r * slippage_probability
