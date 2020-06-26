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

import numpy as np
from collections import Counter
from strategic_exploration.hrl import abstract_state as AS
from strategic_exploration.hrl.rl import Experience


class RewardBonus(object):
  """For visiting abstract state s, gives a bonus reward

    equal to beta / sqrt(n(s)) where n(s) is the visit count
    of s
  """

  def __init__(self, beta=0.63):
    self._beta = beta
    self._state_counts = Counter()

  def __call__(self, experience):
    """Returns another experience with the bonus added to the reward.

        Args: experience (Experience)

        Returns:
            Experience
        """
    next_state = AS.AbstractState(experience.next_state)
    next_state_count = self._state_counts[next_state]
    assert next_state_count > 0
    reward_bonus = self._beta / np.sqrt(next_state_count)
    return Experience(experience.state, experience.action,
                      experience.reward + reward_bonus, experience.next_state,
                      experience.done)

  def observe(self, experience):
    """Updates internal state counts based on observing this experience.

        Args: experience (Experience)
    """
    next_state = AS.AbstractState(experience.next_state)
    self._state_counts[next_state] += 1

  def clear(self):
    """Resets all state counts to 0 and frees up any memory from the state

        counts
    """
    self._state_counts.clear()
