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

"""DMC Wrapper to perform action repeat.

Typically used with suite_dm_control.load_pixels to avoid rendering frames
for skipped observations.
"""

import dm_env


class ActionRepeatDMWrapper(dm_env.Environment):
  """Repeat the same action and return summed rewards."""

  def __init__(self, env, action_repeat = 4):
    super().__init__()
    self._env = env
    self._action_repeat = action_repeat

  def step(self, action):
    """Repeat action, sum rewards."""
    total_reward = 0.0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      total_reward += time_step.reward
      if time_step.step_type.last():
        break

    return dm_env.TimeStep(step_type=time_step.step_type,
                           reward=total_reward,
                           discount=time_step.discount,
                           observation=time_step.observation)

  def reset(self):
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._env.action_spec()

  def __getattr__(self, name):
    return getattr(self._env, name)
