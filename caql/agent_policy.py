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

"""Continuous DQN policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from caql import policy


class AgentPolicy(policy.BasePolicy):
  """Implementation for Continuous Agent policy."""

  def __init__(self, action_spec, agent):
    super(AgentPolicy, self).__init__(action_spec)
    self._agent = agent

  def _action(self, state, use_action_function, batch_mode=False):
    state_dim = state.shape[-1]
    state_tensor = np.reshape(state, [-1, state_dim])
    action_tensor, _, _, success = self._agent.best_action(
        state_tensor, use_action_function)
    if success:
      if not batch_mode:
        return np.reshape(action_tensor, [
            -1,
        ])
      else:
        return action_tensor
    return None

  def _update_params(self):
    pass

  def _params_debug_str(self):
    pass
