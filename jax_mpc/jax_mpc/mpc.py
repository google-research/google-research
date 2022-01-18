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

"""Defines a base model predictive control class for algorithms to extend."""


class MPC(object):
  """The base model predictive control class."""

  def init_state(self, a_shape):
    """Returns the initial state for the MPC algorithm."""
    raise NotImplementedError()

  def update(self, mpc_state, env, env_state, rng, reward_fn=None,
             reward_params=None, reward_rng=None):
    """Process the new env_state and update the mpc_state if necessary."""
    raise NotImplementedError()

  def get_action(self, mpc_state):
    """Return the current recommended action."""
    raise NotImplementedError()
