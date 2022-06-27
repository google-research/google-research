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

"""A specific actor for the Aquadem agent."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import adders
from acme import core
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import jax


AquademPolicy = Callable[
    [network_lib.Params, network_lib.Observation, network_lib.Action],
    network_lib.Action]


@dataclasses.dataclass
class AquademPolicyComponents:
  discrete_policy: actor_core_lib.FeedForwardPolicy
  aquadem_policy: AquademPolicy


class AquademActor(core.Actor):
  """A specific actor for Aquadem."""

  def __init__(
      self,
      wrapped_actor,
      policy,
      variable_client,
      adder = None,
  ):
    """Initializes a feed forward actor.

    Args:
      wrapped_actor: the discrete action actor.
      policy: A policy network taking observation and a discrete action and
        returning an action.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
    """
    self._wrapped_actor = wrapped_actor

    # Adding batch dimension inside jit is much more efficient than outside.
    def batched_policy(
        params,
        observation,
        discrete_action,
    ):
      observation = utils.add_batch_dim(observation)
      action = utils.squeeze_batch_dim(
          policy(params, observation, discrete_action))

      return action

    self._policy = jax.jit(batched_policy, backend='cpu')
    self._adder = adder
    self._client = variable_client
    self._last_discrete_action = None

  def select_action(self,
                    observation):
    discrete_action = self._wrapped_actor.select_action(observation)
    action = self._policy(self._client.params,
                          observation,
                          discrete_action)
    self._last_discrete_action = discrete_action
    return utils.to_numpy(action)

  def observe_first(self, timestep):
    self._wrapped_actor.observe_first(timestep)

  def observe(self, action, next_timestep):
    self._wrapped_actor.observe(self._last_discrete_action, next_timestep)

  def update(self, wait = False):
    self._wrapped_actor.update(wait)
    self._client.update(wait)
