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

"""Aquadem Builder."""

from typing import Any, Callable, Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb

from aquadem import actor
from aquadem import config as aquadem_config
from aquadem import learning
from aquadem import networks as aquadem_networks


def get_aquadem_policy(discrete_rl_policy,
                       networks
                       ):
  """The default behavior for the Aquadem agent.

  Args:
    discrete_rl_policy: the discrete policy choosing between the N candidate
      actions
    networks: a AquademNetworks containing all required networks.

  Returns:
    The aquadem default policy.
  """

  def aquadem_policy(params, observation, discrete_action):
    predicted_actions = networks.encoder.apply(params, observation)
    return predicted_actions[Ellipsis, discrete_action]

  return actor.AquademPolicyComponents(discrete_rl_policy, aquadem_policy)


def discretize_spec(spec, num_actions):
  assert isinstance(spec.actions, specs.BoundedArray)
  return spec._replace(actions=specs.DiscreteArray(num_actions))


class AquademBuilder(builders.ActorLearnerBuilder):
  """Aquadem Builder."""

  def __init__(self,
               rl_agent,
               config,
               make_demonstrations,
               logger_fn = lambda: None,):
    """Builds an Aquadem agent.

    Args:
      rl_agent: the standard discrete RL algorithm used by Aquadem
      config: the configuration for the multicategorical
        offline learner.
      make_demonstrations: A function that returns a dataset of
        acme.types.Transition.
      logger_fn: a logger factory for the learner.
    """
    self._rl_agent = rl_agent
    self._config = config
    self._make_demonstrations = make_demonstrations
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key,
      networks,
      dataset,
      replay_client = None,
      counter = None,
  ):
    """Creates the learner."""
    counter = counter or counting.Counter()
    discrete_rl_counter = counting.Counter(counter, 'direct_rl')

    aquadem_learner_key, discrete_rl_learner_key = jax.random.split(random_key)

    def discrete_rl_learner_factory(
        networks, dataset):
      return self._rl_agent.make_learner(
          discrete_rl_learner_key,
          networks,
          dataset,
          replay_client=replay_client,
          counter=discrete_rl_counter)

    # pytype:disable=attribute-error
    demonstrations_iterator = self._make_demonstrations(
        self._rl_agent._config.batch_size)  # pylint: disable=protected-access
    # pytype:enable=attribute-error

    optimizer = optax.adam(learning_rate=self._config.encoder_learning_rate)
    return learning.AquademLearner(
        random_key=aquadem_learner_key,
        discrete_rl_learner_factory=discrete_rl_learner_factory,
        iterator=dataset,
        demonstrations_iterator=demonstrations_iterator,
        optimizer=optimizer,
        networks=networks,
        make_demonstrations=self._make_demonstrations,
        encoder_num_steps=self._config.encoder_num_steps,
        encoder_batch_size=self._config.encoder_batch_size,
        encoder_eval_every=self._config.encoder_eval_every,
        temperature=self._config.temperature,
        num_actions=self._config.num_actions,
        demonstration_ratio=self._config.demonstration_ratio,
        min_demo_reward=self._config.min_demo_reward,
        counter=counter,
        logger=self._logger_fn())

  def make_replay_tables(
      self, environment_spec):
    discretized_spec = discretize_spec(environment_spec,
                                       self._config.num_actions)
    return self._rl_agent.make_replay_tables(discretized_spec)

  def make_dataset_iterator(
      self,
      replay_client):
    return self._rl_agent.make_dataset_iterator(replay_client)

  def make_adder(self,
                 replay_client):
    return self._rl_agent.make_adder(replay_client)

  def make_actor(
      self,
      random_key,
      policy_network,
      adder = None,
      variable_source = None,
  ):
    assert variable_source is not None
    wrapped_actor = self._rl_agent.make_actor(random_key,
                                              policy_network.discrete_policy,
                                              adder,
                                              variable_source)
    return actor.AquademActor(
        wrapped_actor=wrapped_actor,
        policy=policy_network.aquadem_policy,
        # Inference happens on CPU, so it's better to move variables there too.
        variable_client=variable_utils.VariableClient(
            variable_source,
            'aquadem_encoder',
            device='cpu',
            update_period=1000000000),  # never update what does not change
        adder=adder,
    )
