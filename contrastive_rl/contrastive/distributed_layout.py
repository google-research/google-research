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

"""Program definition for a distributed layout based on a builder."""

import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
from acme.utils import observers as observers_lib
import dm_env
import jax
import launchpad as lp
import numpy as np
import reverb
import tqdm


ActorId = int
AgentNetwork = Any
PolicyNetwork = Any
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
Seed = int
EnvironmentFactory = Callable[[Seed], dm_env.Environment]
MakeActorFn = Callable[[types.PRNGKey, PolicyNetwork, core.VariableSource],
                       core.Actor]
LoggerLabel = str
LoggerStepsKey = str
LoggerFn = Callable[[LoggerLabel, LoggerStepsKey], loggers.Logger]
EvaluatorFactory = Callable[[
    types.PRNGKey,
    core.VariableSource,
    counting.Counter,
    MakeActorFn,
], core.Worker]


def get_default_logger_fn(
    log_to_bigtable = False,
    log_every = 10):
  """Creates an actor logger."""

  def create_logger(actor_id):
    return loggers.make_default_logger(
        'actor',
        save_data=(log_to_bigtable and actor_id == 0),
        time_delta=log_every,
        steps_key='actor_steps')
  return create_logger


def default_evaluator_factory(
    environment_factory,
    network_factory,
    policy_factory,
    observers = (),
    log_to_bigtable = False):
  """Returns a default evaluator process."""
  def evaluator(
      random_key,
      variable_source,
      counter,
      make_actor,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))
    networks = network_factory(specs.make_environment_spec(environment))

    actor = make_actor(actor_key, policy_factory(networks), variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = loggers.make_default_logger('evaluator', log_to_bigtable,
                                         steps_key='actor_steps')

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(environment, actor, counter,
                                            logger, observers=observers)
  return evaluator


@dataclasses.dataclass
class CheckpointingConfig:
  """Configuration options for learner checkpointer."""
  # The maximum number of checkpoints to keep.
  max_to_keep: int = 1
  # Which directory to put the checkpoint in.
  directory: str = '~/acme'
  # If True adds a UID to the checkpoint path, see
  # `paths.get_unique_id()` for how this UID is generated.
  add_uid: bool = True


class DistributedLayout:
  """Program definition for a distributed agent based on a builder."""

  def __init__(
      self,
      seed,
      environment_factory,
      network_factory,
      builder,
      policy_network,
      num_actors,
      environment_spec = None,
      actor_logger_fn = None,
      evaluator_factories = (),
      device_prefetch = True,
      prefetch_size = 1,
      log_to_bigtable = False,
      max_number_of_steps = None,
      observers = (),
      multithreading_colocate_learner_and_reverb = False,
      checkpointing_config = None):

    if prefetch_size < 0:
      raise ValueError(f'Prefetch size={prefetch_size} should be non negative')

    actor_logger_fn = actor_logger_fn or get_default_logger_fn(log_to_bigtable)

    self._seed = seed
    self._builder = builder
    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._policy_network = policy_network
    self._environment_spec = environment_spec
    self._num_actors = num_actors
    self._device_prefetch = device_prefetch
    self._log_to_bigtable = log_to_bigtable
    self._prefetch_size = prefetch_size
    self._max_number_of_steps = max_number_of_steps
    self._actor_logger_fn = actor_logger_fn
    self._evaluator_factories = evaluator_factories
    self._observers = observers
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._checkpointing_config = checkpointing_config

  def replay(self):
    """The replay storage."""
    dummy_seed = 1
    environment_spec = (
        self._environment_spec or
        specs.make_environment_spec(self._environment_factory(dummy_seed)))
    return self._builder.make_replay_tables(environment_spec)

  def counter(self):
    kwargs = {}
    if self._checkpointing_config:
      kwargs = vars(self._checkpointing_config)
    return savers.CheckpointingRunner(
        counting.Counter(),
        key='counter',
        subdirectory='counter',
        time_delta_minutes=5,
        **kwargs)

  def learner(
      self,
      random_key,
      replay,
      counter,
  ):
    """The Learning part of the agent."""

    if self._builder._config.env_name.startswith('offline_ant'):  # pytype: disable=attribute-error, pylint: disable=protected-access
      adder = self._builder.make_adder(replay)
      env = self._environment_factory(0)
      dataset = env.get_dataset()  # pytype: disable=attribute-error
      for t in tqdm.trange(dataset['observations'].shape[0]):
        discount = 1.0
        if t == 0 or dataset['timeouts'][t - 1]:
          step_type = dm_env.StepType.FIRST
        elif dataset['timeouts'][t]:
          step_type = dm_env.StepType.LAST
          discount = 0.0
        else:
          step_type = dm_env.StepType.MID

        ts = dm_env.TimeStep(
            step_type=step_type,
            reward=dataset['rewards'][t],
            discount=discount,
            observation=np.concatenate([dataset['observations'][t],
                                        dataset['infos/goal'][t]]),
        )
        if t == 0 or dataset['timeouts'][t - 1]:
          adder.add_first(ts)  # pytype: disable=attribute-error
        else:
          adder.add(action=dataset['actions'][t-1], next_timestep=ts)  # pytype: disable=attribute-error

        if self._builder._config.local and t > 10_000:  # pytype: disable=attribute-error, pylint: disable=protected-access
          break

    iterator = self._builder.make_dataset_iterator(replay)

    dummy_seed = 1
    environment_spec = (
        self._environment_spec or
        specs.make_environment_spec(self._environment_factory(dummy_seed)))

    # Creates the networks to optimize (online) and target networks.
    networks = self._network_factory(environment_spec)

    if self._prefetch_size > 1:
      # When working with single GPU we should prefetch to device for
      # efficiency. If running on TPU this isn't necessary as the computation
      # and input placement can be done automatically. For multi-gpu currently
      # the best solution is to pre-fetch to host although this may change in
      # the future.
      device = jax.devices()[0] if self._device_prefetch else None
      iterator = utils.prefetch(
          iterator, buffer_size=self._prefetch_size, device=device)
    else:
      logging.info('Not prefetching the iterator.')

    counter = counting.Counter(counter, 'learner')

    learner = self._builder.make_learner(random_key, networks, iterator, replay,
                                         counter)
    kwargs = {}
    if self._checkpointing_config:
      kwargs = vars(self._checkpointing_config)
    # Return the learning agent.
    return savers.CheckpointingRunner(
        learner,
        key='learner',
        subdirectory='learner',
        time_delta_minutes=5,
        **kwargs)

  def actor(self, random_key, replay,
            variable_source, counter,
            actor_id):
    """The actor process."""
    adder = self._builder.make_adder(replay)

    environment_key, actor_key = jax.random.split(random_key)
    # Create environment and policy core.

    # Environments normally require uint32 as a seed.
    environment = self._environment_factory(
        utils.sample_uint32(environment_key))

    networks = self._network_factory(specs.make_environment_spec(environment))
    policy_network = self._policy_network(networks)
    actor = self._builder.make_actor(actor_key, policy_network, adder,
                                     variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'actor')
    # Only actor #0 will write to bigtable in order not to spam it too much.
    logger = self._actor_logger_fn(actor_id)
    # Create the loop to connect environment and agent.
    return environment_loop.EnvironmentLoop(environment, actor, counter,
                                            logger, observers=self._observers)

  def coordinator(self, counter, max_actor_steps):
    if self._builder._config.env_name.startswith('offline_ant'):  # pytype: disable=attribute-error, pylint: disable=protected-access
      steps_key = 'learner_steps'
    else:
      steps_key = 'actor_steps'
    return lp_utils.StepsLimiter(counter, max_actor_steps, steps_key=steps_key)

  def build(self, name='agent', program = None):
    """Build the distributed agent topology."""
    if not program:
      program = lp.Program(name=name)

    key = jax.random.PRNGKey(self._seed)

    replay_node = lp.ReverbNode(self.replay)
    with program.group('replay'):
      if self._multithreading_colocate_learner_and_reverb:
        replay = replay_node.create_handle()
      else:
        replay = program.add_node(replay_node)

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))
      if self._max_number_of_steps is not None:
        _ = program.add_node(
            lp.CourierNode(self.coordinator, counter,
                           self._max_number_of_steps))

    learner_key, key = jax.random.split(key)
    learner_node = lp.CourierNode(self.learner, learner_key, replay, counter)
    with program.group('learner'):
      if self._multithreading_colocate_learner_and_reverb:
        learner = learner_node.create_handle()
        program.add_node(
            lp.MultiThreadingColocation([learner_node, replay_node]))
      else:
        learner = program.add_node(learner_node)

    def make_actor(random_key,
                   policy_network,
                   variable_source):
      return self._builder.make_actor(
          random_key, policy_network, variable_source=variable_source)

    with program.group('evaluator'):
      for evaluator in self._evaluator_factories:
        evaluator_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(evaluator, evaluator_key, learner, counter,
                           make_actor))

    with program.group('actor'):
      for actor_id in range(self._num_actors):
        actor_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(self.actor, actor_key, replay, learner, counter,
                           actor_id))

    return program
