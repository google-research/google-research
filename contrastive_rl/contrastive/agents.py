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

"""Defines distributed contrastive RL agents, using JAX."""

import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.jax import utils
from acme.utils import loggers
from contrastive import builder
from contrastive import config as contrastive_config
from contrastive import distributed_layout
from contrastive import networks
from contrastive import utils as contrastive_utils

import dm_env


NetworkFactory = Callable[[specs.EnvironmentSpec],
                          networks.ContrastiveNetworks]


class DistributedContrastive(distributed_layout.DistributedLayout):
  """Distributed program definition for contrastive RL."""

  def __init__(
      self,
      environment_factory,
      network_factory,
      config,
      seed,
      num_actors,
      max_number_of_steps = None,
      log_to_bigtable = False,
      log_every = 10.0,
      evaluator_factories = None,
  ):
    # Check that the environment-specific parts of the config have been set.
    assert config.max_episode_steps > 0
    assert config.obs_dim > 0

    logger_fn = functools.partial(loggers.make_default_logger,
                                  'learner', log_to_bigtable,
                                  time_delta=log_every, asynchronous=True,
                                  serialize_fn=utils.fetch_devicearray,
                                  steps_key='learner_steps')
    contrastive_builder = builder.ContrastiveBuilder(config,
                                                     logger_fn=logger_fn)
    if evaluator_factories is None:
      eval_policy_factory = (
          lambda n: networks.apply_policy_and_sample(n, True))
      eval_observers = [
          contrastive_utils.SuccessObserver(),
          contrastive_utils.DistanceObserver(
              obs_dim=config.obs_dim,
              start_index=config.start_index,
              end_index=config.end_index)
      ]
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=eval_policy_factory,
              log_to_bigtable=log_to_bigtable,
              observers=eval_observers)
      ]
      if config.local:
        evaluator_factories = []
    actor_observers = [
        contrastive_utils.SuccessObserver(),
        contrastive_utils.DistanceObserver(obs_dim=config.obs_dim,
                                           start_index=config.start_index,
                                           end_index=config.end_index)]
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        network_factory=network_factory,
        builder=contrastive_builder,
        policy_network=networks.apply_policy_and_sample,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
        observers=actor_observers,
        checkpointing_config=distributed_layout.CheckpointingConfig())
