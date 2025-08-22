# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Decentralized multiagent factories.

Used to unify agent initialization for both local and distributed layouts.
"""

import enum
import functools
from typing import Any, Callable, Dict, Mapping, Optional

from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import builders as jax_builders
from acme.agents.jax import ppo
from acme.multiagent import types as ma_types
from acme.multiagent import utils as ma_utils
from acme.utils import loggers

from concept_marl import concept_ppo


class DefaultSupportedAgent(enum.Enum):
  """Agents which have default initializers supported below."""
  PPO = 'PPO'
  CONCEPT_PPO = 'ConceptPPO'


def init_default_network(
    agent_type,
    agent_spec):
  """Returns default networks for a single agent."""
  if agent_type == DefaultSupportedAgent.PPO:
    return ppo.make_networks(agent_spec)
  elif agent_type == DefaultSupportedAgent.CONCEPT_PPO:
    return concept_ppo.make_networks(agent_spec)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def init_default_policy_network(
    agent_type,
    network,
    agent_spec,
    config,
    eval_mode = False):
  """Returns default policy network for a single agent."""
  del agent_spec, config
  if agent_type == DefaultSupportedAgent.PPO:
    return ppo.make_inference_fn(network, evaluation=eval_mode)
  elif agent_type == DefaultSupportedAgent.CONCEPT_PPO:
    return concept_ppo.make_inference_fn(network, evaluation=eval_mode)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def init_default_builder(
    agent_type,
    agent_config,
):
  """Returns default builder for a single agent."""
  if agent_type == DefaultSupportedAgent.PPO:
    assert isinstance(agent_config, ppo.PPOConfig)
    return ppo.PPOBuilder(agent_config)
  elif agent_type == DefaultSupportedAgent.CONCEPT_PPO:
    assert isinstance(agent_config, concept_ppo.ConceptPPOConfig)
    return concept_ppo.ConceptPPOBuilder(agent_config)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def init_default_config(
    agent_type,
    config_overrides):
  """Returns default config for a single agent."""
  if agent_type == DefaultSupportedAgent.PPO:
    return ppo.PPOConfig(**config_overrides)
  elif agent_type == DefaultSupportedAgent.CONCEPT_PPO:
    return concept_ppo.ConceptPPOConfig(**config_overrides)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def default_logger_factory(
    agent_types,
    base_label,
    save_data,
    time_delta = 1.0,
    asynchronous = False,
    print_fn = None,
    serialize_fn = None,
    steps_key = 'steps',
):
  """Returns callable that constructs default logger for all agents."""
  logger_fns = {}
  for agent_id in agent_types.keys():
    logger_fns[agent_id] = functools.partial(
        loggers.make_default_logger,
        f'{base_label}{agent_id}',
        save_data=save_data,
        time_delta=time_delta,
        asynchronous=asynchronous,
        print_fn=print_fn,
        serialize_fn=serialize_fn,
        steps_key=steps_key,
    )
  return logger_fns


def default_config_factory(
    agent_types,
    batch_size,
    config_overrides = None
):
  """Returns default configs for all agents.

  Args:
    agent_types: dict mapping agent IDs to their type.
    batch_size: shared batch size for all agents.
    config_overrides: dict mapping (potentially a subset of) agent IDs to their
      config overrides. This should include any mandatory config parameters for
      the agents that do not have default values.
  """
  configs = {}
  for agent_id, agent_type in agent_types.items():
    agent_config_overrides = dict(
        # batch_size is required by LocalLayout, which is shared amongst
        # the agents. Hence, we enforce a shared batch_size in builders.
        batch_size=batch_size,
        # Unique replay_table_name per agent.
        replay_table_name=f'{adders_reverb.DEFAULT_PRIORITY_TABLE}_agent{agent_id}'
    )
    if config_overrides is not None and agent_id in config_overrides:
      agent_config_overrides = {
          **config_overrides[agent_id],
          **agent_config_overrides  # Comes second to ensure batch_size override
      }
    configs[agent_id] = init_default_config(agent_type, agent_config_overrides)
  return configs


def network_factory(
    environment_spec,
    agent_types,
    init_network_fn = None
):
  """Returns networks for all agents.

  Args:
    environment_spec: environment spec.
    agent_types: dict mapping agent IDs to their type.
    init_network_fn: optional callable that handles the network initialization
      for all sub-agents. If this is not supplied, a default network initializer
      is used (if it is supported for the designated agent type).
  """
  init_fn = init_network_fn or init_default_network
  networks = {}
  for agent_id, agent_type in agent_types.items():
    single_agent_spec = ma_utils.get_agent_spec(environment_spec, agent_id)
    networks[agent_id] = init_fn(agent_type, single_agent_spec)
  return networks


def policy_network_factory(
    networks,
    environment_spec,
    agent_types,
    agent_configs,
    eval_mode,
    init_policy_network_fn = None
):
  """Returns default policy networks for all agents.

  Args:
    networks: dict mapping agent IDs to their networks.
    environment_spec: environment spec.
    agent_types: dict mapping agent IDs to their type.
    agent_configs: dict mapping agent IDs to their config.
    eval_mode: whether the policy should be initialized in evaluation mode (only
      used if an init_policy_network_fn is not explicitly supplied).
    init_policy_network_fn: optional callable that handles the policy network
      initialization for all sub-agents. If this is not supplied, a default
      policy network initializer is used (if it is supported for the designated
      agent type).
  """
  init_fn = init_policy_network_fn or init_default_policy_network
  policy_networks = {}
  for agent_id, agent_type in agent_types.items():
    single_agent_spec = ma_utils.get_agent_spec(environment_spec, agent_id)
    policy_networks[agent_id] = init_fn(agent_type, networks[agent_id],
                                        single_agent_spec,
                                        agent_configs[agent_id], eval_mode)
  return policy_networks


def builder_factory(
    agent_types,
    agent_configs,
    init_builder_fn = None
):
  """Returns default policy networks for all agents."""
  init_fn = init_builder_fn or init_default_builder
  builders = {}
  for agent_id, agent_type in agent_types.items():
    builders[agent_id] = init_fn(agent_type, agent_configs[agent_id])
  return builders
