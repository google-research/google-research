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

"""Example running Concept PPO in meltingpot with Acme."""

from typing import Callable, Dict

from absl import app
from absl import flags
from acme import specs
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.multiagent import types as ma_types
import dm_env
import optax

from concept_marl.experiments import helpers
from concept_marl.experiments.meltingpot.wrappers import wrapper_utils
from concept_marl.utils import builder as mp_builder
from concept_marl.utils import factories as mp_factories


_ENV_NAME = flags.DEFINE_string('env_name', 'cooking_basic',
                                'Name of the environment to run.')
_EPISODE_LENGTH = flags.DEFINE_integer('episode_length', 100,
                                       'Max number of steps in episode.')
_NUM_STEPS = flags.DEFINE_integer('num_steps', 10000,
                                  'Number of env steps to run training for.')
_EVAL_EVERY = flags.DEFINE_integer('eval_every', 1000,
                                   'How often to run evaluation.')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_LR_START = flags.DEFINE_float('learning_rate_start', 5e-4, 'Learning rate.')
_LR_END = flags.DEFINE_float('learning_rate_end', 5e-7, 'Learning rate.')
_LR_DECAY = flags.DEFINE_integer('learning_rate_decay_steps', 100_000,
                                 'Learning rate.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 256, 'Batch size.')
_UNROLL_LENGTH = flags.DEFINE_integer('unroll_length', 16,
                                      'Unroll length for PPO.')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 5, 'Num epochs for PPO.')
_NUM_MINIBATCHES = flags.DEFINE_integer('num_minibatches', 32,
                                        'Num minibatches for PPO.')
_PPO_CLIPPING_EPSILON = flags.DEFINE_float('ppo_clipping_epsilon', 0.2,
                                           'Clipping epsilon for PPO.')
_ENTROPY_COST = flags.DEFINE_float('entropy_cost', 0.01,
                                   'Entropy cost weight for PPO.')
_VALUE_COST = flags.DEFINE_float('value_cost', 1.0,
                                 'Value cost weight for PPO.')
_MAX_GRAD_NORM = flags.DEFINE_float('max_gradient_norm', 0.5,
                                    'Global gradient clip for PPO.')
# concept related flags
_CONCEPT_COST = flags.DEFINE_float('concept_cost', 0.1,
                                   'Concept cost weight for PPO.')
# clean up flags
_EAT_REWARD = flags.DEFINE_float('eat_reward', 0.1,
                                 'Local reward for eating an apple.')
_CLEAN_REWARD = flags.DEFINE_float('clean_reward', 0.005,
                                   'Local reward for cleaning river.')


def get_env(env_name, seed, episode_length):
  """Initializes and returns meltingpot environment."""
  env_config = dict(
      env_name=env_name,
      action_type='flat',
      grayscale=False,
      scale_dims=(40, 40),
      episode_length=episode_length,
      seed=seed)

  # standard melting pot wrapper
  if 'cooking' in env_name:
    env_config['dense_rewards'] = True
    env = wrapper_utils.make_and_wrap_cooking_environment(**env_config)
  elif 'clean' in env_name:
    env_config['dense_rewards'] = True
    env_config['clean_reward'] = _CLEAN_REWARD.value
    env_config['eat_reward'] = _EAT_REWARD.value
    env = wrapper_utils.make_and_wrap_cleanup_environment(**env_config)
  elif 'capture' in env_name:
    env = wrapper_utils.make_and_wrap_capture_environment(**env_config)
  else:
    raise ValueError('Invalid environment choice!')

  # envlogger book-keeping
  env_config['env_name_for_get_env'] = env_name
  env_config['num_steps'] = episode_length
  env.n_agents = env.num_agents
  return env, env_config


def _make_environment_factory(env_name):

  def environment_factory(seed):
    environment, _ = get_env(env_name, seed, _EPISODE_LENGTH.value)
    return environment

  return environment_factory


def _make_network_factory(
    agent_types
):
  """Returns a network factory for meltingpot experiments."""

  def network_factory(
      environment_spec):
    return mp_factories.network_factory(
        environment_spec,
        agent_types,
        init_network_fn=helpers.init_default_meltingpot_network)

  return network_factory


def build_experiment_config():
  """Returns a config for meltingpot experiments."""

  # init environment
  environment_factory = _make_environment_factory(_ENV_NAME.value)
  environment = environment_factory(_SEED.value)

  # init learning rate schedule
  learning_rate = optax.polynomial_schedule(
      init_value=_LR_START.value,
      end_value=_LR_END.value,
      power=1,
      transition_steps=_LR_DECAY.value)

  # init Concept PPO agent
  agent_types = {
      str(i): mp_factories.DefaultSupportedAgent.CONCEPT_PPO
      for i in range(environment.num_agents)  # pytype: disable=attribute-error
  }
  config_overrides = {  # pylint: disable=g-complex-comprehension
      agent_id: {
          'learning_rate': learning_rate,
          'batch_size': _BATCH_SIZE.value,
          'unroll_length': _UNROLL_LENGTH.value,
          'num_minibatches': _NUM_MINIBATCHES.value,
          'num_epochs': _NUM_EPOCHS.value,
          'ppo_clipping_epsilon': _PPO_CLIPPING_EPSILON.value,
          'entropy_cost': _ENTROPY_COST.value,
          'value_cost': _VALUE_COST.value,
          'concept_cost': _CONCEPT_COST.value,
          'max_gradient_norm': _MAX_GRAD_NORM.value,
          'clip_value': False,
      } for agent_id in agent_types.keys()
  }

  # init configs from agents
  configs = mp_factories.default_config_factory(agent_types, _BATCH_SIZE.value,
                                                config_overrides)

  builder = mp_builder.DecentralizedMultiAgentBuilder(
      agent_types=agent_types,
      agent_configs=configs,
      init_policy_network_fn=mp_factories.init_default_policy_network,
      init_builder_fn=mp_factories.init_default_builder)

  return experiments.ExperimentConfig(
      builder=builder,
      environment_factory=environment_factory,
      network_factory=_make_network_factory(agent_types=agent_types),
      seed=_SEED.value,
      max_num_actor_steps=_NUM_STEPS.value)


def main(_):
  config = build_experiment_config()
  experiments.run_experiment(
      experiment=config, eval_every=_EVAL_EVERY.value, num_eval_episodes=5)


if __name__ == '__main__':
  app.run(main)
