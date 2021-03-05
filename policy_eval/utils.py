# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utilities for loading data."""
import functools
import numpy as np
import tensorflow as tf
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import actor_policy
from tf_agents.trajectories import time_step
import tqdm

import policy_eval.actor as actor_lib


def estimate_monte_carlo_returns(env,
                                 discount,
                                 actor,
                                 std,
                                 num_episodes,
                                 max_length = 1000):
  """Estimate policy returns using with Monte Carlo.

  Args:
    env: Learning environment.
    discount: MDP discount.
    actor: Policy to estimate returns for.
    std: Fixed policy standard deviation.
    num_episodes: Number of episodes.
    max_length: Maximum length of episodes.

  Returns:
    A dictionary that contains trajectories.
  """

  durations = []
  for gym_env in env.pyenv.envs:
    if hasattr(gym_env, 'duration'):
      durations.append(gym_env.duration)
      gym_env._duration = max_length  # pylint: disable=protected-access

  episode_return_sum = 0.0
  for _ in tqdm.tqdm(range(num_episodes), desc='Estimation Returns'):
    timestep = env.reset()
    episode_return = 0.0

    t = 0
    while not timestep.is_last():
      _, action, _ = actor(timestep.observation, std)
      timestep = env.step(action)
      episode_return += timestep.reward[0] * (discount**t)
      t += 1

    episode_return_sum += episode_return

  for i, gym_env in enumerate(env.pyenv.envs):
    if hasattr(gym_env, 'duration'):
      gym_env._duration = durations[i]  # pylint: disable=protected-access

  return episode_return_sum / num_episodes * (1 - discount)


def get_d4rl_policy(env, weights, is_dapg=False):
  """Creates TF Agents policy based from D4RL saved weights."""
  hidden_dims = []
  fc_idx = 0
  while 'fc%d/weight' % fc_idx in weights:
    hidden_dims.append(np.shape(weights['fc0/weight'])[0])
    fc_idx += 1

  if is_dapg:
    activation_fn = tf.keras.activations.tanh
    continuous_projection_net = functools.partial(
        normal_projection_network.NormalProjectionNetwork,
        mean_transform=None,
        std_transform=tf.exp,
        state_dependent_std=True)
  else:
    activation_fn = tf.keras.activations.relu
    continuous_projection_net = functools.partial(
        tanh_normal_projection_network.TanhNormalProjectionNetwork,
        std_transform=lambda x: tf.exp(tf.clip_by_value(x, -5., 2.)))

  actor_net = actor_distribution_network.ActorDistributionNetwork(
      env.observation_spec(),
      env.action_spec(),
      fc_layer_params=hidden_dims,
      continuous_projection_net=continuous_projection_net,
      activation_fn=activation_fn)
  policy = actor_policy.ActorPolicy(
      time_step_spec=env.time_step_spec(),
      action_spec=env.action_spec(),
      actor_network=actor_net,
      training=False)

  # Set weights
  # pylint: disable=protected-access
  for fc_idx in range(len(hidden_dims)):
    actor_net._encoder.layers[fc_idx + 1].set_weights(
        [weights['fc%d/weight' % fc_idx].T, weights['fc%d/bias' % fc_idx]])

  if is_dapg:
    actor_net._projection_networks.layers[0].set_weights(
        [weights['last_fc/weight'].T, weights['last_fc/bias']])
    actor_net._projection_networks.layers[1].set_weights(
        [weights['last_fc_log_std/weight'].T, weights['last_fc_log_std/bias']])
  else:
    actor_net._projection_networks.layers[0].set_weights(
        [np.concatenate(
            (weights['last_fc/weight'], weights['last_fc_log_std/weight']),
            axis=0).T,
         np.concatenate(
             (weights['last_fc/bias'], weights['last_fc_log_std/bias']),
             axis=0)])
  # pylint: enable=protected-access
  return policy


class D4rlActor(object):
  """Actor wrapper for D4RL policies."""

  def __init__(self, env, weights, is_dapg=False):
    self.policy = get_d4rl_policy(env, weights, is_dapg=is_dapg)

  def __call__(self, states, std=None, actions=None):
    mode = None
    tfagents_step = time_step.TimeStep(0, 0, 0, states)
    if actions is None:
      samples = self.policy.action(tfagents_step).action
      log_probs = None
    else:
      samples = None
      log_probs = self.policy.distribution(tfagents_step).action.log_prob(
          actions)
    return mode, samples, log_probs
