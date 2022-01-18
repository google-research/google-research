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

"""Utilities for training and evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from behavior_regularized_offline_rl.brac import dataset


def eval_policy_episodes(env, policy, n_episodes):
  """Evaluates policy performance."""
  results = []
  for _ in range(n_episodes):
    time_step = env.reset()
    total_rewards = 0.0
    while not time_step.is_last().numpy()[0]:
      action = policy(time_step.observation)[0]
      time_step = env.step(action)
      total_rewards += time_step.reward
    results.append(total_rewards)
  results = np.array(results)
  return float(np.mean(results)), float(np.std(results))


def eval_policies(env, policies, n_episodes):
  results_episode_return = []
  infos = collections.OrderedDict()
  for name, policy in policies.items():
    mean, _ = eval_policy_episodes(env, policy, n_episodes)
    results_episode_return.append(mean)
    infos[name] = collections.OrderedDict()
    infos[name]['episode_mean'] = mean
  results = results_episode_return
  return results, infos


# TODO(wuyifan): external version for loading environments
def env_factory(env_name):
  py_env = suite_mujoco.load(env_name)
  tf_env = tf_py_environment.TFPyEnvironment(py_env)
  return tf_env


def get_transition(time_step, next_time_step, action, next_action):
  return dataset.Transition(
      s1=time_step.observation,
      s2=next_time_step.observation,
      a1=action,
      a2=next_action,
      reward=next_time_step.reward,
      discount=next_time_step.discount)


class DataCollector(object):
  """Class for collecting sequence of environment experience."""

  def __init__(self, tf_env, policy, data):
    self._tf_env = tf_env
    self._policy = policy
    self._data = data
    self._saved_action = None

  def collect_transition(self):
    """Collect single transition from environment."""
    time_step = self._tf_env.current_time_step()
    if self._saved_action is None:
      self._saved_action = self._policy(time_step.observation)[0]
    action = self._saved_action
    next_time_step = self._tf_env.step(action)
    next_action = self._policy(next_time_step.observation)[0]
    self._saved_action = next_action
    if not time_step.is_last()[0].numpy():
      transition = get_transition(time_step, next_time_step,
                                  action, next_action)
      self._data.add_transitions(transition)
      return 1
    else:
      return 0
