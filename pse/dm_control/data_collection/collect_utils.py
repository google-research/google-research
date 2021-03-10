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

# Lint as: python3
"""Utilities for data collection on Distracting DM control environments."""

import os

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.drivers import py_driver
from tf_agents.metrics import py_metrics
from tf_agents.policies import scripted_py_policy
from tf_agents.specs import tensor_spec

from pse.dm_control import env_utils as utils

gfile = tf.compat.v1.gfile


def get_expanded_dir(current_dir, env_name, trial_id, check=True):
  expanded_dir = os.path.join(current_dir, env_name, trial_id)
  if not gfile.Exists(expanded_dir):
    if check:
      raise ValueError(f'{expanded_dir} doesn\'t exist')
    else:
      gfile.MakeDirs(expanded_dir)
  return expanded_dir


def run_env(env, policy, max_episodes, max_steps=None):
  logging.info('Running policy on env ..')
  replay_buffer = []
  metrics = [
      py_metrics.AverageReturnMetric(),
      py_metrics.AverageEpisodeLengthMetric()
  ]
  observers = [replay_buffer.append]
  observers.extend(metrics)
  driver = py_driver.PyDriver(
      env, policy, observers, max_steps=max_steps, max_episodes=max_episodes)
  initial_time_step = env.reset()
  initial_state = policy.get_initial_state(1)
  driver.run(initial_time_step, initial_state)
  return replay_buffer, metrics


def get_complete_episodes(replay_buffer, num_episodes=2):
  terminal_steps = [int(x.next_step_type) for x in replay_buffer]
  episode_boundaries = np.where(np.array(terminal_steps) == 2)[0]
  episode_boundaries = np.append(episode_boundaries[::-1], [-2])[::-1]
  return [replay_buffer[episode_boundaries[i] + 2: episode_boundaries[i+1] + 1]
          for i in range(num_episodes)]


def collect_pair_episodes(
    policy,
    env_name,
    max_steps=None,
    random_seed=None,
    frame_shape=(84, 84, 3),
    max_episodes=10):
  env = utils.load_dm_env_for_eval(
      env_name,
      frame_shape=frame_shape,
      task_kwargs={'random': random_seed})

  buffer, metrics = run_env(
      env, policy, max_steps=max_steps, max_episodes=max_episodes)

  # Collect episodes with the same optimal policy
  env_copy = utils.load_dm_env_for_eval(
      env_name,
      frame_shape=(84, 84, 3),
      task_kwargs={'random': random_seed})

  actions = [x.action for x in buffer]
  action_script = list(zip([1] * len(actions), actions))
  optimal_policy = scripted_py_policy.ScriptedPyPolicy(
      time_step_spec=env.time_step_spec(), action_spec=env.action_spec(),
      action_script=action_script)
  paired_buffer, paired_metrics = run_env(
      env_copy, optimal_policy, max_steps=max_steps, max_episodes=max_episodes)

  for metric, paired_metric in zip(metrics, paired_metrics):
    assert metric.result() == paired_metric.result(), (
        'Metric results don\'t match')
    logging.info('%s: %.2f', metric.name, metric.result())

  episodes = get_complete_episodes(buffer, max_episodes)
  paired_episodes = get_complete_episodes(paired_buffer, max_episodes)
  return episodes, paired_episodes


def create_tensor_specs(data_spec, episode_len):
  spec = tuple([data_spec for _ in range(episode_len)])
  tensor_data_spec = tensor_spec.from_spec(data_spec)
  tensor_episode_spec = tensor_spec.from_spec((spec, spec))
  return tensor_data_spec, tensor_episode_spec
