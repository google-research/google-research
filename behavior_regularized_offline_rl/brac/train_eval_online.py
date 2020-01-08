# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Training and evaluation in the online mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

from absl import logging

import gin
import numpy as np
import tensorflow.compat.v1 as tf
from behavior_regularized_offline_rl.brac import dataset
from behavior_regularized_offline_rl.brac import policies
from behavior_regularized_offline_rl.brac import train_eval_utils
from behavior_regularized_offline_rl.brac import utils


@gin.configurable
def train_eval_online(
    # Basic args.
    log_dir,
    agent_module,
    env_name='HalfCheetah-v2',
    # Train and eval args.
    total_train_steps=int(1e6),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(1e8),
    eval_freq=5000,
    n_eval_episodes=20,
    # For saving a partially trained policy.
    eval_target=None,  # Target return value to stop training.
    eval_target_n=2,  # Stop after n consecutive evals above eval_target.
    # Agent train args.
    initial_explore_steps=10000,
    replay_buffer_size=int(1e6),
    model_params=(((200, 200),), 2),
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0.0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    ):
  """Training a policy with online interaction."""
  # Create tf_env to get specs.
  tf_env = train_eval_utils.env_factory(env_name)
  tf_env_test = train_eval_utils.env_factory(env_name)
  observation_spec = tf_env.observation_spec()
  action_spec = tf_env.action_spec()

  # Initialize dataset.
  with tf.device('/cpu:0'):
    train_data = dataset.Dataset(
        observation_spec,
        action_spec,
        replay_buffer_size,
        circular=True,
        )
  data_ckpt = tf.train.Checkpoint(data=train_data)
  data_ckpt_name = os.path.join(log_dir, 'replay')

  time_st_total = time.time()
  time_st = time.time()
  timed_at_step = 0

  # Collect data from random policy.
  explore_policy = policies.ContinuousRandomPolicy(action_spec)
  steps_collected = 0
  log_freq = 5000
  logging.info('Collecting data ...')
  collector = train_eval_utils.DataCollector(tf_env, explore_policy, train_data)
  while steps_collected < initial_explore_steps:
    count = collector.collect_transition()
    steps_collected += count
    if (steps_collected % log_freq == 0
        or steps_collected == initial_explore_steps) and count > 0:
      steps_per_sec = ((steps_collected - timed_at_step)
                       / (time.time() - time_st))
      timed_at_step = steps_collected
      time_st = time.time()
      logging.info('(%d/%d) steps collected at %.4g steps/s.', steps_collected,
                   initial_explore_steps, steps_per_sec)

  # Construct agent.
  agent_flags = utils.Flags(
      action_spec=action_spec,
      model_params=model_params,
      optimizers=optimizers,
      batch_size=batch_size,
      weight_decays=weight_decays,
      update_freq=update_freq,
      update_rate=update_rate,
      discount=discount,
      train_data=train_data)
  agent_args = agent_module.Config(agent_flags).agent_args
  agent = agent_module.Agent(**vars(agent_args))

  # Prepare savers for models and results.
  train_summary_dir = os.path.join(log_dir, 'train')
  eval_summary_dir = os.path.join(log_dir, 'eval')
  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_summary_dir)
  eval_summary_writers = collections.OrderedDict()
  for policy_key in agent.test_policies.keys():
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        os.path.join(eval_summary_dir, policy_key))
    eval_summary_writers[policy_key] = eval_summary_writer
  agent_ckpt_name = os.path.join(log_dir, 'agent')
  eval_results = []

  # Train agent.
  logging.info('Start training ....')
  time_st = time.time()
  timed_at_step = 0
  target_partial_policy_saved = False
  collector = train_eval_utils.DataCollector(
      tf_env, agent.online_policy, train_data)
  for _ in range(total_train_steps):
    collector.collect_transition()
    agent.train_step()
    step = agent.global_step
    if step % summary_freq == 0 or step == total_train_steps:
      agent.write_train_summary(train_summary_writer)
    if step % print_freq == 0 or step == total_train_steps:
      agent.print_train_info()

    if step % eval_freq == 0 or step == total_train_steps:
      time_ed = time.time()
      time_cost = time_ed - time_st
      logging.info(
          'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
      eval_result, eval_infos = train_eval_utils.eval_policies(
          tf_env_test, agent.test_policies, n_eval_episodes)
      eval_results.append([step] + eval_result)
      # Cecide whether to save a partially trained policy based on current model
      # performance.
      if (eval_target is not None and len(eval_results) >= eval_target_n
          and not target_partial_policy_saved):
        evals_ = list([eval_results[-(i + 1)][1]
                       for i in range(eval_target_n)])
        evals_ = np.array(evals_)
        if np.min(evals_) >= eval_target:
          agent.save(agent_ckpt_name + '_partial_target')
          dataset.save_copy(train_data, data_ckpt_name + '_partial_target')
          logging.info('A partially trained policy was saved at step %d,'
                       ' with episodic return %.4g.', step, evals_[-1])
          target_partial_policy_saved = True
      logging.info('Testing at step %d:', step)
      for policy_key, policy_info in eval_infos.items():
        logging.info(utils.get_summary_str(
            step=None, info=policy_info, prefix=policy_key + ': '))
        utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
      time_st = time.time()
      timed_at_step = step
    if step % save_freq == 0:
      agent.save(agent_ckpt_name + '-' + str(step))

  # Final save after training.
  agent.save(agent_ckpt_name + '_final')
  data_ckpt.write(data_ckpt_name + '_final')
  time_cost = time.time() - time_st_total
  logging.info('Training finished, time cost %.4gs.', time_cost)
  return np.array(eval_results)
