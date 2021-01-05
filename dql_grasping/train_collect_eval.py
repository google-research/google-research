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

"""Runs the Train, Collection, Evaluation loop for dql_grasping.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import tensorflow.compat.v1 as tf
from dql_grasping import input_data
from dql_grasping import run_env
from dql_grasping import train_q


@gin.configurable
def train_collect_eval(collect_env,
                       eval_env,
                       test_env,
                       policy_class,
                       run_agent_fn=run_env.run_env,
                       train_fn=train_q.train_q,
                       do_collect_eval=True,
                       file_patterns='',
                       get_data_fn=input_data.get_data,
                       onpolicy=True,
                       num_collect=100,
                       num_eval=100,
                       num_test=100,
                       data_format='tfrecord',
                       eval_frequency=5,
                       root_dir=None,
                       task=0,
                       master='',
                       ps_tasks=0):
  """Runs synchronous train, collect, eval loop.

  This loop instantiates the policy instance from policy_class. The policy
  manages its own tf.Session. The train function may create its own session for
  the purpose of updating its variables.

  train_fn reuses graph created by policy, to avoid having to
  configure the same neural net twice (one for policy and one for training.)

  Args:
    collect_env: (gym.Env) Gym environment to collect data from (and train the
      policy on).
    eval_env: (gym.Env) Gym environment to evaluate the policy on. Can be
      another instance of collect_env, or a different environment if one wishes
      to evaluate generalization capability. The only constraint is that the
      action and observation spaces have to be equivalent. If None, eval_env
      is not evaluated.
    test_env: (gym.Env) Another environment to evaluate on.  Either another
      instance of collect_env, or a different environment to evaluate
      generalization.
    policy_class: Policy class that we want to train.
    run_agent_fn: (Optional) Python function that executes the interaction of
      the policy with the environment. Defaults to run_env.run_env.
    train_fn: (Optional) Python function that trains the policy. Defaults to
      train_q.train_q.
    do_collect_eval: If True, performs data collection using the trained policy.
    file_patterns: (str) Comma-separated regex of file patterns to train on.
      This is used to instantiate the file-backed "replay buffer".
    get_data_fn: (Optional) Python function that fetches data from files.
    onpolicy: (bool) If True, appends data from policy_collect directory.
    num_collect: (int) Number of episodes to collect & evaluate from
      collect_env.
    num_eval: (int) Number of episodes to collect & evaluate from eval_env.
    num_test: (int) Number of episodes to collect & evaluate from test_env.
    data_format: (string) File extension of input data files.
    eval_frequency: (int) How many times we run eval/test vs. collect.
      Evaluating is costly compared to training, so we can speed up iteration
      time by not evaluating every time we collect.
    root_dir: (str) Root directory for this training trial. Training directory,
      eval directory are subdirectories of root_dir.
    task: (int) Optional worker task for distributed training. Defaults to solo
      master task on a single machine
    master: (int) Optional address of master worker. Specify this when doing
      distributed training.
    ps_tasks: (int) Optional number of parameter-server tasks. Used only for
      distributed TF training jobs.

  Raises:
    ValueError: If ps_tasks > 0 (implies distributed training) while
      do_collect_eval is set to True.
  """
  # Spaces do not implement `==` operator. Convert to strings to check
  # compatibility between training & eval env representation.
  if ((collect_env and eval_env) and
      (str(collect_env.observation_space), str(collect_env.action_space)) !=
      (str(eval_env.observation_space), str(eval_env.action_space))):
    raise ValueError('Collect and Eval environments have incompatible '
                     'observation or action dimensions.')
  if ps_tasks > 0 and do_collect_eval:
    raise ValueError(
        'Collecting data not supported by distributed training jobs')
  if onpolicy:
    file_patterns += ',' + os.path.join(
        root_dir, 'policy_collect', '*.%s' % data_format)
  train_dir = os.path.join(root_dir, 'train')
  it = 0
  while True:
    tf.reset_default_graph()
    # Re-fresh the source of data.
    with tf.Graph().as_default():
      with tf.device(tf.train.replica_device_setter(ps_tasks)):
        policy = policy_class()
        if train_fn:
          dataset = get_data_fn(file_patterns=file_patterns)
          step, done = train_fn(dataset, policy, log_dir=train_dir, reuse=True,
                                task=task, master=master)
        else:
          step, done = 0, True
        if train_fn:
          tf.logging.info('Evaluating policy at step %d' % step)
          ckpt = tf.train.latest_checkpoint(train_dir)
          tf.logging.info('Restoring model variables from %s' % ckpt)
          policy.restore(ckpt)
          if ckpt:
            step = int(ckpt.split('.ckpt-')[-1])
        if onpolicy:
          run_agent_fn(collect_env, policy=policy, global_step=step,
                       root_dir=root_dir, task=task, num_episodes=num_collect,
                       tag='collect')

        if it % eval_frequency == 0:
          if eval_env:
            run_agent_fn(eval_env, policy=policy, global_step=step,
                         root_dir=root_dir, task=task, explore_schedule=None,
                         num_episodes=num_eval, tag='eval')
          if test_env:
            run_agent_fn(test_env, policy=policy, global_step=step,
                         root_dir=root_dir, task=task, explore_schedule=None,
                         num_episodes=num_test, tag='test')

        it += 1
      if done:
        tf.logging.info('Train-Collect-Eval completed.')
        break
