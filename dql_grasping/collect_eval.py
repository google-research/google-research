# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Collect/Eval a policy on the live environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import tensorflow.compat.v1 as tf

from dql_grasping import run_env


@gin.configurable
def collect_eval(collect_env,
                 eval_env,
                 policy_class,
                 num_collect=2000,
                 num_eval=100,
                 run_agent_fn=run_env.run_env,
                 root_dir=''):
  """Modified version of train_collect_eval() that only does collect and eval.

  Args:
    collect_env: (gym.Env) Gym environment to collect data from (and train the
      policy on).
    eval_env: (gym.Env) Gym environment to evaluate the policy on. Can be
      another instance of collect_env, or a different environment if one wishes
      to evaluate generalization capability. The only constraint is that the
      action and observation spaces have to be equivalent. If None, eval_env
      is not evaluated.
    policy_class: Policy class that we want to train.
    num_collect: (int) Number of episodes to collect from collect_env.
    num_eval: (int) Number of episodes to evaluate from eval_env.
    run_agent_fn: (Optional) Python function that executes the interaction of
      the policy with the environment. Defaults to run_env.run_env.
    root_dir: Base directory where checkpoint, collect data, and eval data are
      stored.
  """
  policy = policy_class()
  ckpt = None
  model_dir = os.path.join(root_dir, 'train')
  collect_dir = os.path.join(root_dir, 'policy_collect')
  eval_dir = os.path.join(root_dir, 'eval')
  ckpt = tf.train.latest_checkpoint(model_dir)
  if ckpt:
    tf.logging.info('Restoring model variables from %s', ckpt)
    policy.restore(ckpt)
  if collect_env:
    run_agent_fn(collect_env, policy=policy, num_episodes=num_collect,
                 root_dir=collect_dir)
  if eval_env:
    run_agent_fn(eval_env, policy=policy, num_episodes=num_eval,
                 root_dir=eval_dir)

