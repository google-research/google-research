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

"""Tests for joint attention training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.system import system_multiprocessing
from social_rl.multiagent_tfagents import multiagent_train_eval
from social_rl.multiagent_tfagents.joint_attention import attention_ppo_agent


class MultiagentTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for multi-agent training in multigrid."""

  def test_import(self):
    self.assertIsNotNone(multiagent_train_eval)

  def test_attention(self):
    agent_class = functools.partial(attention_ppo_agent.MultiagentAttentionPPO,
                                    attention_bonus_type='kld')
    root_dir = '/tmp/attention/'
    if tf.io.gfile.exists(root_dir):
      tf.compat.v1.gfile.DeleteRecursively(root_dir)
    multiagent_train_eval.train_eval(
        root_dir=root_dir,
        agent_class=agent_class,
        env_name='MultiGrid-Meetup-Empty-6x6-v0',
        num_environment_steps=1,
        num_epochs=2,
        replay_buffer_capacity=401,
        collect_episodes_per_iteration=2,
        train_checkpoint_interval=500,
        policy_checkpoint_interval=500,
        log_interval=1500,
        summary_interval=500,
        num_parallel_environments=2,
        num_eval_episodes=1,
        actor_fc_layers=(2,),
        value_fc_layers=(2,),
        lstm_size=(2,),
        conv_filters=2,
        conv_kernel=2,
        direction_fc=2,
        use_attention_networks=True)
    train_exists = tf.io.gfile.exists(os.path.join(root_dir, 'train'))
    self.assertTrue(train_exists)
    saved_policies = tf.io.gfile.listdir(
        os.path.join(root_dir, 'policy_saved_model'))
    self.assertGreaterEqual(len(saved_policies), 1)


if __name__ == '__main__':
  system_multiprocessing.handle_test_main(tf.test.main)
