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

"""Tests for Social RL, including multi-agent and adversarial environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.system import system_multiprocessing

from social_rl.adversarial_env import train_adversarial_env


class AdversarialEnvTest(tf.test.TestCase, parameterized.TestCase):
  """Test adversarial environment generation with PAIRED."""

  def test_import(self):
    self.assertIsNotNone(train_adversarial_env)

  def test_paired(self):
    root_dir = '/tmp/adversarial_env/'
    if tf.io.gfile.exists(root_dir):
      tf.compat.v1.gfile.DeleteRecursively(root_dir)
    train_adversarial_env.train_eval(
        root_dir=root_dir,
        env_name='MultiGrid-MiniAdversarial-v0',
        actor_fc_layers=(2,),
        value_fc_layers=(2,),
        lstm_size=(2,),
        conv_filters=2,
        conv_kernel=3,
        direction_fc=1,
        adversary_env_rnn=False,
        adv_actor_fc_layers=(2,),
        adv_value_fc_layers=(2,),
        adv_lstm_size=(2,),
        adv_conv_filters=2,
        adv_conv_kernel=3,
        adv_timestep_fc=1,
        num_train_steps=3,
        collect_episodes_per_iteration=2,
        num_parallel_envs=2,
        replay_buffer_capacity=401,
        num_epochs=2,
        num_eval_episodes=1,
        eval_interval=10,
        train_checkpoint_interval=500,
        policy_checkpoint_interval=500,
        log_interval=500,
        summary_interval=500,
        debug_summaries=False,
        summarize_grads_and_vars=False)
    train_exists = tf.io.gfile.exists(os.path.join(root_dir, 'train'))
    self.assertTrue(train_exists)
    saved_policies = tf.io.gfile.listdir(
        os.path.join(root_dir, 'policy_saved_model'))
    self.assertGreaterEqual(len(saved_policies), 1)

if __name__ == '__main__':
  system_multiprocessing.handle_test_main(tf.test.main)
