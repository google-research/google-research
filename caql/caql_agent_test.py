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

"""Tests for caql_agent."""

import os.path
from unittest import mock

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf

from caql import agent_policy
from caql import caql_agent
from caql import gaussian_noise_policy
from caql import replay_memory as replay_memory_lib
from caql import utils

tf.disable_v2_behavior()
tf.enable_resource_variables()


def _create_agent(session, state_spec, action_spec):
  return caql_agent.CaqlAgent(
      session=session,
      state_spec=state_spec,
      action_spec=action_spec,
      discount_factor=.99,
      hidden_layers=[32, 16],
      learning_rate=.001,
      learning_rate_action=.005,
      learning_rate_ga=.01,
      action_maximization_iterations=20,
      tau_copy=.001,
      clipped_target_flag=True,
      hard_update_steps=5000,
      batch_size=64,
      l2_loss_flag=True,
      simple_lambda_flag=False,
      dual_filter_clustering_flag=False,
      solver='gradient_ascent',
      dual_q_label=True,
      initial_lambda=1.,
      tolerance_min_max=[1e-4, 100.])


class CaqlAgentTest(tf.test.TestCase):

  def setUp(self):
    super(CaqlAgentTest, self).setUp()
    seed = 9999
    logging.info('Setting the numpy seed to %d', seed)
    np.random.seed(seed)

  def testInitialize(self):
    env = utils.create_env('Pendulum')
    state_spec, action_spec = utils.get_state_and_action_specs(env)
    with self.test_session() as sess:
      agent = _create_agent(sess, state_spec, action_spec)
      self.assertEqual(0, agent.initialize(saver=None))

  def testInitializeWithCheckpoint(self):
    # Create an agent, train 17 steps, and save a checkpoint.
    env = utils.create_env('Pendulum')
    state_spec, action_spec = utils.get_state_and_action_specs(env)
    replay_memory = replay_memory_lib.ReplayMemory(name='ReplayBuffer',
                                                   capacity=100000)
    save_path = os.path.join(self.get_temp_dir(), 'test_checkpoint')
    with self.test_session(graph=tf.Graph()) as sess:
      agent = _create_agent(sess, state_spec, action_spec)
      saver = tf.train.Saver()
      step = agent.initialize(saver)
      self.assertEqual(0, step)

      greedy_policy = agent_policy.AgentPolicy(action_spec, agent)
      behavior_policy = gaussian_noise_policy.GaussianNoisePolicy(
          greedy_policy, 1., .99, .01)

      for _ in range(100):
        env = utils.create_env('Pendulum')
        episode, _, _ = utils._collect_episode(
            env=env, time_out=200, discount_factor=.99,
            behavior_policy=behavior_policy)
        replay_memory.extend(episode)
        if hasattr(env, 'close'):
          env.close()

      while step < 17:
        minibatch = replay_memory.sample_with_replacement(64)
        (_, _, _, best_train_label_batch, _, _) = (
            agent.train_q_function_network(minibatch, None, None))
        agent.train_action_function_network(best_train_label_batch)
        step += 1
      saver.save(sess, save_path)

    # Create an agent and restore TF variables from the checkpoint.
    with self.test_session() as sess:
      agent = _create_agent(sess, state_spec, action_spec)
      saver = tf.train.Saver()
      self.assertEqual(17, agent.initialize(saver, self.get_temp_dir()))

  def testComputeClusterMasks(self):
    states = np.random.uniform(-1., 1., (5, 2))
    mask = np.array([True, False, True, False, True])
    cluster_mask, noncluster_mask, cluster_info = (
        caql_agent.CaqlAgent.compute_cluster_masks(states, mask))
    self.assertEqual([True, False, True, False, True], cluster_mask.tolist())
    self.assertEqual([False] * 5, noncluster_mask.tolist())
    self.assertEmpty(cluster_info)

  def testComputeClusterMasksWithLargeEpsilon(self):
    states = np.random.uniform(-1., 1., (5, 2))
    mask = np.array([True, False, True, False, True])
    cluster_mask, noncluster_mask, cluster_info = (
        caql_agent.CaqlAgent.compute_cluster_masks(states, mask, 10))
    self.assertEqual([False, False, True, False, False], cluster_mask.tolist())
    self.assertEqual([True, False, False, False, True],
                     noncluster_mask.tolist())
    self.assertAllClose(
        [{'non_cluster_index': 0,
          'centroid': (2, np.array([-0.69108876, -0.62607341]))},
         {'non_cluster_index': 4,
          'centroid': (2, np.array([-0.69108876, -0.62607341]))}
        ], cluster_info)

  def testComputeClusterMasksWithAllZeroInputMask(self):
    states = np.random.uniform(-1., 1., (3, 2))
    mask = np.zeros(3)
    cluster_mask, noncluster_mask, cluster_info = (
        caql_agent.CaqlAgent.compute_cluster_masks(states, mask))
    self.assertEqual([False, False, False], cluster_mask.tolist())
    self.assertEqual([False, False, False], noncluster_mask.tolist())
    self.assertIsNone(cluster_info)

  def testComputeTolerance(self):
    env = utils.create_env('Pendulum')
    state_spec, action_spec = utils.get_state_and_action_specs(env)
    agent = _create_agent(None, state_spec, action_spec)
    agent.target_network = mock.Mock()
    agent.target_network.predict_q_function.return_value = 0
    agent.train_network = mock.Mock()
    agent.train_network.predict_q_function.return_value = 0
    agent.train_network.compute_td_rmse.return_value = .2
    self.assertAlmostEqual(
        0.018,
        agent._compute_tolerance(states=None, actions=None, next_states=None,
                                 rewards=None, dones=None, tolerance_init=0.1,
                                 tolerance_decay=0.9))

  def testComputeToleranceUseMin(self):
    env = utils.create_env('Pendulum')
    state_spec, action_spec = utils.get_state_and_action_specs(env)
    agent = _create_agent(None, state_spec, action_spec)
    self.assertEqual(
        1e-4,
        agent._compute_tolerance(states=None, actions=None, next_states=None,
                                 rewards=None, dones=None, tolerance_init=None,
                                 tolerance_decay=None))

  def testComputeToleranceUseInit(self):
    env = utils.create_env('Pendulum')
    state_spec, action_spec = utils.get_state_and_action_specs(env)
    agent = _create_agent(None, state_spec, action_spec)
    self.assertEqual(
        0.123,
        agent._compute_tolerance(states=None, actions=None, next_states=None,
                                 rewards=None, dones=None, tolerance_init=0.123,
                                 tolerance_decay=None))


if __name__ == '__main__':
  tf.test.main()
