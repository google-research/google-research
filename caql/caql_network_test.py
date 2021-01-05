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

"""Tests for caql_network."""

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf

from caql import caql_network
from caql import utils

tf.disable_v2_behavior()
tf.enable_resource_variables()


def _create_network(session,
                    state_spec,
                    action_spec,
                    name="test",
                    l2_loss_flag=True,
                    simple_lambda_flag=True,
                    solver="gradient_ascent",
                    sufficient_ascent_flag=False):
  return caql_network.CaqlNet(
      session=session,
      state_spec=state_spec,
      action_spec=action_spec,
      hidden_layers=[32, 16],
      learning_rate=.001,
      learning_rate_action=.005,
      learning_rate_ga=.01,
      batch_size=3,
      action_maximization_iterations=20,
      name=name,
      l2_loss_flag=l2_loss_flag,
      simple_lambda_flag=simple_lambda_flag,
      solver=solver,
      sufficient_ascent_flag=sufficient_ascent_flag,
      initial_lambda=10.0,
      lambda_max=5e3)


class CaqlNetworkTest(tf.test.TestCase):

  def setUp(self):
    super(CaqlNetworkTest, self).setUp()
    seed = 9999
    logging.info("Setting the numpy seed to %d", seed)
    np.random.seed(seed)
    self.sess = tf.Session()
    env = utils.create_env("Pendulum")
    state_spec, action_spec = utils.get_state_and_action_specs(env)
    self.network = _create_network(self.sess, state_spec, action_spec)

    tf.set_random_seed(seed)

    self.sess.run(
        tf.initializers.variables(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

  def testActionProjection(self):
    action = np.array([[1.0], [3.0], [-10.0]])
    projected_action_np = self.network._action_projection(action)
    self.assertIsInstance(projected_action_np, np.ndarray)
    self.assertArrayNear(
        np.array([[1.0], [2.0], [-2.0]]).flatten(),
        projected_action_np.flatten(),
        err=1e-4)

    action_tf = tf.convert_to_tensor(action)
    projected_action_tf = self.network._action_projection(action_tf)
    self.assertIsInstance(projected_action_tf, tf.Tensor)
    self.assertArrayNear(
        np.array([[1.0], [2.0], [-2.0]]).flatten(),
        self.network._session.run(projected_action_tf).flatten(),
        err=1e-4)

  def testBuildActionFunctionNet(self):
    state = tf.convert_to_tensor(
        np.random.uniform(-1., 1., (3, 3)), dtype=tf.float32)
    action_function_out = self.network._build_action_function_net(state)
    self.assertIsInstance(action_function_out, tf.Tensor)
    self.assertEqual((self.network.batch_size, self.network.action_dim),
                     action_function_out.shape)

  def testBuilQFunctionNet(self):
    state = tf.convert_to_tensor(
        np.random.uniform(-1., 1., (3, 3)), dtype=tf.float32)
    action = tf.convert_to_tensor(
        np.random.uniform(-2., 2., (3, 1)), dtype=tf.float32)
    q_function_out = self.network._build_q_function_net(state, action)
    self.assertIsInstance(q_function_out, tf.Tensor)
    self.assertEqual((self.network.batch_size, 1), q_function_out.shape)

  def testBuilLambdaFunctionNet(self):
    state = tf.convert_to_tensor(
        np.random.uniform(-1., 1., (3, 3)), dtype=tf.float32)
    action = tf.convert_to_tensor(
        np.random.uniform(-2., 2., (3, 1)), dtype=tf.float32)
    lambda_function_out = self.network._build_lambda_function_net(state, action)
    self.assertIsInstance(lambda_function_out, tf.Tensor)
    self.assertEqual((self.network.batch_size, 1), lambda_function_out.shape)

  def testPredictActionFunction(self):
    state = np.random.uniform(-1., 1., (3, 3))
    action_output = self.network.predict_action_function(state)
    self.assertIsInstance(action_output, np.ndarray)
    self.assertEqual((self.network.batch_size, 1), action_output.shape)

    self.assertArrayNear(
        np.array([[-0.1091], [-0.0947], [0.0371]]).flatten(),
        action_output.flatten(),
        err=1e-4)

  def testPredictQFunction(self):
    state = np.random.uniform(-1., 1., (3, 3))
    action = np.random.uniform(-2., 2., (3, 1))
    q_output = self.network.predict_q_function(state, action)
    self.assertIsInstance(q_output, np.ndarray)
    self.assertEqual((self.network.batch_size, 1), q_output.shape)

    self.assertArrayNear(
        np.array([[0.2190], [0.3202], [0.0745]]).flatten(),
        q_output.flatten(),
        err=1e-4)

  def testPredictStatePerturbedQFunction(self):
    centroid_states = np.random.uniform(-1., 1., (3, 3))
    centroid_actions = np.random.uniform(-2., 2., (3, 1))
    state_deviation = np.random.uniform(-0.5, 0.5, (3, 3))
    perturbed_q_output = self.network.predict_state_perturbed_q_function(
        centroid_states, centroid_actions, state_deviation)
    self.assertIsInstance(perturbed_q_output, np.ndarray)
    self.assertEqual((self.network.batch_size, 1), perturbed_q_output.shape)
    self.assertArrayNear(
        np.array([[0.2096], [0.3015], [-0.0401]]).flatten(),
        perturbed_q_output.flatten(),
        err=1e-4)

  def testPredictLambdaFunction(self):
    state = np.random.uniform(-1., 1., (3, 3))
    action = np.random.uniform(-2., 2., (3, 1))
    lambda_output = self.network.predict_lambda_function(state, action)
    self.assertIsInstance(lambda_output, np.ndarray)
    self.assertEqual((self.network.batch_size, 1), lambda_output.shape)
    self.assertArrayNear(
        np.array([[10.0], [10.0], [10.0]]).flatten(),
        lambda_output.flatten(),
        err=1e-4)

  def testComputeBackup(self):
    state = np.random.uniform(-1., 1., (3, 3))
    action = np.random.uniform(-2., 2., (3, 1))
    rewards = np.random.uniform(0., 10., (3, 1))
    dones = np.array([[True], [False], [True]])
    discount_factor = 0.99
    q_labels = self.network.predict_q_function(state, action)
    backup_output = self.network.compute_backup(q_labels, rewards, dones,
                                                discount_factor)

    self.assertIsInstance(backup_output, np.ndarray)
    self.assertArrayNear(
        np.array([[2.8836], [4.4714], [3.3149]]).flatten(),
        backup_output.flatten(),
        err=1e-4)

  def testComputeTDRMSE(self):
    state = np.random.uniform(-1., 1., (3, 3))
    action = np.random.uniform(-2., 2., (3, 1))
    rewards = np.random.uniform(0., 10., (3, 1))
    dones = np.array([[True], [False], [True]])
    discount_factor = 0.99
    q_labels = self.network.predict_q_function(state, action)
    td_rmse_output = self.network.compute_td_rmse(state, action, q_labels,
                                                  rewards, dones,
                                                  discount_factor)
    self.assertAlmostEqual(3.407, td_rmse_output, places=3)

  def testComputeDualActiveConstraintCondition(self):

    states = np.random.uniform(-1., 1., (3, 3))
    actions = np.random.uniform(-2., 2., (3, 1))
    next_state = np.random.uniform(-1., 1., (3, 3))

    dual_maxq_labels = self.network.compute_dual_maxq_label(next_state)
    rewards = np.random.uniform(0., 10., (3, 1))
    dones = np.array([[True], [False], [True]])
    discount_factor = 0.99

    dual_mask = self.network.compute_dual_active_constraint_condition(
        states, actions, dual_maxq_labels, rewards, dones, discount_factor)
    self.assertArrayNear(
        np.array([[True], [True], [True]]).flatten(),
        np.array(dual_mask).flatten(),
        err=1e-4)

  def testGradientAscentBestActions(self):
    state_tensor = np.random.uniform(-1., 1., (3, 3))
    tolerance_tensor = 1e-3
    self.network.solver = "gradient_ascent"
    ga_best_actions = self.network.compute_best_actions(
        state_tensor,
        tolerance_tensor,
        warmstart=False,
        tf_summary_vals=None)
    self.assertArrayNear(
        np.array([[0.5839], [1.6532], [1.0911]]).flatten(),
        np.array(ga_best_actions).flatten(),
        err=1e-4)

  def testCrossEntropyBestActions(self):
    state_tensor = np.random.uniform(-1., 1., (3, 3))
    tolerance_tensor = 1e-3
    self.network.solver = "cross_entropy"
    cem_best_actions = self.network.compute_best_actions(
        state_tensor,
        tolerance_tensor,
        warmstart=False,
        tf_summary_vals=None)
    self.assertArrayNear(
        np.array([[1.5246], [1.9969], [1.7113]]).flatten(),
        np.array(cem_best_actions).flatten(),
        err=1e-4)

  def testComputeDualMaxQLabel(self):
    next_state = np.random.uniform(-1., 1., (3, 3))
    dual_maxq_label = self.network.compute_dual_maxq_label(next_state)

    self.assertArrayNear(
        np.array([[1.0251], [1.0383], [1.0597]]).flatten(),
        dual_maxq_label.flatten(),
        err=1e-4)

  def testBatchTrainActionFunction(self):
    state_tensor_stack = np.random.uniform(-1., 1., (3, 3))
    best_q_stack = np.random.uniform(0., 10., (3, 1))
    action_loss = self.network.batch_train_action_function(
        state_tensor_stack, best_q_stack)
    self.assertAlmostEqual(60.966, action_loss, places=3)

  def testBatchTrainQFunction(self):
    state_tensor_stack = np.random.uniform(-1., 1., (3, 3))
    action_tensor_stack = np.random.uniform(-2., 2., (3, 1))
    true_label_stack = np.random.uniform(0., 10., (3, 1))
    q_loss = self.network.batch_train_q_function(state_tensor_stack,
                                                 action_tensor_stack,
                                                 true_label_stack)
    self.assertAlmostEqual(10.767, q_loss, places=3)

  def testBatchTrainLambdaFunction(self):
    state_tensor_stack = np.random.uniform(-1., 1., (3, 3))
    action_tensor_stack = np.random.uniform(-2., 2., (3, 1))
    true_label_stack = np.random.uniform(0., 10., (3, 1))
    lambda_loss = self.network.batch_train_lambda_function(
        state_tensor_stack, action_tensor_stack, true_label_stack)
    self.assertAlmostEqual(-32.464, lambda_loss, places=3)


if __name__ == "__main__":
  tf.test.main()
