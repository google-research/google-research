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

"""Reinforcement Learning based Locally Interpretable Models (RL-LIM).
"""

# Necessary functions and packages call
from __future__ import print_function

import numpy as np

from sklearn.linear_model import Ridge

import tensorflow as tf

from tqdm import tqdm


def rllim(x_train, y_train, x_valid, y_valid, x_test, parameters, hyperparam):
  """RL-LIM Architecture.

  Args:
    x_train: training feature
    y_train: training labels
    x_valid: validation features
    y_valid: validation labels
    x_test: testing features
    parameters: network parameters such as hidden_dim, iterations,
      activation_fn, layer_number
    hyperparam: hyperparameter lambda

  Returns:
    final_data_value: Estimated values of the samples
  """
  # Reset the graph
  tf.reset_default_graph()

  # Network parameters
  hidden_dim = parameters['hidden_dim']
  outer_iterations = parameters['iterations']
  act_fn = tf.nn.tanh
  layer_number = parameters['layer_number']
  batch_size = parameters['batch_size']
  batch_size_small = parameters['batch_size_small']

  # Basic parameters
  data_dim = len(x_train[0, :])

  # Training inputs
  x_input = tf.placeholder(tf.float32, [None, data_dim], name='x_input')

  # Target input
  xt_input = tf.placeholder(tf.float32, [None, data_dim], name='xt_input')

  # Selection vector
  s_input = tf.placeholder(tf.float32, [None, batch_size_small], name='s_input')

  # Rewards (Reinforcement signal)
  reward_input = tf.placeholder(tf.float32, [batch_size_small, 1],
                                name='reward_input')

  # Data value evaluator
  def data_value_evaluator(x, xt):
    """Returns data value estimations.

    Args:
      x: input features
      xt: target prediction

    Returns:
      dve: data value estimations
    """
    with tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):

      xt_ext = tf.tile(tf.reshape(xt, [1, data_dim]), [tf.size(x[:, 0]), 1])
      inputs = tf.concat((x * xt_ext, x-xt_ext, x, xt_ext), axis=1)

      # Stacks multi-layered perceptron
      inter_layer = tf.contrib.layers.fully_connected(
          inputs, hidden_dim, activation_fn=act_fn)
      for _ in range(layer_number-2):
        inter_layer = tf.contrib.layers.fully_connected(
            inter_layer, hidden_dim, activation_fn=act_fn)

      dve = tf.contrib.layers.fully_connected(
          inter_layer, 1, activation_fn=tf.nn.sigmoid)

    return dve

  # Generates selected probabilities
  est_data_value = data_value_evaluator(x_input, xt_input)

  est_data_value_set = [data_value_evaluator(x_input, xt_input[i, :]) \
                        for i in range(batch_size_small)]

  # Loss for the REINFORCE algorithm
  # 1. Without lambda penalty
  for ktt in range(batch_size_small):
    prob = tf.reduce_mean(s_input[:, ktt] * \
                          tf.log(est_data_value_set[ktt] + 1e-8) + \
                        (1-s_input[:, ktt]) * \
                          tf.log(1 - est_data_value_set[ktt] + 1e-8))
    if ktt == 0:
      dve_loss_curr = (-reward_input[ktt] * prob)
    else:
      dve_loss_curr = dve_loss_curr + (-reward_input[ktt] * prob)

  dve_loss = dve_loss_curr / batch_size_small

  # 2. With lambda penalty
  for ktt in range(batch_size_small):
    prob_hat = tf.reduce_mean(s_input[:, ktt] * \
                              tf.log(est_data_value_set[ktt] + 1e-8) + \
                        (1-s_input[:, ktt]) * \
                              tf.log(1 - est_data_value_set[ktt] + 1e-8))
    if ktt == 0:
      dve_loss_curr_hat = (-reward_input[ktt] * prob_hat) - \
          hyperparam * tf.reduce_mean(est_data_value_set[ktt]) * prob_hat + \
          + 1e3 * tf.maximum(0.01 - tf.reduce_mean(est_data_value_set[ktt]), 0)
    else:
      dve_loss_curr_hat = dve_loss_curr_hat + (-reward_input[ktt] * prob_hat) \
          - hyperparam * tf.reduce_mean(est_data_value_set[ktt]) * prob_hat + \
          + 1e3 * tf.maximum(0.01 - tf.reduce_mean(est_data_value_set[ktt]), 0)

  dve_loss_hat = dve_loss_curr_hat / batch_size_small

  # Gets variables
  dve_vars = [v for v in tf.trainable_variables() \
              if v.name.startswith('data_value_estimator')]

  # Optimization step
  dve_solver = tf.train.AdamOptimizer(0.0001).minimize(
      dve_loss, var_list=dve_vars)

  dve_solver_hat = tf.train.AdamOptimizer(0.0001).minimize(
      dve_loss_hat, var_list=dve_vars)

  # Directly trains on the training set
  ori_model = Ridge(alpha=1, fit_intercept=True)
  ori_model.fit(x_train, y_train)

  # Main session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for itt in tqdm(range(outer_iterations)):

    # Batch selection
    batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]

    x_batch = x_train[batch_idx, :]
    y_batch = y_train[batch_idx]

    val_batch_idx = np.random.permutation(len(x_valid[:, 0]))[:batch_size_small]

    # Initialization
    reward_curr = np.zeros([batch_size_small, 1])
    sel_prob_curr = np.zeros([batch_size, batch_size_small])

    xt_batch = x_valid[val_batch_idx, :]
    yt_batch = y_valid[val_batch_idx]

    for ktt in range(batch_size_small):

      # Generates selection probability
      est_dv_curr = sess.run(
          est_data_value,
          feed_dict={
              x_input: x_batch,
              xt_input: np.reshape(xt_batch[ktt, :], [1, data_dim])
          })

      # Samples based on the selection probability
      sel_prob_curr[:, ktt] = np.random.binomial(1, est_dv_curr,
                                                 est_dv_curr.shape)[:, 0]

      # Exception (When selection probability is 0)
      if np.sum(sel_prob_curr[:, ktt]) == 0:
        est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
        sel_prob_curr[:, ktt] = np.random.binomial(1, est_dv_curr,
                                                   est_dv_curr.shape)[:, 0]

      # Instance-wise locally interpretable model training
      new_model = Ridge(alpha=1, fit_intercept=True)
      new_model.fit(x_batch, y_batch, sel_prob_curr[:, ktt])

      yt_batch_hat_new = new_model.predict(np.reshape(xt_batch[ktt, :],
                                                      [1, data_dim]))
      new_mse = np.abs(yt_batch_hat_new - yt_batch[ktt])

      yt_batch_hat_ori = ori_model.predict(np.reshape(xt_batch[ktt, :],
                                                      [1, data_dim]))
      ori_mse = np.abs(yt_batch_hat_ori - yt_batch[ktt])

      # Reward computation
      reward_curr[ktt] = new_mse - ori_mse

    # Trains the generator
    # Without lambda penalty
    if itt < 500:
      _ = sess.run(
          dve_solver,
          feed_dict={
              x_input: x_batch,
              xt_input: xt_batch,
              s_input: sel_prob_curr,
              reward_input: reward_curr
          })
    # With lambda penalty
    else:
      _ = sess.run(
          dve_solver_hat,
          feed_dict={
              x_input: x_batch,
              xt_input: xt_batch,
              s_input: sel_prob_curr,
              reward_input: reward_curr
          })

  # Final data value computations for the entire training sampls
  final_coef = np.zeros([len(x_test[:, 0]), len(x_test[0, :])+1])
  final_pred = np.zeros([len(x_test[:, 0]),])

  # For each testing sample
  for jtt in range(len(x_test[:, 0])):
    final_data_value = sess.run(
        est_data_value,
        feed_dict={
            x_input: x_train,
            xt_input: np.reshape(x_test[jtt, :], [1, data_dim])
        })[:, 0]

    # Fits locally interpretable model
    new_model = Ridge(alpha=1, fit_intercept=True)
    new_model.fit(x_train, y_train, final_data_value)

    final_coef[jtt, 0] = new_model.intercept_
    final_coef[jtt, 1:] = new_model.coef_
    final_pred[jtt] = new_model.predict(np.reshape(x_test[jtt, :],
                                                   [1, len(x_test[0, :])]))

  sess.close()

  # Returns data values for training samples
  return final_pred, final_coef
