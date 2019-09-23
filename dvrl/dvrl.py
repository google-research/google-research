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

"""Data Valuation using Reinforcement Learning (DVRL).
"""

# Necessary packages and function call
from __future__ import print_function

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

import tensorflow as tf

from tqdm import tqdm


def dvrl(x_train, y_train, x_valid, y_valid, parameters, perf_metric):
  """Returns data values using DVRL framework.

  Args:
    x_train: training feature
    y_train: training labels
    x_valid: validation features
    y_valid: validation labels
    parameters: network parameters such as hidden_dim, iterations,
      activation_fn, layer_number
    perf_metric: 'auc' or 'accuracy'

  Returns:
    final_data_value: Estimated values of the samples
  """
  # Reset the graph
  tf.reset_default_graph()

  # One-hot encoding of training label
  y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)]

  # Network parameters
  hidden_dim = parameters['hidden_dim']
  outer_iterations = parameters['iterations']
  act_fn = tf.nn.relu
  layer_number = parameters['layer_number']
  batch_size = np.min([parameters['batch_size'], len(x_train[:, 0])])

  # Basic parameters
  data_dim = len(x_train[0, :])
  label_dim = len(y_train_onehot[0, :])
  comb_dim = int(hidden_dim/10)

  # Training inputs
  x_input = tf.placeholder(tf.float32, [None, data_dim], name='x_input')
  y_input = tf.placeholder(tf.float32, [None, label_dim], name='y_input')

  # Prediction difference (to enhance noise label detection)
  y_hat_input = tf.placeholder(
      tf.float32, [None, label_dim], name='y_hat_input')

  # Selection vector
  s_input = tf.placeholder(tf.float32, [None, 1], name='s_input')

  # Rewards for reinforcement signal
  reward_input = tf.placeholder(tf.float32, name='reward_input')

  # data value evaluator
  def data_value_evaluator(x, y, y_hat):
    """Returns data value estimations.

    Args:
      x: input features
      y: input labels
      y_hat: prediction differences

    Returns:
      dve: data value estimations
    """
    with tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):

      # Concatenates inputs and outputs
      inputs = tf.concat((x, y), axis=1)

      # Stacks multi-layered perceptron
      inter_layer = tf.contrib.layers.fully_connected(
          inputs, hidden_dim, activation_fn=act_fn)
      for _ in range(int(layer_number - 3)):
        inter_layer = tf.contrib.layers.fully_connected(
            inter_layer, hidden_dim, activation_fn=act_fn)
      inter_layer = tf.contrib.layers.fully_connected(
          inter_layer, comb_dim, activation_fn=act_fn)

      # Combines with auxiliary labels
      comb_layer = tf.concat((inter_layer, y_hat), axis=1)
      comb_layer = tf.contrib.layers.fully_connected(
          comb_layer, comb_dim, activation_fn=act_fn)
      dve = tf.contrib.layers.fully_connected(
          comb_layer, 1, activation_fn=tf.nn.sigmoid)

    return dve

  # Selection probabilities
  est_data_value = data_value_evaluator(x_input, y_input, y_hat_input)

  # Loss for the REINFORCE algorithm
  prob = tf.reduce_sum(s_input * tf.log(est_data_value + 1e-8) + \
                       (1-s_input) * tf.log(1 - est_data_value + 1e-8))
  dve_loss = (-reward_input * prob) + \
              1e3 * (tf.maximum(tf.reduce_mean(est_data_value) - 0.9, 0) + \
                     tf.maximum(0.1 - tf.reduce_mean(est_data_value), 0))

  # Gets variables
  dve_vars = [v for v in tf.trainable_variables() \
              if v.name.startswith('data_value_estimator')]

  # Optimization step
  dve_solver = tf.train.AdamOptimizer(0.01).minimize(
      dve_loss, var_list=dve_vars)

  # Baseline interpretable model
  batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]
  x_batch = x_train[batch_idx, :]
  y_batch = y_train[batch_idx]

  ori_model = LogisticRegression()
  ori_model.fit(x_batch, y_batch)

  y_valid_hat = ori_model.predict_proba(x_valid)

  # Performance
  if perf_metric == 'auc':
    ori_mse = roc_auc_score(y_valid, y_valid_hat[:, 1])
  elif perf_metric == 'accuracy':
    ori_mse = accuracy_score(y_valid, np.argmax(y_valid_hat, axis=1))
  elif perf_metric == 'log_loss':
    ori_mse = -log_loss(y_valid, y_valid_hat)

  # Validation performance
  val_model = LogisticRegression()
  val_model.fit(x_valid, y_valid)

  y_train_valid_pred = val_model.predict_proba(x_train)

  # Prediction differences used to estimate fidelity
  y_pred_diff = np.abs(y_train_onehot - y_train_valid_pred)

  # Main session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for _ in tqdm(range(outer_iterations)):

    # Batch selection
    batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]

    x_batch = x_train[batch_idx, :]
    y_batch_onehot = y_train_onehot[batch_idx]
    y_batch = y_train[batch_idx]
    y_hat_batch = y_pred_diff[batch_idx]

    # Generates selection probability
    est_dv_curr = sess.run(
        est_data_value,
        feed_dict={
            x_input: x_batch,
            y_input: y_batch_onehot,
            y_hat_input: y_hat_batch
        })

    # Samples based on the selection probability
    sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

    # Exception (When selection probability is 0)
    if np.sum(sel_prob_curr) == 0:
      est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
      sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

    new_model = LogisticRegression()
    new_model.fit(x_batch, y_batch, sel_prob_curr[:, 0])

    y_valid_hat = new_model.predict_proba(x_valid)

    if perf_metric == 'auc':
      new_mse = roc_auc_score(y_valid, y_valid_hat[:, 1])
    elif perf_metric == 'accuracy':
      new_mse = accuracy_score(y_valid, np.argmax(y_valid_hat, axis=1))
    elif perf_metric == 'log_loss':
      new_mse = -log_loss(y_valid, y_valid_hat)

    reward_curr = new_mse - ori_mse

    # Trains the generator
    _ = sess.run(
        dve_solver,
        feed_dict={
            x_input: x_batch,
            y_input: y_batch_onehot,
            y_hat_input: y_hat_batch,
            s_input: sel_prob_curr,
            reward_input: reward_curr
        })

  # Final data value computations for the entire training sampls
  final_data_value = sess.run(
      est_data_value,
      feed_dict={
          x_input: x_train,
          y_input: y_train_onehot,
          y_hat_input: y_pred_diff
      })[:, 0]

  sess.close()

  # Returns data values for training samples
  return final_data_value
