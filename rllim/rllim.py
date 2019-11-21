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

"""Reinforcement Learning based Locally Interpretable Modeling (RL-LIM)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tqdm


class Rllim(object):
  """Reinforcement Learning based Locally Interpretable Modeling (RL-LIM) class.

    Attributes:
      x_train: training feature
      y_train: training labels
      x_probe: probe features
      y_probe: probe labels
      parameters: network parameters such as hidden_dim, iterations,
                  activation_fn, num_layers
      interp_model: interpretable model (object)
      baseline_model: interpretable baseline model (object)
      checkpoint_file_name: file name for saving and loading trained model
      hidden_dim: hidden state dimensions
      outer_iterations: number of RL iterations
      act_fn: activation function
      num_layers: number of layers
      batch_size: number of samples in each batch for training interpretable
                  model
      batch_size_inner: number of samples in each batch for RL training
      hyper_lambda: main hyper-parameter of RL-LIM (lambda)
  """

  def __init__(self, x_train, y_train, x_probe, y_probe, parameters,
               interp_model, baseline_model, checkpoint_file_name):
    """Initiallizes RL-LIM."""

    # Resets the graph
    tf.reset_default_graph()

    # Initializes train and probe sets
    self.x_train = x_train
    self.y_train = y_train
    self.x_probe = x_probe
    self.y_probe = y_probe

    # Checkpoint file name
    self.checkpoint_file_name = checkpoint_file_name

    # Network parameters
    self.hidden_dim = parameters['hidden_dim']
    self.outer_iterations = parameters['iterations']
    self.act_fn = tf.nn.tanh
    self.num_layers = parameters['num_layers']
    self.batch_size = parameters['batch_size']
    self.batch_size_inner = parameters['batch_size_inner']
    self.hyper_lambda = parameters['lambda']

    # Basic parameters
    self.data_dim = len(x_train[0, :])

    # Placeholders
    # Training inputs
    self.x_input = tf.placeholder(tf.float32, [None, self.data_dim])

    # Target input
    self.xt_input = tf.placeholder(tf.float32, [None, self.data_dim])

    # Selection vector
    self.s_input = tf.placeholder(tf.float32, [None, self.batch_size_inner])

    # Rewards (Reinforcement signal)
    self.reward_input = tf.placeholder(tf.float32, [self.batch_size_inner, 1])

    # Interpretable models
    self.baseline_model = baseline_model
    self.interp_model = interp_model

  def inst_weight_evaluator(self, x, xt):
    """Returns instance-wise weight estimations.

    Args:
      x: training features
      xt: target features

    Returns:
      inst_weight: instance-wise weights
    """
    with tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):

      # Reshapes xt to have the same dimension with x
      xt_ext = tf.tile(tf.reshape(xt, [1, self.data_dim]),
                       [tf.size(x[:, 0]), 1])
      # Defines input
      inputs = tf.concat((x * xt_ext, x-xt_ext, x, xt_ext), axis=1)

      # Stacks multi-layered perceptron
      inter_layer = tf.contrib.layers.fully_connected(
          inputs, self.hidden_dim, activation_fn=self.act_fn)
      for _ in range(self.num_layers-2):
        inter_layer = tf.contrib.layers.fully_connected(
            inter_layer, self.hidden_dim, activation_fn=self.act_fn)

      inst_weight = tf.contrib.layers.fully_connected(
          inter_layer, 1, activation_fn=tf.nn.sigmoid)

    return inst_weight

  def rllim_train(self):
    """Training instance-wise weight estimator."""

    # Generates selected probabilities
    est_data_value = self.inst_weight_evaluator(self.x_input, self.xt_input)

    # Generates a set of selected probabilities
    est_data_value_set = \
    [self.inst_weight_evaluator(self.x_input, self.xt_input[i, :]) \
     for i in range(self.batch_size_inner)]

    # Loss for the REINFORCE algorithm
    epsilon = 1e-8  # add to log to avoid overflow
    # 1. Without lambda penalty
    for ktt in range(self.batch_size_inner):
      prob = tf.reduce_mean(self.s_input[:, ktt] * \
                            tf.log(est_data_value_set[ktt] + epsilon) + \
                          (1-self.s_input[:, ktt]) * \
                            tf.log(1 - est_data_value_set[ktt] + epsilon))
      if ktt == 0:
        dve_loss_curr = (-self.reward_input[ktt] * prob)
      else:
        dve_loss_curr = dve_loss_curr + (-self.reward_input[ktt] * prob)

    dve_loss = dve_loss_curr / self.batch_size_inner

    # 2. With lambda penalty
    eta = 1e3  # multiplier to the regularizer
    thresh = 0.01  # threshold for the minimum selection

    for ktt in range(self.batch_size_inner):
      prob_hat = tf.reduce_mean(self.s_input[:, ktt] * \
                                tf.log(est_data_value_set[ktt] + epsilon) + \
                          (1-self.s_input[:, ktt]) * \
                                tf.log(1 - est_data_value_set[ktt] + epsilon))
      if ktt == 0:
        dve_loss_curr_hat = (-self.reward_input[ktt] * prob_hat) - \
        self.hyper_lambda * tf.reduce_mean(est_data_value_set[ktt]) * \
        prob_hat + \
        eta * tf.maximum(thresh - tf.reduce_mean(est_data_value_set[ktt]), 0)
      else:
        dve_loss_curr_hat = dve_loss_curr_hat + \
        (-self.reward_input[ktt] * prob_hat) - \
        self.hyper_lambda * tf.reduce_mean(est_data_value_set[ktt]) \
        * prob_hat + \
        eta * tf.maximum(thresh - tf.reduce_mean(est_data_value_set[ktt]), 0)

    dve_loss_hat = dve_loss_curr_hat / self.batch_size_inner

    # Variables
    dve_vars = [v for v in tf.trainable_variables() \
                if v.name.startswith('data_value_estimator')]

    # Optimization step
    dve_solver = tf.train.AdamOptimizer(0.0001).minimize(
        dve_loss, var_list=dve_vars)

    dve_solver_hat = tf.train.AdamOptimizer(0.0001).minimize(
        dve_loss_hat, var_list=dve_vars)

    # Main session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Saves model at the end
    saver = tf.train.Saver()

    # Outer iterations
    for itt in tqdm.tqdm(range(self.outer_iterations)):

      # Batch selection
      batch_idx = \
      np.random.permutation(len(self.x_train[:, 0]))[:self.batch_size]

      x_batch = self.x_train[batch_idx, :]
      y_batch = self.y_train[batch_idx]

      val_batch_idx = \
      np.random.permutation(len(self.x_probe[:, 0]))[:self.batch_size_inner]

      xt_batch = self.x_probe[val_batch_idx, :]
      yt_batch = self.y_probe[val_batch_idx]

      # Initialization
      reward_curr = np.zeros([self.batch_size_inner, 1])
      sel_prob_curr = np.zeros([self.batch_size, self.batch_size_inner])

      # Inner iterations
      for ktt in range(self.batch_size_inner):

        # Generates selection probability
        est_dv_curr = sess.run(
            est_data_value,
            feed_dict={
                self.x_input: x_batch,
                self.xt_input: np.reshape(xt_batch[ktt, :], [1, self.data_dim])
            })

        # Samples based on the selection probability
        sel_prob_curr[:, ktt] = np.random.binomial(1, est_dv_curr,
                                                   est_dv_curr.shape)[:, 0]

        # Exception (When selection probability is 0)
        if np.sum(sel_prob_curr[:, ktt]) == 0:
          est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
          sel_prob_curr[:, ktt] = np.random.binomial(1, est_dv_curr,
                                                     est_dv_curr.shape)[:, 0]

        # Trains instance-wise locally interpretable model
        self.interp_model.fit(x_batch, y_batch, sel_prob_curr[:, ktt])

        # Interpretable predictions
        yt_batch_hat_new = \
        self.interp_model.predict(np.reshape(xt_batch[ktt, :],
                                             [1, self.data_dim]))

        # Fidelity of interpretable model
        new_mse = np.abs(yt_batch_hat_new - yt_batch[ktt])

        # Interpretable baseline prediction
        yt_batch_hat_ori = \
        self.baseline_model.predict(np.reshape(xt_batch[ktt, :],
                                               [1, self.data_dim]))

        # Fidelity of interpretable baseline model
        ori_mse = np.abs(yt_batch_hat_ori - yt_batch[ktt])

        # Computes reward
        reward_curr[ktt] = new_mse - ori_mse

      # Trains the generator
      # Without lambda penalty
      if itt < 500:
        _ = sess.run(
            dve_solver,
            feed_dict={
                self.x_input: x_batch,
                self.xt_input: xt_batch,
                self.s_input: sel_prob_curr,
                self.reward_input: reward_curr
            })

      # With lambda penalty
      else:
        _ = sess.run(
            dve_solver_hat,
            feed_dict={
                self.x_input: x_batch,
                self.xt_input: xt_batch,
                self.s_input: sel_prob_curr,
                self.reward_input: reward_curr
            })

    # Saves model
    saver.save(sess, self.checkpoint_file_name)

  def instancewise_weight_estimator(self, x_train, y_train, x_test):
    """Computes instance-wise weights for a given test input.

    Args:
      x_train: training features
      y_train: training labels
      x_test: testing features

    Returns:
      instancewise_weights: estimated instance-wise weights
    """
    # Calls inst_weight_evaluator function
    est_data_value = self.inst_weight_evaluator(self.x_input, self.xt_input)

    # Restores the saved model
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, self.checkpoint_file_name)

    # For individual testing sample
    x_test = np.reshape(x_test, [1, self.data_dim])

    # Computes instance-wise weight for a given test input
    instancewise_weights = sess.run(
        est_data_value,
        feed_dict={
            self.x_input: x_train,
            self.xt_input: x_test
            })[:, 0]

    return instancewise_weights[:len(y_train)]

  def rllim_interpreter(self, x_train, y_train, x_test, interp_model):
    """Returns interpretable predictions and instance-wise explanations.

    Args:
      x_train: training features
      y_train: training labels
      x_test: testing features
      interp_model: locally interpretable model (object)

    Returns:
      final_pred: interpretable predictions
      final_coef: instance-wise explanations
    """

    # Calls instance-wise weight estimator
    est_data_value = self.inst_weight_evaluator(self.x_input, self.xt_input)

    # Restores the saved model
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, self.checkpoint_file_name)

    # If the number of testing sample is 1
    if len(x_test.shape) == 1:
      x_test = np.reshape(x_test, [1, len(x_test)])

    # Initializes output
    final_coef = np.zeros([len(x_test[:, 0]), len(x_test[0, :])+1])
    final_pred = np.zeros([len(x_test[:, 0]),])

    # For each testing sample
    for jtt in tqdm.tqdm(range(len(x_test[:, 0]))):
      instancewise_weights = sess.run(
          est_data_value,
          feed_dict={
              self.x_input: x_train,
              self.xt_input: np.reshape(x_test[jtt, :], [1, self.data_dim])
          })[:, 0]

      # Fits locally interpretable model
      if np.sum(instancewise_weights > 0):
        interp_model.fit(x_train, y_train, instancewise_weights)

        # Instance-wise explanations
        final_coef[jtt, 0] = interp_model.intercept_
        final_coef[jtt, 1:] = interp_model.coef_

        # Instance-wise prediction
        final_pred[jtt] = \
            interp_model.predict(np.reshape(x_test[jtt, :],
                                            [1, len(x_test[0, :])]))

    return final_pred, final_coef
