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

"""Data Valuation using Reinforcement Learning (DVRL)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import numpy as np
from sklearn import metrics
import tensorflow as tf
import tqdm
from dvrl import dvrl_metrics
from tensorflow.contrib import layers as contrib_layers


class Dvrl(object):
  """Data Valuation using Reinforcement Learning (DVRL) class.

    Attributes:
      x_train: training feature
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      problem: 'regression' or 'classification'
      pred_model: predictive model (object)
      parameters: network parameters such as hidden_dim, iterations,
                  activation function, layer_number, learning rate
      checkpoint_file_name: File name for saving and loading the trained model
      flags: flag for training with stochastic gradient descent (flag_sgd)
             and flag for using pre-trained model (flag_pretrain)
  """

  def __init__(self, x_train, y_train, x_valid, y_valid,
               problem, pred_model, parameters, checkpoint_file_name, flags):
    """Initializes DVRL."""

    # Inputs
    self.x_train = x_train
    self.y_train = y_train
    self.x_valid = x_valid
    self.y_valid = y_valid

    self.problem = problem

    # One-hot encoded labels
    if self.problem == 'classification':
      self.y_train_onehot = \
          np.eye(len(np.unique(y_train)))[y_train.astype(int)]
      self.y_valid_onehot = \
          np.eye(len(np.unique(y_train)))[y_valid.astype(int)]
    elif self.problem == 'regression':
      self.y_train_onehot = np.reshape(y_train, [len(y_train), 1])
      self.y_valid_onehot = np.reshape(y_valid, [len(y_valid), 1])

    # Network parameters
    self.hidden_dim = parameters['hidden_dim']
    self.comb_dim = parameters['comb_dim']
    self.outer_iterations = parameters['iterations']
    self.act_fn = parameters['activation']
    self.layer_number = parameters['layer_number']
    self.batch_size = np.min([parameters['batch_size'], len(x_train[:, 0])])
    self.learning_rate = parameters['learning_rate']

    # Basic parameters
    self.epsilon = 1e-8  # Adds to the log to avoid overflow
    self.threshold = 0.9  # Encourages exploration

    # Flags
    self.flag_sgd = flags['sgd']
    self.flag_pretrain = flags['pretrain']

    # If the pred_model uses stochastic gradient descent (SGD) for training
    if self.flag_sgd:
      self.inner_iterations = parameters['inner_iterations']
      self.batch_size_predictor = np.min([parameters['batch_size_predictor'],
                                          len(x_valid[:, 0])])

    # Checkpoint file name
    self.checkpoint_file_name = checkpoint_file_name

    # Basic parameters
    self.data_dim = len(x_train[0, :])
    self.label_dim = len(self.y_train_onehot[0, :])

    # Training Inputs
    # x_input can be raw input or its encoded representation, e.g. using a
    # pre-trained neural network. Using encoded representation can be beneficial
    # to reduce computational cost for high dimensional inputs, like images.

    self.x_input = tf.placeholder(tf.float32, [None, self.data_dim])
    self.y_input = tf.placeholder(tf.float32, [None, self.label_dim])

    # Prediction difference
    # y_hat_input is the prediction difference between predictive models
    # trained on the training set and validation set.
    # (adding y_hat_input into data value estimator as the additional input
    # is observed to improve data value estimation quality in some cases)
    self.y_hat_input = tf.placeholder(tf.float32, [None, self.label_dim])

    # Selection vector
    self.s_input = tf.placeholder(tf.float32, [None, 1])

    # Rewards (Reinforcement signal)
    self.reward_input = tf.placeholder(tf.float32)

    # Pred model (Note that any model architecture can be used as the predictor
    # model, either randomly initialized or pre-trained with the training data.
    # The condition for predictor model to have fit (e.g. using certain number
    # of back-propagation iterations) and predict functions as its subfunctions.
    self.pred_model = pred_model

    # Final model
    self.final_model = pred_model

    # With randomly initialized predictor
    if (not self.flag_pretrain) & self.flag_sgd:
      if not os.path.exists('tmp'):
        os.makedirs('tmp')
      pred_model.fit(self.x_train, self.y_train_onehot,
                     batch_size=len(self.x_train), epochs=0)
      # Saves initial randomization
      pred_model.save_weights('tmp/pred_model.h5')
      # With pre-trained model, pre-trained model should be saved as
      # 'tmp/pred_model.h5'

    # Baseline model
    if self.flag_sgd:
      self.ori_model = copy.copy(self.pred_model)
      self.ori_model.load_weights('tmp/pred_model.h5')

      # Trains the model
      self.ori_model.fit(x_train, self.y_train_onehot,
                         batch_size=self.batch_size_predictor,
                         epochs=self.inner_iterations, verbose=False)
    else:
      self.ori_model = copy.copy(self.pred_model)
      self.ori_model.fit(x_train, y_train)

    # Valid baseline model
    if 'summary' in dir(self.pred_model):
      self.val_model = copy.copy(self.pred_model)
      self.val_model.load_weights('tmp/pred_model.h5')

      # Trains the model
      self.val_model.fit(x_valid, self.y_valid_onehot,
                         batch_size=self.batch_size_predictor,
                         epochs=self.inner_iterations, verbose=False)
    else:
      self.val_model = copy.copy(self.pred_model)
      self.val_model.fit(x_valid, y_valid)

  def data_value_evaluator(self):
    """Returns data value evaluator model.

    Here, we assume a simple multi-layer perceptron architecture for the data
    value evaluator model. For data types like tabular, multi-layer perceptron
    is already efficient at extracting the relevant information.
    For high-dimensional data types like images or text,
    it is important to introduce inductive biases to the architecture to
    extract information efficiently. In such cases, there are two options:

    (i) Input the encoded representations (e.g. the last layer activations of
    ResNet for images, or the last layer activations of BERT for  text) and use
    the multi-layer perceptron on top of it. The encoded representations can
    simply come from a pre-trained predictor model using the entire dataset.

    (ii) Modify the data value evaluator model definition below to have the
    appropriate inductive bias (e.g. using convolutional layers for images,
    or attention layers text).

    Returns:
      dve: data value estimations
    """
    with tf.variable_scope('data_value_estimator', reuse=tf.AUTO_REUSE):

      inputs = tf.concat((self.x_input, self.y_input), axis=1)

      # Stacks multi-layered perceptron
      inter_layer = contrib_layers.fully_connected(
          inputs, self.hidden_dim, activation_fn=self.act_fn)
      for _ in range(int(self.layer_number - 3)):
        inter_layer = contrib_layers.fully_connected(
            inter_layer, self.hidden_dim, activation_fn=self.act_fn)
      inter_layer = contrib_layers.fully_connected(
          inter_layer, self.comb_dim, activation_fn=self.act_fn)

      # Combines with y_hat
      comb_layer = tf.concat((inter_layer, self.y_hat_input), axis=1)
      comb_layer = contrib_layers.fully_connected(
          comb_layer, self.comb_dim, activation_fn=self.act_fn)
      dve = contrib_layers.fully_connected(
          comb_layer, 1, activation_fn=tf.nn.sigmoid)

    return dve

  def train_dvrl(self, perf_metric):
    """Trains DVRL based on the specified objective function.

    Args:
      perf_metric: 'auc', 'accuracy', 'log-loss' for classification
                   'mae', 'mse', 'rmspe' for regression
    """

    # Generates selected probability
    est_data_value = self.data_value_evaluator()

    # Generator loss (REINFORCE algorithm)
    prob = tf.reduce_sum(self.s_input * tf.log(est_data_value + self.epsilon) +\
                         (1-self.s_input) * \
                         tf.log(1 - est_data_value + self.epsilon))
    dve_loss = (-self.reward_input * prob) + \
                1e3 * (tf.maximum(tf.reduce_mean(est_data_value) \
                                  - self.threshold, 0) + \
                       tf.maximum((1-self.threshold) - \
                                  tf.reduce_mean(est_data_value), 0))

    # Variable
    dve_vars = [v for v in tf.trainable_variables() \
                if v.name.startswith('data_value_estimator')]

    # Solver
    dve_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(
        dve_loss, var_list=dve_vars)

    # Baseline performance
    if self.flag_sgd:
      y_valid_hat = self.ori_model.predict(self.x_valid)
    else:
      if self.problem == 'classification':
        y_valid_hat = self.ori_model.predict_proba(self.x_valid)
      elif self.problem == 'regression':
        y_valid_hat = self.ori_model.predict(self.x_valid)

    if perf_metric == 'auc':
      valid_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
    elif perf_metric == 'accuracy':
      valid_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                  axis=1))
    elif perf_metric == 'log_loss':
      valid_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
    elif perf_metric == 'rmspe':
      valid_perf = dvrl_metrics.rmspe(self.y_valid, y_valid_hat)
    elif perf_metric == 'mae':
      valid_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
    elif perf_metric == 'mse':
      valid_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)

    # Prediction differences
    if self.flag_sgd:
      y_train_valid_pred = self.val_model.predict(self.x_train)
    else:
      if self.problem == 'classification':
        y_train_valid_pred = self.val_model.predict_proba(self.x_train)
      elif self.problem == 'regression':
        y_train_valid_pred = self.val_model.predict(self.x_train)
        y_train_valid_pred = np.reshape(y_train_valid_pred, [-1, 1])

    if self.problem == 'classification':
      y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred)
    elif self.problem == 'regression':
      y_pred_diff = \
          np.abs(self.y_train_onehot - y_train_valid_pred)/self.y_train_onehot

    # Main session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Model save at the end
    saver = tf.train.Saver(dve_vars)

    for _ in tqdm.tqdm(range(self.outer_iterations)):

      # Batch selection
      batch_idx = \
          np.random.permutation(len(self.x_train[:, 0]))[:self.batch_size]

      x_batch = self.x_train[batch_idx, :]
      y_batch_onehot = self.y_train_onehot[batch_idx]
      y_batch = self.y_train[batch_idx]
      y_hat_batch = y_pred_diff[batch_idx]

      # Generates selection probability
      est_dv_curr = sess.run(
          est_data_value,
          feed_dict={
              self.x_input: x_batch,
              self.y_input: y_batch_onehot,
              self.y_hat_input: y_hat_batch
          })

      # Samples the selection probability
      sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

      # Exception (When selection probability is 0)
      if np.sum(sel_prob_curr) == 0:
        est_dv_curr = 0.5 * np.ones(np.shape(est_dv_curr))
        sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)

      # Trains predictor
      # If the predictor is neural network
      if 'summary' in dir(self.pred_model):

        new_model = self.pred_model
        new_model.load_weights('tmp/pred_model.h5')

        # Train the model
        new_model.fit(x_batch, y_batch_onehot,
                      sample_weight=sel_prob_curr[:, 0],
                      batch_size=self.batch_size_predictor,
                      epochs=self.inner_iterations, verbose=False)

        y_valid_hat = new_model.predict(self.x_valid)

      else:
        new_model = self.pred_model
        new_model.fit(x_batch, y_batch, sel_prob_curr[:, 0])

      # Prediction
      if 'summary' in dir(new_model):
        y_valid_hat = new_model.predict(self.x_valid)
      else:
        if self.problem == 'classification':
          y_valid_hat = new_model.predict_proba(self.x_valid)
        elif self.problem == 'regression':
          y_valid_hat = new_model.predict(self.x_valid)

      # Reward computation
      if perf_metric == 'auc':
        dvrl_perf = metrics.roc_auc_score(self.y_valid, y_valid_hat[:, 1])
      elif perf_metric == 'accuracy':
        dvrl_perf = metrics.accuracy_score(self.y_valid, np.argmax(y_valid_hat,
                                                                   axis=1))
      elif perf_metric == 'log_loss':
        dvrl_perf = -metrics.log_loss(self.y_valid, y_valid_hat)
      elif perf_metric == 'rmspe':
        dvrl_perf = dvrl_metrics.rmspe(self.y_valid, y_valid_hat)
      elif perf_metric == 'mae':
        dvrl_perf = metrics.mean_absolute_error(self.y_valid, y_valid_hat)
      elif perf_metric == 'mse':
        dvrl_perf = metrics.mean_squared_error(self.y_valid, y_valid_hat)

      if self.problem == 'classification':
        reward_curr = dvrl_perf - valid_perf
      elif self.problem == 'regression':
        reward_curr = valid_perf - dvrl_perf

      # Trains the generator
      _, _ = sess.run(
          [dve_solver, dve_loss],
          feed_dict={
              self.x_input: x_batch,
              self.y_input: y_batch_onehot,
              self.y_hat_input: y_hat_batch,
              self.s_input: sel_prob_curr,
              self.reward_input: reward_curr
          })

    # Saves trained model
    saver.save(sess, self.checkpoint_file_name)

    # Trains DVRL predictor
    # Generate data values
    final_data_value = sess.run(
        est_data_value, feed_dict={
            self.x_input: self.x_train,
            self.y_input: self.y_train_onehot,
            self.y_hat_input: y_pred_diff})[:, 0]

    # Trains final model
    # If the final model is neural network
    if 'summary' in dir(self.pred_model):
      self.final_model.load_weights('tmp/pred_model.h5')
      # Train the model
      self.final_model.fit(self.x_train, self.y_train_onehot,
                           sample_weight=final_data_value,
                           batch_size=self.batch_size_predictor,
                           epochs=self.inner_iterations, verbose=False)
    else:
      self.final_model.fit(self.x_train, self.y_train, final_data_value)

  def data_valuator(self, x_train, y_train):
    """Returns data values using the data valuator model.

    Args:
      x_train: training features
      y_train: training labels

    Returns:
      final_dat_value: final data values of the training samples
    """

    # One-hot encoded labels
    if self.problem == 'classification':
      y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)]
      y_train_valid_pred = self.val_model.predict_proba(x_train)
    elif self.problem == 'regression':
      y_train_onehot = np.reshape(y_train, [len(y_train), 1])
      y_train_valid_pred = np.reshape(self.val_model.predict(x_train),
                                      [-1, 1])

    # Generates y_train_hat
    if self.problem == 'classification':
      y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)
    elif self.problem == 'regression':
      y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)/y_train_onehot

    # Restores the saved model
    imported_graph = \
        tf.train.import_meta_graph(self.checkpoint_file_name + '.meta')

    sess = tf.Session()
    imported_graph.restore(sess, self.checkpoint_file_name)

    # Estimates data value
    est_data_value = self.data_value_evaluator()

    final_data_value = sess.run(
        est_data_value, feed_dict={
            self.x_input: x_train,
            self.y_input: y_train_onehot,
            self.y_hat_input: y_train_hat})[:, 0]

    return final_data_value

  def dvrl_predictor(self, x_test):
    """Returns predictions using the predictor model.

    Args:
      x_test: testing features

    Returns:
      y_test_hat: predictions of the predictive model with DVRL
    """

    if self.flag_sgd:
      y_test_hat = self.final_model.predict(x_test)
    else:
      if self.problem == 'classification':
        y_test_hat = self.final_model.predict_proba(x_test)
      elif self.problem == 'regression':
        y_test_hat = self.final_model.predict(x_test)

    return y_test_hat
