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

"""Simple feedforward neural network for Mnist."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf


class NetworkMnist(object):
  """Network for Mnist dataset, training and computing the features."""

  def __init__(self, dataset, network_params, hparams, path_model, train_model):
    """Initializes the class.

    Args:
      dataset: (x_train, y_train), (x_test, y_test)
      network_params: parameters of the network
      hparams: hyper parameters
      path_model: path to model.
      train_model: boolean, True if the network has to be trained.
    """
    (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
    self.network_params = network_params
    self.hparams = hparams
    self.n = self.x_train.shape[0]
    self.path_model = path_model
    self.train_model = train_model
    self.build_graph()

  def build_graph(self):
    """Builds the neural network graph."""

    # define parameters of the network
    dim_input = self.network_params['dim_input']
    num_classes = self.network_params['num_classes']
    num_units = self.network_params['num_units']

    # define hyper-parameters
    hparams = self.hparams
    step_sgd = 1e-2 if not hasattr(hparams, 'step_sgd') else hparams.step_sgd

    # define graph
    self.g = tf.Graph()
    with self.g.as_default():

      # create and store a new session for the graph
      self.sess = tf.Session()

      # define placeholders
      self.x = tf.placeholder(shape=[None, dim_input], dtype=tf.float32)
      self.y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32)

      # define simple model
      with tf.variable_scope('representation_network'):
        z0 = tf.layers.dense(
            inputs=self.x, units=num_units[0], activation=tf.nn.relu)
        self.z1 = tf.layers.dense(
            inputs=z0, units=num_units[1], activation=tf.nn.relu)
      with tf.variable_scope('last_layer'):
        z2 = tf.layers.dense(inputs=self.z1, units=num_classes)

      self.loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits_v2(
              labels=self.y, logits=z2))

      # SGD optimizer
      self.sgd_opt = tf.train.GradientDescentOptimizer(step_sgd)
      self.train_sgd_op = self.sgd_opt.minimize(self.loss)

      # Saver for the full network
      self.saver_network = tf.train.Saver()

      if self.train_model:
        init = tf.global_variables_initializer()
        print('Initializing the variables ...')
        self.sess.run(init)
        print('Graph successfully initialized.')
      else:
        print('loading the network ...')
        # Restores from checkpoint
        self.saver_network.restore(self.sess, self.path_model)
        print('Graph successfully loaded.')

  def next_batch(self, batchsize=None):
    """Give the next batch of training data."""
    batch_size = self.hparams['batch_size'] if batchsize is None else batchsize
    indx = np.random.choice(self.x_train.shape[0], batch_size, replace=False)
    return self.x_train[indx, :], self.y_train[indx]

  def train(self):
    """Trains the network, displaying training and test errors."""

    training_loss_h, test_loss_h = [], []
    num_iters = self.hparams['num_iters']

    # training
    init_t = time.time()
    print('-----------------------------------------------------')
    print('Starting training of the classical Neural Network by SGD.')
    for i in range(num_iters):
      batch_x, batch_y = self.next_batch()
      feed_dict = {self.x: batch_x, self.y: batch_y}
      _, loss_v = self.sess.run([self.train_sgd_op, self.loss],
                                feed_dict=feed_dict)
      training_loss_h.append(loss_v)

      if i % 100 == 0:
        feed_dict = {self.x: self.x_test, self.y: self.y_test}
        test_loss_v = self.sess.run(
            self.loss, feed_dict=feed_dict)
        test_loss_h.append(test_loss_v)
        msg = ('{:5.5} steps. Loss = {:6.6}. \t Test Loss = {:6.6}.').format(
            str(i), str(loss_v), str(test_loss_v))
        print(msg)
    # Saving the full network
    # Relative path, move the folder then to an appropriate location.
    self.saver_network.save(self.sess, 'model/full_network')
    print('-----------------------------------------------------')
    print('Training complete after {} seconds.'.format(time.time() - init_t))
    print('-----------------------------------------------------')

    return training_loss_h, test_loss_h

  def features(self, x):
    """Computes the features of the last but one layer."""
    feed_dict = {self.x: x}
    return self.sess.run(self.z1, feed_dict=feed_dict)
