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

"""Implementation of Last Layer Dropout."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import gin.tf


MAX_BYTES_MEM = 10**8


def _variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


@gin.configurable
class LastLayerDropout(object):
  """Implement Last Layer Dropout on the features."""

  def __init__(self, dataset, working_dir, model_dir,
               dim_input, num_classes,
               batch_size=gin.REQUIRED, step_size=gin.REQUIRED,
               dropout_keep_prob=gin.REQUIRED, num_samples=gin.REQUIRED,
               num_training_iters=gin.REQUIRED):
    """Initializes the class.

    Args:
      dataset: (features_train, y_train), (features_test, y_test)
      working_dir: working directory where the outputs are written
      model_dir: directory where the weights of the full network are.
      dim_input: dimension of the input features
      num_classes: number of classes
      batch_size: batch size
      step_size: step size
      dropout_keep_prob: dropout probability p, keeps neuron with probability p
      num_samples: number of samples
      num_training_iters: number of training steps before fixing a model
    """
    (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
    self.n = self.x_train.shape[0]
    self.dim_input = dim_input
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.step_size = step_size
    self.dropout_keep_prob = dropout_keep_prob
    self.num_samples = num_samples
    self.num_training_iters = num_training_iters
    self.working_dir = working_dir
    self.model_dir = model_dir
    self.build_graph()

  def build_graph(self):
    """Builds the neural network graph."""

    # define graph
    self.g = tf.Graph()
    with self.g.as_default():

      # create and store a new session for the graph
      self.sess = tf.Session()

      # define placeholders
      self.x = tf.placeholder(shape=[None, self.dim_input], dtype=tf.float32)
      self.y = tf.placeholder(shape=[None, self.num_classes], dtype=tf.float32)

      # define simple model
      with tf.variable_scope('last_layer'):
        dropout_input = tf.nn.dropout(self.x, keep_prob=self.dropout_keep_prob)
        self.z = tf.layers.dense(inputs=dropout_input, units=self.num_classes)

      self.loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits_v2(
              labels=self.y, logits=self.z))

      self.output_probs = tf.nn.softmax(self.z)

      # Variables of the last layer
      self.ll_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.ll_vars_concat = tf.concat(
          [self.ll_vars[0], tf.expand_dims(self.ll_vars[1], axis=0)], 0)

      # Summary
      _variable_summaries(self.ll_vars_concat)

      # saving the weights of last layer when running SGLD/SGD/MCMC algorithm
      self.saver = tf.train.Saver(var_list=self.ll_vars,
                                  max_to_keep=self.num_samples)

      self.gd_opt = tf.train.GradientDescentOptimizer(self.step_size)

      # SGD optimizer for the last layer
      grads_vars_sgd = self.gd_opt.compute_gradients(self.loss)
      self.train_op = self.gd_opt.apply_gradients(grads_vars_sgd)

      for g, v in grads_vars_sgd:
        if g is not None:
          s = list(v.name)
          s[v.name.rindex(':')] = '_'
          tf.summary.histogram(''.join(s) + '/grad_hist_dropout_sgd', g)

      # Merge all the summaries and write them out
      self.all_summaries = tf.summary.merge_all()
      location = os.path.join(self.working_dir, 'logs')
      self.writer = tf.summary.FileWriter(location, graph=self.g)

      saver_network = tf.train.Saver(var_list=self.ll_vars)
      print('loading the network ...')
      # Restores from checkpoint
      saver_network.restore(self.sess, self.model_dir)
      print('Graph successfully loaded.')

  def next_batch(self):
    """Give the next batch of training data."""
    indx = np.random.choice(self.n, self.batch_size, replace=False)
    return self.x_train[indx, :], self.y_train[indx]

  def sample(self, reinitialize=False):
    """Samples weights after training, for dropout we only need one sample."""

    # we store training loss for sanity check
    training_loss_h, test_loss_h = [], []

    # Keep only one over thinning_interval sample
    num_training_iters = self.num_training_iters

    # Saving the weights of the last layer
    num_ll_weights = int((self.dim_input + 1) * self.num_classes)
    sampled_weights = np.zeros((1, num_ll_weights))

    # random initialization of the weights if needed.
    if reinitialize:
      self.sess.run(tf.variables_initializer(self.ll_vars))

    # sampling
    init_t = time.time()
    print('-----------------------------------------------------')
    print('Starting sampling of the Dropout Neural Network')

    # we first train the model using dropout
    for i in np.arange(num_training_iters):

      # train step
      batch_x, batch_y = self.next_batch()
      feed_dict = {self.x: batch_x, self.y: batch_y}
      self.sess.run([self.train_op], feed_dict=feed_dict)

      # training, summary and test error every 100 iters
      if i % 100 == 0:
        _, summary, loss_v, = self.sess.run([
            self.train_op, self.all_summaries, self.loss], feed_dict=feed_dict)
        self.writer.add_summary(summary, i)
        self.writer.flush()
        training_loss_h.append(loss_v)
        feed_dict_test = {self.x: self.x_test, self.y: self.y_test}
        test_loss_v = self.sess.run([self.loss], feed_dict=feed_dict_test)
        test_loss_h.append(test_loss_v)
        msg = ('{} steps. Loss = {}. \t Test Loss = {}.').format(
            str(i), str(loss_v), str(test_loss_v))
        print(msg)

    # finally, we store the (last) sampled model
    ll_vars_v = self.sess.run(self.ll_vars_concat, feed_dict=feed_dict)
    sampled_weights[0, :] = ll_vars_v.flatten('F')

    self.saver.save(
        self.sess,
        os.path.join(self.working_dir, 'weights/saved-dropout-last-weights'),
        global_step=0,
        write_meta_graph=False)

    print('-----------------------------------------------------')
    print('Training complete after {} seconds.'.format(time.time() - init_t))
    print('-----------------------------------------------------')

    self.writer.close()

    return training_loss_h, test_loss_h, sampled_weights

  def predict(self, x):
    """Predict the output probabilities.

    Assume that self.sample() has been executed before.
    WARNING: please be careful of memory issues if number_samples and/or
             num_test_points are too large.

    Args:
      x: features on which to compute predicted probabilities.
    """

    # set hyper-parameters
    num_samples = self.num_samples
    num_classes = self.num_classes
    n = x.shape[0]

    # initialize tensor of results
    # Need to separate into several pieces for memory issues
    max_num_steps = MAX_BYTES_MEM // (num_classes * n)
    q = num_samples // max_num_steps
    r = num_samples % max_num_steps

    # only one model, but dropout in test time leads to different predictions
    file_name = os.path.join(
        self.working_dir, 'weights/saved-dropout-last-weights-0')
    self.saver.restore(self.sess, file_name)

    for k in np.arange(q):
      probabilities_tab = np.zeros((x.shape[0],
                                    self.num_classes,
                                    max_num_steps))
      for i in np.arange(max_num_steps):
        probabilities_v = self.sess.run(
            self.output_probs, feed_dict={self.x: x})
        probabilities_tab[:, :, i] = probabilities_v
      save_name = os.path.join(self.working_dir,
                               'proba_tab_' + str(k) + '.npy')
      with tf.gfile.Open(save_name, 'wb') as f:
        np.save(f, probabilities_tab)

    if r > 0:
      probabilities_tab = np.zeros((x.shape[0],
                                    self.num_classes,
                                    r))
      for i in np.arange(r):
        probabilities_v = self.sess.run(
            self.output_probs, feed_dict={self.x: x})
        probabilities_tab[:, :, i] = probabilities_v
      save_name = os.path.join(self.working_dir,
                               'proba_tab_' + str(q) + '.npy')
      with tf.gfile.Open(save_name, 'wb') as f:
        np.save(f, probabilities_tab)
