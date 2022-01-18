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

# Lint as: python2, python3
"""Implementation of Last Layer Bayes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import gin.tf
import numpy as np
import six
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers


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
class LastLayerBayesian(object):
  """Implement LLB on the features."""

  def __init__(self, dataset, working_dir, model_dir,
               dim_input, num_classes,
               batch_size=gin.REQUIRED, step_size=gin.REQUIRED,
               thinning_interval=gin.REQUIRED, num_samples=gin.REQUIRED,
               sampler=gin.REQUIRED):
    """Initializes the class.

    Args:
      dataset: (features_train, y_train), (features_test, y_test)
      working_dir: working directory where the outputs are written
      model_dir: directory where the weights of the full network are.
      dim_input: dimension of the input features
      num_classes: number of classes
      batch_size: batch size
      step_size: step size
      thinning_interval: thinning interval
      num_samples: number of samples
      sampler: sampler, eg 'sgd', 'sgld' or 'lmc'
    """
    (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
    self.n = self.x_train.shape[0]
    self.dim_input = dim_input
    self.num_classes = num_classes
    self.sampler = sampler
    self.batch_size = batch_size
    self.step_size = step_size
    self.thinning_interval = thinning_interval
    self.num_samples = num_samples
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
        self.z = tf.layers.dense(inputs=self.x, units=self.num_classes)

      self.loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits_v2(
              labels=self.y, logits=self.z))

      self.output_probs = tf.nn.softmax(self.z)

      # Variables of the last layer
      self.ll_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      self.ll_vars_concat = tf.concat(
          [self.ll_vars[0],
           tf.expand_dims(self.ll_vars[1], axis=0)], 0)

      # Summary
      _variable_summaries(self.ll_vars_concat)

      # add regularization that acts as a unit Gaussian prior on the last layer
      regularizer = contrib_layers.l2_regularizer(1.0)

      # regularization
      prior = contrib_layers.apply_regularization(regularizer, self.ll_vars)
      self.bayesian_loss = self.n * self.loss + prior

      # saving the weights of last layer when running SGLD/SGD/MCMC algorithm
      self.saver = tf.train.Saver(var_list=self.ll_vars,
                                  max_to_keep=self.num_samples)

      # SGLD optimizer for the last layer
      if self.sampler in ['sgld', 'lmc']:
        step = self.step_size / self.n
        gd_opt = tf.train.GradientDescentOptimizer(step)
        grads_vars = gd_opt.compute_gradients(self.bayesian_loss)
        grads_vars_sgld = []

        for g, v in grads_vars:
          if g is not None:
            s = list(v.name)
            s[v.name.rindex(':')] = '_'
            # Adding Gaussian noise to the gradient
            gaussian_noise = (np.sqrt(2. / step)
                              * tf.random_normal(tf.shape(g)))
            g_sgld = g + gaussian_noise
            tf.summary.histogram(''.join(s) + '/grad_hist_mcmc',
                                 g / self.n)
            tf.summary.histogram(''.join(s) + '/gaussian_noise_hist_mcmc',
                                 gaussian_noise / self.n)
            tf.summary.histogram(''.join(s) + '/grad_total_hist_mcmc',
                                 g_sgld / self.n)
            grads_vars_sgld.append((g_sgld, v))

        self.train_op = gd_opt.apply_gradients(grads_vars_sgld)

      # SGD optimizer for the last layer
      if self.sampler == 'sgd':
        gd_opt = tf.train.GradientDescentOptimizer(self.step_size)
        grads_vars_sgd = gd_opt.compute_gradients(self.loss)
        self.train_op = gd_opt.apply_gradients(grads_vars_sgd)

        for g, v in grads_vars_sgd:
          if g is not None:
            s = list(v.name)
            s[v.name.rindex(':')] = '_'
            tf.summary.histogram(''.join(s) + '/grad_hist_sgd', g)

      # Merge all the summaries and write them out
      self.all_summaries = tf.summary.merge_all()
      location = os.path.join(self.working_dir, 'logs')
      self.writer = tf.summary.FileWriter(location, graph=self.g)

      saver_network = tf.train.Saver(var_list=self.ll_vars)
      print('loading the network ...')
      # Restores from checkpoint
      # self.sess.run(tf.global_variables_initializer())
      saver_network.restore(self.sess, self.model_dir)
      print('Graph successfully loaded.')

  def next_batch(self):
    """Give the next batch of training data."""
    indx = np.random.choice(self.n, self.batch_size, replace=False)
    return self.x_train[indx, :], self.y_train[indx]

  def sample(self, reinitialize=False):
    """Samples weights for the network, displaying training and test errors."""

    # we store training loss for sanity check
    training_loss_h, test_loss_h = [], []

    # Keep only one over thinning_interval sample
    num_iters = self.num_samples * self.thinning_interval

    # Saving the weights of the last layer
    num_ll_weights = int((self.dim_input + 1) * self.num_classes)
    sampled_weights = np.zeros((self.num_samples, num_ll_weights))

    # Random initialization of the weights if needed.
    if reinitialize:
      self.sess.run(tf.variables_initializer(self.ll_vars))

    # sampling
    init_t = time.time()
    print('-----------------------------------------------------')
    print('Starting sampling of the Bayesian Neural Network by ' +
          six.ensure_str(self.sampler))
    for i in np.arange(0, num_iters):
      batch_x, batch_y = self.next_batch()
      feed_dict = {self.x: batch_x, self.y: batch_y}
      self.sess.run([self.train_op], feed_dict=feed_dict)

      if (i+1) % self.thinning_interval == 0:
        summary, loss_v, ll_vars_v = self.sess.run([
            self.all_summaries, self.loss,
            self.ll_vars_concat], feed_dict=feed_dict)
        training_loss_h.append(loss_v)
        self.writer.add_summary(summary, i+1)
        self.writer.flush()
        sampled_weights[i // self.thinning_interval,
                        :] = ll_vars_v.flatten('F')
        feed_dict_test = {self.x: self.x_test, self.y: self.y_test}
        test_loss_v = self.sess.run([self.loss], feed_dict=feed_dict_test)
        test_loss_h.append(test_loss_v)
        msg = ('{} steps. Loss = {}. \t Test Loss = {}.').format(
            str(i+1), str(loss_v), str(test_loss_v))
        print(msg)
        self.saver.save(
            self.sess,
            os.path.join(self.working_dir, 'weights/saved-last-weights'),
            global_step=i+1,
            write_meta_graph=False)

    print('-----------------------------------------------------')
    print('Training complete after {} seconds.'.format(time.time() - init_t))
    print('-----------------------------------------------------')

    self.writer.close()

    return training_loss_h, test_loss_h, sampled_weights

  def predict(self, x):
    """Predict the output probabilities.

    Assume that self.sample() has been executed before.

    Args:
      x: features on which to compute predicted probabilities.
    """

    # set hyper-parameters
    thinning_interval = self.thinning_interval
    num_samples = self.num_samples
    num_classes = self.num_classes
    n = x.shape[0]

    # initialize tensor of results
    # Need to separate into several pieces for memory issues
    max_num_steps = MAX_BYTES_MEM // (num_classes * n)
    q = num_samples // max_num_steps
    r = num_samples % max_num_steps

    for k in np.arange(q):
      probabilities_tab = np.zeros((x.shape[0],
                                    self.num_classes,
                                    max_num_steps))
      for i in np.arange(max_num_steps):
        step = (k * max_num_steps + i + 1) * thinning_interval
        file_name = os.path.join(self.working_dir,
                                 'weights/saved-last-weights-' + str(step))
        self.saver.restore(self.sess, file_name)
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
        step = (q * max_num_steps + i + 1) * thinning_interval
        file_name = os.path.join(self.working_dir,
                                 'weights/saved-last-weights-' + str(step))
        self.saver.restore(self.sess, file_name)
        probabilities_v = self.sess.run(
            self.output_probs, feed_dict={self.x: x})
        probabilities_tab[:, :, i] = probabilities_v
      save_name = os.path.join(self.working_dir,
                               'proba_tab_' + str(q) + '.npy')
      with tf.gfile.Open(save_name, 'wb') as f:
        np.save(f, probabilities_tab)
