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

""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
  import special_grads
except KeyError as e:
  print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
        file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics

FLAGS = flags.FLAGS

class MAML:
  def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
    """ must call construct_model() after initializing MAML! """
    self.dim_input = dim_input
    self.dim_output = dim_output
    self.update_lr = FLAGS.update_lr
    self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
    self.classification = False
    self.test_num_updates = test_num_updates
    if FLAGS.datasource == 'sinusoid':
      self.dim_hidden = [40, 40]
      self.loss_func = mse
      self.forward = self.forward_fc
      self.construct_weights = self.construct_fc_weights
    elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
      self.loss_func = xent
      self.classification = True
      if FLAGS.conv:
        self.dim_hidden = FLAGS.num_filters
        self.forward = self.forward_conv
        self.construct_weights = self.construct_conv_weights
      else:
        self.dim_hidden = [256, 128, 64, 64]
        self.forward=self.forward_fc
        self.construct_weights = self.construct_fc_weights
      if FLAGS.datasource == 'miniimagenet':
        self.channels = 3
      else:
        self.channels = 1
      self.img_size = int(np.sqrt(self.dim_input/self.channels))
    else:
      raise ValueError('Unrecognized data source.')

  def construct_model(self, input_tensors=None, prefix='metatrain_'):
    # a: training data for inner gradient, b: test data for meta gradient
    if input_tensors is None:
      self.inputa = tf.placeholder(tf.float32)
      self.inputb = tf.placeholder(tf.float32)
      self.labela = tf.placeholder(tf.float32)
      self.labelb = tf.placeholder(tf.float32)
    else:
      self.inputa = input_tensors['inputa']
      self.inputb = input_tensors['inputb']
      self.labela = input_tensors['labela']
      self.labelb = input_tensors['labelb']

    with tf.variable_scope('model', reuse=None) as training_scope:
      if 'weights' in dir(self):
        training_scope.reuse_variables()
        weights = self.weights
      else:
        # Define the weights
        self.weights = weights = self.construct_weights()

      # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
      lossesa, outputas, lossesb, outputbs = [], [], [], []
      accuraciesa, accuraciesb = [], []
      num_updates = max(self.test_num_updates, FLAGS.num_updates)
      outputbs = [[]]*num_updates
      lossesb = [[]]*num_updates
      accuraciesb = [[]]*num_updates

      def task_metalearn(inp, reuse=True):
        """ Perform gradient descent for one task in the meta-batch. """
        inputa, inputb, labela, labelb = inp
        task_outputbs, task_lossesb = [], []

        if self.classification:
          task_accuraciesb = []

        task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
        task_lossa = self.loss_func(task_outputa, labela)

        grads = tf.gradients(task_lossa, list(weights.values()))
        if FLAGS.stop_grad:
          grads = [tf.stop_gradient(grad) for grad in grads]
        gradients = dict(zip(weights.keys(), grads))
        fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
        output = self.forward(inputb, fast_weights, reuse=True)
        task_outputbs.append(output)
        task_lossesb.append(self.loss_func(output, labelb))

        for j in range(num_updates - 1):
          loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
          grads = tf.gradients(loss, list(fast_weights.values()))
          if FLAGS.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]
          gradients = dict(zip(fast_weights.keys(), grads))
          fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
          output = self.forward(inputb, fast_weights, reuse=True)
          task_outputbs.append(output)
          task_lossesb.append(self.loss_func(output, labelb))

        task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

        if self.classification:
          task_accuracya = contrib_metrics.accuracy(
              tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
          for j in range(num_updates):
            task_accuraciesb.append(
                contrib_metrics.accuracy(
                    tf.argmax(tf.nn.softmax(task_outputbs[j]), 1),
                    tf.argmax(labelb, 1)))
          task_output.extend([task_accuracya, task_accuraciesb])

        return task_output

      if FLAGS.norm != 'None':
        # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
        unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

      out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
      if self.classification:
        out_dtype.extend([tf.float32, [tf.float32]*num_updates])
      result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
      if self.classification:
        outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
      else:
        outputas, outputbs, lossesa, lossesb  = result

    ## Performance & Optimization
    if 'train' in prefix:
      self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
      self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
      # after the map_fn
      self.outputas, self.outputbs = outputas, outputbs
      if self.classification:
        self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
        self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
      self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

      if FLAGS.metatrain_iterations > 0:
        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
        if FLAGS.datasource == 'miniimagenet':
          gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
        self.metatrain_op = optimizer.apply_gradients(gvs)
    else:
      self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
      self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
      if self.classification:
        self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
        self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

    ## Summaries
    tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
    if self.classification:
      tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

    for j in range(num_updates):
      tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
      if self.classification:
        tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

  ### Network construction functions (fc networks and conv networks)
  def construct_fc_weights(self):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
    weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
    for i in range(1,len(self.dim_hidden)):
      weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
      weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
    weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
    weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
    return weights

  def forward_fc(self, inp, weights, reuse=False):
    hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
    for i in range(1,len(self.dim_hidden)):
      hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

  def construct_conv_weights(self):
    weights = {}

    dtype = tf.float32
    conv_initializer = contrib_layers.xavier_initializer_conv2d(dtype=dtype)
    fc_initializer = contrib_layers.xavier_initializer(dtype=dtype)
    k = 3

    weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
    weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
    weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
    weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
    if FLAGS.datasource == 'miniimagenet':
      # assumes max pooling
      weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
      weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
    else:
      weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
      weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
    return weights

  def forward_conv(self, inp, weights, reuse=False, scope=''):
    # reuse is for the normalization parameters.
    channels = self.channels
    inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

    hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
    if FLAGS.datasource == 'miniimagenet':
      # last hidden layer is 6x6x64-ish, reshape to a vector
      hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
    else:
      hidden4 = tf.reduce_mean(hidden4, [1, 2])

    return tf.matmul(hidden4, weights['w5']) + weights['b5']
