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

"""Utility functions for declaring variables and adding summaries.

It adds all different scalars and histograms for each variable and provides
utility functions for weight and bias variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('verbose', False, 'If true, adds summary.')


def weight_variable(shape, stddev=0.1, wd=.0000002):
  """Create a weight variable with appropriate initialization."""
  with tf.device('/cpu:0'):
    with tf.name_scope('weights'):
      weights = tf.get_variable(
          'weights',
          shape,
          initializer=tf.truncated_normal_initializer(
              stddev=stddev, dtype=tf.float32),
          dtype=tf.float32)
      if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
      variable_summaries(weights)
  return weights


def bias_variable(shape, init_value=0.1, name='biases'):
  """Create a bias variable with appropriate initialization."""
  with tf.device('/cpu:0'):
    with tf.name_scope(name):
      biases = tf.get_variable(
          name,
          shape,
          initializer=tf.constant_initializer(init_value),
          dtype=tf.float32)
      variable_summaries(biases)
  return biases


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  if FLAGS.verbose:
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
  else:
    pass


def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  if FLAGS.verbose:
    tf.summary.histogram('activation', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))
  else:
    pass


def loss_summaries(total_loss):
  """Add summaries for losses in a model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
