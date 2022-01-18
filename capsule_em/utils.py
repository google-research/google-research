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

"""Utility functions for declaring variables and adding summaries.

It adds all different scalars and histograms for each variable and provides
utility functions for weight and bias variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('verbose', False, 'If true, adds summary.')


XLA_COMPILE = False
JIT_SCOPE = 'routing'
RECOMPUTE = False


@contextlib.contextmanager
def noop_jit_scope(compile_ops=True, scope_name=JIT_SCOPE):
  tf.logging.info('No-op jit_scope %s %s', compile_ops, scope_name)
  yield


def maybe_jit_scope(*args, **kwargs):
  """Placeholder to enable jit scoping."""

  # kwargs['compile_ops'] = op_filter

  if FLAGS.jit_scopes:
    # if 'scope_name' not in kwargs:
    #   kwargs['scope_name'] = JIT_SCOPE
    # # This makes things really messed up...
    # # kwargs['separate_compiled_gradients'] = True
    # tf.logging.info('Adding jit_scope "%s"', kwargs['scope_name'])
    return tf.xla.experimental.jit_scope(*args, **kwargs)
  else:
    return noop_jit_scope(*args, **kwargs)


def noop_recompute_grad(func):
  tf.logging.info('No-op recompute_grad')
  return func


if RECOMPUTE:
  maybe_recompute_grad = tf.recompute_grad
else:
  maybe_recompute_grad = noop_recompute_grad


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


def kernel_tile(input_tensor, kernel, stride=1, rate=1, name='kernel_tile'):
  """Tiles the input in a convolutional manner based on kernel size.

  Equivalent of:
   output = tf.extract_image_patches(
       input,
       ksizes=[1, kernel, kernel, 1],
       strides=[1, stride, stride, 1],
       rates=[1, rate, rate, 1],
       padding='VALID')
  Args:
    input_tensor: The input feature map to be tiled.
    kernel: The kernel size for the convolutional map.
    stride: The stride for the convolutional feature map.
    rate: The rate of sampling for the convolutional feature map.
    name: The name for this module.

  Returns:
    output: The tiled version of input, considering convolutional dimensions.
  """

  input_shape = input_tensor.get_shape()
  tf.logging.info('kernel_tile: input shape %s', input_shape)

  tile_filter = np.zeros(
      shape=[kernel * rate, kernel * rate, input_shape[3], kernel * kernel],
      dtype=np.float32)
  for i in range(kernel):
    for j in range(kernel):
      tile_filter[i * rate, j * rate, :, i * kernel + j] = 1.0

  with tf.name_scope(name):
    tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
    if rate > 1:
      input_tensor = tf.pad(input_tensor,
                            [[0, 0], [0, rate - 1], [0, rate - 1], [0, 0]])
    output = tf.nn.depthwise_conv2d(
        input_tensor,
        tile_filter_op,
        strides=[1, stride, stride, 1],
        padding='VALID')
    output_shape = output.get_shape()
    output = tf.reshape(
        output,
        shape=[
            int(output_shape[0]),
            int(output_shape[1]),
            int(output_shape[2]),
            int(input_shape[3]), kernel * kernel
        ])
    output = tf.transpose(output, perm=[0, 1, 2, 4, 3])

  tf.logging.info('kernel_tile: output shape %s', output.shape)
  return output
