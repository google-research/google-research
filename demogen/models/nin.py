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

"""Defines model for variants of Network in Network.

Original paper: https://arxiv.org/abs/1312.4400
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers


def _nin_block(
    input_tensor,
    filter_num,
    batch_norm,
    is_training,
    regularizer,
    dropout_prob,
    spatial_dropout):
  """Create one block within the Network in Network architecture.

  Args:
    input_tensor: The input to the block.
    filter_num: Number of filters used in all the convolution layers.
    batch_norm: Whether to use batch normalization after convolution.
    is_training: Indicates if the model is being trained or not. This affects
      the behavior of batch normalization.
    regularizer: Kernel regularizer for the convolution layers.
    dropout_prob: The probability that an individual component is dropped out.
    spatial_dropout: Whether to use spatial dropout where the spatial
      dimensions are not dropped out independently.

  Returns:
    The input tensor transformed by the NIN block.
  """
  activation = input_tensor

  for i in range(3):
    out = tf.layers.conv2d(
        activation,
        filters=filter_num,
        kernel_size=1 if i > 0 else 3,
        strides=1 if i > 0 else 2,
        padding='SAME',
        kernel_regularizer=regularizer)

    if batch_norm:
      out = tf.layers.batch_normalization(out, training=is_training)
    activation = tf.nn.relu(out)

  if dropout_prob > 0.0 and is_training:
    activation = _dropout_block(activation, dropout_prob, spatial_dropout)

  return out, activation


def _dropout_block(input_tensor, dropout_prob, spatial):
  """Apply dropout out or spatial dropout to the input tensor.

  Args:
    input_tensor: The input on which dropout is applied.
    dropout_prob: The probability that an individual component is dropped out.
    spatial:  Whether to use spatial dropout where the spatial dimensions are
      not dropped out independently.

  Returns:
    The input tensor with dropout applied.
  """
  shape = input_tensor.shape
  ns = [shape[0], 1, 1, shape[3]] if spatial else None
  out = tf.nn.dropout(input_tensor, dropout_prob, noise_shape=ns)
  return out


class Nin(object):
  """A class for creating the NIN architecture.

  A callable class that builds a nin model according to given hyperparameters.

  Attributes:
    filter_num: Number of filters in each layer.
    num_classes: Number of classes in the training data.
    variables: All tensorflow variables of the model sorted by name.
    dropout: Whether dropout is used.
    spatial_dropout: Whether spatial dropout is used.
    bn: Whether batchnorm is used.
    decay_fac: Coefficient for l2 weight decay.
    variables: A list of tf.Variable's tha contains tensorflow variables within
      the model's scope sorted by the variable's name.
  """

  def __init__(
      self,
      filter_num=64,
      num_classes=10,
      scope='nin',
      dropout=False,
      batchnorm=False,
      decay_fac=0.01,
      spatial_dropout=False):

    self.filter_num = filter_num
    self.num_classes = num_classes
    self.dropout = dropout
    self.spatial_dropout = spatial_dropout
    self.bn = batchnorm
    self.decay_fac = decay_fac
    self._scope = scope
    self._was_called = False

  @property
  def variables(self):
    """Returns variables within model's scope."""
    if not self._was_called:
      raise ValueError('This model has not been called yet.')
    all_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)
    return sorted(all_vars, key=lambda x: x.name)

  def __call__(
      self,
      inputs,
      is_training,
      end_points_collection=None):

    with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE) as scope:
      self._scope = scope.name

      reg = None
      if self.decay_fac > 0.:
        reg = contrib_layers.l2_regularizer(scale=self.decay_fac)

      all_preactivations = []
      activation = inputs
      for _ in range(3):
        preactivation, activation = _nin_block(
            activation, self.filter_num, self.bn, is_training, reg,
            self.dropout, self.spatial_dropout)
        all_preactivations.append(preactivation)

      h4 = tf.layers.conv2d(
          activation, self.num_classes, 4, 1, 'VALID', kernel_regularizer=reg)
      logits = tf.reshape(h4, [-1, self.num_classes])

      if end_points_collection:
        end_points_collection['inputs'] = inputs
        for name, end_point in zip(['h1', 'h2', 'h3'], all_preactivations):
          end_points_collection[name] = end_point

      self._was_called = True

      return logits
