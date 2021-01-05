# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Contains definitions for MLP Networks.
"""
from __future__ import print_function
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers


class MLP(object):
  """Definition of MLP Networks."""

  def __init__(self, keep_prob, wd, feature_dim):
    """Creates a model for classifying using MLP encoding.

    Args:
      keep_prob: The rate of keeping one neuron in Dropout.
      wd: The co-efficient of weight decay.
      feature_dim: the dimension of the representation space.
    """
    super(MLP, self).__init__()

    self.regularizer = contrib_layers.l2_regularizer(scale=wd)
    self.initializer = contrib_layers.xavier_initializer()
    self.variance_initializer = contrib_layers.variance_scaling_initializer(
        factor=0.1,
        mode='FAN_IN',
        uniform=False,
        seed=None,
        dtype=tf.dtypes.float32)
    self.drop_rate = 1 - keep_prob
    self.feature_dim = feature_dim

  def encoder(self, inputs, training):
    """Forwards a batch of inputs.

    Args:
      inputs: A Tensor representing a batch of inputs.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor. If self.neck is true, the logits Tensor is with shape
      [<batch_size>, self.num_classes]. If self.neck is not true, the logits
      Tensor is with shape [<batch_size>, 256].
    """
    # pylint: disable=unused-argument
    out = tf.layers.dense(
        inputs,
        units=self.feature_dim,
        kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer,
        name='fc')
    out = tf.nn.relu(out)
    out = tf.layers.dense(
        out,
        units=self.feature_dim,
        kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer,
        name='fc2')
    return out

  def confidence_model(self, mu, training):
    """Given a batch of mu, output a batch of variance."""
    out = tf.layers.dropout(mu, rate=self.drop_rate, training=training)
    out = tf.layers.dense(
        out,
        units=self.feature_dim,
        kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer,
        name='fc_variance')
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out, rate=self.drop_rate, training=training)
    out = tf.layers.dense(
        out,
        units=self.feature_dim,
        kernel_initializer=self.initializer,
        kernel_regularizer=self.regularizer,
        name='fc_variance2')
    return out


def mlp(keep_prob, wd, feature_dim):
  net = MLP(keep_prob=keep_prob, wd=wd, feature_dim=feature_dim)
  return net
