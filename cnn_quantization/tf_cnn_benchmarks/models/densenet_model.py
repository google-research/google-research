# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Densenet model configuration.

References:
  "Densely Connected Convolutional Networks": https://arxiv.org/pdf/1608.06993
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from cnn_quantization.tf_cnn_benchmarks.models import model as model_lib


class DensenetCifar10Model(model_lib.CNNModel):
  """Densenet cnn network configuration."""

  def __init__(self, model, layer_counts, growth_rate, params=None):
    self.growth_rate = growth_rate
    super(DensenetCifar10Model, self).__init__(
        model, 32, 64, 0.1, layer_counts=layer_counts, params=params)
    self.batch_norm_config = {'decay': 0.9, 'epsilon': 1e-5, 'scale': True}

  def dense_block(self, cnn, growth_rate):
    input_layer = cnn.top_layer
    c = cnn.batch_norm(input_layer, **self.batch_norm_config)
    c = tf.nn.relu(c)
    c = cnn.conv(growth_rate, 3, 3, 1, 1, stddev=np.sqrt(2.0/9/growth_rate),
                 activation=None, input_layer=c)
    channel_index = 3 if cnn.channel_pos == 'channels_last' else 1
    cnn.top_layer = tf.concat([input_layer, c], channel_index)
    cnn.top_size += growth_rate

  def transition_layer(self, cnn):
    in_size = cnn.top_size
    cnn.batch_norm(**self.batch_norm_config)
    cnn.top_layer = tf.nn.relu(cnn.top_layer)
    cnn.conv(in_size, 1, 1, 1, 1, stddev=np.sqrt(2.0/9/in_size))
    cnn.apool(2, 2, 2, 2)

  def add_inference(self, cnn):
    if self.layer_counts is None:
      raise ValueError('Layer counts not specified for %s' % self.get_model())
    if self.growth_rate is None:
      raise ValueError('Growth rate not specified for %s' % self.get_model())

    cnn.conv(16, 3, 3, 1, 1, activation=None)
    # Block 1
    for _ in xrange(self.layer_counts[0]):
      self.dense_block(cnn, self.growth_rate)
    self.transition_layer(cnn)
    # Block 2
    for _ in xrange(self.layer_counts[1]):
      self.dense_block(cnn, self.growth_rate)
    self.transition_layer(cnn)
    # Block 3
    for _ in xrange(self.layer_counts[2]):
      self.dense_block(cnn, self.growth_rate)
    cnn.batch_norm(**self.batch_norm_config)
    cnn.top_layer = tf.nn.relu(cnn.top_layer)
    channel_index = 3 if cnn.channel_pos == 'channels_last' else 1
    cnn.top_size = cnn.top_layer.get_shape().as_list()[channel_index]
    cnn.spatial_mean()

  def get_learning_rate(self, global_step, batch_size):
    num_batches_per_epoch = 50000 // batch_size
    boundaries = num_batches_per_epoch * np.array([150, 225, 300],
                                                  dtype=np.int64)
    boundaries = [x for x in boundaries]
    values = [0.1, 0.01, 0.001, 0.0001]
    return tf.train.piecewise_constant(global_step, boundaries, values)


def create_densenet40_k12_model():
  return DensenetCifar10Model('densenet40_k12', (12, 12, 12), 12)


def create_densenet100_k12_model():
  return DensenetCifar10Model('densenet100_k12', (32, 32, 32), 12)


def create_densenet100_k24_model():
  return DensenetCifar10Model('densenet100_k24', (32, 32, 32), 24)
