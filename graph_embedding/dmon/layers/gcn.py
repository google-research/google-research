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

# Lint as: python3
"""TODO(tsitsulin): add headers, tests, and improve style."""
from typing import List
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer


class GCN(Layer):  # TODO(tsitsulin): add docstring pylint: disable=missing-class-docstring

  def __init__(self,
               n_channels,
               activation='selu',
               skip_connection = True,
               no_features = False):
    super(GCN, self).__init__()
    self.n_channels = n_channels
    self.skip_connection = skip_connection
    self.no_features = no_features
    if isinstance(activation, str):
      self.activation = Activation(activation)
    elif isinstance(Activation):
      self.activation = activation
    elif activation is None:
      self.activation = Lambda(lambda x: x)
    else:
      raise ValueError('GCN activation of unknown type')

  def build(self, input_shape):
    if self.no_features:
      self.n_features = input_shape[0][0]
    else:
      self.n_features = input_shape[0][-1]
    self.kernel = self.add_variable(
        'kernel', shape=(self.n_features, self.n_channels))
    self.bias = self.add_variable('bias', shape=(self.n_channels,))
    if self.skip_connection:
      self.skip_weight = self.add_variable(
          'skip_weight', shape=(self.n_channels,))
    else:
      self.skip_weight = 0
    super().build(input_shape)

  def compute_output_shape(self, input_shape):
    features_shape = input_shape[0]
    output_shape = features_shape[:-1] + (self.n_channels,)
    return output_shape

  def call(self, inputs):  # TODO(tsitsulin): convert to tuple typing pylint: disable=g-bare-generic
    features, graph = inputs
    if self.no_features:
      output = self.kernel
    else:
      output = tf.matmul(features, self.kernel)
    output = output * self.skip_weight + tf.sparse.sparse_dense_matmul(
        graph, output)
    output = output + self.bias
    return self.activation(output)
