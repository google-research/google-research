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

"""SVDF layer."""
from kws_streaming.layers import depthwiseconv1d
from kws_streaming.layers import modes
from kws_streaming.layers import non_scaling_dropout
from kws_streaming.layers.compat import tf


class Svdf(tf.keras.layers.Layer):
  """svdf streaming with support of both internal and external states.

  It is a sequence of 1D convolutions in vertical and horizontal directions.
  It is used to reduce comutation of 2d convoultion,
  by factoring it into combination of 1d colvolutions in feature and time dims.
  """

  def __init__(self,
               memory_size,
               units1,
               use_bias1=False,
               units2=-1,
               activation='relu',
               use_bias=True,
               inference_batch_size=1,
               mode=modes.Modes.TRAINING,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               kernel_constraint=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               bias_constraint=None,
               dropout=0.0,
               pad=True,
               **kwargs):
    super(Svdf, self).__init__(**kwargs)

    self.memory_size = memory_size
    self.units1 = units1  # number of hidden units in the first dense layer
    self.use_bias1 = use_bias1  # use bias in the first dense layer

    self.units2 = units2  # number of hidden units in the second dense layer
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias  # use bias at depthwiseconv1d
    self.inference_batch_size = inference_batch_size
    self.mode = mode

    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.kernel_constraint = kernel_constraint

    self.bias_initializer = bias_initializer
    self.bias_regularizer = bias_regularizer
    self.bias_constraint = bias_constraint
    self.dropout = min(1., max(0., dropout))
    self.pad = pad

  def build(self, input_shape):
    super(Svdf, self).build(input_shape)

    if self.mode == modes.Modes.TRAINING:
      self.dropout1 = non_scaling_dropout.NonScalingDropout(
          self.dropout, training=True)
    else:
      self.dropout1 = tf.keras.layers.Lambda(lambda x: x)
    self.dense1 = tf.keras.layers.Dense(
        units=self.units1, use_bias=self.use_bias1)
    self.depth_cnn1 = depthwiseconv1d.DepthwiseConv1D(
        memory_size=self.memory_size,
        inference_batch_size=self.inference_batch_size,
        use_bias=self.use_bias,
        mode=self.mode,
        pad=self.pad)
    if self.units2 > 0:
      self.dense2 = tf.keras.layers.Dense(units=self.units2, use_bias=True)
    else:
      self.dense2 = tf.keras.layers.Lambda(lambda x: x)

  def compute_output_shape(self, input_shape):
    if input_shape.rank != 3:
      raise ValueError('input_shape.rank:%d must = 3' % input_shape.rank)
    if self.mode not in (modes.Modes.TRAINING,
                         modes.Modes.NON_STREAM_INFERENCE):
      if input_shape[1] != 1:
        raise ValueError('input_shape[1]:%d must = 1' % input_shape[1])

    output_shape = input_shape
    output_shape[-1] = self.units2
    return output_shape

  def call(self, inputs):
    output = self.dropout1(inputs)
    output = self.dense1(output)
    output = self.depth_cnn1(output)
    output = self.activation(output)
    output = self.dense2(output)
    return output

  def get_config(self):
    config = {
        'memory_size': self.memory_size,
        'units1': self.units1,
        'use_bias1': self.use_bias1,
        'units2': self.units2,
        'activation': self.activation,
        'use_bias': self.use_bias,
        'inference_batch_size': self.inference_batch_size,
        'mode': self.mode,
        'kernel_initializer': self.kernel_initializer,
        'kernel_regularizer': self.kernel_regularizer,
        'kernel_constraint': self.kernel_constraint,
        'bias_initializer': self.bias_initializer,
        'bias_regularizer': self.bias_regularizer,
        'bias_constraint': self.bias_constraint,
        'dropout': self.dropout,
        'pad': self.pad,
    }
    base_config = super(Svdf, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    return self.depth_cnn1.get_input_state()

  def get_output_state(self):
    return self.depth_cnn1.get_output_state()
