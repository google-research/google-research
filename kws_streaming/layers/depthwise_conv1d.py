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

"""Depthwise Conv1D layer for streaming and non streaming use case."""

from kws_streaming.layers import modes
from kws_streaming.layers import temporal_padding
from kws_streaming.layers.compat import tf


class DepthwiseConv1D(tf.keras.layers.Layer):
  """Depthwise 1D convolution with support of streaming inference.

     Input data in training mode has shape [batch, time, feature_dim]
     It computes a convolution of multiple 1d kernels in time direction,
     where number of kernels is equal to feature_dim.
     Input data in inference mode has shape [batch, feature_dim]
     In inference mode it creates a buffer called "self.states"
     with dims [batch_size, memory_size, feature_dim]
     It updates "memory" with every inference iteration:
     by adding new feature_dim and removing the oldest one
     so that memory_size is kept constant.
     Then it computes one step convolution of multiple 1d kernels
     in memory_size direction
  """

  def __init__(self,
               memory_size,
               inference_batch_size=1,
               use_bias=True,
               mode=modes.Modes.TRAINING,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               kernel_constraint=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               bias_constraint=None,
               pad='causal',
               **kwargs):
    super(DepthwiseConv1D, self).__init__(**kwargs)
    self.memory_size = memory_size

    # it has to be set for inference mode only
    self.inference_batch_size = inference_batch_size
    self.use_bias = use_bias
    self.mode = mode

    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.kernel_constraint = kernel_constraint

    self.bias_initializer = bias_initializer
    self.bias_regularizer = bias_regularizer
    self.bias_constraint = bias_constraint
    self.pad = pad

  def build(self, input_shape):
    super(DepthwiseConv1D, self).build(input_shape)
    feature_dim = input_shape[2]  # feature dim is a third dimension
    self.time_kernel = self.add_weight(
        shape=(self.memory_size, feature_dim),
        name='time_kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(feature_dim,),
          name='time_bias',
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # create state varaible for streamable inference mode only
      self.states = self.add_weight(
          name='states',
          shape=[self.inference_batch_size, self.memory_size, feature_dim],
          trainable=False,
          initializer=tf.zeros_initializer)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with external state,
      # state becomes an input output placeholder
      self.input_state = tf.keras.layers.Input(
          shape=(
              self.memory_size,
              feature_dim,
          ),
          batch_size=self.inference_batch_size,
          name=self.name + 'input_state')  # adding names to make it unique
      self.output_state = None

  def call(self, inputs):

    if inputs.shape.rank != 3:  # [batch, time, feature]
      raise ValueError('inputs.shape.rank: %d must be 3 ' % inputs.shape.rank)

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # run streamable inference on input [batch, 1, feature]
      # returns output [batch, 1, feature]
      return self._streaming_internal_state(inputs)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with extrnal state in addition to output
      # we return output state
      output, self.output_state = self._streaming_external_state(
          inputs, self.input_state)
      return output
    elif self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      # on input [batch, time, features],
      # returns output [batch, time, features]
      return self._non_streaming(inputs)
    else:
      raise ValueError('wrong mode', self.mode)

  def get_config(self):
    config = {
        'memory_size': self.memory_size,
        'inference_batch_size': self.inference_batch_size,
        'use_bias': self.use_bias,
        'mode': self.mode,
        'kernel_initializer': self.kernel_initializer,
        'kernel_regularizer': self.kernel_regularizer,
        'kernel_constraint': self.kernel_constraint,
        'bias_initializer': self.bias_initializer,
        'bias_regularizer': self.bias_regularizer,
        'bias_constraint': self.bias_constraint,
        'pad': self.pad,
    }
    base_config = super(DepthwiseConv1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    # input state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return self.input_state
    else:
      raise ValueError('wrong mode', self.mode)

  def get_output_state(self):
    # output state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return self.output_state
    else:
      raise ValueError('wrong mode', self.mode)

  def _streaming_internal_state(self, inputs):
    # depthwise 1D convolution in streaming mode with internal state
    # it is used for streaming inference
    if inputs.shape[1] != 1:  # [batch, 1, feature]
      raise ValueError('inputs.shape[1]: %d must be 1 ' % inputs.shape[1])

    # remove latest row [batch_size, (memory_size-1), feature_dim]
    memory = self.states[:, 1:self.memory_size, :]

    # add new row [batch_size, memory_size, feature_dim]
    memory = tf.keras.backend.concatenate([memory, inputs], 1)
    assign_states = self.states.assign(memory)

    with tf.control_dependencies([assign_states]):
      # elementwise multiplication [batch_size, memory_size, feature_dim]
      output = memory * self.time_kernel

      # [batch_size, feature_dim]
      output_sum = tf.keras.backend.sum(output, axis=1)

      if self.use_bias:
        output_sum = output_sum + self.bias

      output_sum = tf.keras.backend.expand_dims(output_sum, -2)

      return output_sum  # [batch, 1, feature]

  def _streaming_external_state(self, inputs, state):
    # depthwise 1D convolution in streaming mode with external state:
    # state will be received as additional input and then
    # updated state will be returned as an output
    # it is used for streaming inference
    if inputs.shape[1] != 1:  # [batch, 1, feature]
      raise ValueError('inputs.shape[1]: %d must be 1 ' % inputs.shape[1])

    # remove latest row [batch_size, (memory_size-1), feature_dim]
    memory = state[:, 1:self.memory_size, :]

    # add new row [batch_size, memory_size, feature_dim]
    memory = tf.keras.backend.concatenate([memory, inputs], 1)

    # elementwise multiplication [batch_size, memory_size, feature_dim]
    output = memory * self.time_kernel

    # [batch_size, feature_dim]
    output_sum = tf.keras.backend.sum(output, axis=1)

    if self.use_bias:
      output_sum = output_sum + self.bias

    # [batch, 1, feature]
    output_sum = tf.keras.backend.expand_dims(output_sum, -2)

    # return output with memory state,
    # where memory state will be input state on the next inference step
    return output_sum, memory

  def _non_streaming(self, inputs):
    # depthwise 1D convolution in non streaming mode
    # it is used for training or non streaming inference.

    # pad input data
    inputs_pad = temporal_padding.TemporalPadding(
        padding=self.pad, padding_size=self.memory_size - 1)(
            inputs)

    # expand dimensionality for depthwise_conv2d
    # to [memory_size, 1, feature_dim, 1]
    time_kernel_exp = tf.expand_dims(tf.expand_dims(self.time_kernel, 1), -1)

    # run convolution
    depthwise_conv1d = tf.nn.depthwise_conv2d(
        tf.expand_dims(inputs_pad, -2),
        time_kernel_exp,
        strides=[1, 1, 1, 1],
        padding='VALID')  # [batch_size, time_steps, 1, feature_dim]

    # [batch_size, time_steps, feature_dim]
    depthwise_conv1d = tf.squeeze(depthwise_conv1d, [2])

    # [batch_size, time_steps, feature_dim]
    if self.use_bias:
      depthwise_conv1d = depthwise_conv1d + self.bias

    return depthwise_conv1d
