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

"""Wrapper for streaming inference."""

from absl import logging
from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf


class Stream(tf.keras.layers.Layer):
  """Streaming wrapper - it is not a standalone layer.

  It can be used to wrap Keras layer for streaming inference mode.
  Advantage of streaming inference mode - it is more computationally efficient.
  But not all layers are streamable. Some layers require keeping a buffer
  with features in time. We can wrap such layer by Stream().
  Where Stream() will create and keep a temporal buffer called state,
  for both cases: internal state and external state.
  Examples of layers which require temporal buffer/state
  for streaming inference are Conv2D, DepthwiseConv2D, AveragePooling2D,
  Flatten in time dimension, etc.

  This wrapper is generic enough, so that it can be used for any modes:
  1 Streaming with internal state. This wrapper will manage internal state.
  2 Streaming with external state. Developer will have to manage external state
  and feed it as additional input to the model and then receive output with
  updated state.
  3 Non streaming inference mode. In this case wrapper will just call
  a wrapped layer as it is. There will be no difference in efficiency.
  The graph will be the same as in training mode, but some training features
  will be removed (such as dropout, etc)
  4 Training mode.

  Attributes:
    cell: keras layer which has to be streamed or tf.identity
    inference_batch_size: batch size in inference mode
    mode: inference or training mode
    pad_time_dim: padding in time
    state_shape:
    ring_buffer_size_in_time_dim: size of ring buffer in time dim
    use_one_step: True - model will run one sample per one inference step;
      False - model will run multiple per one inference step.
      It is useful for strided streaming
    **kwargs: additional layer arguments

  Raises:
    ValueError: if padding is not 'valid' in streaming mode;
                or if striding is used with use_one_step;
                or cell is not supported
  """

  def __init__(self,
               cell,
               inference_batch_size=1,
               mode=modes.Modes.TRAINING,
               pad_time_dim=None,
               state_shape=None,
               ring_buffer_size_in_time_dim=None,
               use_one_step=True,
               state_name_tag='ExternalState',
               **kwargs):
    super(Stream, self).__init__(**kwargs)

    self.cell = cell
    self.inference_batch_size = inference_batch_size
    self.mode = mode
    self.pad_time_dim = pad_time_dim
    self.state_shape = state_shape
    self.ring_buffer_size_in_time_dim = ring_buffer_size_in_time_dim
    self.use_one_step = use_one_step
    self.state_name_tag = state_name_tag

    if not use_one_step and isinstance(
        self.cell, (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
                    tf.keras.layers.GlobalAveragePooling2D)):
      raise ValueError('Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D '
                       'can be used only with use_one_step = True '
                       'because they are executed one time per inference call '
                       'and produce only one output in time dim, whereas conv '
                       'can produce multiple outputs in time dim, '
                       'so conv can be used with use_one_step = False or True')

    if self.ring_buffer_size_in_time_dim:
      # it is a special case when ring_buffer_size_in_time_dim is specified
      # outside of the layer in this case we just build a ring buffer
      # and do not check what is the type of the cell
      pass
    elif isinstance(
        cell, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D,
               tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D,
               tf.keras.layers.SeparableConv1D)):

      if self.mode not in (modes.Modes.TRAINING,
                           modes.Modes.NON_STREAM_INFERENCE):
        padding = cell.get_config()['padding']
        strides = cell.get_config()['strides']
        if padding != 'valid':
          raise ValueError('conv/cell padding has to be valid,'
                           'padding has to be set by pad_time_dim')

        if self.use_one_step:
          if strides[0] > 1:
            raise ValueError('Stride in time dim greater than 1 '
                             'in streaming mode with use_one_step=True'
                             ' is not supported, set use_one_step=False')

      dilation_rate = cell.get_config()['dilation_rate']
      kernel_size = cell.get_config()['kernel_size']
      if self.use_one_step:
        # effective kernel size in time dimension
        self.ring_buffer_size_in_time_dim = dilation_rate[0] * (kernel_size[0] -
                                                                1) + 1
      else:
        # streaming of strided or 1 step conv
        self.ring_buffer_size_in_time_dim = dilation_rate[0] * (
            kernel_size[0] - 1)

    elif isinstance(self.cell, tf.keras.layers.AveragePooling2D):
      strides = cell.get_config()['strides']
      pool_size = cell.get_config()['pool_size']
      if self.mode not in (
          modes.Modes.TRAINING,
          modes.Modes.NON_STREAM_INFERENCE) and strides[0] != pool_size[0]:
        raise ValueError('Stride in time %d must = pool size in time %d' %
                         (strides[0], pool_size[0]))
      # effective kernel size in time dimension
      self.ring_buffer_size_in_time_dim = pool_size[0]

    elif isinstance(
        self.cell, (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
                    tf.keras.layers.GlobalAveragePooling2D)):
      # effective kernel size in time dimension
      if self.state_shape:
        self.ring_buffer_size_in_time_dim = self.state_shape[1]

    else:
      raise ValueError('Cell is not supported ', cell)

    if self.ring_buffer_size_in_time_dim == 1:
      logging.warn('There is no need to use Stream on time dim with size 1')

  def build(self, input_shape):
    super(Stream, self).build(input_shape)
    if isinstance(
        self.cell,
        (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D,
         tf.keras.layers.DepthwiseConv2D, tf.keras.layers.AveragePooling2D,
         tf.keras.layers.SeparableConv2D, tf.keras.layers.SeparableConv1D)):

      self.state_shape = [
          self.inference_batch_size, self.ring_buffer_size_in_time_dim
      ] + input_shape.as_list()[2:]
    elif isinstance(
        self.cell,
        (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
         tf.keras.layers.GlobalAveragePooling2D)) and not self.state_shape:
      if self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
        # Only in the non-streaming modes we have access to the whole training
        # sequence. In the streaming mode input_shape will not be available.
        # During streaming inference we have access to one sample at a time!
        # So we generate state shape based on input_shape during training.
        # It will be stored in the layer config
        # Then used by clone_streaming_model to create state buffer,
        # during layer initialization.
        # [batch, time, feature, ...]
        self.state_shape = input_shape.as_list()
        self.state_shape[0] = self.inference_batch_size
    elif self.ring_buffer_size_in_time_dim:
      # it is a special case when ring_buffer_size_in_time_dim
      # is defined by user and cell is not defined in Stream wrapper
      self.state_shape = [
          self.inference_batch_size, self.ring_buffer_size_in_time_dim
      ] + input_shape.as_list()[2:]

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # Create a state varaible for streaming inference mode (internal state).
      # Where states become a weight in the layer
      if self.ring_buffer_size_in_time_dim:
        self.states = self.add_weight(
            name='states',
            shape=self.state_shape,
            trainable=False,
            initializer=tf.zeros_initializer)

    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # For streaming inference with extrnal states,
      # the states are passed in as input.
      if self.ring_buffer_size_in_time_dim:
        self.input_state = tf.keras.layers.Input(
            shape=self.state_shape[1:],
            batch_size=self.inference_batch_size,
            name=self.name + '/' +
            self.state_name_tag)  # adding names to make it unique
      else:
        self.input_state = None
      self.output_state = None

  def call(self, inputs):
    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      return self._streaming_internal_state(inputs)

    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming inference mode with external state
      # in addition to the output we return the output state.
      output, self.output_state = self._streaming_external_state(
          inputs, self.input_state)
      return output

    elif self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      return self._non_streaming(inputs)

    else:
      raise ValueError(f'Encountered unexpected mode `{self.mode}`.')

  def get_config(self):
    config = super(Stream, self).get_config()
    config.update({
        'inference_batch_size': self.inference_batch_size,
        'mode': self.mode,
        'pad_time_dim': self.pad_time_dim,
        'cell': self.cell,
        'state_shape': self.state_shape,
        'ring_buffer_size_in_time_dim': self.ring_buffer_size_in_time_dim,
        'use_one_step': self.use_one_step,
        'state_name_tag': self.state_name_tag,
    })
    return config

  def get_input_state(self):
    # input state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.input_state]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def get_output_state(self):
    # output state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.output_state]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def _streaming_internal_state(self, inputs):
    if self.use_one_step:
      # The time dimenstion always has to equal 1 in streaming mode.
      if inputs.shape[1] != 1:
        raise ValueError('inputs.shape[1]: %d must be 1 ' % inputs.shape[1])

      # remove latest row [batch_size, (memory_size-1), feature_dim, channel]
      memory = self.states[:, 1:self.ring_buffer_size_in_time_dim, :]

      # add new row [batch_size, memory_size, feature_dim, channel]
      memory = tf.keras.backend.concatenate([memory, inputs], 1)

      assign_states = self.states.assign(memory)

      with tf.control_dependencies([assign_states]):
        return self.cell(memory)
    else:
      # add new row [batch_size, memory_size, feature_dim, channel]
      if self.ring_buffer_size_in_time_dim:
        memory = tf.keras.backend.concatenate([self.states, inputs], 1)

        state_update = memory[:, -self.ring_buffer_size_in_time_dim:, :]  # pylint: disable=invalid-unary-operand-type

        assign_states = self.states.assign(state_update)

        with tf.control_dependencies([assign_states]):
          return self.cell(memory)
      else:
        return inputs

  def _streaming_external_state(self, inputs, state):
    state = [] if state is None else state
    if self.use_one_step:
      # The time dimenstion always has to equal 1 in streaming mode.
      if inputs.shape[1] != 1:
        raise ValueError('inputs.shape[1]: %d must be 1 ' % inputs.shape[1])

      # remove latest row [batch_size, (memory_size-1), feature_dim, channel]
      memory = state[:, 1:self.ring_buffer_size_in_time_dim, :]

      # add new row [batch_size, memory_size, feature_dim, channel]
      memory = tf.keras.backend.concatenate([memory, inputs], 1)

      output = self.cell(memory)
      return output, memory
    else:
      # add new row [batch_size, memory_size, feature_dim, channel]
      if self.ring_buffer_size_in_time_dim:
        memory = tf.keras.backend.concatenate([state, inputs], 1)

        state_update = memory[:, -self.ring_buffer_size_in_time_dim:, :]  # pylint: disable=invalid-unary-operand-type

        output = self.cell(memory)
        return output, state_update
      else:
        return inputs, []

  def _non_streaming(self, inputs):
    # Pad inputs in time dim: causal or same
    if self.pad_time_dim:
      if isinstance(
          self.cell,
          (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
           tf.keras.layers.GlobalAveragePooling2D)):
        raise ValueError('pad_time_dim can not be used with Flatten')

      # temporal padding
      pad = [[0, 0]] * inputs.shape.rank
      if self.pad_time_dim == 'causal':
        if self.use_one_step:
          pad[1] = [self.ring_buffer_size_in_time_dim - 1, 0]
        else:
          pad[1] = [self.ring_buffer_size_in_time_dim, 0]
      elif self.pad_time_dim == 'same':
        half = self.ring_buffer_size_in_time_dim // 2
        pad[1] = [half, half]
      inputs = tf.pad(inputs, pad, 'constant')

    return self.cell(inputs)
