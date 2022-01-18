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

"""Ring buffer Wrapper for streaming inference."""

from absl import logging
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper


class RingBuffer(tf.keras.layers.Wrapper):
  """RingBuffer wrapper - it is not a standalone layer.

  It can be used to wrap Keras layer for streaming inference mode.
  It is similar with Stream wrapper, but it is applicable only
  for subclass modesl with external state:
  user will have to manage extrenal state

  Attributes:
    layer: keras layer which has to be streamed or tf.identity
    inference_batch_size: batch size in inference mode
    pad_time_dim: padding in time
    state_shape: shape of the state/ring_buffer specified by user
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
               layer,
               inference_batch_size=1,
               pad_time_dim=None,
               state_shape=None,
               ring_buffer_size_in_time_dim=None,
               use_one_step=True,
               **kwargs):
    super(RingBuffer, self).__init__(**kwargs, layer=layer)

    self.inference_batch_size = inference_batch_size
    self.pad_time_dim = pad_time_dim
    self.state_shape = state_shape
    self.ring_buffer_size_in_time_dim = ring_buffer_size_in_time_dim
    self.use_one_step = use_one_step
    self.stride = 1

    wrappped_cell = self.get_core_layer()

    if not use_one_step and isinstance(
        wrappped_cell,
        (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
         tf.keras.layers.GlobalAveragePooling2D)):
      raise ValueError('Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D '
                       'can be used only with use_one_step = True '
                       'because they are executed one time per inference call '
                       'and produce only one output in time dim, whereas conv '
                       'can produce multiple outputs in time dim, '
                       'so conv can be used with use_one_step = False or True')

    if self.ring_buffer_size_in_time_dim is not None:
      # it is a special case when ring_buffer_size_in_time_dim is specified
      # outside of the layer in this case we just build a ring buffer
      # and do not check what is the type of the cell
      pass
    elif isinstance(
        wrappped_cell,
        (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D,
         tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D,
         tf.keras.layers.SeparableConv1D)):
      padding = wrappped_cell.get_config()['padding']
      strides = wrappped_cell.get_config()['strides']
      self.stride = strides[0]

      if padding != 'valid':
        raise ValueError('conv/cell padding has to be valid,'
                         'padding has to be set by pad_time_dim')

      if self.use_one_step:
        if strides[0] > 1:
          raise ValueError('Stride in time dim greater than 1 '
                           'in streaming mode with use_one_step=True'
                           ' is not supported, set use_one_step=False')

      dilation_rate = wrappped_cell.get_config()['dilation_rate']
      kernel_size = wrappped_cell.get_config()['kernel_size']
      if self.use_one_step:
        # effective kernel size in time dimension
        self.ring_buffer_size_in_time_dim = dilation_rate[0] * (kernel_size[0] -
                                                                1) + 1
      else:
        # Streaming of strided or 1 step conv.
        # Assuming input length is a multiple of strides (otherwise streaming
        # conv is not meaningful), setting to this value (instead of
        # dilation_rate[0] * (kernel_size[0] - 1)) ensures that we do not
        # ignore the `strides - 1` rightmost (and hence most recent) valid
        # input samples.
        self.ring_buffer_size_in_time_dim = max(
            0, dilation_rate[0] * (kernel_size[0] - 1) - (strides[0] - 1))

    elif isinstance(wrappped_cell, tf.keras.layers.AveragePooling2D):
      strides = wrappped_cell.get_config()['strides']
      pool_size = wrappped_cell.get_config()['pool_size']
      self.stride = strides[0]
      if strides[0] != pool_size[0]:
        raise ValueError('Stride in time %d must = pool size in time %d' %
                         (strides[0], pool_size[0]))
      # effective kernel size in time dimension
      self.ring_buffer_size_in_time_dim = pool_size[0]

    elif isinstance(
        wrappped_cell,
        (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
         tf.keras.layers.GlobalAveragePooling2D)):
      # effective kernel size in time dimension
      if self.state_shape:
        self.ring_buffer_size_in_time_dim = self.state_shape[1]

    else:
      raise ValueError('Cell is not supported ', wrappped_cell)

    if self.ring_buffer_size_in_time_dim == 1:
      logging.warning('There is no need to use Stream on time dim with size 1')

  def get_core_layer(self):
    core_layer = self.layer
    if isinstance(core_layer, quantize_wrapper.QuantizeWrapper):
      core_layer = core_layer.layer
    return core_layer

  def stride(self):
    return self.stride

  def build(self, input_shape):
    super(RingBuffer, self).build(input_shape)

    wrappped_cell = self.get_core_layer()

    if isinstance(
        wrappped_cell,
        (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D,
         tf.keras.layers.DepthwiseConv2D, tf.keras.layers.AveragePooling2D,
         tf.keras.layers.SeparableConv2D, tf.keras.layers.SeparableConv1D)):

      self.state_shape = [
          self.inference_batch_size, self.ring_buffer_size_in_time_dim
      ] + input_shape.as_list()[2:]
    elif isinstance(
        wrappped_cell,
        (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
         tf.keras.layers.GlobalAveragePooling2D)) and not self.state_shape:

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
      self.ring_buffer_size_in_time_dim = self.state_shape[1]
    elif self.ring_buffer_size_in_time_dim:
      # it is a special case when ring_buffer_size_in_time_dim
      # is defined by user and cell is not defined in Stream wrapper
      self.state_shape = [
          self.inference_batch_size, self.ring_buffer_size_in_time_dim
      ] + input_shape.as_list()[2:]

  def call(self, inputs, state=None):
    if state is None:
      # run non streamable training or non streamable inference
      return self._non_streaming(inputs)
    else:
      if self.ring_buffer_size_in_time_dim:
        # in streaming inference mode with external state
        # in addition to the output we return the output state.
        output, output_state = self._streaming_external_state(inputs, state)
      else:
        # if there is no ring buffer then the input_state isn't needed.
        output = self.layer(inputs)
        output_state = None
      return output, output_state

  def get_config(self):
    config = super(RingBuffer, self).get_config()
    config.update({
        'inference_batch_size': self.inference_batch_size,
        'pad_time_dim': self.pad_time_dim,
        'state_shape': self.state_shape,
        'ring_buffer_size_in_time_dim': self.ring_buffer_size_in_time_dim,
        'use_one_step': self.use_one_step,
    })
    return config

  def get_input_state_shape(self):
    return self.state_shape

  def _streaming_external_state(self, inputs, state):
    if self.use_one_step:
      # The time dimenstion always has to equal 1 in streaming mode.
      if inputs.shape[1] != 1:
        raise ValueError('inputs.shape[1]: %d must be 1 ' % inputs.shape[1])

      # remove latest row [batch_size, (memory_size-1), feature_dim, channel]
      memory = state[:, 1:self.ring_buffer_size_in_time_dim, :]

      # add new row [batch_size, memory_size, feature_dim, channel]
      memory = tf.keras.backend.concatenate([memory, inputs], 1)

      output = self.layer(memory)
      return output, memory
    else:
      # add new row [batch_size, memory_size, feature_dim, channel]
      memory = tf.keras.backend.concatenate([state, inputs], 1)

      state_update = memory[:, -self.ring_buffer_size_in_time_dim:, :]  # pylint: disable=invalid-unary-operand-type

      output = self.layer(memory)
      return output, state_update

  def _non_streaming(self, inputs):
    # Pad inputs in time dim: causal or same
    if self.pad_time_dim:
      if isinstance(
          self.layer,
          (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
           tf.keras.layers.GlobalAveragePooling2D)):
        raise ValueError('pad_time_dim can not be used with Flatten')

      # temporal padding
      pad = [[0, 0]] * inputs.shape.rank
      if self.use_one_step:
        pad_total_amount = self.ring_buffer_size_in_time_dim - 1
      else:
        pad_total_amount = self.ring_buffer_size_in_time_dim
      if self.pad_time_dim == 'causal':
        pad[1] = [pad_total_amount, 0]
      elif self.pad_time_dim == 'same':
        half = pad_total_amount // 2
        pad[1] = [half, pad_total_amount - half]
      inputs = tf.pad(inputs, pad, 'constant')
    return self.layer(inputs)
