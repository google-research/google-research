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

"""Wrapper for streaming inference."""

from absl import logging
from kws_streaming.layers import average_pooling2d
from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf


def frequeny_pad(inputs, dilation, stride, kernel_size):
  """Pads input tensor in frequency domain.

  Args:
    inputs: input tensor
    dilation: dilation in frequency dim
    stride: stride in frequency dim
    kernel_size: kernel_size in frequency dim

  Returns:
    padded tensor

  Raises:
    ValueError: if any of input rank is < 3
  """

  # expected input: [N, Time, Frequency, ...]
  if inputs.shape.rank < 3:
    raise ValueError('input_shape.rank:%d must be at least 3' %
                     inputs.shape.rank)

  kernel_size = (kernel_size - 1) * dilation + 1
  total_pad = kernel_size - stride

  pad_left = total_pad // 2
  pad_right = total_pad - pad_left

  pad = [[0, 0]] * inputs.shape.rank
  pad[2] = [pad_left, pad_right]
  return tf.pad(inputs, pad, 'constant')


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
    pad_time_dim: padding in time: None, causal or same.
      If 'same' then model will be non causal and developer will need to insert
      a delay layer to emulate looking ahead effect. Also there will be edge
      cases with residual connections. Demo of these is shown in delay_test.
      If 'causal' then whole conversion to streaming mode is fully automatic.
    state_shape:
    ring_buffer_size_in_time_dim: size of ring buffer in time dim
    use_one_step: True - model will run one sample per one inference step;
      False - model will run multiple per one inference step.
      It is useful for strided streaming
    state_name_tag: name tag for streaming state
    pad_freq_dim: type of padding in frequency dim: None or 'same'
    transposed_conv_crop_output: this parameter is used for
      transposed convolution only and will crop output tensor aligned by stride
      in time dimension only - it is important for streaming of transposed conv
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
               pad_freq_dim='valid',
               transposed_conv_crop_output=True,
               **kwargs):
    super(Stream, self).__init__(**kwargs)

    if pad_freq_dim not in ['same', 'valid']:
      raise ValueError(f'Unsupported padding in frequency, `{pad_freq_dim}`.')

    self.cell = cell
    self.inference_batch_size = inference_batch_size
    self.mode = mode
    self.pad_time_dim = pad_time_dim
    self.state_shape = state_shape
    self.ring_buffer_size_in_time_dim = ring_buffer_size_in_time_dim
    self.use_one_step = use_one_step
    self.state_name_tag = state_name_tag
    self.stride = 1
    self.pad_freq_dim = pad_freq_dim
    self.transposed_conv_crop_output = transposed_conv_crop_output

    self.stride_freq = 1
    self.dilation_freq = 1
    self.kernel_size_freq = 1

    wrapped_cell = self.get_core_layer()
    # pylint: disable=pointless-string-statement
    # pylint: disable=g-inconsistent-quotes
    padding_error = "Cell padding must be 'valid'. Additional context: "
    "keras does not support paddings in different dimensions, "
    "but in some cases we need different paddings in time and feature dims. "
    "Stream layer wraps conv cell and in streaming mode conv cell must use "
    "'valid' padding only. That is why paddings are managed by Stream wrapper "
    "with pad_time_dim and pad_freq_dim. pad_freq_dim is applied on dims with "
    "index = 2. pad_time_dim is applied on dim 1: time dimension. "
    # pylint: enable=g-inconsistent-quotes
    # pylint: enable=pointless-string-statement

    if not use_one_step and isinstance(
        wrapped_cell,
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
    elif isinstance(wrapped_cell, tf.keras.layers.Conv2DTranspose):
      padding = wrapped_cell.get_config()['padding']
      strides = wrapped_cell.get_config()['strides']
      self.stride = strides[0]
      kernel_size = wrapped_cell.get_config()['kernel_size']

      if padding != 'valid':
        raise ValueError(padding_error)

      # overlap in time domain defines ring buffer size
      self.ring_buffer_size_in_time_dim = max(kernel_size[0] - strides[0], 0)
    elif isinstance(
        wrapped_cell,
        (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D,
         tf.keras.layers.DepthwiseConv1D, tf.keras.layers.DepthwiseConv2D,
         tf.keras.layers.SeparableConv1D, tf.keras.layers.SeparableConv2D,
         average_pooling2d.AveragePooling2D)):
      padding = wrapped_cell.get_config()['padding']
      strides = wrapped_cell.get_config()['strides']
      self.stride = strides[0]

      if self.mode not in (modes.Modes.TRAINING,
                           modes.Modes.NON_STREAM_INFERENCE):
        if padding != 'valid':
          raise ValueError(padding_error)

      if self.mode not in (modes.Modes.TRAINING,
                           modes.Modes.NON_STREAM_INFERENCE):
        if self.use_one_step:
          if strides[0] > 1:
            raise ValueError('Stride in time dim greater than 1 '
                             'in streaming mode with use_one_step=True '
                             'is not supported, set use_one_step=False')

      dilation_rate = wrapped_cell.get_config()['dilation_rate']
      kernel_size = wrapped_cell.get_config()['kernel_size']

      # set parameters in frequency domain
      self.stride_freq = strides[1] if len(strides) > 1 else strides
      self.dilation_freq = dilation_rate[1] if len(
          dilation_rate) > 1 else dilation_rate
      self.kernel_size_freq = kernel_size[1] if len(
          kernel_size) > 1 else kernel_size

      if padding == 'same' and self.pad_freq_dim == 'same':
        raise ValueError('Cell padding and additional padding in frequency dim,'
                         'can not be the same. In this case conv cell will '
                         'pad both time and frequency dims and additional '
                         'frequency padding will be applied due to '
                         'pad_freq_dim')

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

    elif isinstance(wrapped_cell, tf.keras.layers.AveragePooling2D):
      strides = wrapped_cell.get_config()['strides']
      pool_size = wrapped_cell.get_config()['pool_size']
      self.stride = strides[0]
      if self.mode not in (
          modes.Modes.TRAINING,
          modes.Modes.NON_STREAM_INFERENCE) and strides[0] != pool_size[0]:
        raise ValueError('Stride in time %d must = pool size in time %d' %
                         (strides[0], pool_size[0]))
      # effective kernel size in time dimension
      self.ring_buffer_size_in_time_dim = pool_size[0]

    elif isinstance(
        wrapped_cell,
        (tf.keras.layers.Flatten, tf.keras.layers.GlobalMaxPooling2D,
         tf.keras.layers.GlobalAveragePooling2D)):
      # effective kernel size in time dimension
      if self.state_shape:
        self.ring_buffer_size_in_time_dim = self.state_shape[1]

    else:
      raise ValueError('Cell is not supported ', wrapped_cell)

    if self.ring_buffer_size_in_time_dim == 1:
      logging.warning('There is no need to use Stream on time dim with size 1')

  def get_core_layer(self):
    """Get core layer which can be wrapped by quantizer."""
    core_layer = self.cell
    # check two level of wrapping:
    if isinstance(core_layer, tf.keras.layers.Wrapper):
      core_layer = core_layer.layer
    if isinstance(core_layer, tf.keras.layers.Wrapper):
      core_layer = core_layer.layer
    return core_layer

  def stride(self):
    return self.stride

  def build(self, input_shape):
    super(Stream, self).build(input_shape)

    wrapped_cell = self.get_core_layer()
    if isinstance(wrapped_cell, tf.keras.layers.Conv2DTranspose):
      strides = wrapped_cell.get_config()['strides']
      kernel_size = wrapped_cell.get_config()['kernel_size']
      filters = wrapped_cell.get_config()['filters']

      # Only in streaming modes are these shapes and dimensions accessible.
      if self.mode in [
          modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
          modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
      ]:
        self.output_time_dim = input_shape.as_list()[1] * strides[0]

        # here we do not take into account padding, because it is always valid
        # only pad_time_dim can be applied and it does not impact feature dim
        output_feature_size = (input_shape[2] - 1) * strides[1] + kernel_size[1]

        # [batch, time dim(streaming dim), output_feature_size,
        # channels/filters]
        self.state_shape = [
            self.inference_batch_size, self.ring_buffer_size_in_time_dim,
            output_feature_size, filters
        ]
    elif isinstance(
        wrapped_cell,
        (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D,
         tf.keras.layers.DepthwiseConv1D, tf.keras.layers.DepthwiseConv2D,
         tf.keras.layers.SeparableConv1D, tf.keras.layers.SeparableConv2D,
         tf.keras.layers.AveragePooling2D)):

      self.state_shape = [
          self.inference_batch_size, self.ring_buffer_size_in_time_dim
      ] + input_shape.as_list()[2:]
    elif isinstance(
        wrapped_cell,
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

    # For streaming mode we may need different paddings in time
    # and frequency dimensions. When we train streaming aware model it should
    # have causal padding in time, and during streaming inference no padding
    # in time applied. So conv kernel always uses 'valid' padding and we add
    # causal padding in time during training. It is controlled
    # by self.pad_time_dim. In addition we may need 'same' or
    # 'valid' padding in frequency domain. For this case it has to be applied
    # in both training and inference modes. That is why we introduced
    # self.pad_freq_dim.
    if self.pad_freq_dim == 'same':
      inputs = frequeny_pad(inputs, self.dilation_freq, self.stride_freq,
                            self.kernel_size_freq)

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      return self._streaming_internal_state(inputs)

    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      if self.ring_buffer_size_in_time_dim:
        # in streaming inference mode with external state
        # in addition to the output we return the output state.
        output, self.output_state = self._streaming_external_state(
            inputs, self.input_state)
      else:
        # if there is no ring buffer then the input_state isn't needed.
        output = self.cell(inputs)
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
        'state_shape': self.state_shape,
        'ring_buffer_size_in_time_dim': self.ring_buffer_size_in_time_dim,
        'use_one_step': self.use_one_step,
        'state_name_tag': self.state_name_tag,
        'cell': self.cell,
        'pad_freq_dim': self.pad_freq_dim,
        'transposed_conv_crop_output': self.transposed_conv_crop_output,
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
    if isinstance(self.get_core_layer(), tf.keras.layers.Conv2DTranspose):
      outputs = self.cell(inputs)

      if self.ring_buffer_size_in_time_dim == 0:
        if self.transposed_conv_crop_output:
          outputs = outputs[:, 0:self.output_time_dim]
        return outputs

      output_shape = outputs.shape.as_list()

      # need to add remainder state to a specific region of output as below:
      # outputs[:,0:self.ring_buffer_size_in_time_dim,:] =
      # outputs[:,0:self.ring_buffer_size_in_time_dim,:] + self.states
      # but 'Tensor' object does not support item assignment,
      # so doing it through full summation below
      output_shape[1] -= self.state_shape[1]
      padded_remainder = tf.concat(
          [self.states, tf.zeros(output_shape, tf.float32)], 1)
      outputs = outputs + padded_remainder

      # extract remainder state and subtract bias if it is used:
      # bias will be added in the next iteration again and remainder
      # should have only convolution part, so that bias is not added twice
      if self.get_core_layer().get_config()['use_bias']:
        # need to access bias of the cell layer,
        # where cell can be wrapped by wrapper layer
        bias = self.get_core_layer().bias
        new_state = outputs[:, -self.ring_buffer_size_in_time_dim:, :] - bias  # pylint: disable=invalid-unary-operand-type
      else:
        new_state = outputs[:, -self.ring_buffer_size_in_time_dim:, :]  # pylint: disable=invalid-unary-operand-type
      assign_states = self.states.assign(new_state)

      with tf.control_dependencies([assign_states]):
        if self.transposed_conv_crop_output:
          return tf.identity(outputs[:, 0:self.output_time_dim, :])
        else:
          return tf.identity(outputs)
    else:
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
          return self.cell(inputs)

  def _streaming_external_state(self, inputs, state):
    state = [] if state is None else state
    if isinstance(self.get_core_layer(), tf.keras.layers.Conv2DTranspose):
      outputs = self.cell(inputs)

      if self.ring_buffer_size_in_time_dim == 0:
        if self.transposed_conv_crop_output:
          outputs = outputs[:, 0:self.output_time_dim, :]
        return outputs, []

      output_shape = outputs.shape.as_list()

      output_shape[1] -= self.state_shape[1]
      padded_remainder = tf.concat(
          [state, tf.zeros(output_shape, tf.float32)], 1)
      outputs = outputs + padded_remainder

      if self.get_core_layer().get_config()['use_bias']:
        # need to access bias of the cell layer,
        # where cell can be wrapped by wrapper layer
        bias = self.get_core_layer().bias

        new_state = outputs[:, -self.ring_buffer_size_in_time_dim:, :] - bias  # pylint: disable=invalid-unary-operand-type
      else:
        new_state = outputs[:, -self.ring_buffer_size_in_time_dim:, :]  # pylint: disable=invalid-unary-operand-type

      if self.transposed_conv_crop_output:
        outputs = outputs[:, 0:self.output_time_dim, :]
      return outputs, new_state
    else:
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
        memory = tf.keras.backend.concatenate([state, inputs], 1)

        state_update = memory[:, -self.ring_buffer_size_in_time_dim:, :]  # pylint: disable=invalid-unary-operand-type

        output = self.cell(memory)
        return output, state_update

  def _non_streaming(self, inputs):
    # transposed conv is a special case
    if isinstance(self.get_core_layer(), tf.keras.layers.Conv2DTranspose):
      outputs = self.cell(inputs)

      # during training or non streaming inference, input shape can be dynamic
      self.output_time_dim = tf.shape(inputs)[1] * self.stride
      if self.transposed_conv_crop_output:
        if self.pad_time_dim == 'same':
          crop_left = self.ring_buffer_size_in_time_dim // 2
          return outputs[:, crop_left:crop_left + self.output_time_dim, :]
        else:
          return outputs[:, 0:self.output_time_dim, :]
      else:
        return outputs
    else:
      # Pad inputs in time dim: causal or same
      if self.pad_time_dim:
        if isinstance(
            self.cell,
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

      return self.cell(inputs)
