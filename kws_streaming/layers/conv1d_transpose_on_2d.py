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

"""Conv1DTranspose streaming aware layer on 2D data."""

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf


class Conv1DTransposeOn2D(tf.keras.layers.Conv2DTranspose):
  """Streaming aware Conv1DTranspose layer applied on 2D data.

  It is an alternative to Conv1DTranspose.
  For context Conv1DTranspose processes input data with dims
  [batch, time_dim, channels] and is based on tf.keras.layers.Conv1DTranspose:
  in TFlite it will call reshape to [batch, 1, time_dim, channels] and then
  call Conv2DTranspose and then reshape it back to [batch, time_dim, channels].
  So to avoid reshape call we created this layer, which runs Conv1DTranspose
  on 2D data [batch, time_dim, 1, channels] and is based on
  tf.keras.layers.Conv2DTranspose.
  In above example (for simplicity) we assumed that input and output channels
  are the same, but in general they are different as shown in
  conv1d_transpose_on_2d_test.py.

  Attributes:
    mode: Training or inference modes: non streaming, streaming.
    inference_batch_size: batch size in inference mode.
    state_shape: shape of remainder state.
    crop_output: if True output will be cropped: aligned by stride.
    **kwargs: additional layer arguments
  """

  def __init__(self,
               mode=modes.Modes.TRAINING,
               inference_batch_size=1,
               pad_time_dim='causal',
               state_shape=None,
               crop_output=True,
               **kwargs):
    super(Conv1DTransposeOn2D, self).__init__(**kwargs)

    if (kwargs.get('activation') not in [None, 'linear']) and self.use_bias:
      raise ValueError('activation should be disabled because we need to '
                       'subtract bias from remainder state, in streaming mode',
                       kwargs.get('activation'))

    if len(self.kernel_size) != 2:
      raise ValueError('len(kernel_size):%d must be 2' % len(self.kernel_size))

    if self.kernel_size[1] != 1:
      raise ValueError('kernel_size[1]:%d must be 1' % self.kernel_size[1])

    if len(self.strides) != 2:
      raise ValueError('len(strides):%d must be 2' % len(self.strides))

    if self.strides[1] != 1:
      raise ValueError('strides[1]:%d must be 1' % self.strides[1])

    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.pad_time_dim = pad_time_dim
    self.state_shape = state_shape
    self.crop_output = crop_output

    self.overlap = self.kernel_size[0] - self.strides[0]
    self.overlap = max(self.overlap, 0)

    if pad_time_dim not in ['same', 'causal']:
      raise ValueError(
          'pad_time_dim (\'%s\') must be either \'same\' or \'causal\'' %
          pad_time_dim)

    if 'padding' in kwargs and kwargs['padding'] != 'valid':
      raise ValueError(
          'padding (\'%s\') must be \'valid\'. Use pad_time_dim to make the '
          'layer causal (\'causal\') or with lookahead (\'same\')' %
          kwargs['padding'])

  def build(self, input_shape):
    super(Conv1DTransposeOn2D, self).build(input_shape)

    if input_shape.rank != 4:
      raise ValueError('input_shape.rank:%d must 4' % input_shape.rank)

    if self.mode in [
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE
    ]:
      if input_shape.as_list()[1] is None:
        raise ValueError('in streaming mode time dimension of input packet '
                         'should not be dynamic: TFLite limitation')

      self.output_time_dim = input_shape.as_list()[1] * self.strides[0]

      # TODO(rybakov) extend it to 2D for STFT by setting state_shape =
      # input_shape[2] * strides[1] + max(0, strides[1] - kernel_size[1])
      self.state_shape = [
          self.inference_batch_size, self.overlap, 1, self.filters
      ]

      if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
        if self.overlap > 0:
          self.states = self.add_weight(
              name='states',
              shape=self.state_shape,
              trainable=False,
              initializer=tf.zeros_initializer)

      elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
        self.output_state = None
        self.input_state = None
        # For streaming inference with external states,
        # the states are passed in as input.
        if self.overlap > 0:
          self.input_state = tf.keras.layers.Input(
              shape=self.state_shape[1:],
              batch_size=self.inference_batch_size,
              name=self.name + '/input_state_remainder')

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
    config = super(Conv1DTransposeOn2D, self).get_config()
    # only variables which are listed in constructor can be updated here
    # because they will be used to construct the class from config
    config.update({
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'pad_time_dim': self.pad_time_dim,
        'state_shape': self.state_shape,
        'crop_output': self.crop_output,
    })
    return config

  def _streaming_internal_state(self, inputs):
    outputs = super(Conv1DTransposeOn2D, self).call(inputs)

    if self.overlap == 0:
      if self.crop_output:
        outputs = outputs[:, 0:self.output_time_dim]
      return outputs

    output_shape = outputs.shape.as_list()

    # need to add remainder state to a specific region of output as below:
    # outputs[:,0:self.overlap,:] = outputs[:,0:self.overlap,:] + self.states
    # but 'Tensor' object does not support item assignment,
    # so doing it through full summation below
    output_shape[1] -= self.state_shape[1]
    padded_remainder = tf.concat(
        [self.states, tf.zeros(output_shape, tf.float32)], 1)
    outputs = outputs + padded_remainder

    # extract remainder state and substruct bias if it is used:
    # bias will be added in the next iteration again and remainder
    # should have only convolution part, so that bias is not added twice
    if self.use_bias:
      new_state = outputs[:, -self.overlap:, :] - self.bias
    else:
      new_state = outputs[:, -self.overlap:, :]
    assign_states = self.states.assign(new_state)

    with tf.control_dependencies([assign_states]):
      if self.crop_output:
        return tf.identity(outputs[:, 0:self.output_time_dim, :])
      else:
        return tf.identity(outputs)

  def _streaming_external_state(self, inputs, states):
    outputs = super(Conv1DTransposeOn2D, self).call(inputs)

    if self.overlap == 0:
      if self.crop_output:
        outputs = outputs[:, 0:self.output_time_dim, :]
      return outputs, []

    output_shape = outputs.shape.as_list()

    output_shape[1] -= self.state_shape[1]
    padded_remainder = tf.concat(
        [states, tf.zeros(output_shape, tf.float32)], 1)
    outputs = outputs + padded_remainder

    if self.use_bias:
      new_state = outputs[:, -self.overlap:, :] - self.bias
    else:
      new_state = outputs[:, -self.overlap:, :]

    if self.crop_output:
      outputs = outputs[:, 0:self.output_time_dim, :]
    return outputs, new_state

  def _non_streaming(self, inputs):
    outputs = super(Conv1DTransposeOn2D, self).call(inputs)
    # during training or non streaming inference, input shape can be dynamic
    output_time_dim = tf.shape(inputs)[1] * self.strides[0]
    if self.crop_output:
      if self.pad_time_dim == 'same':
        crop_left = self.overlap // 2
        return outputs[:, crop_left:crop_left + output_time_dim, :]
      else:
        return outputs[:, 0:output_time_dim, :]
    else:
      return outputs

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
