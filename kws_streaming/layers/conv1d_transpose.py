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

"""Conv1DTranspose streaming aware layer."""
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes


class Conv1DTranspose(tf.keras.layers.Conv1DTranspose):
  """streaming aware Conv1DTranspose layer.

  Attributes:
    mode: Training or inference modes: non streaming, streaming.
    inference_batch_size: batch size in inference mode
    state_shape: shape of remainder state
    crop_output: if True output will be cropped: aligned by stride
    **kwargs: additional layer arguments
  """

  def __init__(self,
               mode=Modes.TRAINING,
               inference_batch_size=1,
               state_shape=None,
               crop_output=True,
               **kwargs):
    super(Conv1DTranspose, self).__init__(**kwargs)

    if (kwargs.get('activation') not in [None, 'linear']) and self.use_bias:
      raise ValueError('activation should be disabled because we need to '
                       'subtract bias from remainder state, in streaming mode',
                       kwargs.get('activation'))

    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.state_shape = state_shape
    self.crop_output = crop_output

    self.overlap = self.kernel_size[0] - self.strides[0]
    self.overlap = max(self.overlap, 0)

  def build(self, input_shape):
    super(Conv1DTranspose, self).build(input_shape)

    if input_shape.rank < 2:
      raise ValueError('input_shape.rank:%d must at least 2' % input_shape.rank)

    if self.mode in [
        Modes.STREAM_INTERNAL_STATE_INFERENCE,
        Modes.STREAM_EXTERNAL_STATE_INFERENCE
    ]:
      if input_shape.as_list()[1] is None:
        raise ValueError('in streaming mode time dimension of input packet '
                         'should not be dynamic: TFLite limitation')

      self.output_time_dim = input_shape.as_list()[1] * self.strides[0]

      self.input_state = []
      self.output_state = []
      if self.overlap > 0:
        self.state_shape = [
            self.inference_batch_size, self.overlap, self.filters
        ]

        if self.mode == Modes.STREAM_INTERNAL_STATE_INFERENCE:
          self.states = self.add_weight(
              name='states',
              shape=self.state_shape,
              trainable=False,
              initializer=tf.zeros_initializer)

        elif self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
          # For streaming inference with extrnal states,
          # the states are passed in as input.
          self.input_state = tf.keras.layers.Input(
              shape=self.state_shape[1:],
              batch_size=self.inference_batch_size,
              name=self.name + '/input_state_remainder')

  def call(self, inputs):

    if self.mode == Modes.STREAM_INTERNAL_STATE_INFERENCE:
      return self._streaming_internal_state(inputs)

    elif self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming inference mode with external state
      # in addition to the output we return the output state.
      output, self.output_state = self._streaming_external_state(
          inputs, self.input_state)
      return output

    elif self.mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      return self._non_streaming(inputs)

    else:
      raise ValueError('wrong mode', self.mode)

  def get_config(self):
    config = super(Conv1DTranspose, self).get_config()
    # only variables which are listed in constructor can be updated here
    # because they will be used to construct the class from config
    config.update({
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'state_shape': self.state_shape,
        'crop_output': self.crop_output,
    })
    return config

  def _streaming_internal_state(self, inputs):
    outputs = super(Conv1DTranspose, self).call(inputs)

    if self.overlap == 0:
      if self.crop_output:
        return tf.identity(outputs[:, 0:self.output_time_dim, :])
      else:
        return tf.identity(outputs)

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
    outputs = super(Conv1DTranspose, self).call(inputs)

    if self.overlap == 0:
      if self.crop_output:
        return outputs[:, 0:self.output_time_dim, :], []
      else:
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
      return outputs[:, 0:self.output_time_dim, :], new_state
    else:
      return outputs, new_state

  def _non_streaming(self, inputs):
    outputs = super(Conv1DTranspose, self).call(inputs)
    # during training or non streaming inference, input shape can be dynamic
    output_time_dim = tf.shape(inputs)[1] * self.strides[0]
    if self.crop_output:
      return outputs[:, 0:output_time_dim, :]
    else:
      return outputs

  def get_input_state(self):
    # input state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return self.input_state
    else:
      raise ValueError('wrong mode', self.mode)

  def get_output_state(self):
    # output state will be used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return self.output_state
    else:
      raise ValueError('wrong mode', self.mode)
