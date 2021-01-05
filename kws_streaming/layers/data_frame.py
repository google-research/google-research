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

"""A layer which splits input speech signal into frames."""

from kws_streaming.layers import modes
from kws_streaming.layers import temporal_padding
from kws_streaming.layers.compat import tf


class DataFrame(tf.keras.layers.Layer):
  """Frame splitter with support of streaming inference.

  In training mode we use tf.signal.frame.
  It receives input data [batch, time] and
  converts it into [batch, frames, frame_size].
  More details at:
  https://www.tensorflow.org/api_docs/python/tf/signal/frame
  In inference mode we do a streaming version of tf.signal.frame:
  we receive input packet with dims [batch, frame_step].
  Then we use it to update internal state buffer in a sliding window manner.
  Return output data with size [batch, frame_size].

  Attributes:
    mode: inference or training mode
    frame_size: size of sliding window
    frame_step: stride of sliding window
    use_one_step:
      True, it will produce only one frame per one inference call
      False, it will produce multiple frames per one inference call - in this
      case input data should have enough samples to produce multiple frames.
      It can process batch of frames in streaming mode - we need it for
      models with striding or pooling.
    padding: apply padding on input data - only causal mode is supported
    **kwargs: additional layer arguments

  Raises:
    ValueError: if use_one_step and frame_step can not be bigger than frame_size
      if padding is non causal

  """

  def __init__(self,
               mode=modes.Modes.TRAINING,
               inference_batch_size=1,
               frame_size=400,
               frame_step=160,
               use_one_step=True,
               padding=None,
               **kwargs):
    super(DataFrame, self).__init__(**kwargs)

    if use_one_step and frame_step > frame_size:
      raise ValueError('frame_step:%d must be <= frame_size:%d' %
                       (frame_step, frame_size))

    if padding and padding != 'causal':
      raise ValueError('only causal padding is supported')

    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.frame_size = frame_size
    self.frame_step = frame_step
    self.use_one_step = use_one_step
    self.padding = padding

    if self.use_one_step:
      self.ring_buffer_size_in_time_dim = frame_size
    else:
      self.ring_buffer_size_in_time_dim = frame_size - 1

    if self.padding:
      self.padding_layer = temporal_padding.TemporalPadding(
          padding_size=self.ring_buffer_size_in_time_dim, padding=self.padding)
    else:
      self.padding_layer = tf.keras.layers.Lambda(lambda x: x)

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # create state varaible for inference streaming with internal state
      self.states = self.add_weight(
          name='frame_states',
          shape=[self.inference_batch_size, self.ring_buffer_size_in_time_dim],
          trainable=False,
          initializer=tf.zeros_initializer)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with external state,
      # state becomes an input output placeholders
      self.input_state = tf.keras.layers.Input(
          shape=(self.ring_buffer_size_in_time_dim,),
          batch_size=self.inference_batch_size,
          name=self.name + 'input_state')
      self.output_state = None

  def call(self, inputs):
    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # run streamable inference on input [batch, frame_step]
      # returns output [batch, 1, frame_size]
      return self._streaming_internal_state(inputs)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with external state in addition to the output
      # we return output state
      output, self.output_state = self._streaming_external_state(
          inputs, self.input_state)
      return output
    elif self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      # on input [batch, time], returns output [batch, frames, frame_size]
      return self._non_streaming(inputs)
    else:
      raise ValueError(f'Encountered unexpected mode `{self.mode}`.')

  def get_config(self):
    config = {
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'frame_size': self.frame_size,
        'frame_step': self.frame_step,
        'use_one_step': self.use_one_step,
        'padding': self.padding,
    }
    base_config = super(DataFrame, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    # input state is used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.input_state]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def get_output_state(self):
    # output state is used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.output_state]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def _streaming_internal_state(self, inputs):
    # first dimension is batch size
    if inputs.shape[0] != self.inference_batch_size:
      raise ValueError(
          'inputs.shape[0]:%d must be = self.inference_batch_size:%d' %
          (inputs.shape[0], self.inference_batch_size))

    if self.use_one_step:
      # use_one_step is used only for backward compatibility
      # it assumes that frame_size and frame_step overlap
      # below version with tf.signal.frame is more generic

      # second dimension is frame_step
      if inputs.shape[1] != self.frame_step:
        raise ValueError('inputs.shape[1]:%d must be = self.frame_step:%d' %
                         (inputs.shape[1], self.frame_step))

      # remove latest rows [batch_size, (frame_size-frame_step)]
      memory = self.states[:, self.frame_step:self.frame_size]

      # add new rows [batch_size, memory_size]
      memory = tf.keras.backend.concatenate([memory, inputs], 1)

      assign_states = self.states.assign(memory)

      with tf.control_dependencies([assign_states]):
        # add time dim
        output_frame = tf.keras.backend.expand_dims(memory, -2)
        return output_frame
    else:
      memory = tf.keras.backend.concatenate([self.states, inputs], 1)
      state_update = memory[:, -self.ring_buffer_size_in_time_dim:]
      assign_states = self.states.assign(state_update)

      with tf.control_dependencies([assign_states]):
        output_frame = tf.signal.frame(
            memory, frame_length=self.frame_size, frame_step=self.frame_step)
        return output_frame

  def _streaming_external_state(self, inputs, states):
    # first dimension is batch size
    if inputs.shape[0] != self.inference_batch_size:
      raise ValueError(
          'inputs.shape[0]:%d must be = self.inference_batch_size:%d' %
          (inputs.shape[0], self.inference_batch_size))

    if self.use_one_step:
      # second dimension is frame_step
      if inputs.shape[1] != self.frame_step:
        raise ValueError('inputs.shape[1]:%d must be = self.frame_step:%d' %
                         (inputs.shape[1], self.frame_step))

      # remove latest rows [batch_size, (frame_size-frame_step)]
      memory = states[:, self.frame_step:self.frame_size]

      # add new rows [batch_size, frame_size]
      memory = tf.keras.backend.concatenate([memory, inputs], 1)

      # add time dim to have [batch, 1, frame_size]
      output_frame = tf.keras.backend.expand_dims(memory, -2)

      return output_frame, memory
    else:
      memory = tf.keras.backend.concatenate([states, inputs], 1)
      state_update = memory[:, -self.ring_buffer_size_in_time_dim:]  # pylint: disable=invalid-unary-operand-type
      output_frame = tf.signal.frame(
          memory, frame_length=self.frame_size, frame_step=self.frame_step)
      return output_frame, state_update

  def _non_streaming(self, inputs):
    if inputs.shape.rank != 2:  # [Batch, Time]
      raise ValueError('inputs.shape.rank:%d must be 2' % inputs.shape.rank)

    net = inputs
    net = self.padding_layer(net)
    # Extract frames from [Batch, Time] -> [Batch, Frames, frame_size]
    net = tf.signal.frame(
        net, frame_length=self.frame_size, frame_step=self.frame_step)
    return net
