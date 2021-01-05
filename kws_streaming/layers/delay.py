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

"""Dealy layer."""

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf


class Delay(tf.keras.layers.Layer):
  """Delay layer.

  It is useful for introducing delay in streaming mode for non causal filters.
  For example in residual connections with multiple conv layers

  Attributes:
    mode: Training or inference modes: non streaming, streaming.
    delay: delay value
    inference_batch_size: batch size in inference mode
    also_in_non_streaming: Apply delay also in training and non-streaming
      inference mode.
    **kwargs: additional layer arguments
  """

  def __init__(self,
               mode=modes.Modes.TRAINING,
               delay=0,
               inference_batch_size=1,
               also_in_non_streaming=False,
               **kwargs):
    super(Delay, self).__init__(**kwargs)
    self.mode = mode
    self.delay = delay
    self.inference_batch_size = inference_batch_size
    self.also_in_non_streaming = also_in_non_streaming

    if delay < 0:
      raise ValueError('delay (%d) must be non-negative' % delay)

  def build(self, input_shape):
    super(Delay, self).build(input_shape)

    if self.delay > 0:
      self.state_shape = [self.inference_batch_size, self.delay
                         ] + input_shape.as_list()[2:]
      if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
        self.states = self.add_weight(
            name='states',
            shape=self.state_shape,
            trainable=False,
            initializer=tf.zeros_initializer)

      elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
        # For streaming inference with extrnal states,
        # the states are passed in as input.
        self.input_state = tf.keras.layers.Input(
            shape=self.state_shape[1:],
            batch_size=self.inference_batch_size,
            name=self.name + '/input_state_delay')
        self.output_state = None

  def call(self, inputs):
    if self.delay == 0:
      return inputs

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
    config = super(Delay, self).get_config()
    config.update({
        'mode': self.mode,
        'delay': self.delay,
        'inference_batch_size': self.inference_batch_size,
        'also_in_non_streaming': self.also_in_non_streaming,
    })
    return config

  def _streaming_internal_state(self, inputs):
    memory = tf.keras.backend.concatenate([self.states, inputs], 1)
    outputs = memory[:, 0:inputs.shape.as_list()[1], :]
    new_memory = memory[:, -self.delay:, :]
    assign_states = self.states.assign(new_memory)

    with tf.control_dependencies([assign_states]):
      return tf.identity(outputs)

  def _streaming_external_state(self, inputs, states):
    memory = tf.keras.backend.concatenate([states, inputs], 1)
    outputs = memory[:, 0:inputs.shape.as_list()[1], :]
    new_memory = memory[:, -self.delay:, :]
    return outputs, new_memory

  def _non_streaming(self, inputs):
    if self.also_in_non_streaming:
      return tf.pad(inputs,
                    ((0, 0), (self.delay, 0), (0, 0)))[:, :-self.delay, :]
    else:
      return inputs

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
