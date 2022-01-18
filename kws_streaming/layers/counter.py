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

"""Counter layer for zeroing its input tensor depending on counter state."""

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf


class Counter(tf.keras.layers.Layer):
  """Counter layer.

     In training or non streaming inference it returns input as it is.
     But during streaming inference it will count number of calls and if
     its number <= max_counter then it will return zeros with the shape
     of input tensor. If number of calls is > max_counter then it will
     return input data. This layer can be convenient for models in streaming
     mode where they just started execution and returned data do not have any
     value because streaming buffers are not fully initialized, so model will
     return zeros.
  """

  def __init__(self,
               max_counter=0,
               mode=modes.Modes.TRAINING,
               state_name_tag='counter',
               **kwargs):
    super(Counter, self).__init__(**kwargs)
    self.max_counter = max_counter
    self.mode = mode
    self.state_name_tag = state_name_tag

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # create state variable for streamable inference mode only
      self.state = self.add_weight(
          shape=[1, 1, 1],
          name='counter',
          trainable=False,
          initializer=tf.zeros_initializer)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with external state,
      # state becomes an input output placeholder
      self.input_state = tf.keras.layers.Input(
          shape=(1, 1),
          batch_size=1,
          name=self.name + '/' +
          self.state_name_tag)  # adding names to make it unique
      self.output_state = None

  def call(self, inputs):
    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      return self._streaming_internal_state(inputs)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with extrnal state in addition to output
      # we return output state
      output, self.output_state = self._streaming_external_state(
          inputs, self.input_state)
      return output
    elif self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      return inputs
    else:
      raise ValueError(f'Encountered unexpected mode `{self.mode}`.')

  def get_config(self):
    config = {
        'max_counter': self.max_counter,
        'mode': self.mode,
        'state_name_tag': self.state_name_tag,
    }
    base_config = super(Counter, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

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
    new_state = self.state + 1.0
    new_state = tf.math.minimum(new_state, self.max_counter + 1)

    assign_state = self.state.assign(new_state)

    with tf.control_dependencies([assign_state]):
      multiplier = tf.keras.activations.relu(
          new_state[0][0][0], max_value=1.0, threshold=self.max_counter)
      outputs = tf.multiply(inputs, multiplier)
      return outputs

  def _streaming_external_state(self, inputs, state):
    state_one = state + 1.0

    # overflow protection
    new_state = tf.math.minimum(state_one, self.max_counter + 1)

    # create multiplier which will return zeros for all input values < threshold
    # otherwise it will return one
    multiplier = tf.keras.activations.relu(
        new_state[0][0][0], max_value=1.0, threshold=self.max_counter)
    outputs = tf.multiply(inputs, multiplier)
    return outputs, new_state
