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

"""GRU layer."""
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes


class GRU(tf.keras.layers.Layer):
  """GRU with support of streaming inference with internal/external state.

  In training mode we use GRU.
  It receives input data [batch, time, feature] and
  returns [batch, time, units] if return_sequences==True or
  returns [batch, 1, units] if return_sequences==False

  In inference mode we use GRUCell
  In streaming mode with internal state
  it receives input data [batch, 1, feature]
  In streaming mode with internal state it returns: [batch, 1, units]

  In streaming mode with external state it receives input data with states:
    [batch, 1, feature] + state1[batch, units]
  In streaming mode with external state it returns:
    (output[batch, 1, units], state1[batch, units]
  We use layer and parameter description from:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell

  Attributes:
    units: dimensionality of the output space.
    mode: Training or inference modes: non streaming, streaming.
    inference_batch_size: batch size for inference mode
    return_sequences: Whether to return the last output in the output sequence,
      or the full sequence.
    unroll:  If True, the network will be unrolled, else a symbolic loop will be
      used. For any inference mode it will be set True inside.
    stateful: If True, the last state for each sample at index i in a batch will
      be used as initial state for the sample of index i in the following batch.
      If model will be in streaming mode then it is better to train model with
      stateful=True. This flag is about stateful training and applied during
      training only.
  """

  def __init__(self,
               units=64,
               mode=Modes.TRAINING,
               inference_batch_size=1,
               return_sequences=False,
               unroll=False,
               stateful=False,
               name='GRU',
               **kwargs):
    super(GRU, self).__init__(**kwargs)

    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.units = units
    self.return_sequences = return_sequences
    self.stateful = stateful

    if mode != Modes.TRAINING:  # in any inference mode
      # let's unroll gru, so there is no symbolic loops / control flow
      unroll = True

    self.unroll = unroll

    if self.mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
      self.gru = tf.keras.layers.GRU(
          units=units,
          return_sequences=return_sequences,
          name='cell',
          stateful=self.stateful,
          unroll=self.unroll)
    if self.mode == Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # create state varaible for stateful streamable inference
      self.input_state = self.add_weight(
          name='input_state',
          shape=[inference_batch_size, units],
          trainable=False,
          initializer=tf.zeros_initializer)
      self.gru_cell = tf.keras.layers.GRUCell(units=units, name='cell')
      self.gru = None
    elif self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in stateless mode state becomes an input output placeholders
      self.input_state = tf.keras.layers.Input(
          shape=(units,),
          batch_size=inference_batch_size,
          name=self.name + 'input_state')
      self.gru_cell = tf.keras.layers.GRUCell(units=units, name='cell')
      self.gru = None
      self.output_state = None

  def call(self, inputs):
    if inputs.shape.rank != 3:  # [batch, time, feature]
      raise ValueError('inputs.shape.rank:%d must be 3' % inputs.shape.rank)

    if self.mode == Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # run streamable inference on input [batch, 1, features]
      # returns output [batch, 1, units]
      return self._streaming_internal_state(inputs)
    elif self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in stateless mode in addition to output we return output state
      output, self.output_state = self._streaming_external_state(
          inputs, self.input_state)
      return output
    elif self.mode in (Modes.TRAINING, Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      # on input [batch, time, features], returns [batch, time, units]
      return self._non_streaming(inputs)
    else:
      raise ValueError('wrong mode', self.mode)

  def get_config(self):
    config = {
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'units': self.units,
        'return_sequences': self.return_sequences,
        'unroll': self.unroll,
        'stateful': self.stateful,
    }
    base_config = super(GRU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    # input state is used only for STREAM_INTERNAL_STATE_INFERENCE mode
    if self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.input_state]
    else:
      raise ValueError('wrong mode', self.mode)

  def get_output_state(self):
    # output state is used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.output_state]
    else:
      raise ValueError('wrong mode', self.mode)

  def _streaming_internal_state(self, inputs):
    # first dimension is batch size
    if inputs.shape[0] != self.inference_batch_size:
      raise ValueError(
          'inputs.shape[0]:%d must be = self.inference_batch_size:%d' %
          (inputs.shape[0], self.inference_batch_size))

    # receive inputs: [batch, 1, feature]
    # convert it for gru cell to inputs1: [batch, feature]
    inputs = tf.keras.backend.squeeze(inputs, axis=1)
    output, states = self.gru_cell(inputs, [self.input_state])
    # update internal states
    assign_state = self.input_state.assign(states[0])

    with tf.control_dependencies([assign_state]):
      # output [batch, 1, feature]
      output = tf.keras.backend.expand_dims(output, axis=1)
      return output

  def _streaming_external_state(self, inputs, state):
    # first dimension is batch size
    if inputs.shape[0] != self.inference_batch_size:
      raise ValueError(
          'inputs.shape[0]:%d must be = self.inference_batch_size:%d' %
          (inputs.shape[0], self.inference_batch_size))

    # receive inputs: [batch, 1, feature]
    # convert it for gru cell to inputs1: [batch, feature]
    inputs = tf.keras.backend.squeeze(inputs, axis=1)
    output, states = self.gru_cell(inputs, [state])

    # output [batch, 1, feature]
    output = tf.keras.backend.expand_dims(output, axis=1)
    return output, states[0]

  def _non_streaming(self, inputs):
    # inputs [batch, time, feature]
    output = self.gru(inputs)  # [batch, time, units]

    if not self.return_sequences:
      # if we do not return sequence the output will be [batch, units]
      # for consistency make it [batch, 1, units]
      output = tf.keras.backend.expand_dims(output, axis=1)
    return output
