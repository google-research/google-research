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

"""LSTM layer."""

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1


class LSTM(tf.keras.layers.Layer):
  """LSTM with support of streaming inference with internal/external state.

  In training mode we use LSTM.
  It receives input data [batch, time, feature] and
  returns [batch, time, units] if return_sequences==True or
  returns [batch, 1, units] if return_sequences==False

  In inference mode we use LSTMCell
  In streaming mode with internal state
  it receives input data [batch, 1, feature]
  In streaming mode with internal state it returns: [batch, 1, units]

  In streaming mode with external state it receives input data with states:
    [batch, 1, feature] + state1[batch, units] + state2[batch, units]
  In streaming mode with external state it returns:
    (output[batch, 1, units], state1[batch, units], state2[batch, units])
  We use layer and parameter description from:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
  https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

  Attributes:
    units: dimensionality of the output space.
    mode: Training or inference modes: non streaming, streaming.
    inference_batch_size: batch size for inference mode
    return_sequences: Whether to return the last output. in the output sequence,
      or the full sequence.
    use_peepholes: True to enable diagonal/peephole connections
    num_proj: The output dimensionality for the projection matrices. If None, no
      projection is performed. It will be used only if use_peepholes is True.
    unroll:  If True, the network will be unrolled, else a symbolic loop will be
      used. For any inference mode it will be set True inside.
    stateful: If True, the last state for each sample at index i in a batch will
      be used as initial state for the sample of index i in the following batch.
      If model will be in streaming mode then it is better to train model with
      stateful=True This flag is about stateful training and applied during
      training only.
  """

  def __init__(self,
               units=64,
               mode=modes.Modes.TRAINING,
               inference_batch_size=1,
               return_sequences=False,
               use_peepholes=False,
               num_proj=128,
               unroll=False,
               stateful=False,
               name='LSTM',
               **kwargs):
    super(LSTM, self).__init__(**kwargs)

    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.units = units
    self.return_sequences = return_sequences
    self.num_proj = num_proj
    self.use_peepholes = use_peepholes
    self.stateful = stateful

    if mode != modes.Modes.TRAINING:  # in any inference mode
      # let's unroll lstm, so there is no symbolic loops / control flow
      unroll = True

    self.unroll = unroll

    if self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      if use_peepholes:
        self.lstm_cell = tf1.nn.rnn_cell.LSTMCell(
            num_units=units, use_peepholes=True, num_proj=num_proj, name='cell')
        self.lstm = tf.keras.layers.RNN(
            cell=self.lstm_cell,
            return_sequences=return_sequences,
            unroll=self.unroll,
            stateful=self.stateful)
      else:
        self.lstm = tf.keras.layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            name='cell',
            unroll=self.unroll,
            stateful=self.stateful)
    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # create state varaible for stateful streamable inference
      self.input_state1 = self.add_weight(
          name='input_state1',
          shape=[inference_batch_size, units],
          trainable=False,
          initializer=tf.zeros_initializer)
      if use_peepholes:
        # second state in peepholes LSTM has different dimensions with
        # the first state due to projection layer with dim num_proj
        self.input_state2 = self.add_weight(
            name='input_state2',
            shape=[inference_batch_size, num_proj],
            trainable=False,
            initializer=tf.zeros_initializer)
        self.lstm_cell = tf1.nn.rnn_cell.LSTMCell(
            num_units=units, use_peepholes=True, num_proj=num_proj, name='cell')
      else:
        # second state in the standard LSTM has the same dimensions with
        # the first state
        self.input_state2 = self.add_weight(
            name='input_state2',
            shape=[inference_batch_size, units],
            trainable=False,
            initializer=tf.zeros_initializer)
        self.lstm_cell = tf.keras.layers.LSTMCell(units=units, name='cell')
      self.lstm = None
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with external state state,
      # state becomes an input output placeholders
      self.input_state1 = tf.keras.layers.Input(
          shape=(units,),
          batch_size=inference_batch_size,
          name=self.name + 'input_state1')
      if use_peepholes:
        self.input_state2 = tf.keras.layers.Input(
            shape=(num_proj,),
            batch_size=inference_batch_size,
            name=self.name + 'input_state2')
        self.lstm_cell = tf1.nn.rnn_cell.LSTMCell(
            num_units=units, use_peepholes=True, num_proj=num_proj)
      else:
        self.input_state2 = tf.keras.layers.Input(
            shape=(units,),
            batch_size=inference_batch_size,
            name=self.name + 'input_state2')
        self.lstm_cell = tf.keras.layers.LSTMCell(units=units, name='cell')
      self.lstm = None
      self.output_state1 = None
      self.output_state2 = None

  def call(self, inputs):
    if inputs.shape.rank != 3:  # [batch, time, feature]
      raise ValueError('inputs.shape.rank:%d must be 3' % inputs.shape.rank)

    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE:
      # run streamable inference on input [batch, 1, features]
      # returns output [batch, 1, units]
      return self._streaming_internal_state(inputs)
    elif self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      # in streaming mode with external state
      # in addition to output we return output state
      output, self.output_state1, self.output_state2 = self._streaming_external_state(
          inputs, self.input_state1, self.input_state2)
      return output
    elif self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      # on input [batch, time, features], returns [batch, time, units]
      return self._non_streaming(inputs)
    else:
      raise ValueError(f'Encountered unexpected mode `{self.mode}`.')

  def get_config(self):
    config = {
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'units': self.units,
        'return_sequences': self.return_sequences,
        'unroll': self.unroll,
        'num_proj': self.num_proj,
        'use_peepholes': self.use_peepholes,
        'stateful': self.stateful,
    }
    base_config = super(LSTM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    # input state is used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.input_state1, self.input_state2]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def get_output_state(self):
    # output state is used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.output_state1, self.output_state2]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def _streaming_internal_state(self, inputs):
    # first dimension is batch size
    if inputs.shape[0] != self.inference_batch_size:
      raise ValueError(
          'inputs.shape[0]:%d must be = self.inference_batch_size:%d' %
          (inputs.shape[0], self.inference_batch_size))

    # receive inputs: [batch, 1, feature]
    # convert it for lstm cell to inputs1: [batch, feature]
    inputs1 = tf.keras.backend.squeeze(inputs, axis=1)
    output, states = self.lstm_cell(inputs1,
                                    [self.input_state1, self.input_state2])
    # update internal states
    assign_state1 = self.input_state1.assign(states[0])
    assign_state2 = self.input_state2.assign(states[1])

    with tf.control_dependencies([assign_state1, assign_state2]):
      # output [batch, 1, feature]
      output = tf.keras.backend.expand_dims(output, axis=1)
      return output

  def _streaming_external_state(self, inputs, state1, state2):
    # first dimension is batch size
    if inputs.shape[0] != self.inference_batch_size:
      raise ValueError(
          'inputs.shape[0]:%d must be = self.inference_batch_size:%d' %
          (inputs.shape[0], self.inference_batch_size))

    # receive inputs: [batch, 1, feature]
    # convert it for lstm cell to inputs1: [batch, feature]
    inputs1 = tf.keras.backend.squeeze(inputs, axis=1)
    output, states = self.lstm_cell(inputs1, [state1, state2])

    # output [batch, 1, feature]
    output = tf.keras.backend.expand_dims(output, axis=1)
    return output, states[0], states[1]

  def _non_streaming(self, inputs):
    # inputs [batch, time, feature]
    output = self.lstm(inputs)  # [batch, time, units]

    if not self.return_sequences:
      # if we do not return sequence the output will be [batch, units]
      # for consistency make it [batch, 1, units]
      output = tf.keras.backend.expand_dims(output, axis=1)
    return output
