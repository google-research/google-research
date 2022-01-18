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

"""Sequence-to-sequence RNN model with attention mechanism.

Based on:
  https://www.tensorflow.org/tutorials/text/nmt_with_attention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np

import tensorflow.compat.v1 as tf  # tf

import homophonous_logography.neural.corpus as data
import homophonous_logography.neural.utils as utils


def _check_rnn_cell_type(rnn_cell_type, component_type):
  # Our decoder is always unidirectional.
  cell_type = rnn_cell_type
  if component_type == "decoder":
    if cell_type == "BiLSTM":
      cell_type = "LSTM"
    elif cell_type == "BiGRU":
      cell_type = "GRU"
  return cell_type


def _get_rnn(rnn_cell_type, num_units, component_type):
  """Manufactures network layer given the configuration."""
  cell_type = _check_rnn_cell_type(rnn_cell_type, component_type)
  if cell_type == "GRU":
    return tf.keras.layers.GRU(num_units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer="glorot_uniform")
  elif cell_type == "BiGRU":
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(num_units,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer="glorot_uniform"))
  elif cell_type == "LSTM":
    return tf.keras.layers.LSTM(num_units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer="glorot_uniform")
  elif cell_type == "BiLSTM":
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(num_units,
                             return_sequences=True,
                             return_state=True,
                             recurrent_initializer="glorot_uniform"))
  else:
    raise ValueError("Invalid RNN cell type: {}".format(cell_type))


class Encoder(tf.keras.Model):
  """Encoder component."""

  def __init__(self, vocab_size, embedding_dim, enc_units, rnn_cell_type):
    super(Encoder, self).__init__()
    self._enc_units = enc_units
    self._rnn_cell_type = rnn_cell_type
    self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self._rnn = _get_rnn(rnn_cell_type, self._enc_units, "encoder")

  def call(self, x, hidden):
    x = self._embedding(x)
    if self._rnn_cell_type == "BiLSTM":
      output, forward_h, forward_c, back_h, back_c = self._rnn(
          x, initial_state=hidden)
      state = [forward_h + back_h, forward_c + back_c]
    elif self._rnn_cell_type == "LSTM":
      output, state_h, state_c = self._rnn(x, initial_state=hidden)
      state = [state_h, state_c]
    elif self._rnn_cell_type == "BiGRU":
      output, forward_state, backward_state = self._rnn(
          x, initial_state=hidden)
      state = forward_state + backward_state
    else:  # GRU
      output, state = self._rnn(x, initial_state=hidden)
    return output, state

  def init_hidden_state(self, batch_size):
    rnn_cell_type = self._rnn_cell_type
    num_units = self._enc_units
    if rnn_cell_type == "BiLSTM":
      # [forward memory state, forward carry state,
      #  backward memory state, backward carry state].
      return [tf.zeros((batch_size, num_units)) for _ in range(4)]
    elif rnn_cell_type == "LSTM" or rnn_cell_type == "BiGRU":
      # [memory state, carry state].
      return [tf.zeros((batch_size, num_units)) for _ in range(2)]
    else:
      return tf.zeros((batch_size, num_units))


class BahdanauAttention(tf.keras.layers.Layer):
  """Attention by Bahdanau et. al."""

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self._w1 = tf.keras.layers.Dense(units)
    self._w2 = tf.keras.layers.Dense(units)
    self._v = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # > query hidden shape == (batch_size, hidden size)
    # > query_with_time_axis shape == (batch_size, 1, hidden size)
    # > values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate
    # the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1) we get 1 at the last axis
    # because we are applying score to self.V the shape of the tensor before
    # applying self.V is (batch_size, max_length, units)
    score = self._v(tf.nn.tanh(
        self._w1(values) + self._w2(query_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class LuongAttention(tf.keras.layers.Layer):
  """Attention by Luong, et. al."""

  def __init__(self, hidden_dim):
    super(LuongAttention, self).__init__()
    self._w = tf.keras.layers.Dense(hidden_dim)

  def call(self, query, values):
    # > query hidden shape == (batch_size, 1, hidden_size)
    # > values shape == (batch_size, max_len, hidden_size)

    # score shape: (batch_size, 1, max_length)
    scores = tf.matmul(query, self._w(values), transpose_b=True)

    # attention_weights shape == (batch_size, 1, max_length)
    attention_weights = tf.nn.softmax(scores, axis=2)

    # Final context vector shape: (batch_size, 1, hidden_size)
    context_vector = tf.matmul(attention_weights, values)
    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  """Decoder component."""

  def __init__(self, vocab_size, embedding_dim, dec_units, attention_type,
               rnn_cell_type):
    super(Decoder, self).__init__()
    self._dec_units = dec_units
    self._attention_type = attention_type
    self._rnn_cell_type = _check_rnn_cell_type(rnn_cell_type, "decoder")
    self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self._rnn = _get_rnn(rnn_cell_type, self._dec_units, "decoder")
    self._attention = self._get_attention(attention_type)
    self._fc = tf.keras.layers.Dense(vocab_size)
    self._luong_concat = tf.keras.layers.Dense(self._dec_units,
                                               activation="tanh")

  def call(self, x, hidden, last_context, enc_output):
    if self._attention_type == "LUONG":
      return self._luong_lstm_arch(x, hidden, last_context, enc_output)
    else:
      return self._bahdanau_arch(x, hidden, enc_output)

  def _bahdanau_arch(self, x, hidden_state, encoder_outputs):
    # Note: In Bahdanau's model of attention we ignore the last context vector.

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self._embedding(x)

    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vec, attention_weights = self._apply_attention(
        hidden_state, encoder_outputs)

    # x shape after concatenation == (batch_size, 1,
    # embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)

    # passing the concatenated vector to the RNN.
    output, state = self._apply_rnn(x, hidden_state)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self._fc(output)

    return x, state, None, attention_weights

  def _luong_lstm_arch(self, x, hidden_state, last_context, encoder_outputs):
    """RNN with Luong attention mechanism."""
    x = self._embedding(x)

    # Step RNN. Here `last_context` is the attention vector h~_{t-1} from the
    # previous timestep. Feed it back to the current input (see `Input Feeding`
    # section 3.3 of Luong's paper).
    x = tf.concat([x, tf.expand_dims(last_context, 1)], axis=-1)
    output, state = self._apply_rnn(x, hidden_state)

    # Apply attention to the current RNN state.
    context, attention_weights = self._attention(output, encoder_outputs)

    # Get attention hidden state h~_t = tanh(W_c[c_t;h_t]) (see Equation (5)
    # in Luong's paper). Get rid of the time dimension before concatenating.
    x = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], axis=-1)
    tilde_h = self._luong_concat(x)
    logits = self._fc(tilde_h)
    return logits, state, tilde_h, attention_weights

  def _get_attention(self, attention_type):
    if attention_type == "BAHDANAU":
      return BahdanauAttention(self._dec_units)
    elif attention_type == "LUONG":
      return LuongAttention(self._dec_units)
    else:
      raise ValueError("Unknown attention type: {}".format(attention_type))

  def _apply_attention(self, hidden_state, encoder_outputs):
    # enc_outputs shape == (batch_size, max_length, hidden_size)
    if self._rnn_cell_type == "GRU":
      query = hidden_state
    elif self._rnn_cell_type == "LSTM":
      query = hidden_state[0]  # forward_h
    else:
      raise ValueError("Decoder should be unidirectional")
    return self._attention(query, encoder_outputs)

  def _apply_rnn(self, x, hidden_state):
    if self._rnn_cell_type == "LSTM":
      output, state_h, state_c = self._rnn(x, initial_state=hidden_state)
      state = [state_h, state_c]
    elif self._rnn_cell_type == "GRU":
      output, state = self._rnn(x)
    else:
      raise ValueError("Decoder should be unidirectional")
    return output, state

  def init_context(self, batch_size):
    return tf.zeros([batch_size, self._dec_units], dtype=tf.float32)


class Seq2SeqRnnModel(object):
  """Full RNN model."""

  def __init__(self,
               batch_size=64,
               embedding_dim=64,
               enc_units=256,
               dec_units=256,
               attention_type="BAHDANAU",
               rnn_cell_type="GRU",
               input_symbols=None,
               output_symbols=None,
               model_dir=".",
               name="model"):
    self._optimizer = tf.keras.optimizers.Adam()
    self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")
    self._batch_size = batch_size
    self._embedding_dim = embedding_dim
    self._enc_units = enc_units
    self._dec_units = dec_units
    self._attention_type = attention_type
    self._rnn_cell_type = rnn_cell_type
    self._input_symbols = input_symbols
    self._input_vocab_size = len(input_symbols)
    self._output_symbols = output_symbols
    self._output_vocab_size = len(output_symbols)
    self._encoder = Encoder(self._input_vocab_size,
                            self._embedding_dim,
                            self._enc_units,
                            self._rnn_cell_type)
    self._decoder = Decoder(self._output_vocab_size,
                            self._embedding_dim,
                            self._dec_units,
                            self._attention_type,
                            self._rnn_cell_type)
    self._name = name
    self._checkpoint_dir = os.path.join(model_dir, self._name)
    self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
    self._checkpoint = tf.train.Checkpoint(optimizer=self._optimizer,
                                           encoder=self._encoder,
                                           decoder=self._decoder)
    # Lengths of padded inputs and outputs:
    self._input_length = -1
    self._output_length = -1

  def _loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = self._loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)

  @tf.function
  def _train_step(self, inp, targ, encoder_hidden_state):
    """Single training step."""
    loss = 0
    with tf.GradientTape() as tape:
      enc_output, enc_hidden = self._encoder(inp, encoder_hidden_state)
      dec_hidden = enc_hidden
      dec_context = self._decoder.init_context(self._batch_size)
      dec_input = tf.expand_dims(
          [self._output_symbols.find("<s>")] * self._batch_size, 1)
      for t in range(1, targ.shape[1]):
        predictions, dec_hidden, dec_context, _ = self._decoder(
            dec_input, dec_hidden, dec_context, enc_output)
        loss += self._loss_function(targ[:, t], predictions)
        dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = (self._encoder.trainable_variables +
                 self._decoder.trainable_variables)
    gradients = tape.gradient(loss, variables)
    self._optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

  def train(self, corpus, epochs=10, direction="pronounce", window=-1):
    """Full training."""
    # Create training log that also redirects to stdout.
    stdout_file = sys.stdout
    logfile = os.path.join(self._checkpoint_dir, "train.log")
    print("Training log: {}".format(logfile))
    sys.stdout = utils.DualLogger(logfile)

    # Dump some parameters.
    print("           Direction: {}".format(direction))
    print("            # Epochs: {}".format(epochs))
    print("          Batch size: {}".format(self._batch_size))
    print("         Window size: {}".format(window))
    print("      Attention type: {}".format(self._attention_type))
    print("       RNN cell type: {}".format(self._rnn_cell_type))
    print("     Max written len: {}".format(corpus.max_written_len))
    print("        Max pron len: {}".format(corpus.max_pronounce_len))
    print("Max written word len: {}".format(corpus.max_written_word_len))
    print("   Max pron word len: {}".format(corpus.max_pronounce_word_len))

    # Perform training.
    best_total_loss = 1000000
    nbatches = data.num_batches(corpus, self._batch_size, direction=direction,
                                window=window)
    for epoch in range(epochs):
      start = time.time()
      total_loss = 0
      steps = 0
      batches = data.batchify(corpus, self._batch_size, direction,
                              window=window)
      batch, (inp, targ) = next(batches)
      enc_hidden = self._encoder.init_hidden_state(self._batch_size)
      if self._input_length == -1:
        # TODO(agutkin,rws): Following two lines will break if batchify()
        # returns an empty tuple.
        if isinstance(inp, np.ndarray):
          self._input_length = inp.shape[1]
        if isinstance(targ, np.ndarray):
          self._output_length = targ.shape[1]
      while batch > -1:
        batch_loss = self._train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        if batch % 10 == 0:
          print("Epoch {} Batch {} (/{}) Loss {:.4f}".format(
              epoch + 1,
              batch,
              nbatches,
              batch_loss.numpy()))
        steps += 1
        batch, (inp, targ) = next(batches)
      total_loss /= steps
      print("Epoch {} Loss {:.4f}".format(epoch + 1,
                                          total_loss))
      if total_loss < best_total_loss:
        self._checkpoint.save(file_prefix=self._checkpoint_prefix)
        print("Saved checkpoint to {}".format(self._checkpoint_prefix))
        best_total_loss = total_loss
      print("Time taken for 1 epoch {} sec\n".format(
          time.time() - start))
    print("Best total loss: {:.4f}".format(best_total_loss))

    # Restore stdout.
    sys.stdout = stdout_file

  def decode(self, inputs, joiner=""):
    """Decodes inputs."""
    def expand_dims(inp):
      return tf.expand_dims(tf.convert_to_tensor(inp), 0)

    inputs = expand_dims(inputs)
    enc_hidden = self._encoder.init_hidden_state(1)
    enc_out, enc_hidden = self._encoder(inputs, enc_hidden)
    dec_hidden = enc_hidden
    dec_context = self._decoder.init_context(1)
    dec_input = tf.expand_dims([self._output_symbols.find("<s>")], 0)
    result = []
    attention_plot = np.zeros([self._output_length, self._input_length])
    for t in range(self._output_length):
      predictions, dec_hidden, dec_context, attention_weights = self._decoder(
          dec_input,
          dec_hidden,
          dec_context,
          enc_out,
          training=False)
      attention_weights = tf.reshape(attention_weights, [-1])
      attention_plot[t] = attention_weights.numpy()
      predicted_id = int(tf.argmax(predictions[0]).numpy())
      outsym = self._output_symbols.find(predicted_id)
      if outsym == "</s>" or outsym == "</targ>":
        return joiner.join(result), attention_plot
      else:
        result.append(outsym)
      dec_input = tf.expand_dims([predicted_id], 0)
    return joiner.join(result), attention_plot

  def update_property(self, property_name, value):
    setattr(self, property_name, value)

  @property
  def checkpoint(self):
    return self._checkpoint

  @property
  def checkpoint_dir(self):
    return self._checkpoint_dir

  @property
  def input_symbols(self):
    return self._input_symbols

  @property
  def output_symbols(self):
    return self._output_symbols

  @property
  def input_length(self):
    return self._input_length

  @property
  def eval_mode(self):
    return ""
