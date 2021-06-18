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

"""Sequence-to-sequence model.

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


class Encoder(tf.keras.Model):
  """Simple encoder."""

  def __init__(self, vocab_size, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self._enc_units = enc_units
    self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self._gru = tf.keras.layers.GRU(self._enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer="glorot_uniform")

  def call(self, x, hidden):
    x = self._embedding(x)
    output, state = self._gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self, batch_size):
    return tf.zeros((batch_size, self._enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  """Additive Bahdanau et al. attention."""

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self._w1 = tf.keras.layers.Dense(units)
    self._w2 = tf.keras.layers.Dense(units)
    self._v = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1) we get 1 at the last axis
    # because we are applying score to self.V the shape of the tensor before
    # applying self.V is (batch_size, max_length, units)
    score = self._v(tf.nn.tanh(
        self._w1(values) + self._w2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  """Simple decoder."""

  def __init__(self, vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    self._dec_units = dec_units
    self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self._gru = tf.keras.layers.GRU(self._dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer="glorot_uniform")
    self._fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self._attention = BahdanauAttention(self._dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self._attention(hidden, enc_output)

    # x shape after passing through embedding: (batch_size, 1, embedding_dim)
    x = self._embedding(x)

    # x shape after concatenation: (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self._gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self._fc(output)

    return x, state, attention_weights


class Seq2SeqModel(object):
  """Sequence-to-sequence model implementing encoder-decoder."""

  def __init__(self,
               batch_size=64,
               embedding_dim=256,
               enc_units=256,
               dec_units=256,
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
    self._input_symbols = input_symbols
    self._input_vocab_size = len(input_symbols)
    self._output_symbols = output_symbols
    self._output_vocab_size = len(output_symbols)
    self._encoder = Encoder(self._input_vocab_size,
                            self._embedding_dim,
                            self._enc_units)
    self._decoder = Decoder(self._output_vocab_size,
                            self._embedding_dim,
                            self._dec_units)
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
  def _train_step(self, inp, targ, enc_hidden):
    """Single training step."""
    loss = 0
    with tf.GradientTape() as tape:
      enc_output, enc_hidden = self._encoder(inp, enc_hidden)
      dec_hidden = enc_hidden
      dec_input = tf.expand_dims(
          [self._output_symbols.find("<s>")] * self._batch_size, 1)
      for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = self._decoder(
            dec_input, dec_hidden, enc_output)
        loss += self._loss_function(targ[:, t], predictions)
        dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = (self._encoder.trainable_variables +
                 self._decoder.trainable_variables)
    gradients = tape.gradient(loss, variables)
    self._optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

  def train(self, corpus, epochs=10, direction="pronounce", window=-1):
    """Main entry point for running the training."""
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
      enc_hidden = self._encoder.initialize_hidden_state(self._batch_size)
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
              epoch + 1, batch, nbatches, batch_loss.numpy()))
        steps += 1
        batch, (inp, targ) = next(batches)
      total_loss /= steps
      print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss))
      if total_loss < best_total_loss:
        self._checkpoint.save(file_prefix=self._checkpoint_prefix)
        print("Saved checkpoint to {}".format(self._checkpoint_prefix))
        best_total_loss = total_loss
      print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    print("Best total loss: {:.4f}".format(best_total_loss))

    # Restore stdout.
    sys.stdout = stdout_file

  def decode(self, inputs, joiner=""):
    """Decoder step."""
    def expand_dims(inp):
      return tf.expand_dims(tf.convert_to_tensor(inp), 0)

    inputs = expand_dims(inputs)
    hidden = tf.zeros((1, self._enc_units))
    enc_out, enc_hidden = self._encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([self._output_symbols.find("<s>")], 0)
    result = []
    attention_plot = np.zeros([self._output_length, self._input_length])
    for t in range(self._output_length):
      predictions, dec_hidden, attention_weights = self._decoder(
          dec_input, dec_hidden, enc_out)
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
  def input_length(self):
    return self._input_length

  @property
  def output_length(self):
    return self._output_length

  @property
  def input_symbols(self):
    return self._input_symbols

  @property
  def output_symbols(self):
    return self._output_symbols
