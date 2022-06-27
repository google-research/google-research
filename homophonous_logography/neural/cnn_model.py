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

"""CNN model based on work described by Gehring, et. al. (2017).

See https://arxiv.org/pdf/1705.03122.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np

import tensorflow as tf  # tf

import homophonous_logography.neural.corpus as data
import homophonous_logography.neural.positional_embedding as pos
import homophonous_logography.neural.utils as utils

tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


# Maximum conceivable sequence length.
_MAX_SEQUENCE_LENGTH = 500

# Residual scaling factor: np.sqrt(0.5) in Gehring et. al. (2017).
_RESIDUAL_SCALING_FACTOR = np.sqrt(0.5)


def _glu(x):
  """Gated linear unit."""
  dim = x.shape[2] // 2
  return tf.math.multiply(x[:, :, :dim], tf.math.sigmoid(x[:, :, dim:]))


def _encoder_conv_block(x, num_hidden, kernel_size, dilation_rate):
  return tf.keras.layers.Conv1D(num_hidden, kernel_size=kernel_size,
                                padding="same", dilation_rate=dilation_rate)(x)


def _decoder_conv_block(x, num_hidden, kernel_size, dilation_rate):
  return tf.keras.layers.Conv1D(num_hidden, kernel_size=kernel_size,
                                padding="causal",
                                dilation_rate=dilation_rate)(x)


def _layer(x, conv_block, kernel_size, num_hidden, dilation_rate,
           residual=None):
  """Single convolution block followed by a GLU nonlinarity.

  The GLU halfs the dimension of the input features.
  See https://arxiv.org/pdf/1705.03122.pdf.

  Args:
    x: Inputs.
    conv_block: Convolutional block.
    kernel_size: Size of the convolution kernel.
    num_hidden: Number of features.
    dilation_rate: Dilation rate.
    residual: Residual connections.

  Returns:
    Outputs from the convolutional blocks.
  """
  z = conv_block(x, num_hidden * 2, kernel_size, dilation_rate)
  z = _glu(z)
  if residual is not None:
    z = (z + residual) * _RESIDUAL_SCALING_FACTOR
  return z


def _embedding(inputs, vocab_size, embedding_dim, dropout_rate):
  """Embedding layer."""
  x = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                mask_zero=True)(inputs)
  positions = pos.PositionEmbedding(max_length=_MAX_SEQUENCE_LENGTH)(x)
  x += positions
  x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
  return x


def _encoder(encoder_inputs, input_vocab_size, embedding_dim, num_layers,
             num_hidden_units, kernel_size, dropout_rate, use_residuals):
  """Encoder component."""
  x_encoder = _embedding(encoder_inputs, input_vocab_size, embedding_dim,
                         dropout_rate)

  if x_encoder.shape[2] != num_hidden_units:
    x_encoder = tf.keras.layers.Conv1D(num_hidden_units, 1)(x_encoder)
  e = x_encoder
  for i in range(num_layers):
    z = _layer(x_encoder, _encoder_conv_block, kernel_size,
               num_hidden=num_hidden_units, dilation_rate=2 ** i,
               residual=(x_encoder if use_residuals else None))
    x_encoder = tf.keras.layers.Dropout(rate=dropout_rate)(z)

  # Encoder output and its state (denoted `z^u` and `z^u + e` in the paper).
  encoder_state = (x_encoder + e) * _RESIDUAL_SCALING_FACTOR
  return x_encoder, encoder_state


def _decoder(decoder_inputs, encoder_inputs, encoder_outputs, encoder_state,
             target_vocab_size, embedding_dim, num_layers,
             num_hidden_units, kernel_size, dropout_rate):
  """Decoder component."""
  x_decoder = _embedding(decoder_inputs, target_vocab_size, embedding_dim,
                         dropout_rate)

  encoder_padding = tf.expand_dims(tf.math.equal(encoder_inputs, 0), axis=-1)
  real_lengths = tf.cast(tf.shape(encoder_outputs)[1], tf.float32)
  real_lengths -= tf.math.reduce_sum(
      tf.cast(encoder_padding, tf.float32), axis=1, keepdims=True)
  encoder_mask = tf.expand_dims(tf.math.not_equal(encoder_inputs, 0), axis=-1)
  encoder_mask = tf.cast(encoder_mask, tf.float32)
  decoder_mask = tf.cast(tf.expand_dims(tf.math.not_equal(
      decoder_inputs, 0), axis=-1), tf.float32)
  encoder_outputs *= encoder_mask
  encoder_state *= encoder_mask

  attention_scale = real_lengths * tf.math.rsqrt(real_lengths)
  attention_mask = tf.linalg.matmul(decoder_mask, encoder_mask,
                                    transpose_b=True)
  if x_decoder.shape[2] != num_hidden_units:
    x_decoder = tf.keras.layers.Conv1D(num_hidden_units, 1)(x_decoder)
  g = x_decoder
  layer_attention = []
  for i in range(num_layers):
    # Output of the current decoder layer `h_i`.
    h = _layer(x_decoder, _decoder_conv_block, kernel_size,
               num_hidden=num_hidden_units, dilation_rate=2 ** i)
    h *= decoder_mask
    h_residual = h
    # See Eq. (1) in Gehring et. al. (2017).
    d = tf.keras.layers.Dense(num_hidden_units)(h) + g
    d *= _RESIDUAL_SCALING_FACTOR
    d *= decoder_mask
    dz = tf.linalg.matmul(d, tf.transpose(encoder_outputs, [0, 2, 1]))
    a = tf.keras.layers.Softmax()(dz, mask=attention_mask)
    layer_attention.append(a)
    # See Eq. (2) in Gehring et. al. (2017).
    c = tf.linalg.matmul(a, encoder_state)
    c *= attention_scale
    h = (h_residual + c) * _RESIDUAL_SCALING_FACTOR
    h = tf.keras.layers.Dense(num_hidden_units)(h)
    h = tf.keras.layers.Dropout(rate=dropout_rate)(h)
    x_decoder = h

  return x_decoder, layer_attention


def _build_model(input_vocab_size,
                 target_vocab_size,
                 embedding_dim,
                 use_residuals=True,
                 num_layers=3,
                 num_hidden_units=256,
                 kernel_size=3,
                 dropout_rate=0.1,
                 name="cnn"):
  """Constructs Keras model."""
  # Encoder.
  encoder_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="inputs")
  x_encoder, encoder_state = _encoder(
      encoder_inputs, input_vocab_size, embedding_dim, num_layers,
      num_hidden_units, kernel_size, dropout_rate, use_residuals)

  # Decoder.
  decoder_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32,
                                  name="decoder_inputs")
  x_decoder, layer_attention = _decoder(
      decoder_inputs, encoder_inputs, x_encoder, encoder_state,
      target_vocab_size, embedding_dim, num_layers, num_hidden_units,
      kernel_size, dropout_rate)

  # Outputs.
  decoder_outputs = tf.keras.layers.Dense(target_vocab_size)(x_decoder)
  model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs],
                                outputs=[decoder_outputs, layer_attention],
                                name=name)
  model.summary()
  return model


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Custom learning schedule."""

  def __init__(self, model_dim, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.model_dim = model_dim
    self.model_dim = tf.cast(self.model_dim, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)


_TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


class Seq2SeqCnnModel(object):
  """Full convolutional model."""

  def __init__(self,
               batch_size=64,
               input_symbols=None,
               output_symbols=None,
               multilayer_retrieval_strategy="AVERAGE",
               embedding_dim=64,
               use_residuals=True,
               num_layers=2,
               num_hidden_units=256,
               kernel_size=3,
               dropout_rate=0.1,
               model_dir=".",
               name="model"):
    self._batch_size = batch_size
    self._input_symbols = input_symbols
    self._input_vocab_size = len(input_symbols)
    self._output_symbols = output_symbols
    self._output_vocab_size = len(output_symbols)
    self._use_residuals = use_residuals
    self._multilayer_retrieval = multilayer_retrieval_strategy
    self._cnn = _build_model(self._input_vocab_size,
                             self._output_vocab_size,
                             embedding_dim,
                             use_residuals=use_residuals,
                             num_layers=num_layers,
                             num_hidden_units=num_hidden_units,
                             kernel_size=kernel_size,
                             dropout_rate=dropout_rate)

    self._learning_rate = CustomSchedule(num_hidden_units * 2)
    self._optimizer = tf.keras.optimizers.Adam(
        self._learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")
    self._train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    self._name = name
    self._checkpoint_dir = os.path.join(model_dir, self._name)
    self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
    self._checkpoint = tf.train.Checkpoint(optimizer=self._optimizer,
                                           cnn=self._cnn)
    # Length of the current output tensor (for eval).
    self._input_length = -1
    self._output_length = -1

  @tf.function
  def _loss_function(self, y_true, y_pred):
    loss = self._loss_object(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

  @tf.function
  def _accuracy_function(self, real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, output_type=tf.int32, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

  @tf.function(input_signature=_TRAIN_STEP_SIGNATURE)
  def _train_step(self, inputs, targets):
    """Single training step."""
    # Teacher forcing.
    target_inputs = targets[:, :-1]
    target_real = targets[:, 1:]
    with tf.GradientTape() as tape:
      predictions, _ = self._cnn(
          inputs=[inputs, target_inputs], training=True)
      loss = self._loss_function(target_real, predictions)

    gradients = tape.gradient(loss, self._cnn.trainable_variables)
    self._optimizer.apply_gradients(zip(gradients,
                                        self._cnn.trainable_variables))
    self._train_accuracy(self._accuracy_function(target_real, predictions))
    return loss

  def train(self, corpus, epochs=10, direction="pronounce", window=-1):
    """Runs training."""
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
    print("       Use residuals: {}".format(self._use_residuals))

    # Perform training.
    best_total_loss = 1000000
    nbatches = data.num_batches(corpus, self._batch_size, direction=direction,
                                window=window)
    for epoch in range(epochs):
      self._train_accuracy.reset_states()

      start = time.time()
      total_loss = 0
      steps = 0
      batches = data.batchify(corpus, self._batch_size, direction,
                              window=window)
      batch, (inputs, targ) = next(batches)
      while batch > -1:
        bos = np.expand_dims(
            [self._output_symbols.find("<s>")] * np.shape(targ)[0], 1)
        targets = np.concatenate((bos, targ), axis=-1)
        batch_loss = self._train_step(inputs, targets)
        total_loss += batch_loss
        if batch % 10 == 0:
          print("Epoch {} Batch {} (/{}) Loss {:.4f}".format(
              epoch + 1,
              batch,
              nbatches,
              batch_loss))
        steps += 1
        batch, (inputs, targ) = next(batches)
      total_loss /= steps
      print("Epoch {} Loss {:.4f} Accuracy {:.4f}".format(
          epoch + 1, total_loss, self._train_accuracy.result()))

      if total_loss < best_total_loss:
        self._checkpoint.save(file_prefix=self._checkpoint_prefix)
        print("Saved checkpoint to {}".format(self._checkpoint_prefix))
        best_total_loss = total_loss
      print("Time taken for 1 epoch {} sec\n".format(
          time.time() - start))
    print("Best total loss: {:.4f}".format(best_total_loss))

    # Restore stdout.
    sys.stdout = stdout_file

  @tf.function(reduce_retracing=True)
  def _predict_step(self, encoder_input, output):
    return self._cnn(
        inputs=[encoder_input, output], training=False)

  def _combine_attention(self, attentions):
    """Combines attentions for a given timestep."""
    if self._multilayer_retrieval == "AVERAGE":
      return np.sum(attentions, axis=0) / len(attentions)
    elif self._multilayer_retrieval == "MAX":
      att = np.max(attentions, axis=0)
      return att / np.sum(att)
    elif self._multilayer_retrieval == "JOINT":
      att = np.prod(np.vstack(attentions), axis=0, dtype=np.float64)
      if np.isclose(np.sum(att), 0.0):
        att.fill(1.0 / len(att))  # Make uniform.
      else:
        att /= np.sum(att)
      return att
    elif self._multilayer_retrieval == "TOP":
      return attentions[-1]
    elif self._multilayer_retrieval == "BOTTOM":
      return attentions[0]
    else:
      raise ValueError("Invalid multi-layer retrieval: {}".format(
          self._multilayer_retrieval))

  def decode(self, inputs, joiner=""):
    """Runs inference."""
    encoder_input = tf.convert_to_tensor([inputs], dtype=tf.int32)

    # The first input to the transformer will be the start token.
    start = [self._output_symbols.find("<s>")]
    output = tf.convert_to_tensor(start, dtype=tf.int32)
    output = tf.expand_dims(output, 0)

    result = []
    attention_plot = np.zeros([self._output_length, self._input_length])
    for t in range(self._output_length):
      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, layer_attention = self._predict_step(
          encoder_input, output)
      attentions = [att[0, t, :].numpy() for att in layer_attention]
      attention_plot[t] = self._combine_attention(attentions)

      # Select the last word from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
      predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

      # Concatentate the predicted_id to the output which is given to the
      # decoder as its input.
      output = tf.concat([output, predicted_id], axis=-1)

      outsym = self._output_symbols.find(int(predicted_id.numpy()))
      if outsym == "</s>" or outsym == "</targ>":
        break
      else:
        result.append(outsym)

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
    return "_%s" % self._multilayer_retrieval.lower()
