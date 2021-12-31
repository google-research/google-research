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

"""Simple sequence-to-sequence transformer model.

Loosely based on:
  https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
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
import homophonous_logography.neural.utils as utils

tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


def _create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]


def _create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = _create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)


def _scaled_dot_product_attention(query, key, value, mask):
  """Actual attention function using dot product."""
  matmul_qk = tf.matmul(query, key, transpose_b=True)
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask zero out padding tokens.
  if mask is not None:
    logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(logits, axis=-1)
  return tf.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-head attention implementation."""

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads
    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)
    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs["query"], inputs["key"], inputs[
        "value"], inputs["mask"]
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    scaled_attention, attention_weights = _scaled_dot_product_attention(
        query, key, value, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))
    outputs = self.dense(concat_attention)
    return outputs, attention_weights


class PositionalEncoding(tf.keras.layers.Layer):
  """Trigonometric positional encoding."""

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, Ellipsis]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def _encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  """One layer of the encoder."""
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention, _ = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          "query": inputs,
          "key": inputs,
          "value": inputs,
          "mask": padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation="relu")(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


# Limit the lengths of input sequences.
_MAX_SEQUENCE_LENGTH = 500


def _encoder(vocab_size,
             num_layers,
             units,
             d_model,
             num_heads,
             dropout,
             name="encoder"):
  """Encoder component."""
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(_MAX_SEQUENCE_LENGTH, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = _encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def _decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  """Single decoder layer."""
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention1, attention_weights_block1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          "query": inputs,
          "key": inputs,
          "value": inputs,
          "mask": look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2, attention_weights_block2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          "query": attention1,
          "key": enc_outputs,
          "value": enc_outputs,
          "mask": padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation="relu")(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=[outputs, attention_weights_block1, attention_weights_block2],
      name=name)


def _decoder(vocab_size,
             num_layers,
             units,
             d_model,
             num_heads,
             dropout,
             name="decoder"):
  """Decoder component."""
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(_MAX_SEQUENCE_LENGTH, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  attention_weights = {}
  for i in range(num_layers):
    outputs, attn_w_block1, attn_w_block2 = _decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="decoder_layer_{}".format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
    attention_weights["decoder_layer{}_block1".format(i+1)] = attn_w_block1
    attention_weights["decoder_layer{}_block2".format(i+1)] = attn_w_block2

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=[outputs, attention_weights],
      name=name)


def _transformer(input_vocab_size,
                 target_vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 dropout,
                 name="transformer"):
  """Transformer network."""
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      _create_padding_mask, output_shape=(1, 1, None),
      name="enc_padding_mask")(inputs)
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = tf.keras.layers.Lambda(
      _create_look_ahead_mask,
      output_shape=(1, None, None),
      name="look_ahead_mask")(dec_inputs)
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = tf.keras.layers.Lambda(
      _create_padding_mask, output_shape=(1, 1, None),
      name="dec_padding_mask")(inputs)

  enc_outputs = _encoder(
      vocab_size=input_vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs, attention_weights = _decoder(
      vocab_size=target_vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=target_vocab_size, name="outputs")(
      dec_outputs)

  model = tf.keras.Model(inputs=[inputs, dec_inputs],
                         outputs=[outputs, attention_weights], name=name)
  model.summary()
  return model


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule."""

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


_TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


class Seq2SeqTransformerModel(object):
  """Full transformer model."""

  def __init__(self,
               batch_size=64,
               num_heads=8,
               ff_dim=512,
               num_layers=4,
               model_dim=128,
               input_symbols=None,
               output_symbols=None,
               multihead_retrieval_strategy="AVERAGE",
               model_dir=".",
               name="model"):
    self._batch_size = batch_size
    self._input_symbols = input_symbols
    self._input_vocab_size = len(input_symbols)
    self._output_symbols = output_symbols
    self._output_vocab_size = len(output_symbols)
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._multihead_retrieval = multihead_retrieval_strategy
    self._transformer = _transformer(
        input_vocab_size=self._input_vocab_size,
        target_vocab_size=self._output_vocab_size,
        num_layers=num_layers,
        units=ff_dim,
        d_model=model_dim,
        num_heads=num_heads,
        dropout=0.1)

    self._learning_rate = CustomSchedule(model_dim)
    self._optimizer = tf.keras.optimizers.Adam(
        self._learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")
    self._train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    self._name = name
    self._checkpoint_dir = os.path.join(model_dir, self._name)
    self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
    self._checkpoint = tf.train.Checkpoint(optimizer=self._optimizer,
                                           transformer=self._transformer)
    # Length of the current output tensor (for eval).
    self._input_length = -1
    self._output_length = -1

  def _loss_function(self, y_true, y_pred):
    loss = self._loss_object(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

  def _accuracy_function(self, real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, output_type=tf.int32, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

  @tf.function(input_signature=_TRAIN_STEP_SIGNATURE)
  def _train_step(self, inputs, targets):
    """One step of the training."""
    target_inputs = targets[:, :-1]
    target_real = targets[:, 1:]
    with tf.GradientTape() as tape:
      predictions, _ = self._transformer(
          inputs=[inputs, target_inputs], training=True)
      loss = self._loss_function(target_real, predictions)

    gradients = tape.gradient(loss, self._transformer.trainable_variables)
    self._optimizer.apply_gradients(zip(gradients,
                                        self._transformer.trainable_variables))
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

  def _get_attention(self, attention_weights):
    """Prepare attention for consumption.

    Args:
      attention_weights: tensor with shape:
         (batch=1, num_heads, seq_len_q, seq_len_k).

    Returns:
      Accumulated attention.
    """
    attention_heads = tf.squeeze(  # Remove batch dimension.
        attention_weights["decoder_layer%d_block2" % self._num_layers], 0)

    # Basic sanity checks.
    if len(attention_heads) != self._num_heads:
      raise ValueError("Invalid number of attention heads: {}".format(
          len(attention_heads)))
    if len(attention_heads.shape) != 3:
      raise ValueError("Invalid shape of attention weights: {}".format(
          len(attention_heads.shape)))
    if attention_heads.shape[1] > self._output_length:
      raise ValueError("Expected output length <= {} for dim 1, got {}".format(
          self._output_length, attention_heads.shape[1]))
    elif attention_heads.shape[1] < self._output_length:
      output_len_diff = self._output_length - attention_heads.shape[1]
      attention_heads = tf.pad(attention_heads,
                               [[0, 0], [0, output_len_diff], [0, 0]])
    if attention_heads.shape[2] != self._input_length:
      raise ValueError("Expected input length {} for dim 2, got {}".format(
          self._input_length, attention_heads.shape[2]))

    # Combine.
    if self._multihead_retrieval == "AVERAGE":
      attention = tf.reduce_sum(attention_heads, axis=0) / self._num_heads
    elif self._multihead_retrieval == "MAX":
      attention = tf.reduce_max(attention_heads, axis=0)
    else:
      raise ValueError("Unknown retrieval strategy: {}".format(
          self._multihead_retrieval))
    return attention

  @tf.function(experimental_relax_shapes=True)
  def _predict_step(self, encoder_input, output):
    """One prediction step."""
    return self._transformer(
        inputs=[encoder_input, output], training=False)

  def decode(self, inputs, joiner=""):
    """Decodes the inputs."""
    encoder_input = tf.convert_to_tensor([inputs], dtype=tf.int32)

    # The first input to the transformer will be the start token.
    start = [self._output_symbols.find("<s>")]
    output = tf.convert_to_tensor(start, dtype=tf.int32)
    output = tf.expand_dims(output, 0)

    result = []
    for _ in range(self._output_length):
      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, attention_weights = self._predict_step(
          encoder_input, output)

      # select the last word from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
      predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

      # concatentate the predicted_id to the output which is given to the
      # decoder as its input.
      output = tf.concat([output, predicted_id], axis=-1)

      outsym = self._output_symbols.find(int(predicted_id.numpy()))
      if outsym == "</s>" or outsym == "</targ>":
        break
      else:
        result.append(outsym)

    # Accumulate attention over all the heads.
    attention = self._get_attention(attention_weights)
    return joiner.join(result), attention.numpy()

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
    return "_%s" % self._multihead_retrieval.lower()
