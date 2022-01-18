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

# Lint as: python3
"""Custom Keras Models for the RL agent."""

import tensorflow.compat.v1 as tf
from tensorflow.contrib import checkpoint as contrib_checkpoint
from tensorflow.contrib.eager.python import tfe as contrib_eager


def create_checkpoint_manager(agent,
                              ckpt_dir,
                              restore=False,
                              include_optimizer=False,
                              meta_learn=False):
  """Helper function for checkpointing."""
  objects_to_save = dict(model=agent.pi, global_step=agent.global_step)
  if include_optimizer:
    objects_to_save.update(optimizer=agent.optimizer)
  if meta_learn:
    objects_to_save.update(
        score_optimizer=agent.score_optimizer, score_fn=agent.score_fn)
  checkpoint = tf.train.Checkpoint(**objects_to_save)
  manager = contrib_checkpoint.CheckpointManager(
      checkpoint, directory=ckpt_dir, max_to_keep=2)
  if restore:
    if manager.latest_checkpoint is not None:
      status = checkpoint.restore(manager.latest_checkpoint)
      tf.logging.info('Loaded checkpoint {}'.format(manager.latest_checkpoint))
      if not include_optimizer:
        status.assert_consumed()
  return manager


def entropy_from_logits(logits):
  """Calculate the sequence entropy using the logits."""
  a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
  ea0 = tf.exp(a0)
  z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
  p0 = ea0 / z0
  seq_entropy = p0 * (tf.log(z0) - a0)
  entropy = tf.reduce_sum(seq_entropy, axis=-1)
  return entropy


def gru(units, return_sequences=True, return_state=True, use_cudnn=True):
  """GRU Helper Function."""
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than
  # GRU) the code automatically does that.
  if tf.test.is_gpu_available() and use_cudnn:
    return tf.keras.layers.CuDNNGRU(
        units,
        return_sequences=return_sequences,
        return_state=return_state,
        recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(
        units,
        return_sequences=return_sequences,
        return_state=return_state,
        recurrent_activation='sigmoid',
        recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
  """Simple Encoder model with GRU cell."""

  def __init__(self, embedding_dim, enc_units, vocab_size=4):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    # Valid input takes values from 0 to `vocab_size` - 1
    self.embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=True)
    self.gru = gru(self.enc_units, return_state=False, use_cudnn=False)

  @contrib_eager.defun
  def call(self, x, initial_state=None):
    x = self.embedding(x)
    output = self.gru(x, initial_state=initial_state)
    return output

  def initialize_hidden_state(self, batch_sz):
    return tf.zeros((batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
  """Decoder model with attention inspired by NMT colab.

  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb
  """

  def __init__(self, num_outputs, dec_units):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.gru = gru(self.dec_units, use_cudnn=False)
    self.dense = tf.keras.layers.Dense(num_outputs)
    self._num_outputs = num_outputs

    # used for attention
    self.w1 = tf.keras.layers.Dense(self.dec_units, use_bias=False)
    self.w2 = tf.keras.layers.Dense(self.dec_units)
    self.w3 = tf.keras.layers.Dense(1)

  @contrib_eager.defun(input_signature=[
      contrib_eager.TensorSpec(shape=[None, 16], dtype=tf.float32),
      contrib_eager.TensorSpec(shape=[None, None, 16], dtype=tf.float32),
  ])
  def _call(self, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)

    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H))
    # to self.V
    score = self.w3(
        tf.nn.tanh(self.w1(enc_output) + self.w2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * enc_output
    context_vector = tf.reduce_sum(context_vector, axis=1)

    # x shape after expansion (batch_size, 1, observation_dim)
    # x = tf.expand_dims(x, 1)
    # x shape after concatenation is (batch_size, 1, obs_dim + hidden_size)
    # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    x = tf.expand_dims(context_vector, 1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size * 1, vocab)
    x = self.dense(output)

    return x, state

  def call(self, num_inputs, enc_output, hidden, return_state=False):
    outputs_per_timestep = []
    for _ in range(num_inputs):
      out, hidden = self._call(hidden, enc_output)
      outputs_per_timestep.append(out)

    outputs = tf.stack(outputs_per_timestep, axis=1)
    if self._num_outputs == 1:
      outputs = tf.squeeze(outputs, axis=-1)  # Value Function
    if return_state:
      return outputs, hidden
    else:
      return outputs


class EncoderDecoder(tf.keras.Model):
  """Encoder Decoder architecture for generating actions given a context."""

  def __init__(self, embedding_dim, units, num_outputs):
    super(EncoderDecoder, self).__init__()
    self._encoder = Encoder(embedding_dim, units)
    self._decoder = Decoder(num_outputs, units)

  def encode_context(self, context, context_lengths=None):
    enc_output = self._encoder(context)
    # Extract the correct hidden state using the sequence_length
    batch_size, maxlen, _ = [i.value for i in enc_output.get_shape()]
    if context_lengths is None:
      context_lengths = tf.fill([batch_size], maxlen)
    else:
      sequence_mask = tf.sequence_mask(
          context_lengths, maxlen, dtype=enc_output.dtype)
      enc_output = enc_output * tf.expand_dims(sequence_mask, axis=-1)
    hidden_indices = tf.transpose(
        tf.stack([tf.range(batch_size), context_lengths - 1]))
    enc_hidden = tf.gather_nd(enc_output, hidden_indices)
    return enc_output, enc_hidden

  def call(self,
           contexts,
           num_inputs=1,
           context_lengths=None,
           return_state=False):
    enc_output, initial_state = self.encode_context(contexts, context_lengths)
    return self._call(num_inputs, enc_output, initial_state, return_state)

  def _call(self, num_inputs, enc_output, initial_state, return_state):
    return self._decoder(
        num_inputs, enc_output, hidden=initial_state, return_state=return_state)


class SimpleLinearNN(tf.keras.Model):
  """Simple one layer neural network based on dense layer."""

  def __init__(self):
    super(SimpleLinearNN, self).__init__()
    self.dense = tf.keras.layers.Dense(
        1, use_bias=True, bias_initializer=tf.initializers.ones())

  @contrib_eager.defun
  def call(self, inputs):
    out = tf.squeeze(self.dense(inputs), axis=-1)
    return out


class Linear(tf.keras.layers.Layer):
  """Simple layer for constructing the auxiliary reward function."""

  def __init__(self, units=16, use_bias=True):
    super(Linear, self).__init__()
    self.units = units

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    kernel = self.kernel
    w_arr = tf.split(kernel, self.units)
    pair_weights = []
    for x1 in range(self.units):
      a1, b1 = x1 // 4, x1 % 4
      for x2 in range(self.units):
        a2, b2 = x2 // 4, x2 % 4
        indices = [4 * a1 + a2, 4 * a1 + b2, 4 * b1 + a2, 4 * b1 + b2]
        weight = ((w_arr[indices[0]] * w_arr[indices[-1]]) *
                  self.w1) + (w_arr[indices[1]] * w_arr[indices[2]]) * self.w2
        pair_weights.append(weight)
    pair_kernel = tf.concat(pair_weights, axis=0)
    weights = tf.concat([kernel, pair_kernel], axis=0, name='weights')
    out = tf.tensordot(inputs, weights, axes=1) + self.bias
    return out

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    trainable = True
    tf.logging.info('Trainable is set to {}'.format(trainable))
    self.kernel = self.add_weight(
        name='kernel',
        shape=[self.units],
        trainable=trainable,
        initializer=tf.initializers.random_uniform(0.0, 1.0))
    self.w1 = self.add_weight('w1', shape=[1], trainable=trainable)
    self.w2 = self.add_weight('w2', shape=[1], trainable=trainable)
    self.bias = self.add_weight(
        'bias', shape=[1], initializer=tf.initializers.ones())
    super(Linear, self).build(input_shape)


class LinearNN(tf.keras.Model):
  """Simple one layer neural network based on Linear layer."""

  def __init__(self, **kwargs):
    super(LinearNN, self).__init__()
    self._linear = Linear(**kwargs)

  @contrib_eager.defun(input_signature=[
      contrib_eager.TensorSpec(shape=[None, 16 * 17], dtype=tf.float32)
  ])
  def call(self, inputs):
    return self._linear(inputs)
