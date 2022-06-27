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

"""The actual Contrack model."""

import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple

import tensorflow as tf

from contrack import custom_ops
from contrack import encoding
from contrack.env import Env


def _pad_and_clip_seq_batch(seq_batch, seq_len_batch,
                            pad_value, maxlen,
                            data_vec_len):
  """Pads a batch of sequences with a padding value up to a length."""
  with tf.name_scope('pad_seq_batch'):
    seq_mask = tf.sequence_mask(lengths=seq_len_batch, maxlen=maxlen)
    seq_mask = tf.expand_dims(seq_mask, 2)
    seq_mask = tf.tile(seq_mask, [1, 1, data_vec_len])
    # Trim or pad seq_batch as needed to make the shapes compatible
    padded_shape = [tf.shape(seq_batch)[0], maxlen, data_vec_len]
    seq_batch = seq_batch[:, :maxlen, :]
    seq_dim_pad_len = tf.constant(maxlen) - tf.shape(seq_batch)[1]
    seq_batch = tf.pad(
        seq_batch, paddings=[[0, 0], [0, seq_dim_pad_len], [0, 0]])
    seq_batch.set_shape([seq_batch.shape[0], maxlen, data_vec_len])
    pad_value = tf.cast(pad_value, dtype=seq_batch.dtype)
    pad_batch = tf.fill(padded_shape, value=pad_value)
    padded_seq_batch = tf.where(seq_mask, seq_batch, pad_batch)
    return padded_seq_batch


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def split_heads(x, n):
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  y = tf.reshape(x, x_shape[:-1] + [n, m // n])

  return tf.transpose(y, [0, 2, 1, 3])


def combine_heads(x):
  x = tf.transpose(x, [0, 2, 1, 3])
  x_shape = shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])


class ConvertToSequenceLayer(tf.keras.layers.Layer):
  """Concatenates input data into a sequence suitable for prediction."""

  def __init__(self, input_vec_len):
    super(ConvertToSequenceLayer, self).__init__()
    self.config = Env.get().config
    self.input_vec_len = input_vec_len

  @classmethod
  def from_config(cls, config):
    return ConvertToSequenceLayer(config['input_vec_len'])

  def get_config(self):
    return {'input_vec_len': self.input_vec_len}

  def compute_mask(self, inputs, mask=None):
    state_seq_len = inputs['state_seq_length']
    token_seq_len = inputs['token_seq_length']

    input_seq_len = tf.add(state_seq_len, token_seq_len)

    return tf.sequence_mask(input_seq_len, maxlen=self.config.max_seq_len)

  def call(self,
           inputs,
           training = None):
    with tf.name_scope('convert_to_sequence'):
      state_seq_len = tf.cast(inputs['state_seq_length'], tf.int32)
      state_seq = inputs['state_seq']

      token_seq_len = tf.cast(inputs['token_seq_length'], tf.int32)
      token_seq = inputs['token_seq']

      input_seq, input_seq_len = custom_ops.sequence_concat(
          sequences=[state_seq, token_seq],
          lengths=[state_seq_len, token_seq_len])

      # Clip and pad seq
      input_seq_len = tf.minimum(
          input_seq_len, self.config.max_seq_len, name='input_seq_len')
      input_seq = _pad_and_clip_seq_batch(
          input_seq,
          input_seq_len,
          pad_value=0,
          maxlen=self.config.max_seq_len,
          data_vec_len=self.input_vec_len)

      # Add timing signal
      if self.config.timing_signal_size > 0:
        num_channels = self.config.timing_signal_size
        positions = tf.cast(tf.range(self.config.max_seq_len), dtype=tf.int64)

        min_timescale = 1.0
        max_timescale = 1.0e4

        with tf.name_scope('TimingSignal'):
          num_timescales = num_channels // 2
          log_timescale_increment = (
              math.log(max_timescale / min_timescale) /
              (tf.cast(num_timescales, tf.float32) - 1))
          inv_timescales = min_timescale * tf.exp(
              tf.cast(tf.range(num_timescales), tf.float32) *
              -log_timescale_increment)
          scaled_time = (
              tf.expand_dims(tf.cast(positions, tf.float32), 1) *
              tf.expand_dims(inv_timescales, 0))
          signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
          time_signal = tf.pad(
              signal, [[0, 0], [0, tf.math.floormod(num_channels, 2)]])

          time_signal = tf.expand_dims(time_signal, 0)
          time_signal = tf.tile(time_signal, [tf.shape(input_seq)[0], 1, 1])
          seq_mask = tf.cast(
              tf.sequence_mask(
                  lengths=input_seq_len, maxlen=self.config.max_seq_len),
              dtype=tf.float32)
          seq_mask = tf.expand_dims(seq_mask, 2)
          seq_mask = tf.tile(seq_mask, [1, 1, num_channels])
          time_signal = time_signal * seq_mask
          input_seq = tf.concat([input_seq, time_signal],
                                axis=2,
                                name='add_time_signal')

      return input_seq, input_seq_len


class IdentifyNewEntityLayer(tf.keras.layers.Layer):
  """The layer identifying new entities in a message."""

  def __init__(self, seq_shape):
    super(IdentifyNewEntityLayer, self).__init__()
    self.config = Env.get().config.new_id_attention
    self.supports_masking = True
    self.seq_shape = seq_shape

    self.batch_size = seq_shape[0]
    self.seq_len = seq_shape[1]
    key_dim = math.ceil(seq_shape[2] / self.config.num_heads)
    output_shape = [
        self.batch_size, self.seq_len, key_dim * self.config.num_heads
    ]
    self.attention = tf.keras.layers.MultiHeadAttention(
        num_heads=self.config.num_heads,
        key_dim=key_dim,
        value_dim=key_dim,
        use_bias=True,
        dropout=self.config.dropout_rate,
        output_shape=[output_shape[2]])

    self.layer_norm = tf.keras.layers.LayerNormalization(
        axis=2, epsilon=1e-6, name='SelfAttentionNorm')

    self.affine = tf.keras.layers.Dense(2, use_bias=True)

    self.q_dense = tf.keras.layers.Dense(output_shape[2], use_bias=True)
    self.k_dense = tf.keras.layers.Dense(output_shape[2], use_bias=True)
    self.v_dense = tf.keras.layers.Dense(output_shape[2], use_bias=True)

    hidden_size = 100
    filter_size = 800
    self.attention_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
    self.match_residual_dense = tf.keras.layers.Dense(
        hidden_size, use_bias=True)
    self.ffn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=2, epsilon=1e-6, name='FFNNorm')

    self.ff_relu_dense = tf.keras.layers.Dense(
        filter_size, use_bias=True, activation='relu')
    self.ff_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)

  @classmethod
  def from_config(cls, config):
    return IdentifyNewEntityLayer(config['seq_shape'])

  def get_config(self):
    return {'seq_shape': self.seq_shape}

  def compute_mask(self,
                   inputs,
                   mask = None):
    return mask

  def call(self,
           inputs,
           training = None,
           mask = None):
    x = inputs

    # Make feature space size a multiple of num_heads
    num_heads = self.config.num_heads
    if x.shape[-1] % num_heads > 0:
      with tf.name_scope('PadForMultipleOfHeads'):
        fill_size = num_heads - x.shape[-1] % num_heads
        fill_mat = tf.tile(tf.zeros_like(x[:, :, :1]), [1, 1, fill_size])
        x = tf.concat([x, fill_mat], 2)

    # Multihead Attention
    input_depth = x.shape[-1]
    x = self.layer_norm(x, training=training)
    q = split_heads(self.q_dense(x, training=training), num_heads)
    k = split_heads(self.k_dense(x, training=training), num_heads)
    v = split_heads(self.v_dense(x, training=training), num_heads)

    key_depth_per_head = input_depth // num_heads
    q *= key_depth_per_head**-0.5

    logits = tf.matmul(q, k, transpose_b=True)
    weights = tf.nn.softmax(logits, name='attention_weights')
    y = tf.matmul(weights, v)

    y = combine_heads(y)
    y = self.attention_dense(y)

    r = self.match_residual_dense(x)
    y = self.ffn_layer_norm(y + r, training=training)

    # Feed forward
    z = self.ff_relu_dense(y, training=training)
    z = self.ff_dense(z, training=training)
    z += y

    # Affine layer
    z = tf.concat([inputs[:, :, :68], z], 2)
    logits = self.affine(z, training=training)

    # Apply mask
    logits *= tf.expand_dims(tf.cast(mask, tf.float32), -1)

    return logits


class ComputeIdsLayer(tf.keras.layers.Layer):
  """Compute Ids for the (new entity) tokens in the input sequence."""

  def __init__(self):
    super(ComputeIdsLayer, self).__init__()
    self.encodings = Env.get().encodings
    self.config = Env.get().config

  @classmethod
  def from_config(cls, config):
    del config
    return ComputeIdsLayer()

  def get_config(self):
    return {}

  def compute_mask(self,
                   inputs,
                   mask = None):
    _, seq_len, _ = inputs
    return tf.sequence_mask(seq_len, maxlen=self.config.max_seq_len)

  def call(self,
           inputs,
           mask = None):
    seq, enref_seq_len, is_new_logits = inputs

    enref_seq_len = tf.cast(enref_seq_len, dtype=tf.int32)

    enref_ids = self.encodings.as_enref_encoding(seq).enref_id.slice()

    is_new_entity = tf.cast(is_new_logits[:, :, 0] > 0.0, tf.float32)

    new_id_one_hot = custom_ops.new_id(
        state_ids=enref_ids, state_len=enref_seq_len, is_new=is_new_entity)
    new_id_one_hot = tf.stop_gradient(new_id_one_hot)

    return new_id_one_hot


class TrackEnrefsLayer(tf.keras.layers.Layer):
  """Predict enref Ids, properties, ane group membership."""

  def __init__(self, seq_shape):
    super(TrackEnrefsLayer, self).__init__()

    self.config = Env.get().config.tracking_attention
    self.encodings = Env.get().encodings
    self.supports_masking = True
    self.seq_shape = seq_shape

    self.batch_size = seq_shape[0]
    self.seq_len = seq_shape[1]
    attention_input_length = (
        seq_shape[2] + 2 + self.encodings.new_enref_encoding().enref_id.SIZE)
    key_dim = math.ceil(attention_input_length / self.config.num_heads)
    output_shape = [
        self.batch_size, self.seq_len, key_dim * self.config.num_heads
    ]
    self.attention = tf.keras.layers.MultiHeadAttention(
        num_heads=self.config.num_heads,
        key_dim=key_dim,
        value_dim=key_dim,
        use_bias=True,
        dropout=self.config.dropout_rate,
        output_shape=[output_shape[2]])

    self.layer_norm = tf.keras.layers.LayerNormalization(
        axis=2, epsilon=1e-6, name='SelfAttentionNorm')

    self.affine = tf.keras.layers.Dense(
        self.encodings.prediction_encoding_length, use_bias=True)

    self.q_dense = tf.keras.layers.Dense(output_shape[2], use_bias=True)
    self.k_dense = tf.keras.layers.Dense(output_shape[2], use_bias=True)
    self.v_dense = tf.keras.layers.Dense(output_shape[2], use_bias=True)

    hidden_size = 100
    filter_size = 800
    self.attention_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
    self.match_residual_dense = tf.keras.layers.Dense(
        hidden_size, use_bias=True)
    self.ffn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=2, epsilon=1e-6, name='FFNNorm')

    self.ff_relu_dense = tf.keras.layers.Dense(
        filter_size, use_bias=True, activation='relu')
    self.ff_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)

  @classmethod
  def from_config(cls, config):
    return IdentifyNewEntityLayer(config['seq_shape'])

  def get_config(self):
    return {'seq_shape': self.seq_shape}

  def compute_mask(self,
                   inputs,
                   mask = None):
    return mask[0]

  def call(self,
           inputs,
           training = None,
           mask = None):
    seq, is_new_entity, new_ids = inputs

    is_new_entity = tf.stop_gradient(is_new_entity)

    x = tf.concat([seq, is_new_entity, new_ids], axis=2)

    # Make feature space size a multiple of num_heads
    num_heads = self.config.num_heads
    if x.shape[-1] % num_heads > 0:
      with tf.name_scope('PadForMultipleOfHeads'):
        fill_size = num_heads - x.shape[-1] % num_heads
        fill_mat = tf.tile(tf.zeros_like(x[:, :, :1]), [1, 1, fill_size])
        x = tf.concat([x, fill_mat], 2)

    # Multihead Attention
    input_depth = x.shape[-1]
    x = self.layer_norm(x, training=training)
    q = split_heads(self.q_dense(x, training=training), num_heads)
    k = split_heads(self.k_dense(x, training=training), num_heads)
    v = split_heads(self.v_dense(x, training=training), num_heads)

    key_depth_per_head = input_depth // num_heads
    q *= key_depth_per_head**-0.5

    logits = tf.matmul(q, k, transpose_b=True)
    weights = tf.nn.softmax(logits, name='attention_weights')
    y = tf.matmul(weights, v)

    y = combine_heads(y)
    y = self.attention_dense(y)

    r = self.match_residual_dense(x)
    y = self.ffn_layer_norm(y + r, training=training)

    # Feed forward
    z = self.ff_relu_dense(y, training=training)
    z = self.ff_dense(z, training=training)
    z += y

    # Affine layer
    logits = self.affine(z, training=training)

    # Apply mask
    logits *= tf.expand_dims(tf.cast(mask[0], tf.float32), -1)

    return logits


class MergeIdsLayer(tf.keras.layers.Layer):
  """Layer merging the ids from new entities and existing entities."""

  def __init__(self):
    super(MergeIdsLayer, self).__init__()

    self.encodings = Env.get().encodings

  @classmethod
  def from_config(cls, config):
    del config
    return MergeIdsLayer()

  def get_config(self):
    return {}

  def call(self,
           inputs,
           training = None,
           mask = None):
    is_new_entity, new_ids, logits = inputs

    logits_encoding = self.encodings.as_prediction_encoding(logits)
    existing_ids = logits_encoding.enref_id.slice()

    is_new_id = tf.cast(is_new_entity > 0.0, tf.float32)
    is_new_id = tf.reduce_max(is_new_id, 2, keepdims=True)
    ids = is_new_id * new_ids
    ids += (1.0 - is_new_id) * existing_ids

    logits = logits_encoding.enref_id.replace(ids)
    logits = self.encodings.as_prediction_encoding(
        logits).enref_meta.replace_is_new_slice(is_new_entity)

    return logits


class ContrackModel(tf.keras.Model):
  """The Contrack model."""

  def __init__(self, mode, print_predictions = False):
    super(ContrackModel, self).__init__()

    self.config = Env.get().config
    self.encodings = Env.get().encodings
    self.mode = mode
    self.print_predictions = print_predictions
    self.teacher_forcing = True

    self.convert_to_sequence_layer = ConvertToSequenceLayer(
        self.encodings.enref_encoding_length)

    input_shape = [
        self.config.batch_size, self.config.max_seq_len,
        self.encodings.enref_encoding_length + self.config.timing_signal_size
    ]
    self.new_entity_layer = IdentifyNewEntityLayer(input_shape)

    self.compute_ids_layer = ComputeIdsLayer()

    self.track_enrefs_layer = TrackEnrefsLayer(input_shape)

    self.merge_ids_layer = MergeIdsLayer()

  @classmethod
  def from_config(cls, config):
    return ContrackModel(config['mode'])

  def get_config(self):
    return {'mode': self.mode}

  def init_weights_from_new_entity_model(self, model):
    # Call the model once to create weights
    input_vec_shape = [
        self.config.batch_size, self.config.max_seq_len,
        self.encodings.token_encoding_length
    ]
    null_input = {
        'state_seq_length': tf.ones([self.config.batch_size]),
        'state_seq': tf.zeros(input_vec_shape),
        'token_seq_length': tf.ones([self.config.batch_size]),
        'token_seq': tf.zeros(input_vec_shape)
    }
    self(null_input)

    # Then copy over layer weights
    self.new_entity_layer.set_weights(model.new_entity_layer.get_weights())

  def call(self,
           inputs,
           training = False):
    # Step 1: Concatenate input data into a single sequence
    seq, _ = self.convert_to_sequence_layer(inputs)

    # Step 2: Identify new entities.
    is_new_entity = self.new_entity_layer(seq)

    if self.mode == 'only_new_entities':
      res = tf.zeros_like(seq[:, :, :self.encodings.prediction_encoding_length])
      res_enc = self.encodings.as_prediction_encoding(res)
      res = res_enc.enref_meta.replace_is_new_slice(is_new_entity)
      return res
    elif self.mode == 'only_tracking':
      is_new_entity = tf.stop_gradient(is_new_entity)

    # Step 3: Compute enref ids for new entities
    new_ids = self.compute_ids_layer(
        (seq, inputs['state_seq_length'], is_new_entity))

    # Step 4: Determine enref predictions
    logits = self.track_enrefs_layer((seq, is_new_entity, new_ids))

    # Step 5: Merge ids from new and existing enrefs
    logits = self.merge_ids_layer((is_new_entity, new_ids, logits))

    return logits

  def train_step(self, data):
    """The training step."""
    x = data

    # Shift true labels seq to align with tokens in input_seq
    state_seq_len = tf.cast(data['state_seq_length'], tf.int32)
    token_seq_len = tf.cast(data['token_seq_length'], tf.int32)
    state_seq_dims = tf.shape(data['state_seq'])
    enref_padding = tf.zeros([
        state_seq_dims[0], state_seq_dims[1],
        self.encodings.prediction_encoding_length
    ])
    y, y_len = custom_ops.sequence_concat(
        sequences=[enref_padding, data['annotation_seq']],
        lengths=[state_seq_len, token_seq_len])

    # Clip and pad true labels seq
    y_len = tf.minimum(y_len, self.config.max_seq_len, name='y_len')
    y = _pad_and_clip_seq_batch(
        y,
        y_len,
        pad_value=0,
        maxlen=self.config.max_seq_len,
        data_vec_len=self.encodings.prediction_encoding_length)

    input_seq_len = tf.add(state_seq_len, token_seq_len)
    seq_mask = tf.sequence_mask(
        input_seq_len, maxlen=self.config.max_seq_len, dtype=tf.float32)
    enref_mask = tf.sequence_mask(
        state_seq_len, maxlen=self.config.max_seq_len, dtype=tf.float32)
    sample_weight = seq_mask - enref_mask

    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(
          y, y_pred, sample_weight, regularization_losses=self.losses)
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

  def predict_step(
      self, data):
    """The logic for one inference step."""
    if not self.teacher_forcing:
      data, y_pred = self.call_without_teacher_forcing(data)
    else:
      y_pred = self(data, training=False)

    x = {
        'state_seq_length': data['state_seq_length'],
        'token_seq_length': data['token_seq_length'],
        'scenario_id': data['scenario_id']
    }
    x['state_seq'] = _pad_and_clip_seq_batch(
        data['state_seq'],
        data['state_seq_length'],
        pad_value=0,
        maxlen=self.config.max_seq_len,
        data_vec_len=self.encodings.enref_encoding_length)
    x['token_seq'] = _pad_and_clip_seq_batch(
        data['token_seq'],
        data['token_seq_length'],
        pad_value=0,
        maxlen=self.config.max_seq_len,
        data_vec_len=self.encodings.token_encoding_length)
    x['word_seq'] = _pad_and_clip_seq_batch(
        tf.expand_dims(data['word_seq'], -1),
        data['token_seq_length'],
        pad_value='',
        maxlen=self.config.max_seq_len,
        data_vec_len=1)
    x['annotation_seq'] = _pad_and_clip_seq_batch(
        data['annotation_seq'],
        data['token_seq_length'],
        pad_value=0,
        maxlen=self.config.max_seq_len,
        data_vec_len=self.encodings.prediction_encoding_length)

    return (x, y_pred)

  def make_test_function(self):
    """Creates a function that executes one step of evaluation."""
    test_fn = super(ContrackModel, self).make_test_function()

    def adapted_test_fn(iterator):
      outputs = test_fn(iterator)
      if 'print_prediction' in outputs:
        pred_msgs = outputs['print_prediction']
        if self.print_predictions:
          logging.info(pred_msgs.numpy().decode('utf-8'))
        del outputs['print_prediction']
      return outputs

    self.test_function = adapted_test_fn
    return adapted_test_fn

  def print_prediction(self, seq_len, state_seq_len,
                       words, tokens,
                       predictions, true_targets):
    res = ''

    for batch_index, num_token in enumerate(seq_len.numpy()):
      res += '---------------------------------------\n'
      for i in range(num_token):
        word = words[batch_index, i].numpy().decode('utf-8')
        res += word + ': '

        seq_index = state_seq_len[batch_index] + i
        if seq_index >= self.config.max_seq_len:
          break

        true_target = self.encodings.as_prediction_encoding(
            true_targets[batch_index, seq_index, :].numpy())

        pred = self.encodings.as_prediction_encoding(
            predictions[batch_index, seq_index, :].numpy())

        if self.mode == 'only_new_entities':
          true_label = '%s%s' % (
              'n' if true_target.enref_meta.is_new() > 0.0 else '',
              'c' if true_target.enref_meta.is_new_continued() > 0.0 else '')
          predicted_label = '%s%s' % (
              'n' if pred.enref_meta.is_new() > 0.0 else '',
              'c' if pred.enref_meta.is_new_continued() > 0.0 else '')
          if true_label != predicted_label:
            res += '*** %s != %s' % (predicted_label, true_label)
            res += ' ' + str(pred.enref_meta.slice())
          else:
            res += predicted_label
        else:
          token = self.encodings.as_token_encoding(tokens[batch_index, i, :])
          true_enref = self.encodings.build_enref_from_prediction(
              token, true_target)
          pred_enref = self.encodings.build_enref_from_prediction(token, pred)
          if str(true_enref) != str(pred_enref):
            res += '*** %s != %s' % (str(pred_enref), str(true_enref))
            res += ' %s' % str([
                round(a, 2)
                for a in true_targets[batch_index, seq_index, :].numpy()
            ])
          else:
            res += str(pred_enref) if pred_enref is not None else ''

        res += '\n'

    return res

  def test_step(self, data):
    """The logic for one evaluation step."""
    x = data

    # Shift true labels seq to align with tokens in input_seq
    state_seq_len = tf.cast(data['state_seq_length'], tf.int32)
    token_seq_len = tf.cast(data['token_seq_length'], tf.int32)
    state_seq_dims = tf.shape(data['state_seq'])
    enref_padding = tf.zeros([
        state_seq_dims[0], state_seq_dims[1],
        self.encodings.prediction_encoding_length
    ])
    y, y_len = custom_ops.sequence_concat(
        sequences=[enref_padding, data['annotation_seq']],
        lengths=[state_seq_len, token_seq_len])

    # Clip and pad true labels seq
    y_len = tf.minimum(y_len, self.config.max_seq_len, name='y_len')
    y = _pad_and_clip_seq_batch(
        y,
        y_len,
        pad_value=0,
        maxlen=self.config.max_seq_len,
        data_vec_len=self.encodings.prediction_encoding_length)

    input_seq_len = tf.add(state_seq_len, token_seq_len)
    seq_mask = tf.sequence_mask(
        input_seq_len, maxlen=self.config.max_seq_len, dtype=tf.float32)
    enref_mask = tf.sequence_mask(
        state_seq_len, maxlen=self.config.max_seq_len, dtype=tf.float32)
    sample_weight = seq_mask - enref_mask

    y_pred = self(x, training=False)
    # Updates stateful loss metrics.
    self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)

    self.compiled_metrics.update_state(y, y_pred, sample_weight)

    # Print prediction to log
    output_tensors = {m.name: m.result() for m in self.metrics}

    print_prediction_fn = tf.py_function(self.print_prediction, [
        x['token_seq_length'], x['state_seq_length'], x['word_seq'],
        x['token_seq'], y_pred, y
    ], tf.string)

    output_tensors['print_prediction'] = print_prediction_fn
    return output_tensors

  def disable_teacher_forcing(self):
    self.teacher_forcing = False
    self.current_scenario = None
    self.current_enrefs = []
    self.current_participants = []
    assert self.config.batch_size == 1

  def call_without_teacher_forcing(self, data):
    scenario_id = data['scenario_id'][0].numpy().decode('utf-8')
    # logging.info(scenario_id)
    if scenario_id == self.current_scenario:
      # Continue existing conversations, create state_seq from enrefs
      enrefs = self.current_enrefs
      logging.info('Continue conversation with %d enrefs', len(enrefs))

      data['state_seq_length'] = tf.constant([len(enrefs)], dtype=tf.int64)
      sender = data['sender'][0].numpy().decode('utf-8')
      for enref in enrefs:
        entity_name = enref.entity_name
        enref.enref_context.set_is_sender(entity_name == sender)
        enref.enref_context.set_is_recipient(
            entity_name != sender and entity_name in self.current_participants)
        enref.enref_context.set_message_offset(
            enref.enref_context.get_message_offset() + 1)

      # for i, e in enumerate(enrefs):
      #   diff = e.array - data['state_seq'][0, i, :].numpy()
      #   if np.sum(np.abs(diff)) > 0.1:
      #     logging.info('diff for %s: %s', str(e), diff.tolist())

      enref_seq = [e.array for e in enrefs]
      data['state_seq'] = tf.constant([enref_seq], dtype=tf.float32)
    else:
      # Start new conversation, obtain initial enrefs from participants
      logging.info('New conversation, participants %s',
                   data['participants'].values)
      self.current_scenario = scenario_id
      self.current_participants = [
          p.numpy().decode('utf-8') for p in data['participants'].values
      ]
      self.current_enrefs = []
      for i in range(0, data['state_seq_length'][0].numpy()):
        enref_array = data['state_seq'][0, i].numpy()
        enref = self.encodings.as_enref_encoding(enref_array)
        enref_name = self.current_participants[i]
        enref.populate(enref_name, (i, i + 1), enref_name)
        self.current_enrefs.append(enref)
      # logging.info('Enrefs: %s', str(self.current_enrefs))

    # Run model
    y_pred = self(data, training=False)

    # Update set of enrefs from prediction
    num_tokens = len(data['word_seq'][0])
    num_enrefs = len(data['state_seq'][0])

    token_encs = [self.encodings.as_token_encoding(
        data['token_seq'][0, i, :].numpy()) for i in range(0, num_tokens)]
    pred_encs = [self.encodings.as_prediction_encoding(y_pred[0, i, :].numpy())
                 for i in range(num_enrefs, min(num_enrefs + num_tokens,
                                                self.config.max_seq_len))]
    words = [data['word_seq'][0, i].numpy().decode('utf-8')
             for i in range(0, num_tokens)]
    enrefs = self.encodings.build_enrefs_from_predictions(
        token_encs, pred_encs, words, self.current_enrefs)
    # logging.info('New Enrefs for %s: %s', words, enrefs)
    logging.info('%d new enrefs', len(enrefs))
    self.current_enrefs += enrefs

    return (data, y_pred)


class ContrackLoss(tf.keras.losses.Loss):
  """The loss function used for contrack training."""

  def __init__(self, mode):
    super(
        ContrackLoss,
        self).__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.encodings = Env.get().encodings
    self.config = Env.get().config
    self.mode = mode

  @classmethod
  def from_config(cls, config):
    return ContrackLoss(config['mode'])

  def get_config(self):
    return {'mode': self.mode}

  def _compute_hinge_losses(self, labels,
                            logits):
    """Computes hinge loss."""
    all_ones = tf.ones_like(labels)
    labels = tf.math.subtract(2 * labels, all_ones)
    losses = tf.nn.relu(
        tf.math.subtract(all_ones, tf.math.multiply(labels, logits)))
    if len(losses.get_shape()) > 2:
      losses = tf.math.reduce_sum(losses, [2])
    return losses

  def _compute_annotation_losses(self, target,
                                 predicted):
    """Compute loss comparing predicted and actual annotations."""
    target_enc = self.encodings.as_prediction_encoding(target)
    predicted_enc = self.encodings.as_prediction_encoding(predicted)

    # Membership loss only for groups
    loss = self._compute_hinge_losses(
        labels=target_enc.enref_membership.slice(),
        logits=predicted_enc.enref_membership.slice())
    loss *= target_enc.enref_properties.is_group()

    # Properties loss
    loss += self._compute_hinge_losses(
        labels=target_enc.enref_properties.slice(),
        logits=predicted_enc.enref_properties.slice())

    # Entity ID loss
    is_new = (
        target_enc.enref_meta.is_new() +
        target_enc.enref_meta.is_new_continued())

    existing_entity_loss = self._compute_hinge_losses(
        labels=target_enc.enref_id.slice(),
        logits=predicted_enc.enref_id.slice())
    existing_entity_loss *= 1.0 - is_new

    new_entity_loss = self._compute_hinge_losses(
        labels=target_enc.enref_meta.get_is_new_slice(),
        logits=predicted_enc.enref_meta.get_is_new_slice())
    new_entity_loss *= is_new
    fp_cost = self.config.new_id_false_negative_cost - 1.0
    new_entity_loss *= tf.ones_like(new_entity_loss) + fp_cost * is_new

    loss += existing_entity_loss + new_entity_loss

    # Is_entity loss
    not_an_entity_loss = self._compute_hinge_losses(
        labels=target_enc.enref_meta.slice(),
        logits=predicted_enc.enref_meta.slice())
    loss *= target_enc.enref_meta.is_enref()
    loss += not_an_entity_loss

    return loss

  def _compute_new_id_losses(self, target,
                             predicted):
    """Computes the new id losses for each token in each turn."""
    target_meta = self.encodings.as_prediction_encoding(target).enref_meta
    predicted_meta = self.encodings.as_prediction_encoding(predicted).enref_meta

    losses = self._compute_hinge_losses(
        labels=target_meta.get_is_new_slice(),
        logits=predicted_meta.get_is_new_slice())
    fn_cost = self.config.new_id_false_negative_cost - 1.0
    new_entity_positives = target_meta.is_new() + target_meta.is_new_continued()
    losses *= tf.ones_like(losses) + fn_cost * new_entity_positives
    return losses

  def call(self, target, predicted):
    """Compute loss comparing predicted and actual annotations."""
    with tf.name_scope('contrack_loss'):
      if self.mode == 'only_new_entities':
        return self._compute_new_id_losses(target, predicted)
      else:
        return self._compute_annotation_losses(target, predicted)


def _get_named_slices(y_true, logits,
                      section_name):
  """Returns the slices (given by name) of true and predictied vector."""
  is_entity = tf.expand_dims(y_true.enref_meta.is_enref(), 2)
  if section_name == 'new_entity':
    return (y_true.enref_meta.get_is_new_slice(),
            is_entity * logits.enref_meta.get_is_new_slice())
  elif section_name == 'entities':
    return (y_true.enref_id.slice(), is_entity * logits.enref_id.slice())
  elif section_name == 'properties':
    return (y_true.enref_properties.slice(),
            is_entity * logits.enref_properties.slice())
  elif section_name == 'membership':
    is_group = tf.expand_dims(y_true.enref_properties.is_group(), 2)
    return (y_true.enref_membership.slice(),
            is_entity * is_group * logits.enref_membership.slice())
  else:
    raise ValueError('Unknown section name %s' % section_name)


class ContrackAccuracy(tf.keras.metrics.Mean):
  """Computes zero-one accuracy on a given slice of the result vector."""

  def __init__(self, section_name, dtype=None):
    self.encodings = Env.get().encodings
    self.section_name = section_name
    super(ContrackAccuracy, self).__init__(
        name=f'{section_name}/accuracy', dtype=dtype)

  @classmethod
  def from_config(cls, config):
    return ContrackAccuracy(config['section_name'])

  def get_config(self):
    return {'section_name': self.section_name}

  def update_state(self,
                   y_true,
                   logits,
                   sample_weight = None):
    y_true, logits = _get_named_slices(
        self.encodings.as_prediction_encoding(y_true),
        self.encodings.as_prediction_encoding(logits), self.section_name)
    y_pred = tf.cast(logits > 0.0, tf.float32)

    matches = tf.reduce_max(tf.cast(y_true == y_pred, tf.float32), -1)

    super(ContrackAccuracy, self).update_state(matches, sample_weight)


class ContrackPrecision(tf.keras.metrics.Precision):
  """Computes precision on a given slice of the result vector."""

  def __init__(self, section_name, dtype=None):
    self.encodings = Env.get().encodings
    self.section_name = section_name
    super(ContrackPrecision, self).__init__(
        name=f'{section_name}/precision', dtype=dtype)

  @classmethod
  def from_config(cls, config):
    return ContrackPrecision(config['section_name'])

  def get_config(self):
    return {'section_name': self.section_name}

  def update_state(self,
                   y_true,
                   logits,
                   sample_weight = None):
    y_true, logits = _get_named_slices(
        self.encodings.as_prediction_encoding(y_true),
        self.encodings.as_prediction_encoding(logits), self.section_name)
    y_pred = tf.cast(logits > 0.0, tf.float32)

    super(ContrackPrecision, self).update_state(y_true, y_pred, sample_weight)


class ContrackRecall(tf.keras.metrics.Recall):
  """Computes recall on a given slice of the result vector."""

  def __init__(self, section_name, dtype=None):
    self.encodings = Env.get().encodings
    self.section_name = section_name
    super(ContrackRecall, self).__init__(
        name=f'{section_name}/recall', dtype=dtype)

  @classmethod
  def from_config(cls, config):
    return ContrackRecall(config['section_name'])

  def get_config(self):
    return {'section_name': self.section_name}

  def update_state(self,
                   y_true,
                   logits,
                   sample_weight = None):
    y_true, logits = _get_named_slices(
        self.encodings.as_prediction_encoding(y_true),
        self.encodings.as_prediction_encoding(logits), self.section_name)
    y_pred = tf.cast(logits > 0.0, tf.float32)

    super(ContrackRecall, self).update_state(y_true, y_pred, sample_weight)


class ContrackF1Score(tf.keras.metrics.Metric):
  """Computes the f1 score on a given slice of the result vector."""

  def __init__(self, section_name, dtype=None):
    self.encodings = Env.get().encodings
    self.section_name = section_name
    self.precision = ContrackPrecision(section_name, dtype=dtype)
    self.recall = ContrackRecall(section_name, dtype=dtype)
    super(ContrackF1Score, self).__init__(
        name=f'{section_name}/f1score', dtype=dtype)

  @classmethod
  def from_config(cls, config):
    return ContrackF1Score(config['section_name'])

  def get_config(self):
    return {'section_name': self.section_name}

  def add_weight(self, **kwargs):
    self.precision.add_weight(**kwargs)
    self.recall.add_weight(**kwargs)

  def reset_states(self):
    self.precision.reset_states()
    self.recall.reset_states()

  def result(self):
    precision = self.precision.result()
    recall = self.recall.result()
    return 2.0 * (precision * recall) / (precision + recall +
                                         tf.keras.backend.epsilon())

  def update_state(self,
                   y_true,
                   logits,
                   sample_weight = None):
    self.precision.update_state(y_true, logits, sample_weight)
    self.recall.update_state(y_true, logits, sample_weight)


def build_metrics(mode):
  """Creates list of metrics for all metric types and sections."""
  if mode == 'only_new_entities':
    sections = ['new_entity']
  else:
    sections = ['new_entity', 'entities', 'properties', 'membership']

  metrics = []
  for section in sections:
    metrics += [
        ContrackAccuracy(section),
        ContrackPrecision(section),
        ContrackRecall(section),
        ContrackF1Score(section)
    ]
  return metrics


def get_custom_objects():
  return {
      'ContrackModel': ContrackModel,
      'ConvertToSequenceLayer': ConvertToSequenceLayer,
      'IdentifyNewEntityLayer': IdentifyNewEntityLayer,
      'ComputeIdsLayer': ComputeIdsLayer,
      'TrackEnrefsLayer': TrackEnrefsLayer,
      'ContrackLoss': ContrackLoss,
      'ContrackAccuracy': ContrackAccuracy,
      'ContrackPrecision': ContrackPrecision,
      'ContrackRecall': ContrackRecall,
      'ContrackF1Score': ContrackF1Score,
  }
