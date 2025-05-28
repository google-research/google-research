# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Transformer model."""

import dataclasses

from typing import Optional, Tuple, Callable

import tensorflow.compat.v1 as tf

LARGE_NUMBER = 1e+8

TwoTensors = Tuple[tf.Tensor, tf.Tensor]
ThreeTensors = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
ActivationFn = Callable[[tf.Tensor], tf.Tensor]


@dataclasses.dataclass
class TransformerParams:
  """Transformer model parameters.

  Attributes:
     query_key_dim: the dimension of the query/key
     internal_dim: dimension of the embedding in pointwise layer
     num_layers: number of transformer layers
     value_dim: the dimension of the value embedding (defaults to query_key_dim)
     mha_output_dim: the dimension of the final output of pointwise layer
     heads: number of heads in multi-head attention value_dim and query_key_dim
       need to be divisible by
     dropout_rate: dropout applied to the output of each transformer block
     activation_fn: activation to use in feed forward blocks.
     attention_activation_fn: activation function to use in the attention
       module (default is softmax).
  """

  query_key_dim: int
  internal_dim: int
  num_layers: int
  value_dim: Optional[int] = None
  mha_output_dim: Optional[int] = None
  heads: int = 1
  dropout_rate: float = 0.1
  activation_fn: ActivationFn = tf.nn.relu
  attention_activation_fn: ActivationFn = tf.nn.softmax

  def __post_init__(self):
    if self.value_dim is None:
      self.value_dim = self.query_key_dim
    if self.mha_output_dim is None:
      self.mha_output_dim = self.value_dim
    assert self.value_dim % self.heads == 0
    assert self.query_key_dim % self.heads == 0


def attention(q,
              k,
              v,
              mask = None,
              act_fn = tf.nn.softmax):
  """Simple attention module.

  Args:
    q: batch x head x seq x embedding
    k: same dimension as q.
    v: batch x head x seq x key_embedding
    mask: which examples in seq to ignore
    act_fn: Activation function to use for attention (defaults to softmax).
  Returns:
    batch x head x patch x embedding
  """
  attention_product = tf.matmul(q, k, transpose_b=True)
  key_dim = tf.cast(tf.shape(k)[-1], tf.float32)
  attention_logits = attention_product / tf.math.sqrt(key_dim)

  if mask is not None:
    attention_logits -= mask * LARGE_NUMBER

  attention_weights = act_fn(attention_logits)
  return tf.matmul(attention_weights, v), attention_weights


class PWFeedForward(tf.Module):
  """Pointwise feedforward layer."""

  def __init__(self,
               dim,
               internal_dim,
               name = None,
               activation=tf.nn.relu):
    super(PWFeedForward, self).__init__(name=name)
    self.layer_1 = tf.layers.Dense(
        internal_dim, activation=activation, name='layer_1')
    self.layer_2 = tf.layers.Dense(dim, name='layer_2')

  def __call__(self, input_tensor):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      return self.layer_2(self.layer_1(input_tensor))


class MultiHeadAttention(tf.Module):
  """Multi-head attention layer."""

  def __init__(self,
               params,
               name = None):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = params.heads
    self.qk_depth = params.query_key_dim // params.heads
    self.v_depth = params.value_dim // params.heads
    self.v_dim = params.value_dim

    self.w_q = tf.layers.Dense(params.query_key_dim, name='q')
    self.w_k = tf.layers.Dense(params.query_key_dim, name='k')
    self.w_v = tf.layers.Dense(params.value_dim, name='v')
    self.dense = tf.layers.Dense(params.mha_output_dim, name='fc')
    self.attn_act_fn = params.attention_activation_fn

  def _split_heads(self, x, batch_size,
                   depth):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def __call__(self,
               v,
               k,
               q,
               mask = None):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      batch_size = tf.shape(q)[0]
      q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
      q = self._split_heads(q, batch_size, self.qk_depth)
      k = self._split_heads(k, batch_size, self.qk_depth)
      v = self._split_heads(v, batch_size, self.v_depth)
      scaled_attention, attention_weights = attention(q, k, v, mask,
                                                      act_fn=self.attn_act_fn)
      scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
      concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.v_dim))

      return self.dense(concat_attention), attention_weights


class EncoderLayer(tf.Module):
  """Encoder layer."""

  def __init__(self,
               params,
               name = None,
               **kwargs):
    super(EncoderLayer, self).__init__(name=name)
    self.mha = MultiHeadAttention(params, name='attention')
    self.ffn = None
    if params.internal_dim > 0:
      self.ffn = PWFeedForward(
          dim=params.mha_output_dim,
          internal_dim=params.internal_dim,
          activation=params.activation_fn,
          name='fc')

    self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout_1 = tf.layers.Dropout(params.dropout_rate)
    self.dropout_2 = tf.layers.Dropout(params.dropout_rate)

  def __call__(self,
               x,
               is_training = True,
               mask = None):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      attn_output, attention_weights = self.mha(x, x, x, mask)
      self.attention_weights = attention_weights
      attn_output = self.dropout_1(attn_output, training=is_training)
      out_1 = self.layer_norm_1(x + attn_output)
      if self.ffn is None:
        return out_1
      ffn_output = self.ffn(out_1)
      ffn_output = self.dropout_2(ffn_output, training=is_training)
      return self.layer_norm_2(out_1 + ffn_output)


class DecoderLayer(tf.Module):
  """Decoder layer."""

  def __init__(self, params, name = None):
    super(DecoderLayer, self).__init__(name=name)

    self.mha_1 = MultiHeadAttention(params, name='attention_1')
    self.mha_2 = MultiHeadAttention(params, name='attention_2')
    self.ffn = None
    if params.internal_dim > 0:
      self.ffn = PWFeedForward(
          dim=params.mha_output_dim, internal_dim=params.internal_dim,
          name='fc')

    self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout_1 = tf.layers.Dropout(params.dropout_rate)
    self.dropout_2 = tf.layers.Dropout(params.dropout_rate)
    self.dropout_3 = tf.layers.Dropout(params.dropout_rate)

  def __call__(self,
               x,
               enc_output,
               is_training = True,
               look_ahead_mask = None,
               padding_mask = None):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      attn_1, attn_weights_block_1 = self.mha_1(x, x, x, look_ahead_mask)
      attn_1 = self.dropout_1(attn_1, training=is_training)
      out_1 = self.layer_norm_1(attn_1 + x)
      attn_2, attn_weights_block_2 = self.mha_2(enc_output, enc_output, out_1,
                                                padding_mask)
      attn_2 = self.dropout_2(attn_2, training=is_training)
      out_2 = self.layer_norm_2(attn_2 + out_1)
      if self.ffn is not None:
        ffn_output = self.ffn(out_2)
        ffn_output = self.dropout_3(ffn_output, training=is_training)
        out_3 = self.layer_norm_3(ffn_output + out_2)
      else:
        out_3 = out_2
    return out_3, attn_weights_block_1, attn_weights_block_2


class Encoder(tf.Module):
  """Transformer encoder."""

  def __init__(self,
               params,
               layer_dropout_prob = 0.0,
               skip_last_nonlinearity = False,
               name = None):
    super(Encoder, self).__init__(name=name)
    self.num_layers = params.num_layers
    self.enc_layers = [
        EncoderLayer(params, name=f'layer_{i+1}')
        for i in range(params.num_layers - 1)
    ]

    if skip_last_nonlinearity:
      params = dataclasses.replace(params, activation_fn=None)  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
    self.enc_layers.append(EncoderLayer(
        params, name=f'layer_{params.num_layers}'))

    self.layer_dropout_prob = layer_dropout_prob

  def __call__(self,
               x,
               is_training,
               mask = None):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      dropout_prob = self.layer_dropout_prob if is_training else 0
      for i in range(self.num_layers):
        select = tf.cast(tf.random_uniform((), 0, 1) > dropout_prob, tf.float32)
        x = self.enc_layers[i](x, is_training, mask) * select + x * (1 - select)
    return x


class Decoder(tf.Module):
  """Transformer decoder."""

  def __init__(self,
               params,
               skip_last_nonlinearity = False,
               name = None):
    super(Decoder, self).__init__(name=name)
    self.num_layers = params.num_layers
    self.dec_layers = [
        DecoderLayer(params, name=f'layer_{i+1}')
        for i in range(params.num_layers - 1)
    ]
    if skip_last_nonlinearity:
      params = dataclasses.replace(params, activation_fn=None)  # pytype: disable=wrong-arg-types  # dataclasses-replace-types
    self.dec_layers.append(DecoderLayer(
        params, name=f'layer_{params.num_layers}'))

  def __call__(self,
               x,
               enc_output,
               is_training,
               look_ahead_mask = None,
               padding_mask = None):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      for i in range(self.num_layers):
        x, _, _ = self.dec_layers[i](x, enc_output, is_training,
                                     look_ahead_mask, padding_mask)
    return x


class EncoderDecoderModel(tf.Module):
  """Transformer model."""

  def __init__(self,
               params,
               skip_last_nonlinearity = False,
               name = None):
    super(EncoderDecoderModel, self).__init__(name=name)
    self.encoder = Encoder(params, name='encoder')
    self.decoder = Decoder(params, name='decoder',
                           skip_last_nonlinearity=skip_last_nonlinearity)

  def __call__(self,
               sequence,
               mask = None,
               is_training = True):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      single_sequence = (len(sequence.shape) == 2)
      if single_sequence:
        sequence = tf.expand_dims(sequence, axis=0)
      encoding = self.encoder(sequence, is_training=is_training,
                              mask=mask)
      output = self.decoder(sequence, encoding, is_training=is_training,
                            look_ahead_mask=mask, padding_mask=mask)
      if single_sequence:
        output = tf.squeeze(output, axis=0)
    return output


class EncoderModel(tf.Module):
  """Transformer model."""

  def __init__(self,
               params,
               skip_last_nonlinearity = False,
               name = None):
    super(EncoderModel, self).__init__(name=name)
    self.encoder = Encoder(params, name='encoder',
                           skip_last_nonlinearity=skip_last_nonlinearity)

  def __call__(self,
               sequence,
               mask = None,
               is_training = True):
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      single_sequence = (len(sequence.shape) == 2)
      if single_sequence:
        sequence = tf.expand_dims(sequence, axis=0)
      output = self.encoder(sequence, mask=mask, is_training=is_training)
      if single_sequence:
        output = tf.squeeze(output, axis=0)
    return output


class SeparateEncoderDecoderModel(tf.Module):
  """Model using encoder for samples and decoder for weights."""

  def __init__(self,
               encoder_params,
               decoder_params,
               skip_last_nonlinearity = False,
               name = None):
    super(SeparateEncoderDecoderModel, self).__init__(name=name)
    self.encoder = Encoder(encoder_params, name='encoder')
    self.decoder = Decoder(decoder_params, name='decoder',
                           skip_last_nonlinearity=skip_last_nonlinearity)

  def __call__(self,
               sample_sequence,
               weight_sequence,
               mask = None,
               is_training = True):
    assert mask is None
    with self.name_scope, tf.variable_scope(None, default_name=self.name):
      single_sequence = (len(sample_sequence.shape) == 2)
      if single_sequence:
        sample_sequence = tf.expand_dims(sample_sequence, axis=0)
        weight_sequence = tf.expand_dims(weight_sequence, axis=0)
      encoding = self.encoder(sample_sequence, is_training=is_training)
      output = self.decoder(weight_sequence, encoding, is_training=is_training)
      if single_sequence:
        output = tf.squeeze(output, axis=0)
    return output
