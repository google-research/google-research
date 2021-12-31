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

"""Implements models to embed biological sequences as vector sequences."""

import functools
from typing import Optional, Tuple, Type

import gin
import tensorflow as tf

from dedal import vocabulary
from dedal.models import activations
from dedal.models import initializers
try:
  # pytype: disable=import-error
  from official.nlp.modeling import layers as nlp_layers  # pylint: disable=g-import-not-at-top
  # pytype: enable=import-error
except Exception:  # pylint: disable=broad-except
  pass


@gin.configurable
class Encoder(tf.keras.Model):
  """A generic sequence encoder."""

  def __init__(self,
               vocab = None,
               mask_special_tokens = True,
               trainable = True,
               **kwargs):
    super().__init__(trainable=trainable, **kwargs)
    self._vocab = vocabulary.get_default() if vocab is None else vocab
    self._mask_special_tokens = mask_special_tokens

  def compute_mask(self,
                   inputs,
                   mask = None):
    """Standard keras method."""
    del mask
    mask = self._vocab.padding_mask(inputs)
    if self._mask_special_tokens:
      mask = tf.math.logical_and(mask, self._vocab.special_token_mask(inputs))
    return mask


@gin.configurable
class OneHotEncoder(Encoder):
  """Embeds sequences using non-contextual, one-hot embeddings."""

  def call(self, inputs):
    return tf.one_hot(inputs, len(self._vocab))


@gin.configurable
class LookupEncoder(Encoder):
  """Encoder using a lookup."""

  def __init__(
      self,
      emb_dim = 768,
      dropout = 0.0,
      use_layer_norm = False,
      use_positional_embedding = False,
      position_embed_init=initializers.HarmonicEmbeddings(
          scale_factor=1e-4, max_freq=1.0),
      train_position_embed = True,
      aaemb_init=tf.initializers.TruncatedNormal(stddev=1.0),
      aaemb_scale_factor = None,
      max_len = 1024,
      **kwargs):
    super().__init__(**kwargs)
    self._use_layer_norm = use_layer_norm

    if use_positional_embedding:
      self._positional_embedding = nlp_layers.PositionEmbedding(
          max_length=max_len,
          initializer=position_embed_init,
          trainable=train_position_embed,
          name='embeddings/positional')
    else:
      self._positional_embedding = None

    self._aa_embed = nlp_layers.OnDeviceEmbedding(
        vocab_size=len(self._vocab),
        embedding_width=emb_dim,
        initializer=aaemb_init,
        scale_factor=aaemb_scale_factor,
        name='embeddings/aminoacid')

    if use_layer_norm:
      self._layer_norm = tf.keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, name='embeddings/layer_norm')
    else:
      self._layer_norm = None

    self._dropout = tf.keras.layers.Dropout(
        rate=dropout, name='embeddings/dropout')

  def call(self, inputs):
    embeddings = self._aa_embed(inputs)
    if self._positional_embedding is not None:
      pos_embeddings = self._positional_embedding(embeddings)
      embeddings += pos_embeddings
    if self._layer_norm is not None:
      embeddings = self._layer_norm(embeddings)
    embeddings = self._dropout(embeddings)
    return embeddings


@gin.configurable
class RecurrentEncoder(Encoder):
  """RNN based Encoder."""

  def __init__(
      self,
      emb_dim = 512,
      num_layers = 3,
      rnn_cls = tf.keras.layers.GRU,
      rnn_input_dropout = 0.0,
      rnn_recurrent_dropout = 0.0,
      causal = False,
      aaemb_init=tf.initializers.TruncatedNormal(stddev=1.0),
      kernel_init=tf.initializers.GlorotUniform(),
      recurrent_init=tf.initializers.Orthogonal(),
      aaemb_scale_factor = None,
      **kwargs):
    super().__init__(**kwargs)

    self._aaemb_layer = nlp_layers.OnDeviceEmbedding(
        vocab_size=len(self._vocab),
        embedding_width=emb_dim,
        initializer=aaemb_init,
        scale_factor=aaemb_scale_factor,
        name='embeddings/aminoacid')

    self._rnn_layers = []
    for i in range(num_layers):
      layer = rnn_cls(
          units=self.config.emb_dim,
          kernel_initializer=kernel_init,
          recurrent_initializer=recurrent_init,
          dropout=rnn_input_dropout,
          recurrent_dropout=rnn_recurrent_dropout,
          return_sequences=True,
          name=f'RNN/layer_{i}')
      if not causal:
        layer = tf.keras.layers.Bidirectional(layer, name=f'BiRNN/layer_{i}')
      self._rnn_layers.append(layer)

  def call(self, inputs):
    embeddings = self._aaemb_layer(inputs)
    mask = self._vocab.padding_mask(inputs)
    for layer in self._rnn_layers:
      embeddings = layer(embeddings, mask=mask)
    return embeddings


@gin.configurable
class TransformerEncoder(Encoder):
  """Encoder with a transformer."""

  def __init__(
      self,
      emb_dim = 768,
      num_layers = 6,
      num_heads = 12,
      mlp_dim = 3072,
      mlp_act=activations.approximate_gelu,
      output_dropout = 0.1,
      attention_dropout = 0.1,
      mlp_dropout = 0.1,
      norm_first = True,
      norm_input = False,
      norm_output = True,
      causal = False,
      trainable_posemb = False,
      posemb_init=initializers.HarmonicEmbeddings(
          scale_factor=1e-4, max_freq=1.0),
      aaemb_init=tf.initializers.RandomNormal(stddev=1.0),
      kernel_init=tf.initializers.GlorotUniform(),
      aaemb_scale_factor = None,
      max_len = 1024,
      **kwargs):
    super().__init__(**kwargs)
    self._causal = causal
    self.posemb_layer = nlp_layers.PositionEmbedding(
        max_length=max_len,
        initializer=posemb_init,
        trainable=trainable_posemb,
        name='embeddings/positional')
    self.aaemb_layer = nlp_layers.OnDeviceEmbedding(
        vocab_size=len(self._vocab),
        embedding_width=emb_dim,
        initializer=aaemb_init,
        scale_factor=aaemb_scale_factor,
        name='embeddings/aminoacid')
    layer_norm_cls = functools.partial(
        tf.keras.layers.LayerNormalization, axis=-1, epsilon=1e-12)
    self._input_norm_layer = (
        layer_norm_cls(name='embeddings/layer_norm') if norm_input else None)
    self._output_norm_layer = (
        layer_norm_cls(name='output/layer_norm') if norm_output else None)
    self._dropout_layer = tf.keras.layers.Dropout(
        rate=output_dropout, name='embeddings/dropout')
    self._attention_mask = nlp_layers.SelfAttentionMask()
    self._transformer_layers = []
    for i in range(num_layers):
      self._transformer_layers.append(nlp_layers.TransformerEncoderBlock(
          num_attention_heads=num_heads,
          inner_dim=mlp_dim,
          inner_activation=mlp_act,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          inner_dropout=mlp_dropout,
          kernel_initializer=kernel_init,
          norm_first=norm_first,
          name=f'transformer/layer_{i}'))

  def call(self, inputs):
    aa_embeddings = self.aaemb_layer(inputs)
    pos_embeddings = self.posemb_layer(aa_embeddings)
    embeddings = aa_embeddings + pos_embeddings
    if self._input_norm_layer is not None:
      embeddings = self._input_norm_layer(embeddings)  # pylint: disable=not-callable
    embeddings = self._dropout_layer(embeddings)

    mask = self._vocab.padding_mask(inputs)
    attention_mask = self._attention_mask(
        embeddings, tf.cast(mask, embeddings.dtype))
    if self._causal:
      attention_shape = tf.shape(attention_mask)
      len1, len2 = attention_shape[1], attention_shape[2]
      causal_mask = tf.range(len1)[:, None] >= tf.range(len2)[None, :]
      causal_mask = tf.cast(tf.expand_dims(causal_mask, 0), embeddings.dtype)
      attention_mask *= causal_mask

    for layer in self._transformer_layers:
      embeddings = layer((embeddings, attention_mask))

    if self._output_norm_layer is not None:
      embeddings = self._output_norm_layer(embeddings)

    return embeddings
