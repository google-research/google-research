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

# coding=utf-8
"""Defines the different models."""

import os
import sys

from etcmodel.layers.embedding import EmbeddingLookup
from etcmodel.layers.transformer import RelativeTransformerLayers
from modules import datasets
from official.modeling import tf_utils
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_tutorial.transformer import positional_encoding
from tensorflow_tutorial.transformer import Transformer

# For third_party dependency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party'))


def build_lstm_model(hparams):
  """Defines LSTM model."""
  model = tf.keras.Sequential()

  # Embedding layer
  model.add(
      layers.Embedding(
          input_dim=hparams.vocab_size,
          output_dim=hparams.hidden_size,
          mask_zero=True))

  # Stacked LSTM
  for i in range(hparams.num_hidden_layers):
    return_sequences = i < hparams.num_hidden_layers - 1
    model.add(
        layers.LSTM(
            hparams.hidden_size,
            dropout=hparams.dropout,
            return_sequences=return_sequences,
            kernel_initializer=hparams.initializer))

  # Dense layers
  for i in range(hparams.num_dense_layers):
    dense_dim = hparams.hidden_size if i < hparams.num_dense_layers - 1 else 1
    model.add(layers.Dense(dense_dim, kernel_initializer=hparams.initializer))
  model.add(layers.Reshape(()))

  return model


def build_transformer_model(hparams):
  """Defines Transformer model."""
  model = Transformer(hparams.num_encoder_layers, hparams.hidden_size,
                      hparams.num_heads, hparams.filter_size,
                      hparams.vocab_size, hparams.pad_index, hparams.sep_index,
                      hparams.maximum_position_encoding, hparams.dropout)
  return model


class SinusoidalPositionEmbedding(tf.keras.layers.Layer):
  """Fixed sinusoidal positioanl embeddings."""

  def __init__(self, maximum_position_encoding, embedding_size):
    super(SinusoidalPositionEmbedding, self).__init__()

    self.maximum_position_encoding = maximum_position_encoding
    self.embedding_size = embedding_size

    # Build non-trainable table to lookup
    self.embedding_table = positional_encoding(maximum_position_encoding,
                                               embedding_size)[0]

  def call(self,
           input_ids,
           input_mask=None):
    """Calls the layer."""
    if input_mask is not None:
      input_ids *= input_mask

    output = tf.nn.embedding_lookup(self.embedding_table, input_ids)

    if input_mask is not None:
      output *= tf.expand_dims(tf.cast(input_mask, dtype=output.dtype), axis=-1)

    return output


class RelativeTransformerEncoder(tf.keras.layers.Layer):
  """Relative Transformer encoder with embeddings."""

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               hidden_act,
               input_vocab_size,
               relative_vocab_size,
               maximum_position_encoding,
               restart_query_pos,
               unique_structure_token_pos,
               learned_position_encoding,
               share_pos_embed,
               rate=0.1,
               rate2=0.0,
               initializer_range=0.1):
    super(RelativeTransformerEncoder, self).__init__()

    self.relative_vocab_size = relative_vocab_size
    self.restart_query_pos = restart_query_pos
    self.unique_structure_token_pos = unique_structure_token_pos
    self.learned_position_encoding = learned_position_encoding
    self.share_pos_embed = share_pos_embed

    # Token embeddings
    self.embedding = layers.Embedding(input_vocab_size, d_model, mask_zero=True)

    # Segment embeddings
    self.seg_embedding = layers.Embedding(datasets.SEGMENT_VOCAB_SIZE, d_model)

    # Positional embeddings (question / query / additional)
    if learned_position_encoding:
      embedding = EmbeddingLookup
    else:
      embedding = SinusoidalPositionEmbedding

    self.question_pos_embedding = embedding(maximum_position_encoding, d_model)
    if restart_query_pos:
      if share_pos_embed:
        self.query_pos_embedding = self.question_pos_embedding
      else:
        self.query_pos_embedding = embedding(maximum_position_encoding, d_model)
    else:
      self.query_pos_embedding = None
    if unique_structure_token_pos:
      # We use only learned positional embeddings for structure tokens
      self.additional_pos_embedding = EmbeddingLookup(
          vocab_size=maximum_position_encoding, embedding_size=d_model)
    else:
      self.additional_pos_embedding = None

    self.relative_transformer_layer = RelativeTransformerLayers(
        hidden_size=d_model,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=dff,
        hidden_act=tf_utils.get_activation(hidden_act),
        hidden_dropout_prob=rate,
        attention_probs_dropout_prob=rate2,
        initializer_range=initializer_range,
        relative_vocab_size=relative_vocab_size,
        use_pre_activation_order=True)

  def call(self,
           inputs,
           training=False):
    """Calls the layer."""
    if not isinstance(inputs, dict):
      raise ValueError('Use parse tree input')

    # Token embedding
    token_embed = self.embedding(inputs['token_ids'])

    # Segment embedding
    seg_embed = self.seg_embedding(inputs['segment_ids'])

    # Positional embedding
    pos_embed = self.question_pos_embedding(inputs['position_ids'],
                                            inputs['question_mask'])
    if self.restart_query_pos:
      pos_embed += self.query_pos_embedding(inputs['position_ids'],
                                            inputs['query_mask'])
    if self.unique_structure_token_pos:
      pos_embed += self.additional_pos_embedding(inputs['position_ids'],
                                                 inputs['additional_mask'])

    x = token_embed + seg_embed + pos_embed

    # Don't pass relative_att_ids if relative vocab is not defined
    if self.relative_vocab_size is None:
      relative_att_ids = None
    else:
      relative_att_ids = inputs['relative_att_ids']

    # Transformer layers
    x = self.relative_transformer_layer(
        inputs=x,
        att_mask=inputs['attention_mask'],
        relative_att_ids=relative_att_ids,
        training=training)

    return x


class RelativeTransformer(tf.keras.Model):
  """Relative Transfomer for CFQ binary classification."""

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               hidden_act,
               input_vocab_size,
               relative_vocab_size,
               pe_input,
               restart_query_pos=True,
               unique_structure_token_pos=True,
               learned_position_encoding=True,
               share_pos_embed=False,
               rate=0.1,
               rate2=0.0,
               initializer_range=0.1):
    super(RelativeTransformer, self).__init__()

    self.encoder = RelativeTransformerEncoder(
        num_layers, d_model, num_heads, dff, hidden_act, input_vocab_size,
        relative_vocab_size, pe_input, restart_query_pos,
        unique_structure_token_pos, learned_position_encoding, share_pos_embed,
        rate, rate2, initializer_range)
    self.final_layer = layers.Dense(1)
    self.reshape_layer = layers.Reshape(())

  def call(self, inputs, training=False):
    # (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(inputs, training)

    # (batch_size, 1)
    final_output = self.final_layer(enc_output[:, 0, :])

    # (batch_size)
    final_output = self.reshape_layer(final_output)

    return final_output


def build_relative_transformer_model(
    hparams):
  """Defines relative transformer model."""
  # Relative attention is only used in soft mask
  if hparams.use_relative_attention:
    relative_vocab_size = datasets.RELATIVE_VOCAB_SIZE
  else:
    relative_vocab_size = None

  model = RelativeTransformer(
      hparams.num_encoder_layers, hparams.hidden_size, hparams.num_heads,
      hparams.filter_size, hparams.hidden_act, hparams.vocab_size,
      relative_vocab_size, hparams.maximum_position_encoding,
      hparams.restart_query_pos, hparams.unique_structure_token_pos,
      hparams.learned_position_encoding, hparams.share_pos_embed,
      hparams.dropout, hparams.attn_dropout, hparams.initializer_range)
  return model
