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

"""Felix tagger."""

import math
from typing import List, Sequence, Union

import numpy as np
from official.modeling import activations
from official.nlp.bert import configs
from official.nlp.modeling import layers
import tensorflow as tf


class FelixTagger(tf.keras.Model):
  """Felix tagger model based on a BERT-style transformer-based encoder.

   It adds a edit tagger (classification) and optionally a pointer network
   (self attention) to a BERT encoder.
  """

  def __init__(self,
               network,
               bert_config,
               initializer = 'glorot_uniform',
               seq_length = 128,
               use_pointing = True,
               is_training = True):
    """Creates Felix Tagger.

    Setting up all of the layers needed for call.

    Args:
      network: An encoder network, which should output a sequence of hidden
               states.
      bert_config: A config file which in addition to the  BertConfig values
      also includes: num_classes, hidden_dropout_prob, and query_transformer.
      initializer: The initializer (if any) to use in the classification
                   networks. Defaults to a Glorot uniform initializer.
      seq_length:  Maximum sequence length.
      use_pointing: Whether a pointing network is used.
      is_training: The model is being trained.
    """

    super(FelixTagger, self).__init__()
    self._network = network
    self._seq_length = seq_length
    self._bert_config = bert_config
    self._use_pointing = use_pointing
    self._is_training = is_training

    self._tag_logits_layer = tf.keras.layers.Dense(
        self._bert_config.num_classes)
    if not self._use_pointing:
      return

    # An arbitrary heuristic (sqrt vocab size) for the tag embedding dimension.
    self._tag_embedding_layer = tf.keras.layers.Embedding(
        self._bert_config.num_classes,
        int(math.ceil(math.sqrt(self._bert_config.num_classes))),
        input_length=seq_length)

    self._position_embedding_layer = layers.PositionEmbedding(
        max_length=seq_length)
    self._edit_tagged_sequence_output_layer = tf.keras.layers.Dense(
        self._bert_config.hidden_size, activation=activations.gelu)

    if self._bert_config.query_transformer:
      self._self_attention_mask_layer = layers.SelfAttentionMask()
      self._transformer_query_layer = layers.TransformerEncoderBlock(
          num_attention_heads=self._bert_config.num_attention_heads,
          inner_dim=self._bert_config.intermediate_size,
          inner_activation=activations.gelu,
          output_dropout=self._bert_config.hidden_dropout_prob,
          attention_dropout=self._bert_config.hidden_dropout_prob,
          output_range=seq_length,
      )

    self._query_embeddings_layer = tf.keras.layers.Dense(
        self._bert_config.query_size)

    self._key_embeddings_layer = tf.keras.layers.Dense(
        self._bert_config.query_size)

  def _attention_scores(self, query, key, mask=None):
    """Calculates attention scores as a query-key dot product.

    Args:
      query: Query tensor of shape `[batch_size, sequence_length, Tq]`.
      key: Key tensor of shape `[batch_size, sequence_length, Tv]`.
      mask: mask tensor of shape `[batch_size, sequence_length]`.

    Returns:
      Tensor of shape `[batch_size, sequence_length, sequence_length]`.
    """
    scores = tf.linalg.matmul(query, key, transpose_b=True)

    if mask is not None:
      mask = layers.SelfAttentionMask()(scores, mask)
      # Prevent pointing to self (zeros down the diagonal).
      diagonal_mask = tf.linalg.diag(
          tf.zeros((tf.shape(mask)[0], self._seq_length)), padding_value=1)
      diagonal_mask = tf.cast(diagonal_mask, tf.float32)
      mask = tf.math.multiply(diagonal_mask, mask)
      # As this is pre softmax (exp) as such we set the values very low.
      mask_add = -1e9 * (1. - mask)
      scores = scores * mask + mask_add

    return scores

  def call(self,
           inputs,
           ):
    """Forward pass of the model.

    Args:
      inputs:
        A list of tensors. In training the following 4 tensors are required,
        [input_word_ids, input_mask, input_type_ids,edit_tags].Only the first 3
        are required in test. input_word_ids[batch_size, seq_length],
        input_mask[batch_size, seq_length], input_type_ids[batch_size,
        seq_length], edit_tags[batch_size, seq_length]. If using output
        variants, these should also be provided. output_variant_ids[batch_size,
        1].

    Returns:
      The logits of the edit tags and optionally the logits of the pointer
        network.
    """
    if self._is_training:
      input_word_ids, input_mask, input_type_ids, edit_tags = inputs
    else:
      input_word_ids, input_mask, input_type_ids = inputs
    bert_output = self._network([input_word_ids, input_mask, input_type_ids])[0]
    tag_logits = self._tag_logits_layer(bert_output)

    if not self._use_pointing:
      return [tag_logits]

    if not self._is_training:
      edit_tags = tf.argmax(tag_logits, axis=-1)

    tag_embedding = self._tag_embedding_layer(edit_tags)
    position_embedding = self._position_embedding_layer(tag_embedding)
    edit_tagged_sequence_output = self._edit_tagged_sequence_output_layer(
        tf.keras.layers.concatenate(
            [bert_output, tag_embedding, position_embedding]))

    intermediate_query_embeddings = edit_tagged_sequence_output
    if self._bert_config.query_transformer:
      attention_mask = self._self_attention_mask_layer(
          intermediate_query_embeddings, input_mask)
      for _ in range(int(self._bert_config.query_transformer)):
        intermediate_query_embeddings = self._transformer_query_layer(
            [intermediate_query_embeddings, attention_mask])

    query_embeddings = self._query_embeddings_layer(
        intermediate_query_embeddings)

    key_embeddings = self._key_embeddings_layer(edit_tagged_sequence_output)

    pointing_logits = self._attention_scores(query_embeddings, key_embeddings,
                                             tf.cast(input_mask, tf.float32))
    return [tag_logits, pointing_logits]
