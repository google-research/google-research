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

"""The main ReadItTwiceBert model and related functions."""

from typing import Optional, Text

import dataclasses
import tensorflow.compat.v1 as tf

from readtwice import layers as readtwice_layers
from readtwice.layers import tensor_utils
from readtwice.layers import tpu_utils
from readtwice.models import config as model_config


@dataclasses.dataclass(frozen=True)
class Summary:
  """Summary of block representations.

    states: <float32>[num_summaries, hidden_size]
    processed_states: <float32>[num_summaries, hidden_size]
    block_ids: <int32>[num_summaries, hidden_size]
    block_pos: <int32>[num_summaries, hidden_size]
    labels: <float32>[num_summaries, hidden_size]
  """
  states: tf.Tensor
  processed_states: Optional[tf.Tensor]
  block_ids: tf.Tensor
  block_pos: tf.Tensor
  labels: tf.Tensor


@dataclasses.dataclass(frozen=True)
class SummaryExtractionOutput:
  """Outputs of SummaryExtraction layer.

    local_summary: Summary
    global_summary: Summary
    global_summary_to_token_att_map: <int32>[batch_size, num_summaries]
  """
  local_summary: Summary
  global_summary: Summary
  token_to_global_summary_att_map: tf.Tensor


@dataclasses.dataclass(frozen=True)
class ReadItTwiceBertModelOutput:
  """Outputs of ReadItTwiceBertModel layer.

    final_hidden_states: <float32>[batch_size, seq_length, hidden_size]
    summary_states: <float32>[batch_size, hidden_size]
    processed_summary_states: <float32>[global_batch_size, hidden_size]
    block_ids: <int32>[batch_size]
    block_pos: <int32>[batch_size]
    global_summary_states: <float32>[global_batch_size, hidden_size]
    global_block_ids <int32>[global_batch_size]
    global_block_pos <int32>[global_batch_size]
  """
  final_hidden_states: tf.Tensor
  local_summary: Summary
  global_summary: Summary
  first_read: Optional[tf.Tensor] = None
  token_to_global_summary_att_map: Optional[tf.Tensor] = None


class ReadItTwiceBertModel(tf.keras.layers.Layer):
  """ReadItTwice BERT model."""

  def __init__(self,
               config,
               use_one_hot_embeddings=False,
               name = "read_it_twice_bert",
               **kwargs):
    """Constructor for ReadItTwiceBertModel.

    Args:
      config: `model_config.ReadItTwiceBertConfig` instance.
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.nn.embedding_lookup() for the word embeddings.
      name: (Optional) name of the layer.
      **kwargs: Forwarded to super.

    Raises:
      ValueError: The config is invalid.
    """
    super(ReadItTwiceBertModel, self).__init__(name=name, **kwargs)

    self.use_one_hot_embeddings = use_one_hot_embeddings

    if config.cross_attention_top_k is not None:
      assert config.second_read_type == "cross_attend_once"

    if config.embedding_size is None:
      config = dataclasses.replace(config, embedding_size=config.hidden_size)

    self.config = config

    self.token_embedding = readtwice_layers.EmbeddingLookup(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        projection_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=use_one_hot_embeddings,
        name="token_emb_lookup")

    self.token_embedding_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name="emb_layer_norm")
    self.token_embedding_dropout = tf.keras.layers.Dropout(
        rate=config.hidden_dropout_prob)

    self.position_embedding = readtwice_layers.EmbeddingLookup(
        vocab_size=config.max_seq_length,
        embedding_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=use_one_hot_embeddings,
        name="position_emb_lookup_long")
    # Call layers to force variable initialization.
    self.position_embedding(tf.ones([1, 1], tf.int32))

    if config.cross_attention_pos_emb_mode is not None:
      # We would end up adding block position embeddings multiple times.
      assert config.summary_postprocessing_type not in ["pos", "transformer"]

    if config.second_read_type == "from_scratch":
      share_kv_projections_first_read = config.share_kv_projections
    else:
      # Summaries are not going to be used by the first read model anyway.
      share_kv_projections_first_read = True

    self.transformer_with_side_inputs = readtwice_layers.TransformerWithSideInputLayers(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=tensor_utils.get_activation(config.hidden_act),
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        initializer_range=config.initializer_range,
        share_kv_projections=share_kv_projections_first_read,
        name="transformer_layers")
    # grad_checkpointing_period=config.grad_checkpointing_period)

    self.summary_extraction = SummaryExtraction(
        config=config, use_one_hot_embeddings=use_one_hot_embeddings)

    if config.second_read_type == "new_layers":
      if config.second_read_num_new_layers is None:
        raise ValueError("Must specify `second_read_num_new_layers`"
                         "when `second_read_type` is new_layers")

      self.second_read_transformer = readtwice_layers.TransformerWithSideInputLayers(
          hidden_size=config.hidden_size,
          num_hidden_layers=config.second_read_num_new_layers,
          num_attention_heads=config.num_attention_heads,
          intermediate_size=config.intermediate_size,
          hidden_act=tensor_utils.get_activation(config.hidden_act),
          hidden_dropout_prob=config.hidden_dropout_prob,
          attention_probs_dropout_prob=config.attention_probs_dropout_prob,
          initializer_range=config.initializer_range,
          share_kv_projections=config.share_kv_projections,
          name="transformer_layers")
    elif config.second_read_type == "cross_attend_once":
      if config.second_read_num_new_layers is None:
        raise ValueError("Must specify `second_read_num_new_layers`"
                         "when `second_read_type` is cross_attend_once")
      if config.second_read_num_cross_attention_heads is None:
        raise ValueError("Must specify `second_read_num_cross_attention_heads`"
                         "when `second_read_type` is cross_attend_once")
      if config.second_read_enable_default_side_input is None:
        raise ValueError("Must specify `second_read_enable_default_side_input`"
                         "when `second_read_type` is cross_attend_once")

      self.cross_attention_layer = readtwice_layers.ResidualBlock(
          inner_layer=readtwice_layers.SideAttention(
              hidden_size=config.hidden_size,
              num_heads=config.second_read_num_cross_attention_heads,
              att_dropout_prob=0,
              initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=config.initializer_range),
              top_k_attention=config.cross_attention_top_k,
              pos_embed_mode=config.cross_attention_pos_emb_mode,
              pos_embed_size=config.max_num_blocks_per_document,
              use_one_hot_embeddings=use_one_hot_embeddings,
              enable_default_side_input=config
              .second_read_enable_default_side_input),
          dropout_probability=config.hidden_dropout_prob,
          use_pre_activation_order=False,
          name="cross_attention_layer")

      self.second_read_transformer = readtwice_layers.TransformerWithSideInputLayers(
          hidden_size=config.hidden_size,
          num_hidden_layers=config.second_read_num_new_layers,
          num_attention_heads=config.num_attention_heads,
          intermediate_size=config.intermediate_size,
          hidden_act=tensor_utils.get_activation(config.hidden_act),
          hidden_dropout_prob=config.hidden_dropout_prob,
          attention_probs_dropout_prob=config.attention_probs_dropout_prob,
          initializer_range=config.initializer_range,
          share_kv_projections=True,
          name="transformer_layers")
    elif config.second_read_type == "new_layers_cross_attention":
      if config.second_read_num_new_layers is None:
        raise ValueError("Must specify `second_read_num_new_layers`"
                         "when `second_read_type` is cross_attend_once")
      if config.second_read_num_cross_attention_heads is None:
        raise ValueError("Must specify `second_read_num_cross_attention_heads`"
                         "when `second_read_type` is cross_attend_once")
      if config.second_read_enable_default_side_input is None:
        raise ValueError("Must specify `second_read_enable_default_side_input`"
                         "when `second_read_type` is cross_attend_once")

      self.second_read_transformer = readtwice_layers.TransformerWithSideInputLayers(
          hidden_size=config.hidden_size,
          num_hidden_layers=config.second_read_num_new_layers,
          num_attention_heads=config.num_attention_heads,
          intermediate_size=config.intermediate_size,
          hidden_act=tensor_utils.get_activation(config.hidden_act),
          hidden_dropout_prob=config.hidden_dropout_prob,
          attention_probs_dropout_prob=config.attention_probs_dropout_prob,
          initializer_range=config.initializer_range,
          share_kv_projections=True,
          num_cross_attention_heads=(
              config.second_read_num_cross_attention_heads),
          enable_default_side_input=(
              config.second_read_enable_default_side_input),
          name="transformer_layers")
    else:
      if config.second_read_type != "from_scratch":
        raise ValueError("Unknown `second_read_type`: '{}'".format(
            config.second_read_type))

  def call(self,
           token_ids,
           training,
           enable_side_inputs,
           cross_block_attention_mode,
           position_ids = None,
           block_ids = None,
           block_pos = None,
           att_mask = None,
           annotation_begins = None,
           annotation_ends = None,
           annotation_labels = None,
           num_replicas_concat = None):
    """Calls the layer.

    Args:
      token_ids: <int32>[batch_size, main_seq_len] Tensor of token ids.
      training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      enable_side_inputs: If True, enables read-it-twice model. Otherwise, the
        model becomes equivalent to the standard Transformer model.
      cross_block_attention_mode: The policy on how summaries between different
        blocks are allowed to interact with each other.
      position_ids: <int32>[batch_size, main_seq_len] Optional Tensor of
        absolute position ids. By default we use `tf.range(main_seq_len)` if
        `max_absolute_position_embeddings` is nonzero. The maximum position id
        must not be larger than `max_absolute_position_embeddings`.
      block_ids: <int32>[batch_size] Block IDs of every sample in the batch.
      block_pos: <int32>[batch_size] Optional Tensor of
        absolute position ids of blocks in the original document.
      att_mask: <int32>[batch_size, main_seq_len, main_seq_len]
      annotation_begins: <int32>[batch_size, max_num_annotations] Begin index of
        annotations.
      annotation_ends: <int32>[batch_size, max_num_annotations] End index of
        annotations (inclusive)
      annotation_labels: <int32>[batch_size, max_num_annotations] Label for
        annotations.
      num_replicas_concat: Number of replicas to gather summaries from. If None
        (default) then cross-replicas summaries are not used.

    Returns:
        main_output: ReadItTwiceBertModelOutput object
    """
    batch_size = tf.shape(token_ids)[0]
    main_seq_length = tf.shape(token_ids)[1]
    main_input = self.token_embedding(token_ids)
    if self.position_embedding is not None:
      if position_ids is None:
        main_input += self.position_embedding.embedding_table[
            tf.newaxis, :tf.shape(main_input)[1], :]
      else:
        main_input += self.position_embedding(position_ids)

    main_input = self.token_embedding_norm(main_input)
    main_input = self.token_embedding_dropout(main_input, training=training)

    # [batch_size, main_seq_len, hidden_size]
    first_read = self.transformer_with_side_inputs(
        main_input=main_input,
        side_input=None,
        att_mask=att_mask,
        training=training)

    summary_output = self.summary_extraction(
        hidden_states=first_read,
        block_ids=block_ids,
        block_pos=block_pos,
        annotation_begins=annotation_begins,
        annotation_ends=annotation_ends,
        annotation_labels=annotation_labels,
        main_seq_length=main_seq_length,
        num_replicas_concat=num_replicas_concat,
        cross_block_attention_mode=cross_block_attention_mode,
        training=training,
        token_ids=token_ids)

    if not enable_side_inputs:
      return ReadItTwiceBertModelOutput(
          first_read=first_read,
          final_hidden_states=first_read,
          token_to_global_summary_att_map=summary_output
          .token_to_global_summary_att_map,
          local_summary=summary_output.local_summary,
          global_summary=summary_output.global_summary)

    if att_mask is None:
      # [batch_size, main_seq_length, main_seq_length]
      token_to_token_att_mask = tf.ones(
          [batch_size, main_seq_length, main_seq_length], dtype=tf.int32)
    else:
      token_to_token_att_mask = att_mask

    att_mask_with_side_input = tf.concat([
        token_to_token_att_mask, summary_output.token_to_global_summary_att_map
    ],
                                         axis=2)

    if self.config.second_read_type == "from_scratch":
      # [batch_size, main_seq_len, hidden_size]
      second_read = self.transformer_with_side_inputs(
          main_input=main_input,
          side_input=summary_output.global_summary.processed_states,
          att_mask=att_mask_with_side_input,
          training=training)
    elif self.config.second_read_type in [
        "new_layers", "new_layers_cross_attention"
    ]:
      # [batch_size, main_seq_len, hidden_size]
      second_read = self.second_read_transformer(
          main_input=first_read,
          side_input=summary_output.global_summary.processed_states,
          att_mask=att_mask_with_side_input,
          training=training)
    elif self.config.second_read_type == "cross_attend_once":
      first_read_for_attention = first_read

      att_value_mask = tf.reduce_sum(
          summary_output.token_to_global_summary_att_map, axis=-1)
      att_value_mask = tf.cast(tf.greater(att_value_mask, 0), tf.float32)

      first_read_after_attention = self.cross_attention_layer(
          first_read_for_attention,
          side_input=summary_output.global_summary.processed_states,
          att_mask=summary_output.token_to_global_summary_att_map,
          main_pos=block_pos,
          side_pos=summary_output.global_summary.block_pos,
          att_value_mask=att_value_mask,
          training=training)
      # [batch_size, main_seq_len, hidden_size]
      second_read = self.second_read_transformer(
          main_input=first_read_after_attention,
          side_input=None,
          att_mask=att_mask,
          training=training)
    else:
      raise ValueError("Unknown `second_read_type`: '{}'".format(
          self.config.second_read_type))

    return ReadItTwiceBertModelOutput(
        first_read=first_read,
        final_hidden_states=second_read,
        token_to_global_summary_att_map=summary_output
        .token_to_global_summary_att_map,
        local_summary=summary_output.local_summary,
        global_summary=summary_output.global_summary)

  def get_token_embedding_table(self):
    """Returns the token embedding table, but only if the model is built."""
    if not hasattr(self.token_embedding, "embedding_table"):
      raise ValueError(
          "Cannot call `get_token_embedding_table()` until the model has been "
          "called so that all variables are built.")
    return self.token_embedding.embedding_table


def get_cross_block_att(block_ids,
                        block_pos,
                        all_block_ids,
                        all_block_pos,
                        cross_block_attention_mode,
                        cast_to_int32 = True):
  """Computes attention mask between blocks based on their document IDs."""
  # [batch_size, 1]
  block_ids_expanded = tf.expand_dims(block_ids, 1)
  # [1, global_batch_size]
  all_block_ids_expanded = tf.expand_dims(all_block_ids, 0)

  # [batch_size, 1]
  block_pos_expanded = tf.expand_dims(block_pos, 1)
  # [1, global_batch_size]
  all_block_pos_expanded = tf.expand_dims(all_block_pos, 0)

  # [batch_size, global_batch_size]
  cross_block_attention = tf.logical_and(
      tf.not_equal(block_ids_expanded, 0),
      tf.not_equal(all_block_ids_expanded, 0))

  if cross_block_attention_mode == "doc":
    # [batch_size, global_batch_size]
    cross_block_attention = tf.logical_and(
        tf.equal(block_ids_expanded, all_block_ids_expanded),
        cross_block_attention)
  elif cross_block_attention_mode == "block":
    # [batch_size, global_batch_size]
    cross_block_attention = tf.logical_and(
        tf.equal(block_ids_expanded, all_block_ids_expanded),
        cross_block_attention)
    cross_block_attention = tf.logical_and(
        tf.equal(block_pos_expanded, all_block_pos_expanded),
        cross_block_attention)
  elif cross_block_attention_mode == "other_blocks":
    is_the_same_doc = tf.equal(block_ids_expanded, all_block_ids_expanded)
    is_the_same_block = tf.logical_and(
        tf.equal(block_pos_expanded, all_block_pos_expanded), is_the_same_doc)
    is_the_same_doc_but_not_block = tf.logical_and(
        is_the_same_doc, tf.logical_not(is_the_same_block))
    cross_block_attention = tf.logical_and(is_the_same_doc_but_not_block,
                                           cross_block_attention)
  elif cross_block_attention_mode == "batch":
    pass
  else:
    raise ValueError("Unknown cross_block_attention_mode: " +
                     cross_block_attention_mode)

  if cast_to_int32:
    cross_block_attention = tf.cast(cross_block_attention, dtype=tf.int32)
  return cross_block_attention


class SummaryExtraction(tf.keras.Model):
  """Layer for the extracting summaries after the first read."""

  def __init__(self,
               config,
               use_one_hot_embeddings,
               name = "summary_extraction",
               **kwargs):
    """Constructor for SummaryExtraction.

    Args:
      config: `model_config.ReadItTwiceBertConfig` instance.
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.nn.embedding_lookup() for the word embeddings.
      name: (Optional) name of the layer.
      **kwargs: Forwarded to super.

    Raises:
      ValueError: The config is invalid.
    """
    super(SummaryExtraction, self).__init__(name=name, **kwargs)
    self.mode = config.summary_mode
    self.hidden_size = config.hidden_size
    self.postprocessing_type = config.summary_postprocessing_type
    self.use_sparse_memory_attention = (config.use_sparse_memory_attention)

    self.embedding_norm = None

    if self.mode == "cls":
      pass
    elif self.mode == "text_block":
      self.text_block_extract_every_x = config.text_block_extract_every_x
      assert self.text_block_extract_every_x is not None
      self.extraction_linear = tf.keras.layers.Dense(
          config.hidden_size,
          activation=None,
          kernel_initializer=(tf.truncated_normal_initializer(
              stddev=config.initializer_range)),
          name="entity_pool_linear")
    elif self.mode == "entity":
      self.extraction_linear = tf.keras.layers.Dense(
          config.hidden_size,
          activation=None,
          kernel_initializer=(tf.truncated_normal_initializer(
              stddev=config.initializer_range)),
          name="entity_pool_linear")
    else:
      raise ValueError("Unknown summary mode: {}".format(self.mode))

    if self.postprocessing_type == "none":
      self.postprocessing = None
    elif self.postprocessing_type == "linear":
      self.postprocessing = tf.keras.layers.Dense(
          config.hidden_size,
          activation=tf.tanh,
          kernel_initializer=(tf.truncated_normal_initializer(
              stddev=config.initializer_range)),
          name="cls_pool")
    elif self.postprocessing_type in ["pos", "transformer"]:
      self.position_embedding = readtwice_layers.EmbeddingLookup(
          vocab_size=config.max_num_blocks_per_document,
          embedding_size=config.hidden_size,
          initializer_range=config.initializer_range,
          use_one_hot_lookup=use_one_hot_embeddings,
          name="block_position_emb_lookup")
      # Call layers to force variable initialization.
      self.position_embedding(tf.ones([1, 1], tf.int32))
      self.embedding_norm = tf.keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, name="summary_emb_layer_norm")
      self.embedding_dropout = tf.keras.layers.Dropout(
          rate=config.hidden_dropout_prob)

      if self.postprocessing_type == "transformer":
        if config.summary_postprocessing_num_layers is None:
          raise ValueError("Must specify `postprocessing_num_layers`"
                           "when `postprocessing_type` is \"transformer\"")

        self.postprocessing = readtwice_layers.TransformerWithSideInputLayers(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.summary_postprocessing_num_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=tensor_utils.get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            share_kv_projections=True)
    else:
      raise ValueError("Unknown summary type: {}".format(
          self.postprocessing_type))

  def _extract_summary(self, hidden_states, annotation_begins, annotation_ends,
                       annotation_labels, token_ids):
    if self.mode == "cls":
      return tf.expand_dims(hidden_states[:, 0, :], 1)
    elif self.mode == "entity":
      # [batch_size, local_num_summaries, 2 * hidden_size]
      x = tf.concat([
          tf.gather(hidden_states, annotation_begins, batch_dims=1),
          tf.gather(hidden_states, annotation_ends, batch_dims=1)
      ],
                    axis=2)
      # [batch_size, local_num_summaries, hidden_size]
      x = self.extraction_linear(x)
      # [batch_size, local_num_summaries, 1]
      annotation_mask = tf.expand_dims(
          tf.cast(tf.not_equal(annotation_labels, 0), tf.float32), -1)
      annotation_mask = tf.ensure_shape(annotation_mask, [None, None, 1])
      # [batch_size, local_num_summaries, hidden_size]
      x = x * annotation_mask
      return x
    elif self.mode == "text_block":
      token_block_offset = self.text_block_extract_every_x - 1
      token_block_len = self.text_block_extract_every_x
      x = tf.concat([
          hidden_states[:, ::token_block_len, :],
          hidden_states[:, token_block_offset::token_block_len, :]
      ],
                    axis=2)
      x = self.extraction_linear(x)
      labels = tf.cast(
          tf.logical_and(
              tf.not_equal(token_ids[:, ::token_block_len], 0),
              tf.not_equal(token_ids[:, token_block_offset::token_block_len],
                           0),
          ), tf.float32)
      x = x * tf.expand_dims(labels, -1)
      return x
    else:
      raise ValueError("Unknown summary mode: {}".format(self.mode))

  def call(self,
           hidden_states,
           block_ids,
           block_pos,
           annotation_begins,
           annotation_ends,
           annotation_labels,
           main_seq_length,
           num_replicas_concat,
           cross_block_attention_mode,
           training,
           token_ids = None):
    """Calls the layer.

    Args:
      hidden_states: <int32>[batch_size, main_seq_len, hidden size]. Final
        hidden states of the input after the first pass of the model.
      block_ids: <int32>[batch_size] Block IDs of every sample in the batch.
      block_pos: <int32>[batch_size] Optional Tensor of absolute position ids of
        blocks in the original document.
      annotation_begins: <int32>[batch_size, max_num_annotations] Begin index of
        annotations.
      annotation_ends: <int32>[batch_size, max_num_annotations] End index of
        annotations (inclusive)
      annotation_labels: <int32>[batch_size, max_num_annotations] Label for
        annotations.
      main_seq_length: Length of the input text
      num_replicas_concat: Number of replicas to gather summaries from. If None
        (default) then cross-replicas summaries are not used.
      cross_block_attention_mode: The policy on how summaries between different
        blocks are allowed to interact with each other.
      training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      token_ids: <int32>[batch_size, main_seq_len] Tokens.

    Returns:
        summary_output: SummaryExtractionOutput object
    """
    # [batch_size, local_num_summaries, hidden_size]
    first_token_tensor = self._extract_summary(
        hidden_states=hidden_states,
        annotation_begins=annotation_begins,
        annotation_ends=annotation_ends,
        annotation_labels=annotation_labels,
        token_ids=token_ids)
    batch_size = tf.shape(first_token_tensor)[0]
    local_num_summaries = tf.shape(first_token_tensor)[1]
    original_block_ids = block_ids
    original_block_pos = block_pos
    block_ids = tf.tile(tf.expand_dims(block_ids, 1), [1, local_num_summaries])
    block_pos = tf.tile(tf.expand_dims(block_pos, 1), [1, local_num_summaries])

    first_token_tensor = tf.reshape(
        first_token_tensor,
        [batch_size * local_num_summaries, self.hidden_size])
    block_ids = tf.reshape(block_ids, [batch_size * local_num_summaries])
    block_pos = tf.reshape(block_pos, [batch_size * local_num_summaries])

    all_first_token_tensor = first_token_tensor
    all_block_ids = block_ids
    all_block_pos = block_pos
    if num_replicas_concat:
      # Concatenate all the required tensors across tpu cores.
      all_first_token_tensor = tpu_utils.cross_replica_concat(
          tensor=all_first_token_tensor,
          num_replicas=num_replicas_concat,
          name="cls_token_concat")
      all_block_ids = tpu_utils.cross_replica_concat(
          tensor=all_block_ids,
          num_replicas=num_replicas_concat,
          name="block_ids_concat")
      all_block_ids = tf.stop_gradient(all_block_ids)

      all_block_pos = tpu_utils.cross_replica_concat(
          tensor=all_block_pos,
          num_replicas=num_replicas_concat,
          name="block_pos_concat")
      all_block_pos = tf.stop_gradient(all_block_pos)

    first_token_tensor.set_shape([None, self.hidden_size])
    all_first_token_tensor.set_shape([None, self.hidden_size])

    if self.mode == "cls":
      labels = block_ids
      all_labels = all_block_ids
    elif self.mode == "text_block":
      token_block_offset = self.text_block_extract_every_x - 1
      token_block_len = self.text_block_extract_every_x
      labels = tf.cast(
          tf.logical_and(
              tf.not_equal(token_ids[:, ::token_block_len], 0),
              tf.not_equal(token_ids[:, token_block_offset::token_block_len],
                           0),
          ), tf.int32)
      labels = tf.reshape(labels, [batch_size * local_num_summaries])
      all_labels = labels
      if num_replicas_concat:
        all_labels = tpu_utils.cross_replica_concat(
            tensor=all_labels,
            num_replicas=num_replicas_concat,
            name="labels_concat")
        all_labels = tf.stop_gradient(all_labels)
    else:
      assert self.mode == "entity"
      labels = tf.reshape(annotation_labels, [batch_size * local_num_summaries])
      all_labels = labels
      if num_replicas_concat:
        all_labels = tpu_utils.cross_replica_concat(
            tensor=all_labels,
            num_replicas=num_replicas_concat,
            name="labels_concat")
        all_labels = tf.stop_gradient(all_labels)

    # TODO(urikz): Consider using this
    # Filter out all padding summaries -- the convention is that
    # padding summaries will have label 0.
    # non_padding_summary_mask = tf.not_equal(all_labels, 0)
    # all_first_token_tensor = tf.boolean_mask(all_first_token_tensor,
    #                                          non_padding_summary_mask)
    # all_block_pos = tf.boolean_mask(all_block_pos, non_padding_summary_mask)
    # all_block_ids = tf.boolean_mask(all_block_ids, non_padding_summary_mask)
    # all_labels = tf.boolean_mask(all_labels, non_padding_summary_mask)

    if self.postprocessing_type == "none":
      all_cls_summary = all_first_token_tensor
    elif self.postprocessing_type == "linear":
      all_cls_summary = self.postprocessing(all_first_token_tensor)
    elif self.postprocessing_type in ["pos", "transformer"]:
      # We treat sequence of summaries as just a single sentence.
      # [1, global_num_summaries, hidden_dim]
      all_cls_summary = tf.expand_dims(all_first_token_tensor, 0)

      # Add positional embeddings based on positions of blocks in their
      # original documents.
      all_cls_summary += self.position_embedding(all_block_pos)
      all_cls_summary = self.embedding_norm(all_cls_summary)
      # Note, we don't apply dropout here
      # all_cls_summary = self.embedding_dropout(
      #     all_cls_summary, training=training)
      if self.postprocessing_type == "transformer":
        # Create cross block attention map
        # according to the `cross_block_attention_mode`.
        # [global_num_summaries, global_num_summaries]
        block_att_mask = get_cross_block_att(all_block_ids, all_block_pos,
                                             all_block_ids, all_block_pos,
                                             cross_block_attention_mode)
        # [1, global_num_summaries, global_num_summaries]
        block_att_mask = tf.expand_dims(block_att_mask, 0)

        all_cls_summary = self.postprocessing(
            main_input=all_cls_summary,
            side_input=None,
            att_mask=block_att_mask,
            training=training)
      all_cls_summary = tf.squeeze(all_cls_summary, 0)
    else:
      raise ValueError("Unknown `postprocessing_type`: '{}'".format(
          self.postprocessing_type))

    # [batch_size, global_num_summaries]
    token_to_global_summary_att_map = get_cross_block_att(
        original_block_ids, original_block_pos, all_block_ids, all_block_pos,
        cross_block_attention_mode)
    # [batch_size, main_seq_length, global_num_summaries]
    token_to_global_summary_att_map = tf.tile(
        tf.expand_dims(token_to_global_summary_att_map, 1),
        [1, main_seq_length, 1])

    # Do not attend over pad entity summaries
    # [1, 1, global_num_summaries]
    is_not_pad_summary = tf.expand_dims(
        tf.expand_dims(tf.cast(tf.not_equal(all_labels, 0), tf.int32), 0), 0)
    token_to_global_summary_att_map *= is_not_pad_summary

    if self.use_sparse_memory_attention:
      if self.mode == "entity":
        # 2. Only allow entity mentions to attend summaries
        # [batch_size, max_num_annotations, 1]
        annotation_mask = tf.expand_dims(
            tf.cast(tf.not_equal(annotation_labels, 0), tf.int32), -1)
        # [batch_size, max_num_annotations, main_seq_length]
        mask_begin = tf.sequence_mask(
            annotation_begins, main_seq_length, dtype=tf.int32)
        mask_end = tf.sequence_mask(
            annotation_ends + 1, main_seq_length, dtype=tf.int32)

        def make_mask(x):
          x = x * annotation_mask
          x = tf.reduce_sum(x, 1)
          x = tf.minimum(x, 1)
          return x

        # [batch_size, main_seq_length, 1]
        is_token_belongs_to_entity = tf.expand_dims(
            make_mask(mask_end - mask_begin), -1)
        token_to_global_summary_att_map *= is_token_belongs_to_entity
      elif self.mode == "cls":
        # [batch_size, main_seq_length]
        only_cls_mask = tf.concat([
            tf.cast(tf.fill(dims=[batch_size, 1], value=1), dtype=tf.int32),
            tf.cast(
                tf.fill(dims=[batch_size, main_seq_length - 1], value=0),
                dtype=tf.int32)
        ],
                                  axis=1)
        # [batch_size, main_seq_length, 1]
        only_cls_mask = tf.expand_dims(only_cls_mask, -1)
        # [batch_size, main_seq_length, global_num_summaries]
        token_to_global_summary_att_map *= only_cls_mask
      elif self.mode == "text_block":
        # [main_seq_length]
        text_block_mask = tf.range(main_seq_length, delta=1, dtype=tf.int32)
        # [main_seq_length]
        text_block_mask = tf.math.floormod(text_block_mask,
                                           self.text_block_extract_every_x)
        # [main_seq_length]
        text_block_mask = tf.cast(tf.equal(text_block_mask, 0), tf.int32)
        # [batch_size, main_seq_length]
        text_block_mask = tf.tile(
            tf.expand_dims(text_block_mask, 0), [batch_size, 1])
        # [batch_size, main_seq_length, 1]
        text_block_mask = tf.expand_dims(text_block_mask, -1)
        # [batch_size, main_seq_length, global_num_summaries]
        token_to_global_summary_att_map *= text_block_mask
      else:
        raise ValueError("Unknown summary mode: %s" % self.mode)

    return SummaryExtractionOutput(
        local_summary=Summary(
            states=first_token_tensor,
            processed_states=None,
            block_ids=block_ids,
            block_pos=block_pos,
            labels=labels),
        global_summary=Summary(
            states=all_first_token_tensor,
            processed_states=all_cls_summary,
            block_ids=all_block_ids,
            block_pos=all_block_pos,
            labels=all_labels),
        token_to_global_summary_att_map=token_to_global_summary_att_map)


class SpanPredictionHead(tf.keras.layers.Layer):
  """Layer for the ReadItTwiceBert model for span predictions over input."""

  def __init__(self,
               intermediate_size,
               intermediate_activation=tensor_utils.get_activation("gelu"),
               dropout_rate = 0.0,
               name = "span_prediction_head",
               **kwargs):
    """Constructor for SpanPredictionHead.

    Args:
      intermediate_size: dimension of the intermediate representation of MLP. If
        None then only a single linear layer will be applied
      intermediate_activation: activation function for MLP
      dropout_rate: dropout rate
      name: (Optional) name of the layer.
      **kwargs: Forwarded to super.
    """
    super(SpanPredictionHead, self).__init__(name=name, **kwargs)
    if intermediate_size is not None:
      self._intermediate_dense = tf.keras.layers.Dense(intermediate_size)
      self._intermediate_activation = tf.keras.layers.Activation(
          intermediate_activation)
      self._output_dropout = tf.keras.layers.Dropout(dropout_rate)
      self._output_layer_norm = tf.keras.layers.LayerNormalization()
    else:
      self._intermediate_dense = None
      self._intermediate_activation = None
      self._output_dropout = None
      self._output_layer_norm = None
    self._logits_dense = tf.keras.layers.Dense(2)

  def build(self, input_shape):
    """Keras build function.

    Args:
      input_shape: TensorShape of the input.
    """
    hidden_size = input_shape.as_list()[-1]
    if hidden_size is None:
      raise ValueError("`input_shape[-1]` must be statically known.")
    self._output_dense = tf.keras.layers.Dense(hidden_size)
    super(SpanPredictionHead, self).build(input_shape)

  def call(self,
           hidden_states,
           token_ids = None,
           padding_token_id = None,
           ignore_prefix_length = None,
           training=None):

    if self._intermediate_dense is not None:
      intermediate_outputs = self._intermediate_dense(hidden_states)
      intermediate_outputs = self._intermediate_activation(intermediate_outputs)
      outputs = self._output_dense(intermediate_outputs)
      outputs = self._output_dropout(outputs, training=training)
      outputs = self._output_layer_norm(outputs + hidden_states)
    else:
      outputs = hidden_states
    logits = self._logits_dense(outputs)

    if token_ids is not None or padding_token_id is not None:
      if token_ids is None or padding_token_id is None:
        raise ValueError("Both `token_ids` and `padding_token_id` needs to be "
                         "specified in order to compute mask for logits")
      logits -= tf.expand_dims(tf.cast(tf.equal(token_ids, 0), tf.float32),
                               -1) * 1e6

    if ignore_prefix_length is not None:
      seq_length = tf.shape(logits)[-2]
      logits -= tf.expand_dims(
          tf.sequence_mask(ignore_prefix_length, seq_length, dtype=tf.float32),
          -1) * 1e6

    return logits


class ReadItTwiceDecoderModel(tf.keras.layers.Layer):
  """ReadItTwice decoder model."""

  def __init__(self,
               config,
               num_layers_override,
               num_cross_attention_heads,
               enable_default_side_input = False,
               use_one_hot_embeddings=False,
               name = "read_it_twice_decoder",
               **kwargs):
    """Constructor for ReadItTwiceDecoderModel.

    Args:
      config: `model_config.ReadItTwiceBertConfig` instance.
      num_layers_override: int. Number of Transformer layers.
      num_cross_attention_heads: int. Number of cross-attention heads.
      enable_default_side_input: Add a default side input, which acts like a
        no-op attention, effective allowing attention weights to sum up
        to something less than 1.
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.nn.embedding_lookup() for the word embeddings.
      name: (Optional) name of the layer.
      **kwargs: Forwarded to super.

    Raises:
      ValueError: The config is invalid.
    """
    super(ReadItTwiceDecoderModel, self).__init__(name=name, **kwargs)

    self.use_one_hot_embeddings = use_one_hot_embeddings
    self.num_layers_override = num_layers_override
    self.num_cross_attention_heads = num_cross_attention_heads
    self.enable_default_side_input = enable_default_side_input

    if config.embedding_size is None:
      config = dataclasses.replace(config, embedding_size=config.hidden_size)
    self.config = config

    self.token_embedding = readtwice_layers.EmbeddingLookup(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        projection_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=use_one_hot_embeddings,
        name="token_emb_lookup")

    self.token_embedding_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name="emb_layer_norm")
    self.token_embedding_dropout = tf.keras.layers.Dropout(
        rate=config.hidden_dropout_prob)

    self.position_embedding = readtwice_layers.EmbeddingLookup(
        vocab_size=config.max_seq_length,
        embedding_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=use_one_hot_embeddings,
        name="position_emb_lookup_long")
    # Call layers to force variable initialization.
    self.position_embedding(tf.ones([1, 1], tf.int32))

    self.transformer_with_side_inputs = readtwice_layers.TransformerWithSideInputLayers(
        hidden_size=config.hidden_size,
        num_hidden_layers=self.num_layers_override,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=tensor_utils.get_activation(config.hidden_act),
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        initializer_range=config.initializer_range,
        share_kv_projections=False,
        num_cross_attention_heads=self.num_cross_attention_heads,
        enable_default_side_input=self.enable_default_side_input)

  def call(self,
           token_ids,
           side_input,
           token2side_input_att_mask,
           training,
           position_ids = None):
    """Calls the layer.

    Args:
      token_ids: <int32>[batch_size, main_seq_len] Tensor of token ids.
      side_input: <float32>[side_seq_len, hidden_size] Tensor of side input.
      token2side_input_att_mask: <int32>[batch_size, main_seq_len, side_seq_len]
      training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      position_ids: <int32>[batch_size, main_seq_len] Optional Tensor of
        absolute position ids. By default we use `tf.range(main_seq_len)` if
        `max_absolute_position_embeddings` is nonzero. The maximum position id
        must not be larger than `max_absolute_position_embeddings`.

    Returns:
        main_output: <float32>[batch_size, main_seq_len, hidden_size]
    """
    main_input = self.token_embedding(token_ids)
    if self.position_embedding is not None:
      if position_ids is None:
        main_input += self.position_embedding.embedding_table[
            tf.newaxis, :tf.shape(main_input)[1], :]
      else:
        main_input += self.position_embedding(position_ids)

    main_input = self.token_embedding_norm(main_input)
    main_input = self.token_embedding_dropout(main_input, training=training)

    batch_size = tf.shape(main_input)[0]
    main_seq_len = tf.shape(main_input)[1]
    att_mask = tf.linalg.band_part(
        tf.ones([main_seq_len, main_seq_len], dtype=tf.int32), -1, 0)
    att_mask = tf.tile(tf.expand_dims(att_mask, 0), [batch_size, 1, 1])

    token2side_input_att_mask = tf.tile(
        tf.expand_dims(token2side_input_att_mask, 1), [1, main_seq_len, 1])
    att_mask = tf.concat([att_mask, token2side_input_att_mask], axis=2)

    # [batch_size, main_seq_len, hidden_size]
    main_output = self.transformer_with_side_inputs(
        main_input=main_input,
        side_input=side_input,
        att_mask=att_mask,
        training=training)

    return main_output

  def get_token_embedding_table(self):
    """Returns the token embedding table, but only if the model is built."""
    if not hasattr(self.token_embedding, "embedding_table"):
      raise ValueError(
          "Cannot call `get_token_embedding_table()` until the model has been "
          "called so that all variables are built.")
    return self.token_embedding.embedding_table
