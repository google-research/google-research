# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""The main ETC model and config."""

import copy
import json
from typing import Optional, Text

import tensorflow.compat.v1 as tf

from etcmodel import feature_utils
from etcmodel import layers as etc_layers
from etcmodel import tensor_utils


# Number of other relative attention ids we use besides those from relative
# position representations.  Other uses are for global-to-long (g2l) and
# long-to-global (l2g) cross-attention.  The relative embeddings for these 2
# are distinct and their ids are transposes of each other, so we can just count
# the g2l direction without loss of generality.  Right now we have the
# following 2 uses:
#   1. Cross attention from a sentence global token to long tokens belong to it.
#   2. Cross attention from a sentence global token to long tokens NOT belonging
#     to it.
# Also note there is a 3rd use for global-to-global (g2g) in HotpotQA and
# WikiHop for global tokens in one context attending to global tokens in other
# contexts in a permutation-invariant way.
_NUM_OTHER_RELATIVE_IDS = 3


class EtcConfig(object):
  """Configuration for `EtcModel`."""

  def __init__(self,
               vocab_size,
               segment_vocab_size=16,
               embedding_size=None,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               local_radius=84,
               att_size_per_head=None,
               relative_pos_max_distance=12,
               relative_vocab_size=32,
               max_absolute_position_embeddings=0,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               initializer_range=0.02,
               share_feed_forward_params=True,
               share_kv_projections=False,
               share_qkv_projections=True,
               share_att_output_projection=False,
               use_pre_activation_order=False,
               use_hard_g2l_mask=False,
               use_hard_l2g_mask=False,
               grad_checkpointing_period=0):
    """Constructs `EtcConfig`.

    Args:
      vocab_size: Vocabulary size of `token_ids` and `global_token_ids`.
      segment_vocab_size: Vocabulary size of `segment_ids` and
        `global_segment_ids`.
      embedding_size: Size of `token_ids` and `global_token_ids` embeddings.
        The default of `None` makes this equal to `hidden_size` like original
        BERT, but it can be set to a smaller value (e.g. 128) like ALBERT. Must
        be positive.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      local_radius: How many tokens to the left/right for input tokens to
        locally self-attend to. For example, a value of 1 would allow each token
        to only attend to 1 token to the left and 1 token to the right of it.
      att_size_per_head: Size of attention query/key/value vectors per head.
        By default this will be `hidden_size / num_attention_heads`, so
        `num_attention_heads` must evenly divide `hidden_size` in this case.
      relative_pos_max_distance: Maximum distance to use for relative position
        representations. All larger distances will be clipped to this value.
      relative_vocab_size: The total vocabulary size for relative positions.
        This must be at least `2 * relative_pos_max_distance + 1`.
      max_absolute_position_embeddings: The maximum sequence length that this
        model might ever be used with; used for absolute position embeddings
        like BERT. If set to 0 (the default), we skip absolute position
        embeddings entirely. If nonzero, inputs larger than this value are not
        allowed.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      share_feed_forward_params: If True, we share the same feed-forward
        parameters for the long and global inputs.
      share_kv_projections: If True, key and value projections will be shared
        between long-to-long and long-to-global components, as well as between
        global-to-global and global-to-long components. This results in 2 key
        projections per layer instead of 4 (and similarly for value
        projections). Note that if `share_qkv_projections` is True, then
        `share_kv_projections` is completely ignored since the former results
        in even more sharing.
      share_qkv_projections: If True, all 4 attention operations (long-to-long,
        global-to-global, long-to-global, and global-to-long) will share the
        same query, key, and value projections. The 3 projections will still be
        different from each other and different per layer.
      share_att_output_projection: If True, all 4 attention operations
        (long-to-long, global-to-global, long-to-global, and global-to-long)
        will share the same output projection per layer.
      use_pre_activation_order: If True, use "pre-activation" order for residual
        blocks.
      use_hard_g2l_mask: If True, global tokens only attend to
        tokens of the corresponding sentences in the long input. If False,
        global tokens attend to all sentences within the corresponding global
        example.
      use_hard_l2g_mask: If True, long tokens only attend to
        tokens of the corresponding global tokens. If False, long tokens attend
        to all the global tokens within the corresponding global example.
      grad_checkpointing_period: How often to checkpoint activations. The
        default of 0 stores all activations. If greater than 0, activations are
        recomputed as necessary when calculating gradients to save memory. As an
        optimization, we avoid recomputing the last `grad_checkpointing_period`
        layers, so larger values result in less computational overhead but
        reduced memory savings. Using a value of `1` results in potentially the
        greatest memory savings but with the highest recompute cost.

    Raises:
      ValueError: If config options are invalid.
    """
    self.vocab_size = vocab_size
    self.segment_vocab_size = segment_vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.local_radius = local_radius
    self.att_size_per_head = att_size_per_head
    self.relative_pos_max_distance = relative_pos_max_distance
    self.relative_vocab_size = relative_vocab_size
    self.max_absolute_position_embeddings = max_absolute_position_embeddings
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.share_feed_forward_params = share_feed_forward_params
    self.share_kv_projections = share_kv_projections
    self.share_qkv_projections = share_qkv_projections
    self.share_att_output_projection = share_att_output_projection
    self.use_pre_activation_order = use_pre_activation_order
    self.use_hard_g2l_mask = use_hard_g2l_mask
    self.use_hard_l2g_mask = use_hard_l2g_mask
    self.grad_checkpointing_period = grad_checkpointing_period

  @classmethod
  def from_dict(cls, json_object):
    """Constructs `EtcConfig` from Python dictionary of parameters."""
    config = EtcConfig(vocab_size=None)
    for key, value in json_object.items():
      if hasattr(config, key):
        config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs `EtcConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class EtcModel(tf.keras.layers.Layer):
  """ETC model."""

  def __init__(self,
               config: EtcConfig,
               is_training: Optional[bool] = None,
               use_one_hot_embeddings=False,
               use_one_hot_relative_embeddings=False,
               name: Text = "etc_document_bert",
               **kwargs):
    """Constructor for `EtcModel`.

    Args:
      config: `EtcConfig` instance.
      is_training: Optional bool. True for training model, False for eval model.
        The None default will defer to the typical Keras `training` argument in
        `call` instead. When `is_training` is specified here, the `training`
        argument from `call` must not be used.
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.nn.embedding_lookup() for the word embeddings.
      use_one_hot_relative_embeddings: (optional) bool. Whether to use one-hot
        word embeddings or tf.nn.embedding_lookup() for the relative position
        embeddings.
      name: (Optional) name of the layer.
      **kwargs: Forwarded to super.

    Raises:
      ValueError: The config is invalid.
    """
    super(EtcModel, self).__init__(name=name, **kwargs)

    config = copy.deepcopy(config)
    if is_training is not None and not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    self.config = config
    self.is_training = is_training
    self.use_one_hot_embeddings = use_one_hot_embeddings
    self.use_one_hot_relative_embeddings = use_one_hot_relative_embeddings

    if config.relative_vocab_size is None:
      if config.relative_pos_max_distance != 0:
        raise ValueError(
            "`relative_pos_max_distance` must be 0 when `relative_vocab_size` "
            "is None.")
    elif config.relative_vocab_size < (feature_utils.RelativePositionGenerator(
        config.relative_pos_max_distance).relative_vocab_size +
                                       _NUM_OTHER_RELATIVE_IDS):
      raise ValueError("`relative_vocab_size` ({}) too small for "
                       "`relative_pos_max_distance` ({})".format(
                           config.relative_vocab_size,
                           config.relative_pos_max_distance))
    if config.embedding_size is None:
      config.embedding_size = config.hidden_size

    self.token_embedding = etc_layers.EmbeddingLookup(
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        projection_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=use_one_hot_embeddings,
        name="token_emb_lookup")

    self.token_embedding_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name="long_emb_layer_norm")
    self.token_embedding_dropout = tf.keras.layers.Dropout(
        rate=config.hidden_dropout_prob)

    self.segment_embedding = etc_layers.EmbeddingLookup(
        vocab_size=config.segment_vocab_size,
        embedding_size=config.hidden_size,
        initializer_range=config.initializer_range,
        use_one_hot_lookup=True,
        name="segment_emb_lookup")

    if config.max_absolute_position_embeddings != 0:
      self.position_embedding = etc_layers.EmbeddingLookup(
          vocab_size=config.max_absolute_position_embeddings,
          embedding_size=config.hidden_size,
          initializer_range=config.initializer_range,
          use_one_hot_lookup=use_one_hot_embeddings,
          name="position_emb_lookup_long")
      # We use `max_absolute_position_embeddings` for the maximum global input
      # length even though it's larger than we need. This makes it easier to
      # initialize both long and global position embedding tables with the same
      # values if desired.
      self.global_position_embedding = etc_layers.EmbeddingLookup(
          vocab_size=config.max_absolute_position_embeddings,
          embedding_size=config.hidden_size,
          initializer_range=config.initializer_range,
          use_one_hot_lookup=use_one_hot_embeddings,
          name="position_emb_lookup_global")
      # Call layers to force variable initialization.
      self.position_embedding(tf.ones([1, 1], tf.int32))
      self.global_position_embedding(tf.ones([1, 1], tf.int32))
    else:
      self.position_embedding = None
      self.global_position_embedding = None

    # We use the same embedding table for global tokens to make it easy to place
    # WordPieces in the global memory for finetuning tasks downstream.
    self.global_token_embedding = self.token_embedding
    self.global_token_embedding_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name="global_emb_layer_norm")
    self.global_token_embedding_dropout = tf.keras.layers.Dropout(
        rate=config.hidden_dropout_prob)

    self.global_local_transformer = etc_layers.GlobalLocalTransformerLayers(
        long_hidden_size=config.hidden_size,
        global_hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        local_radius=config.local_radius,
        att_size_per_head=config.att_size_per_head,
        long_intermediate_size=config.intermediate_size,
        global_intermediate_size=config.intermediate_size,
        hidden_act=tensor_utils.get_activation(config.hidden_act),
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        initializer_range=config.initializer_range,
        relative_vocab_size=config.relative_vocab_size,
        share_feed_forward_params=config.share_feed_forward_params,
        share_kv_projections=config.share_kv_projections,
        share_qkv_projections=config.share_qkv_projections,
        share_att_output_projection=config.share_att_output_projection,
        use_pre_activation_order=config.use_pre_activation_order,
        use_one_hot_lookup=use_one_hot_relative_embeddings,
        grad_checkpointing_period=config.grad_checkpointing_period)

  def call(self,
           token_ids: tf.Tensor,
           global_token_ids: tf.Tensor,
           segment_ids: Optional[tf.Tensor] = None,
           global_segment_ids: Optional[tf.Tensor] = None,
           position_ids: Optional[tf.Tensor] = None,
           global_position_ids: Optional[tf.Tensor] = None,
           l2l_att_mask: Optional[tf.Tensor] = None,
           g2g_att_mask: Optional[tf.Tensor] = None,
           l2g_att_mask: Optional[tf.Tensor] = None,
           g2l_att_mask: Optional[tf.Tensor] = None,
           l2l_relative_att_ids: Optional[tf.Tensor] = None,
           g2g_relative_att_ids: Optional[tf.Tensor] = None,
           l2g_relative_att_ids: Optional[tf.Tensor] = None,
           g2l_relative_att_ids: Optional[tf.Tensor] = None,
           long_embedding_adder: Optional[tf.Tensor] = None,
           global_embedding_adder: Optional[tf.Tensor] = None,
           training=None):
    """Calls the layer.

    Args:
      token_ids: <int32>[batch_size, long_seq_len] Tensor of token ids.
      global_token_ids: <int32>[batch_size, global_seq_len] Tensor of global
        token ids.
      segment_ids: <int32>[batch_size, long_seq_len] Optional Tensor of segment
        ids. By default we just fill all elements with segment id 1.
      global_segment_ids: <int32>[batch_size, global_seq_len] Optional Tensor of
        global segment ids. By default we just fill all elements with segment id
        0. This is deliberately different from the `segment_ids` default of 1
        so that we distinguish between global vs. long segments.
      position_ids: <int32>[batch_size, long_seq_len] Optional Tensor of
        absolute position ids. By default we use `tf.range(long_seq_len)` if
        `max_absolute_position_embeddings` is nonzero. The maximum position id
        must not be larger than `max_absolute_position_embeddings`.
      global_position_ids: <int32>[batch_size, global_seq_len] Optional Tensor
        of absolute position ids. By default we use `tf.range(global_seq_len)`
        if `max_absolute_position_embeddings` is nonzero. The maximum position
        id must not be larger than `max_absolute_position_embeddings`. Note that
        global and long absolute position embeddings are separate variables.
      l2l_att_mask: <int32>[batch_size, long_seq_len,  2*local_radius + 1]
      g2g_att_mask: <int32>[batch_size, global_seq_len, global_seq_len]
      l2g_att_mask: <int32>[batch_size, long_seq_len, global_seq_len]
      g2l_att_mask: <int32>[batch_size, global_seq_len, long_seq_len]
      l2l_relative_att_ids: <int32>[batch_size, long_seq_len, 2*local_radius+1]
      g2g_relative_att_ids: <int32>[batch_size, global_seq_len, global_seq_len]
      l2g_relative_att_ids: <int32>[batch_size, long_seq_len, global_seq_len]
      g2l_relative_att_ids: <int32>[batch_size, global_seq_len, long_seq_len]
      long_embedding_adder: <float32>[batch_size, long_seq_len, hidden_size]
        Tensor of additional values to add to the long input embedding before
        layer normalization of the embeddings. By default this is skipped.
      global_embedding_adder: <float32>[batch_size, global_seq_len, hidden_size]
        Tensor of additional values to add to the glboal input embedding before
        layer normalization of the embeddings. By default this is skipped.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference. Must
        be None if `is_training` was not None in `__init__`.

    Returns:
      A list of Tensors, [long_output, global_output]:
        long_output: <float32>[batch_size, long_seq_len, hidden_size]
        global_output: <float32>[batch_size, global_seq_len, hidden_size]
    """
    if self.is_training is not None:
      if training is not None:
        raise ValueError(
            "`training` must be None when `is_training` is given in `__init__`."
        )
      training = self.is_training

    if segment_ids is None:
      segment_ids = tf.ones_like(token_ids)
    if global_segment_ids is None:
      global_segment_ids = tf.zeros_like(global_token_ids)

    if self.config.max_absolute_position_embeddings == 0 and (
        position_ids is not None or global_position_ids is not None):
      raise ValueError(
          "Cannot specify `position_ids` or `global_position_ids` arguments "
          "when `max_absolute_position_embeddings` is 0.")

    long_input = self.token_embedding(token_ids)
    long_input += self.segment_embedding(segment_ids)
    if self.position_embedding is not None:
      if position_ids is None:
        long_input += self.position_embedding.embedding_table[
            tf.newaxis, :long_input.shape[1], :]
      else:
        long_input += self.position_embedding(position_ids)
    if long_embedding_adder is not None:
      long_input += long_embedding_adder
    long_input = self.token_embedding_norm(long_input)
    long_input = self.token_embedding_dropout(long_input, training=training)

    global_input = self.global_token_embedding(global_token_ids)
    global_input += self.segment_embedding(global_segment_ids)
    if self.global_position_embedding is not None:
      if global_position_ids is None:
        global_input += self.global_position_embedding.embedding_table[
            tf.newaxis, :global_input.shape[1], :]
      else:
        global_input += self.global_position_embedding(global_position_ids)
    if global_embedding_adder is not None:
      global_input += global_embedding_adder
    global_input = self.global_token_embedding_norm(global_input)
    global_input = self.global_token_embedding_dropout(
        global_input, training=training)

    return self.global_local_transformer(
        long_input=long_input,
        global_input=global_input,
        l2l_att_mask=l2l_att_mask,
        g2g_att_mask=g2g_att_mask,
        l2g_att_mask=l2g_att_mask,
        g2l_att_mask=g2l_att_mask,
        l2l_relative_att_ids=l2l_relative_att_ids,
        g2g_relative_att_ids=g2g_relative_att_ids,
        l2g_relative_att_ids=l2g_relative_att_ids,
        g2l_relative_att_ids=g2l_relative_att_ids,
        training=training)

  def get_token_embedding_table(self):
    """Returns the token embedding table, but only if the model is built."""
    if not hasattr(self.token_embedding, "embedding_table"):
      raise ValueError(
          "Cannot call `get_token_embedding_table()` until the model has been "
          "called so that all variables are built.")
    return self.token_embedding.embedding_table
