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

"""Attention layers for ETC."""

from typing import List, Optional, Text, Tuple

import tensorflow as tf

from etcmodel import feature_utils
from etcmodel import tensor_utils
from etcmodel.layers import recomputing_dropout


class RelativeAttention(tf.keras.layers.Layer):
  """Layer for multi-head attention with relative position representations.

  This layer projects to multi-head queries, keys, and values before calling
  `QkvRelativeAttention` for relative attention, so see `QkvRelativeAttention`
  for more details. A final output projection is performed to mix the attention
  results for each head back to `hidden_size` dimensions.

  Note that the relative position representations are optional. In their
  absence, this is just scaled dot-product attention from Transformer.
  """

  def __init__(self,
               hidden_size: int,
               num_heads: int,
               total_key_size: Optional[int] = None,
               total_value_size: Optional[int] = None,
               relative_vocab_size: Optional[int] = None,
               att_dropout_prob: float = 0.0,
               initializer=None,
               query_projection: Optional['ProjectAttentionHeads'] = None,
               key_projection: Optional['ProjectAttentionHeads'] = None,
               value_projection: Optional['ProjectAttentionHeads'] = None,
               qkv_relative_attention: Optional['QkvRelativeAttention'] = None,
               output_projection: Optional[tf.keras.layers.Dense] = None,
               use_one_hot_lookup: bool = False,
               name: Text = 'relative_attention',
               **kwargs):
    """Init.

    Note: For typical Transformer setups, `hidden_size`, `total_key_size`, and
    `total_value_size` are all the same, but they need not be in general.

    Args:
      hidden_size: Size of the output hidden dimension.
      num_heads: Number of attention heads.
      total_key_size: Total size of the attention key (and query) vectors after
        concatenating all heads. Defaults to `hidden_size`. Must be a multiple
        of `num_heads`.
      total_value_size: Total size of the attention value vectors after
        concatenating all heads. Defaults to `hidden_size`. Must be a multiple
        of `num_heads`.
      relative_vocab_size: Size of relative position vocabulary. If left
        unspecified, relative positions will be ignored for attention.
      att_dropout_prob: Dropout probability for attention probabilities. Must be
        between 0.0 and 1.0. The default of 0.0 skips dropout.
      initializer: Initializer to use for non-bias variables other than the
        relative embedding table. Bias variables will be initialized to 0,
        and the relative embedding table has its own default initialization
        scale.
      query_projection: (Advanced use) optional `ProjectAttentionHeads` layer to
        use for the query projection instead of creating a new one by default.
        This is exposed to enable sharing across multiple `RelativeAttention`
        layers. The expected input/output shapes of this layer must be
        consistent with `total_key_size` and `num_heads`.
      key_projection: (Advanced use) optional `ProjectAttentionHeads` layer to
        use for the key projection instead of creating a new one by default.
        This is exposed to enable sharing across multiple `RelativeAttention`
        layers. The expected input/output shapes of this layer must be
        consistent with `total_key_size` and `num_heads`.
      value_projection: (Advanced use) optional `ProjectAttentionHeads` layer to
        use for the value projection instead of creating a new one by default.
        This is exposed to enable sharing across multiple `RelativeAttention`
        layers. The expected input/output shapes of this layer must be
        consistent with `total_value_size` and `num_heads`.
      qkv_relative_attention: (Advanced use) optional `QkvRelativeAttention`
        layer to use for attention instead of creating a new one by default.
        This is exposed to enable sharing across multiple `RelativeAttention`
        layers. The expected input/output shapes of this layer must be
        consistent with `total_key_size`, `total_value_size`, and `num_heads`.
        If this is given, then the following arguments (that would otherwise be
        used to create a new `QkvRelativeAttention` layer) are ignored:
          `relative_vocab_size`, `att_dropout_prob`, `use_one_hot_lookup`.
      output_projection: (Advanced use) optional Keras Dense layer to use for
        the output projection instead of creating a new one by default. This is
        exposed to enable sharing across multiple `RelativeAttention` layers.
        The expected input/output shapes of this layer must be consistent with
        `total_value_size` and `hidden_size`, respectively.
      use_one_hot_lookup: Whether to use tf.one_hot for embedding lookup instead
        of tf.gather. Default is False, but setting to True may be more
        efficient on TPUs for vocab sizes that aren't too large.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(RelativeAttention, self).__init__(name=name, **kwargs)

    if total_key_size is None:
      total_key_size = hidden_size
    if total_value_size is None:
      total_value_size = hidden_size

    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.total_key_size = total_key_size
    self.total_value_size = total_value_size
    self.relative_vocab_size = relative_vocab_size
    self.att_dropout_prob = att_dropout_prob
    self.initializer = initializer
    self.query_projection = query_projection
    self.key_projection = key_projection
    self.value_projection = value_projection
    self.qkv_relative_attention = qkv_relative_attention
    self.output_projection = output_projection
    self.use_one_hot_lookup = use_one_hot_lookup

    if total_key_size % num_heads != 0:
      raise ValueError('`total_key_size` must be a multiple of `num_heads`.')
    if total_value_size % num_heads != 0:
      raise ValueError('`total_value_size` must be a multiple of `num_heads`.')

    if self.query_projection is None:
      self.query_projection = ProjectAttentionHeads(
          num_heads=num_heads,
          size_per_head=total_key_size // num_heads,
          use_bias=True,
          initializer=initializer,
          name='query_projection')
    if self.key_projection is None:
      self.key_projection = ProjectAttentionHeads(
          num_heads=num_heads,
          size_per_head=total_key_size // num_heads,
          use_bias=True,
          initializer=initializer,
          name='key_projection')
    if self.value_projection is None:
      self.value_projection = ProjectAttentionHeads(
          num_heads=num_heads,
          size_per_head=total_value_size // num_heads,
          use_bias=True,
          initializer=initializer,
          name='value_projection')
    if self.qkv_relative_attention is None:
      self.qkv_relative_attention = QkvRelativeAttention(
          relative_vocab_size=relative_vocab_size,
          att_dropout_prob=att_dropout_prob,
          use_one_hot_lookup=use_one_hot_lookup)
    if self.output_projection is None:
      self.output_projection = _make_output_projection(
          output_size=hidden_size,
          name='output_projection',
          kernel_initializer=initializer)

  def call(self,
           from_seq: tf.Tensor,
           to_seq: Optional[tf.Tensor] = None,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           training=None) -> tf.Tensor:
    """Calls the layer, attending from `from_seq` to `to_seq`.

    Args:
      from_seq: <float32>[batch_size, from_seq_len, from_hidden_size].
      to_seq: <float32>[batch_size, to_seq_len, to_hidden_size]. If left as
        None, we use `from_seq` as `to_seq`, resulting in self-attention.
      att_mask: <int32>[batch_size, from_seq_len, to_seq_len]. Should have only
        0 and 1 values, with 0 for entries that should be masked and 1
        otherwise. Leave as None to allow all elements to attend to all other
        elements within each example.
      relative_att_ids: <int32>[batch_size, from_seq_len, to_seq_len]. Leave as
        None to skip the relative portion of attention.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
      <float32>[batch_size, from_seq_len, hidden_size].
    """
    if to_seq is None:
      # Perform self-attention.
      to_seq = from_seq

    # [batch_size, from_seq_len, num_heads, query_size_per_head]
    # Note: query_size_per_head = key_size_per_head by definition.
    queries = self.query_projection(from_seq, training=training)

    # [batch_size, to_seq_len, num_heads, key_size_per_head]
    keys = self.key_projection(to_seq, training=training)

    # [batch_size, to_seq_len, num_heads, value_size_per_head]
    values = self.value_projection(to_seq, training=training)

    # [batch_size, from_seq_len, num_heads, value_size_per_head]
    att_output = self.qkv_relative_attention(
        queries=queries,
        keys=keys,
        values=values,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        training=training)

    # [batch_size, from_seq_len, num_heads * value_size_per_head]
    flat_att_output = tensor_utils.flatten_dims(att_output, first_dim=-2)

    # [batch_size, from_seq_len, hidden_size]
    return self.output_projection(flat_att_output, training=training)


class FusedGlobalLocalAttention(tf.keras.layers.Layer):
  """Global-local attention used in `GlobalLocalTransformerLayers`.

  We call this layer "fused" since the l2l and l2g attention operations are
  fused together under 1 attention softmax, as are the g2g and g2l attention
  operations.  This formulation makes standard Transformer attention a
  special case of fused attention given the following conditions:
    1. The local_radius for local self-attention covers the entire "long" input.
    2. The query, key, value, and output projections are shared between l2l,
      g2g, l2g, and g2l.
    3. The global memory tokens would be concatenated with the long input tokens
      to form the input to standard Transformer.

  The connection with standard Transformer raises the possibility of directly
  lifting the weights of a standard Transformer model into a fused attention
  `GlobalLocalTransformerLayers` model to fine-tune on larger inputs.

  See `GlobalLocalTransformerLayers` for more details about the long and global
  inputs expected.
  """

  def __init__(self,
               long_hidden_size: int,
               global_hidden_size: int,
               num_heads: int,
               local_radius: int,
               long_total_att_size: Optional[int] = None,
               global_total_att_size: Optional[int] = None,
               relative_vocab_size: Optional[int] = None,
               att_dropout_prob: float = 0.0,
               initializer=None,
               share_kv_projections: bool = False,
               share_qkv_projections: bool = False,
               share_att_output_projection: bool = False,
               use_one_hot_lookup: bool = False,
               name: Text = 'fused_global_local_att',
               **kwargs):
    """Init.

    Args:
      long_hidden_size: Size of the long input hidden dimension. This will also
        be the size of the long output and intermediate queries/keys/values.
      global_hidden_size: Size of the global input hidden dimension. This will
        also be the size of the global output and intermediate
        queries/keys/values.
      num_heads: Number of attention heads.
      local_radius: How many tokens to the left/right to locally attend to for
        long-to-long attention. For example, a value of 1 would allow each token
        to only attend to 1 token to the left and 1 token to the right of it.
      long_total_att_size: Total size of the long attention query/key/value
        vectors after concatenating all heads. Defaults to `long_hidden_size`.
        Must be a multiple of `num_heads`.
      global_total_att_size: Total size of the global attention query/key/value
        vectors after concatenating all heads. Defaults to `global_hidden_size`.
        Must be a multiple of `num_heads`.
      relative_vocab_size: Size of relative position vocabulary. If left
        unspecified, relative positions will be ignored for attention.
      att_dropout_prob: Dropout probability for attention probabilities. Must be
        between 0.0 and 1.0. The default of 0.0 skips dropout.
      initializer: Initializer to use for non-bias variables other than the
        relative embedding table. Bias variables will be initialized to 0,
        and the relative embedding table has its own default initialization
        scale.
      share_kv_projections: If True, key and value projections will be shared
        between long-to-long and long-to-global components, as well as between
        global-to-global and global-to-long components. This results in 2 key
        projections per layer instead of 4 (and similarly for value
        projections). Note that if `share_qkv_projections` is True, then
        `share_kv_projections` is completely ignored since the former results
        in even more sharing.
      share_qkv_projections: If True, all attention components (long-to-long,
        global-to-global, long-to-global, and global-to-long) will share the
        same query, key, and value projections.
      share_att_output_projection: If True, both long and global attention
        results will share the same output projection per layer.
      use_one_hot_lookup: Whether to use tf.one_hot for embedding lookup instead
        of tf.gather. Default is False, but setting to True may be more
        efficient on TPUs for vocab sizes that aren't too large. Currently this
        is only used during lookup of relative position embeddings.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(FusedGlobalLocalAttention, self).__init__(name=name, **kwargs)

    if long_total_att_size is None:
      long_total_att_size = long_hidden_size
    if global_total_att_size is None:
      global_total_att_size = global_hidden_size

    self.long_hidden_size = long_hidden_size
    self.global_hidden_size = global_hidden_size
    self.num_heads = num_heads
    self.local_radius = local_radius
    self.long_total_att_size = long_total_att_size
    self.global_total_att_size = global_total_att_size
    self.relative_vocab_size = relative_vocab_size
    self.att_dropout_prob = att_dropout_prob
    self.initializer = initializer
    self.share_kv_projections = share_kv_projections
    self.share_qkv_projections = share_qkv_projections
    self.share_att_output_projection = share_att_output_projection
    self.use_one_hot_lookup = use_one_hot_lookup

    self._validate_init_parameters()

    def make_att_head_projection(total_att_size, name):
      return ProjectAttentionHeads(
          num_heads=num_heads,
          size_per_head=total_att_size // num_heads,
          use_bias=True,
          initializer=initializer,
          name=name)

    # Long attention layers

    self.long_query_projection = make_att_head_projection(
        long_total_att_size, 'long_query_projection')

    self.l2l_key_projection = make_att_head_projection(long_total_att_size,
                                                       'l2l_key_projection')
    if share_qkv_projections or share_kv_projections:
      self.l2g_key_projection = self.l2l_key_projection
    else:
      self.l2g_key_projection = make_att_head_projection(
          long_total_att_size, 'l2g_key_projection')

    self.l2l_value_projection = make_att_head_projection(
        long_total_att_size, 'l2l_value_projection')
    if share_qkv_projections or share_kv_projections:
      self.l2g_value_projection = self.l2l_value_projection
    else:
      self.l2g_value_projection = make_att_head_projection(
          long_total_att_size, 'l2g_value_projection')

    self.long_qkv_attention = QkvRelativeLocalAttention(
        local_radius=local_radius,
        relative_vocab_size=relative_vocab_size,
        att_dropout_prob=att_dropout_prob,
        use_one_hot_lookup=use_one_hot_lookup,
        name='long_qkv_attention')

    self.long_output_projection = _make_output_projection(
        output_size=long_hidden_size,
        name='long_output_projection',
        kernel_initializer=initializer)

    # Global attention layers

    if share_qkv_projections:
      self.global_query_projection = self.long_query_projection
      self.g2g_key_projection = self.l2l_key_projection
      self.g2l_key_projection = self.l2l_key_projection
      self.g2g_value_projection = self.l2l_value_projection
      self.g2l_value_projection = self.l2l_value_projection
    else:
      self.global_query_projection = make_att_head_projection(
          global_total_att_size, 'global_query_projection')
      self.g2g_key_projection = make_att_head_projection(
          global_total_att_size, 'g2g_key_projection')
      self.g2g_value_projection = make_att_head_projection(
          global_total_att_size, 'g2g_value_projection')
      if share_kv_projections:
        self.g2l_key_projection = self.g2g_key_projection
        self.g2l_value_projection = self.g2g_value_projection
      else:
        self.g2l_key_projection = make_att_head_projection(
            global_total_att_size, 'g2l_key_projection')
        self.g2l_value_projection = make_att_head_projection(
            global_total_att_size, 'g2l_value_projection')

    self.global_qkv_attention = QkvRelativeAttention(
        relative_vocab_size=relative_vocab_size,
        att_dropout_prob=att_dropout_prob,
        use_one_hot_lookup=use_one_hot_lookup,
        name='global_qkv_attention')

    if share_att_output_projection:
      self.global_output_projection = self.long_output_projection
    else:
      self.global_output_projection = _make_output_projection(
          output_size=global_hidden_size,
          name='global_output_projection',
          kernel_initializer=initializer)

  def call(self,
           long_input: tf.Tensor,
           global_input: tf.Tensor,
           l2l_att_mask: Optional[tf.Tensor] = None,
           g2g_att_mask: Optional[tf.Tensor] = None,
           l2g_att_mask: Optional[tf.Tensor] = None,
           g2l_att_mask: Optional[tf.Tensor] = None,
           l2l_relative_att_ids: Optional[tf.Tensor] = None,
           g2g_relative_att_ids: Optional[tf.Tensor] = None,
           l2g_relative_att_ids: Optional[tf.Tensor] = None,
           g2l_relative_att_ids: Optional[tf.Tensor] = None,
           att_implementation: Text = 'auto',
           training=None) -> List[tf.Tensor]:
    """Calls the layer.

    We use abbreviations like "l2g" to mean "long-to-global".

    Args:
      long_input: <float32>[batch_size, long_seq_len, long_hidden_size].
      global_input: <float32>[batch_size, global_seq_len, global_hidden_size].
      l2l_att_mask: <int32>[batch_size, long_seq_len,  2*local_radius + 1]
        long-to-long attention mask for local attention. Should have only 0 and
        1 values, with 0 for entries that should be masked and 1 otherwise.
        Leave as None to allow all long elements to attend to all other long
        elements within the local radius.
      g2g_att_mask: <int32>[batch_size, global_seq_len, global_seq_len]
        global-to-global attention mask. Should have only 0 and 1 values, with 0
        for entries that should be masked and 1 otherwise. Leave as None to
        allow all global elements to attend to all other global elements within
        each example.
      l2g_att_mask: <int32>[batch_size, long_seq_len, global_seq_len]
        long-to-global attention mask. Should have only 0 and 1 values, with 0
        for entries that should be masked and 1 otherwise. Leave as None to
        allow all long elements to attend to all global elements within each
        example.
      g2l_att_mask: <int32>[batch_size, global_seq_len, long_seq_len]
        global-to-long attention mask. Should have only 0 and 1 values, with 0
        for entries that should be masked and 1 otherwise. Leave as None to
        allow all global elements to attend to all long elements within each
        example.
      l2l_relative_att_ids: <int32>[batch_size, long_seq_len, 2*local_radius+1]
        long-to-long relative local self-attention ids. Leave as None to skip
        the relative portion of l2l attention.
      g2g_relative_att_ids: <int32>[batch_size, global_seq_len, global_seq_len]
        global-to-global relative attention ids. Leave as None to skip the
        relative portion of g2g attention.
      l2g_relative_att_ids: <int32>[batch_size, long_seq_len, global_seq_len]
        long-to-global relative attention ids. Leave as None to skip the
        relative portion of l2g attention.
      g2l_relative_att_ids: <int32>[batch_size, global_seq_len, long_seq_len]
        global-to-long relative attention ids. Leave as None to skip the
        relative portion of g2l attention.
      att_implementation: String representing which internal attention
        implementation to use. Valid values include 'auto' (the default),
        'sparse', and 'full'. 'sparse' is preferred for sequences longer than
        about 1k tokens, but 'full' may be faster for sequences shorter than
        this. 'auto' attempts to automatically decide when to use full
        attention. See `QkvRelativeLocalAttention` for more details.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
      A list of Tensors, [long_output, global_output]:
        long_output: <float32>[batch_size, long_seq_len, long_hidden_size]
        global_output: <float32>[batch_size, global_seq_len, global_hidden_size]
    """
    if (g2g_relative_att_ids is None) != (g2l_relative_att_ids is None):
      raise ValueError(
          '`g2g_relative_att_ids` and `g2l_relative_att_ids` must be either '
          'both present or both absent.')

    batch_size = tf.shape(long_input)[0]
    long_seq_len = tf.shape(long_input)[1]
    global_seq_len = tf.shape(global_input)[1]

    # Make sure the global attention masks are not None so we can concatenate
    # these masks together.  The local attention masks are handled separately
    # within `QkvRelativeLocalAttention`.
    if g2g_att_mask is None:
      g2g_att_mask = tf.ones([batch_size, global_seq_len, global_seq_len],
                             dtype=tf.int32)
    if g2l_att_mask is None:
      g2l_att_mask = tf.ones([batch_size, global_seq_len, long_seq_len],
                             dtype=tf.int32)

    # Long attention

    long_queries = self.long_query_projection(long_input, training=training)
    l2l_keys = self.l2l_key_projection(long_input, training=training)
    l2g_keys = self.l2g_key_projection(global_input, training=training)
    l2l_values = self.l2l_value_projection(long_input, training=training)
    l2g_values = self.l2g_value_projection(global_input, training=training)

    # [batch_size, long_seq_len, num_heads, long_hidden_size // num_heads]
    long_att_output = self.long_qkv_attention(
        queries=long_queries,
        keys=l2l_keys,
        values=l2l_values,
        att_mask=l2l_att_mask,
        relative_att_ids=l2l_relative_att_ids,
        side_keys=l2g_keys,
        side_values=l2g_values,
        side_att_mask=l2g_att_mask,
        side_relative_att_ids=l2g_relative_att_ids,
        att_implementation=att_implementation,
        training=training)

    # [batch_size, long_seq_len, long_hidden_size]
    flat_long_att_output = tensor_utils.flatten_dims(
        long_att_output, first_dim=-2)

    # [batch_size, long_seq_len, long_hidden_size]
    long_output = self.long_output_projection(
        flat_long_att_output, training=training)

    # Global attention

    global_queries = self.global_query_projection(
        global_input, training=training)
    # TODO(jainslie): Consider using the long attention key/value projection
    # results above when `share_qkv_projections` is True, although graph
    # optimization may already dedup these computations.
    g2g_keys = self.g2g_key_projection(global_input, training=training)
    g2l_keys = self.g2l_key_projection(long_input, training=training)
    g2g_values = self.g2g_value_projection(global_input, training=training)
    g2l_values = self.g2l_value_projection(long_input, training=training)

    # [batch_size, global_seq_len + long_seq_len, num_heads,
    #   global_hidden_size // num_heads]
    concat_keys = tf.concat([g2g_keys, g2l_keys], axis=1)
    concat_values = tf.concat([g2g_values, g2l_values], axis=1)

    # [batch_size, global_seq_len, global_seq_len + long_seq_len]
    concat_att_mask = tf.concat([g2g_att_mask, g2l_att_mask], axis=-1)
    if g2g_relative_att_ids is None:
      concat_relative_att_ids = None
    else:
      concat_relative_att_ids = tf.concat(
          [g2g_relative_att_ids, g2l_relative_att_ids], axis=-1)

    # [batch_size, global_seq_len, num_heads, global_hidden_size // num_heads]
    global_att_output = self.global_qkv_attention(
        queries=global_queries,
        keys=concat_keys,
        values=concat_values,
        att_mask=concat_att_mask,
        relative_att_ids=concat_relative_att_ids,
        training=training)

    # [batch_size, global_seq_len, global_hidden_size]
    flat_global_att_output = tensor_utils.flatten_dims(
        global_att_output, first_dim=-2)

    # [batch_size, global_seq_len, global_hidden_size]
    global_output = self.global_output_projection(
        flat_global_att_output, training=training)

    return [long_output, global_output]

  def _validate_init_parameters(self) -> None:
    if self.long_total_att_size % self.num_heads != 0:
      raise ValueError(
          '`long_total_att_size` must be a multiple of `num_heads`.')
    if self.global_total_att_size % self.num_heads != 0:
      raise ValueError(
          '`global_total_att_size` must be a multiple of `num_heads`.')

    share_qkv_or_output_projections = (
        self.share_qkv_projections or self.share_att_output_projection)
    if (share_qkv_or_output_projections and
        self.long_hidden_size != self.global_hidden_size):
      raise ValueError(
          '`long_hidden_size` must equal `global_hidden_size` when '
          '`share_qkv_projections` or `share_att_output_projection` is True.')
    if (share_qkv_or_output_projections and
        self.long_total_att_size != self.global_total_att_size):
      raise ValueError(
          '`long_total_att_size` must equal `global_total_att_size` when '
          '`share_qkv_projections` or `share_att_output_projection` is True.')
    if (self.share_kv_projections and
        self.long_hidden_size != self.global_hidden_size):
      raise ValueError(
          '`long_hidden_size` must equal `global_hidden_size` when '
          '`share_kv_projections` is True.')


class ProjectAttentionHeads(tf.keras.layers.Layer):
  """Layer for projecting a sequence to multi-head queries/keys/values."""

  def __init__(self,
               num_heads: int,
               size_per_head: int,
               use_bias: bool = True,
               initializer=None,
               name: Text = 'attention_head_projection',
               **kwargs):
    """Init.

    Args:
      num_heads: Number of attention heads.
      size_per_head: Output size of each head.
      use_bias: Whether to add a bias to the result. Default True.
      initializer: Initializer to use for the kernel. The bias will be
        initialized to 0.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(ProjectAttentionHeads, self).__init__(name=name, **kwargs)

    if num_heads < 1:
      raise ValueError('`num_heads` must be positive.')
    if size_per_head < 1:
      raise ValueError('`size_per_head` must be positive.')

    self.num_heads = num_heads
    self.size_per_head = size_per_head
    self.use_bias = use_bias
    self.initializer = initializer

    self.linear = tf.keras.layers.Dense(
        units=num_heads * size_per_head,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=initializer,
        bias_initializer='zeros',
        name='linear')

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls the layer.

    Args:
      inputs: <float32>[batch_size, ..., hidden_size].

    Returns:
      <float32>[batch_size, ..., num_heads, size_per_head].
    """
    # [batch_size, ..., num_heads * size_per_head]
    x = self.linear(inputs)

    output_shape = tf.concat(
        [tf.shape(inputs)[:-1], [self.num_heads, self.size_per_head]], 0)
    return tf.reshape(x, output_shape)


class QkvRelativeAttention(tf.keras.layers.Layer):
  """Relative attention layer over queries, keys, and values ("qkv").

  This implements scaled dot-product attention with (optional) relative
  position representations. We allow the user to supply arbitrary
  relative edges in `relative_att_ids` rather than forcing an edge pattern
  based on relative distance.

  For the original scaled dot-product formulation without relative position,
  see https://arxiv.org/abs/1706.03762 .
  For the original formulation of relative position representations, see
  https://arxiv.org/abs/1803.02155 .
  For the relative position formulation with bias terms included, see
  https://arxiv.org/abs/1901.02860 . Note we don't include a global content
  bias term in this implementation since it may already be included in
  the projection that creates `queries`.
  """

  def __init__(
      self,
      relative_vocab_size: Optional[int] = None,
      att_dropout_prob: float = 0.0,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.5),
      use_one_hot_lookup: bool = False,
      name: Text = 'qkv_relative_attention',
      **kwargs):
    """Init.

    Args:
      relative_vocab_size:  Size of relative position vocabulary. If left
        unspecified, relative positions will be ignored for attention.
      att_dropout_prob: Dropout probability for attention probabilities. Must be
        between 0.0 and 1.0. The default of 0.0 skips dropout.
      initializer: Initializer to use for relative embedding table. Ideally, the
        initialization scale should probably be comparable to the expected scale
        of the `keys`.
      use_one_hot_lookup: Whether to use tf.one_hot for relative embedding
        lookup instead of tf.gather. Default is False, but setting to True may
        be more efficient on TPUs for vocab sizes that aren't too large.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(QkvRelativeAttention, self).__init__(name=name, **kwargs)

    if relative_vocab_size is not None and relative_vocab_size < 1:
      raise ValueError('`relative_vocab_size` must be positive.')

    self.relative_vocab_size = relative_vocab_size
    self.att_dropout_prob = att_dropout_prob
    self.initializer = initializer
    self.use_one_hot_lookup = use_one_hot_lookup

    if att_dropout_prob != 0.0:
      self.att_dropout = recomputing_dropout.RecomputingDropout(
          rate=att_dropout_prob)

    self._is_custom_built = False

  # The Keras `build` method only receives the shape for `queries`, but we need
  # the shape of `values` also, so we created this custom build method.
  def _custom_build(self, queries_shape: tf.TensorShape,
                    values_shape: tf.TensorShape) -> None:
    """Build function with custom shape arguments.

    Args:
      queries_shape: [batch_size, query_len, num_heads, key_size_per_head] shape
        of queries Tensor.
      values_shape: [batch_size, key_len, num_heads, value_size_per_head] shape
        of values Tensor.
    """
    num_heads = queries_shape.as_list()[-2]
    key_size_per_head = queries_shape.as_list()[-1]

    with tf.init_scope():  # Make sure this happens in eager.
      if self.relative_vocab_size is not None:
        self.relative_emb_table = self.add_weight(
            name='relative_emb_table',
            shape=[self.relative_vocab_size, num_heads, key_size_per_head],
            initializer=self.initializer,
            trainable=True)
        self.relative_bias_table = self.add_weight(
            name='relative_bias_table',
            shape=[self.relative_vocab_size, num_heads],
            initializer='zeros',
            trainable=True)

    self._is_custom_built = True

  def call(self,
           queries: tf.Tensor,
           keys: tf.Tensor,
           values: tf.Tensor,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           training=None) -> tf.Tensor:
    """Calls the layer.

    Args:
      queries: <float32>[batch_size, query_len, num_heads, key_size_per_head].
      keys: <float32>[batch_size, key_len, num_heads, key_size_per_head].
      values: <float32>[batch_size, key_len, num_heads, value_size_per_head].
      att_mask: <int32>[batch_size, query_len, key_len]. Should have only 0 and
        1 values, with 0 for keys that should be masked and 1 otherwise. Leave
        as None to allow all elements to attend to all other elements within
        each example.
      relative_att_ids: <int32>[batch_size, query_len, key_len]. Leave as None
        to skip the relative portion of attention.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
      <float32>[batch_size, query_len, num_heads, value_size_per_head].
    """
    if not self._is_custom_built:
      self._custom_build(queries_shape=queries.shape, values_shape=values.shape)

    if relative_att_ids is not None and self.relative_vocab_size is None:
      raise ValueError('Cannot use `relative_att_ids` without specifying '
                       '`relative_vocab_size`.')

    # `queries` shape: [batch_size, query_len, num_heads, key_size_per_head]
    # `keys` shape: [batch_size, key_len, num_heads, key_size_per_head]

    # [batch_size, query_len, key_len, num_heads]
    content_att_scores = tf.einsum('bqhd,bkhd->bqkh', queries, keys)

    # TODO(jainslie): Add option to overlay sinusoid encoding matrix in
    # addition to (or instead of) explicit relative position embeddings.
    if relative_att_ids is None:
      att_scores = content_att_scores
    else:
      # `self.relative_emb_table` shape:
      # [relative_vocab_size, num_heads, key_size_per_head]

      # [batch_size, query_len, relative_vocab_size, num_heads]
      all_relative_scores = tf.einsum('bqhd,rhd->bqrh', queries,
                                      self.relative_emb_table)
      all_relative_scores += self.relative_bias_table

      # `relative_att_ids` shape: [batch_size, query_len, key_len]

      # `relative_att_scores` shape: [batch_size, query_len, key_len, num_heads]
      if self.use_one_hot_lookup:
        relative_att_scores = tensor_utils.batch_gather_by_one_hot(
            all_relative_scores, relative_att_ids, batch_dims=2)
      else:
        relative_att_scores = tf.gather(
            all_relative_scores, relative_att_ids, batch_dims=2)

      att_scores = content_att_scores + relative_att_scores

    key_size_per_head = tf.shape(keys)[-1]

    # `att_scores` shape: [batch_size, query_len, key_len, num_heads]
    att_scores /= tf.sqrt(tf.cast(key_size_per_head, att_scores.dtype))

    if att_mask is not None:
      # `att_mask` shape: [batch_size, query_len, key_len]

      # [batch_size, query_len, key_len, 1]
      mask_adder = -10000.0 * (
          1.0 - tf.cast(att_mask[:, :, :, tf.newaxis], att_scores.dtype))

      att_scores += mask_adder

    # [batch_size, query_len, key_len, num_heads]
    att_probs = tf.nn.softmax(att_scores, axis=-2)

    if self.att_dropout_prob != 0.0:
      # Transpose `att_probs` for a memory layout with no padding on TPUs.
      # [batch_size, num_heads, query_len, key_len]
      att_probs = self.att_dropout(
          tf.transpose(att_probs, [0, 3, 1, 2]), training=training)
      return tf.einsum('bhqk,bkhd->bqhd', att_probs, values)

    # `values` shape: [batch_size, key_len, num_heads, value_size_per_head]

    # [batch_size, query_len, num_heads, value_size_per_head]
    return tf.einsum('bqkh,bkhd->bqhd', att_probs, values)


class QkvRelativeLocalAttention(tf.keras.layers.Layer):
  """Relative local attention layer over queries, keys, and values ("qkv").

  This layer is similar to the `QkvRelativeAttention` layer except it's
  specialized for efficient self-attention over a long input sequence via
  a locality constraint. The layer assumes the long input is already projected
  to queries, keys, and values, and it accepts optional side keys and values
  for every query to attend to also. Efficiency is maintained for long inputs
  by only allowing tokens to attend to other tokens within a `local_radius`,
  resulting in complexity that scales linearly in the long input length
  (for a fixed `local_radius`) rather than quadratically in the input length.

  If the input sequence isn't actually that long (e.g. ~1k tokens or less),
  it may be faster to use the full-attention implementation internally,
  which is available via the `att_implementation` argument to `call`.

  Just like `QkvRelativeAttention`, attention masking and relative attention ids
  can further constrain or customize how attention behaves.
  """

  def __init__(
      self,
      local_radius: int,
      relative_vocab_size: Optional[int] = None,
      att_dropout_prob: float = 0.0,
      initializer=tf.keras.initializers.TruncatedNormal(stddev=0.5),
      use_one_hot_lookup: bool = False,
      name: Text = 'qkv_relative_local_att',
      **kwargs):
    """Init.

    Args:
      local_radius: How many tokens to the left/right to locally attend to. For
        example, a value of 1 would allow each token to only attend to 1 token
        to the left and 1 token to the right of it.
      relative_vocab_size:  Size of relative position vocabulary. If left
        unspecified, relative positions will be ignored for attention.
      att_dropout_prob: Dropout probability for attention probabilities. Must be
        between 0.0 and 1.0. The default of 0.0 skips dropout.
      initializer: Initializer to use for relative embedding table. Ideally, the
        initialization scale should probably be comparable to the expected scale
        of the `keys`.
      use_one_hot_lookup: Whether to use tf.one_hot for relative embedding
        lookup instead of tf.gather. Default is False, but setting to True may
        be more efficient on TPUs for vocab sizes that aren't too large.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(QkvRelativeLocalAttention, self).__init__(name=name, **kwargs)

    if local_radius < 1:
      raise ValueError('`local_radius` must be positive.')

    self.local_radius = local_radius
    self.relative_vocab_size = relative_vocab_size
    self.att_dropout_prob = att_dropout_prob
    self.initializer = initializer
    self.use_one_hot_lookup = use_one_hot_lookup

    self.qkv_relative_attention = QkvRelativeAttention(
        relative_vocab_size=relative_vocab_size,
        att_dropout_prob=att_dropout_prob,
        initializer=initializer,
        use_one_hot_lookup=use_one_hot_lookup)

  def call(self,
           queries: tf.Tensor,
           keys: tf.Tensor,
           values: tf.Tensor,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           side_keys: Optional[tf.Tensor] = None,
           side_values: Optional[tf.Tensor] = None,
           side_att_mask: Optional[tf.Tensor] = None,
           side_relative_att_ids: Optional[tf.Tensor] = None,
           att_implementation: Text = 'auto',
           training=None) -> tf.Tensor:
    """Calls the layer.

    Args:
      queries: <float32>[batch_size, long_len, num_heads, key_size_per_head].
      keys: <float32>[batch_size, long_len, num_heads, key_size_per_head].
      values: <float32>[batch_size, long_len, num_heads, value_size_per_head].
      att_mask: <int32>[batch_size, long_len, 2*local_radius + 1]. For the i-th
        example and j-th long token (with 0-based indexing), `att_mask[i, j, :]`
          is the local attention mask centered around j. It should have only 0
          and 1 values, with 0 for keys that should be masked and 1 otherwise.
          Leave as None to allow all tokens to attend to all other local tokens.
      relative_att_ids: <int32>[batch_size, long_len, 2*local_radius + 1]. Leave
        as None to skip the relative portion of attention.
      side_keys: <float32>[batch_size, side_len, num_heads, key_size_per_head].
        Keys of the optional side inputs for all queries to attend to.
      side_values: <float32>[batch_size, side_len, num_heads,
        value_size_per_head]. Values of the optional side inputs for all queries
        to attend to.
      side_att_mask: <int32>[batch_size, long_len, side_len]. Analogous 0/1 mask
        for side inputs with 0 for side keys that should be masked and 1
        otherwise. Leave as None to allow attention to all side inputs.
      side_relative_att_ids: <int32>[batch_size, long_len, side_len]. Relative
        attention for side inputs. Must be None if and only if
        `relative_att_ids` is None.
      att_implementation: String representing which internal attention
        implementation to use. Valid values include 'auto' (the default),
        'sparse', and 'full'. 'sparse' is preferred for sequences longer than
        about 1k tokens, but 'full' may be faster for sequences shorter than
        this. 'auto' defaults to 'sparse' but chooses 'full' if `long_len` is
        statically known and is no more than 1024 (a number subject to change
        in the future). The 'full' implementation has quadratic time and
        memory complexity in the `long_len`, whereas 'sparse' is roughly
        linear (for fixed `side_len`).
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
      <float32>[batch_size, long_len, num_heads, value_size_per_head].
    """
    if (side_keys is None) != (side_values is None):
      raise ValueError(
          '`side_keys` and `side_values` must be either both present or both '
          'absent.')
    if side_att_mask is not None and side_keys is None:
      raise ValueError(
          '`side_keys` must be present when specifying `side_att_mask`.')
    if side_relative_att_ids is not None and side_keys is None:
      raise ValueError('`side_keys` must be present when specifying '
                       '`side_relative_att_ids`.')
    if (side_keys is not None and (relative_att_ids is None) !=
        (side_relative_att_ids is None)):
      raise ValueError(
          '`relative_att_ids` and `side_relative_att_ids` must be either both '
          'present or both absent when using side inputs.')
    if att_implementation not in ['auto', 'sparse', 'full']:
      raise ValueError(
          '`att_implementation` must be one of ["auto", "sparse", "full"], '
          'but got: "{}"'.format(att_implementation))

    (batch_size, long_len, num_heads,
     value_size_per_head) = tensor_utils.get_shape_list(values)

    static_long_len = values.shape.as_list()[1]
    if att_implementation == 'full' or (att_implementation == 'auto' and
                                        static_long_len is not None and
                                        static_long_len <= 1024):
      return self._call_full_att_implementation(
          queries=queries,
          keys=keys,
          values=values,
          att_mask=att_mask,
          relative_att_ids=relative_att_ids,
          side_keys=side_keys,
          side_values=side_values,
          side_att_mask=side_att_mask,
          side_relative_att_ids=side_relative_att_ids,
          training=training)

    # We add 1 since the token itself is part of the block.
    block_len = self.local_radius + 1

    # [batch_size, num_blocks, block_len, num_heads, key_size_per_head]
    # where num_blocks = ceiling(long_len / block_len)
    blocked_queries = tensor_utils.split_into_blocks(
        queries, block_len=block_len, axis=1)

    num_blocks = tensor_utils.get_shape_list(blocked_queries)[1]

    # [batch_size, num_blocks, 3*block_len, num_heads, key_size_per_head]
    blocked_keys = tensor_utils.concat_3_blocks(
        tensor_utils.split_into_blocks(keys, block_len=block_len, axis=1))

    # [batch_size, num_blocks, 3*block_len, num_heads, value_size_per_head]
    blocked_values = tensor_utils.concat_3_blocks(
        tensor_utils.split_into_blocks(values, block_len=block_len, axis=1))

    if att_mask is None:
      # Create a locality mask so tokens can't attend beyond `local_radius`.
      att_mask = tf.ones([batch_size, long_len, 2 * self.local_radius + 1],
                         dtype=tf.int32)

    # [batch_size, num_blocks, block_len, 3*block_len]
    blocked_att_mask = _expand_local_ids_to_blocks(att_mask)

    # [batch_size, num_blocks, block_len, 3*block_len]
    blocked_relative_att_ids = (None if relative_att_ids is None else
                                _expand_local_ids_to_blocks(relative_att_ids))

    if side_keys is not None:
      if side_att_mask is None:
        side_len = tf.shape(side_keys)[1]
        side_att_mask = tf.ones([batch_size, long_len, side_len],
                                dtype=tf.int32)
      (
          # [batch_size, num_blocks, 3*block_len + side_len, num_heads,
          #   key_size_per_head]
          blocked_keys,
          # [batch_size, num_blocks, 3*block_len + side_len, num_heads,
          #   value_size_per_head]
          blocked_values,
          # [batch_size, num_blocks, block_len, 3*block_len + side_len]
          blocked_att_mask,
          # [batch_size, num_blocks, block_len, 3*block_len + side_len] or None
          blocked_relative_att_ids  #
      ) = self._concat_side_inputs(blocked_keys, blocked_values,
                                   blocked_att_mask, blocked_relative_att_ids,
                                   side_keys, side_values, side_att_mask,
                                   side_relative_att_ids)

    # [batch_size * num_blocks, block_len, num_heads, key_size_per_head]
    flat_blocked_queries = tensor_utils.flatten_dims(
        blocked_queries, last_dim=1)

    # [batch_size * num_blocks, 3*block_len + side_len, num_heads,
    #   key_size_per_head]
    flat_blocked_keys = tensor_utils.flatten_dims(blocked_keys, last_dim=1)

    # [batch_size * num_blocks, 3*block_len + side_len, num_heads,
    #   value_size_per_head]
    flat_blocked_values = tensor_utils.flatten_dims(blocked_values, last_dim=1)

    # [batch_size * num_blocks, block_len, 3*block_len + side_len]
    flat_blocked_att_mask = tensor_utils.flatten_dims(
        blocked_att_mask, last_dim=1)

    # [batch_size * num_blocks, block_len, 3*block_len + side_len]
    flat_blocked_relative_att_ids = (None if blocked_relative_att_ids is None
                                     else tensor_utils.flatten_dims(
                                         blocked_relative_att_ids, last_dim=1))

    # [batch_size * num_blocks, block_len, num_heads, value_size_per_head]
    flat_blocked_att_result = self.qkv_relative_attention(
        queries=flat_blocked_queries,
        keys=flat_blocked_keys,
        values=flat_blocked_values,
        att_mask=flat_blocked_att_mask,
        relative_att_ids=flat_blocked_relative_att_ids,
        training=training)

    # [batch_size, num_blocks * block_len, num_heads, value_size_per_head]
    padded_att_result = tf.reshape(
        flat_blocked_att_result,
        [batch_size, num_blocks * block_len, num_heads, value_size_per_head])

    # [batch_size, long_len, num_heads, value_size_per_head]
    return padded_att_result[:, :long_len, :, :]

  def _concat_side_inputs(
      self, blocked_keys: tf.Tensor, blocked_values: tf.Tensor,
      blocked_att_mask: tf.Tensor,
      blocked_relative_att_ids: Optional[tf.Tensor], side_keys: tf.Tensor,
      side_values: tf.Tensor, side_att_mask: tf.Tensor,
      side_relative_att_ids: Optional[tf.Tensor]
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
    """Concatenates side inputs to blocked long inputs, returning the result."""

    def concat_side_vectors(blocked_vectors, side_vectors):
      """Concatenates side keys or values to blocked keys or values.

      Args:
        blocked_vectors: <float>[batch_size, num_blocks, 3*block_len, num_heads,
          size_per_head]
        side_vectors: <float>[batch_size, side_len, num_heads, size_per_head]

      Returns:
        <float>[batch_size, num_blocks, 3*block_len + side_len, num_heads,
          size_per_head]
      """
      num_blocks = tf.shape(blocked_vectors)[1]

      # [batch_size, num_blocks, side_len, num_heads, size_per_head]
      expanded_side_vectors = tf.tile(side_vectors[:, tf.newaxis, :, :, :],
                                      [1, num_blocks, 1, 1, 1])

      return tf.concat([blocked_vectors, expanded_side_vectors], axis=2)

    # [batch_size, num_blocks, 3*block_len + side_len, num_heads,
    #   key_size_per_head]
    blocked_keys = concat_side_vectors(blocked_keys, side_keys)

    # [batch_size, num_blocks, 3*block_len + side_len, num_heads,
    #   value_size_per_head]
    blocked_values = concat_side_vectors(blocked_values, side_values)

    def concat_side_ids(blocked_ids, side_ids):
      """Concatenates side mask or relative attention ids to blocked ones.

      Args:
        blocked_ids: <int32>[batch_size, num_blocks, block_len, 3*block_len]
        side_ids: <int32>[batch_size, long_len, side_len]

      Returns:
        <int32>[batch_size, num_blocks, block_len, 3*block_len + side_len]
      """
      block_len = blocked_ids.shape.as_list()[2]

      # [batch_size, num_blocks, block_len, side_len]
      blocked_side_ids = tensor_utils.split_into_blocks(
          side_ids, block_len=block_len, axis=1)

      return tf.concat([blocked_ids, blocked_side_ids], axis=-1)

    # [batch_size, num_blocks, block_len, 3*block_len + side_len]
    blocked_att_mask = concat_side_ids(blocked_att_mask, side_att_mask)

    if blocked_relative_att_ids is not None:
      # [batch_size, num_blocks, block_len, 3*block_len + side_len]
      blocked_relative_att_ids = concat_side_ids(blocked_relative_att_ids,
                                                 side_relative_att_ids)

    return (blocked_keys, blocked_values, blocked_att_mask,
            blocked_relative_att_ids)

  def _call_full_att_implementation(
      self,
      queries: tf.Tensor,
      keys: tf.Tensor,
      values: tf.Tensor,
      att_mask: Optional[tf.Tensor] = None,
      relative_att_ids: Optional[tf.Tensor] = None,
      side_keys: Optional[tf.Tensor] = None,
      side_values: Optional[tf.Tensor] = None,
      side_att_mask: Optional[tf.Tensor] = None,
      side_relative_att_ids: Optional[tf.Tensor] = None,
      training=None) -> tf.Tensor:
    """Calls the full-attention implementation.

    Args:
      queries: <float32>[batch_size, long_len, num_heads, key_size_per_head].
      keys: <float32>[batch_size, long_len, num_heads, key_size_per_head].
      values: <float32>[batch_size, long_len, num_heads, value_size_per_head].
      att_mask: <int32>[batch_size, long_len, 2*local_radius + 1].
      relative_att_ids: <int32>[batch_size, long_len, 2*local_radius + 1].
      side_keys: <float32>[batch_size, side_len, num_heads, key_size_per_head].
      side_values: <float32>[batch_size, side_len, num_heads,
        value_size_per_head].
      side_att_mask: <int32>[batch_size, long_len, side_len].
      side_relative_att_ids: <int32>[batch_size, long_len, side_len].
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
      <float32>[batch_size, long_len, num_heads, value_size_per_head].
    """
    batch_size = tensor_utils.get_shape_list(queries)[0]
    long_len = tensor_utils.get_shape_list(queries)[1]

    if att_mask is None:
      att_mask = tf.ones([batch_size, long_len, 2 * self.local_radius + 1],
                         dtype=tf.int32)

    # [batch_size, long_len, long_len]
    skewed_att_mask = tensor_utils.skew_elements_right(
        att_mask, axis=-1)[..., self.local_radius:-self.local_radius]

    if relative_att_ids is None:
      skewed_relative_att_ids = None
    else:
      # [batch_size, long_len, long_len]
      skewed_relative_att_ids = tensor_utils.skew_elements_right(
          relative_att_ids, axis=-1)[..., self.local_radius:-self.local_radius]

    concat_keys = keys
    concat_values = values
    concat_att_mask = skewed_att_mask
    concat_relative_att_ids = skewed_relative_att_ids

    if side_keys is not None:
      concat_keys = tf.concat([concat_keys, side_keys], axis=1)
      concat_values = tf.concat([concat_values, side_values], axis=1)

      if side_att_mask is None:
        side_len = tf.shape(side_keys)[1]
        side_att_mask = tf.ones([batch_size, long_len, side_len],
                                dtype=tf.int32)
      concat_att_mask = tf.concat([concat_att_mask, side_att_mask], axis=-1)

      if concat_relative_att_ids is not None:
        concat_relative_att_ids = tf.concat(
            [concat_relative_att_ids, side_relative_att_ids], axis=-1)

    # [batch_size, long_len, num_heads, value_size_per_head]
    return self.qkv_relative_attention(
        queries=queries,
        keys=concat_keys,
        values=concat_values,
        att_mask=concat_att_mask,
        relative_att_ids=concat_relative_att_ids,
        training=training)


def _make_output_projection(output_size: int, name: Text, kernel_initializer):
  """Helper for output projection."""
  return tf.keras.layers.Dense(
      units=output_size,
      activation=None,
      use_bias=True,
      kernel_initializer=kernel_initializer,
      bias_initializer='zeros',
      name=name)


def _expand_local_ids_to_blocks(local_ids: tf.Tensor,
                                mask_padding_ids: bool = True) -> tf.Tensor:
  """Helper to expand local ids to blocked format.

  Args:
    local_ids: [batch_size, seq_len, 2*local_radius + 1] shaped Tensor. This is
      the shape of the `att_mask` and `relative_att_id` Tensors in the
      `QkvRelativeLocalAttention` layer.
    mask_padding_ids: If True (the default), zero out ids representing attention
      from the first tokens to padding tokens before the start of the sequence
      and from the last tokens to padding tokens after the end of the sequence.
      When the ids are actually attention masks (such that 0 means no attention)
      this removes attention to any padding tokens beyond the boundaries of the
      sequence.

  Returns:
    A Tensor of shape [batch_size, num_blocks, block_len, 3*block_len],
    where block_len = local_radius + 1, and
    num_blocks = ceiling(seq_len / block_len).
  """
  batch_size = tf.shape(local_ids)[0]
  local_window_size = local_ids.shape.as_list()[-1]
  if local_window_size is None:
    raise ValueError('`local_ids.shape[-1]` must be known statically.')
  block_len = local_window_size // 2 + 1

  if mask_padding_ids:
    padding_mask = feature_utils.make_local_segmented_att_mask(
        tf.ones(tf.shape(local_ids)[:-1]), local_radius=block_len - 1)
    local_ids *= tf.cast(padding_mask, local_ids.dtype)

  # [batch_size, num_blocks, block_len, 2*local_radius + 1]
  blocked_ids = tensor_utils.split_into_blocks(
      local_ids, block_len=block_len, axis=1)

  # [batch_size, num_blocks, block_len, 3*block_len - 2]
  skewed_ids = tensor_utils.skew_elements_right(blocked_ids, axis=-1)

  # [batch_size, num_blocks, block_len, 3*block_len]
  result = tf.pad(skewed_ids, [[0, 0], [0, 0], [0, 0], [1, 1]])

  if not mask_padding_ids:
    return result

  ones = tf.ones_like(result, dtype=result.dtype)

  # [batch_size, 1, block_len, 3*block_len]
  leftmost_mask = tf.concat([
      tf.zeros([batch_size, 1, block_len, block_len], dtype=ones.dtype),
      tf.ones([batch_size, 1, block_len, 2 * block_len], dtype=ones.dtype)
  ], -1)

  # [batch_size, 1, block_len, 3*block_len]
  rightmost_mask = tf.concat([
      tf.ones([batch_size, 1, block_len, 2 * block_len], dtype=ones.dtype),
      tf.zeros([batch_size, 1, block_len, block_len], dtype=ones.dtype)
  ], -1)

  result *= tf.concat([leftmost_mask, ones[:, 1:, :, :]], 1)
  result *= tf.concat([ones[:, :-1, :, :], rightmost_mask], 1)
  return result
