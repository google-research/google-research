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

"""Attention layers for ReadTwice."""

from typing import List, Optional, Text, Tuple, Union

import tensorflow.compat.v1 as tf

from readtwice.layers import embedding
from readtwice.layers import recomputing_dropout
from readtwice.layers import tensor_utils


class FusedSideAttention(tf.keras.layers.Layer):
  """Attention layer for Read-It-Twice model, which supports both passes.

  The layer resembles standard Transformer self-attention layer with an optional
  side input. There are key differences with the rest of attentions
  in this file:
  1. `FusedSideAttention` is NOT using relative attention. Thus, it's a
  direct generalization of a standard Transformer layer. This also makes it
  easier to lift weights from pre-trained BERT/RoBERTa models.
  2. While side input is similar to the persistent memory in
  all-attention networks https://arxiv.org/abs/1907.01470, the layer doesn't
  allocate new weights for the side input. Thus, the side input could
  be either learnable parameters or some activations.
  3. There are only main-main and main-side attentions. Side input's
  representation is not being updated by the layer.
  4. Finally, the side input is the same for all main inputs within a batch.
  If one wants to restrict certain main inputs to attend a part of side inputs
  that could be achieved by providing an appropriate att_mask.
  """

  def __init__(self,
               hidden_size,
               num_heads,
               att_dropout_prob = 0.0,
               share_kv_projections = True,
               initializer=None,
               name = 'fused_side_attention',
               **kwargs):
    """Init.

    Args:
      hidden_size: Size of the main input hidden dimension. This will also be
        the size of the main output and intermediate queries/keys/values.
      num_heads: Number of attention heads.
      att_dropout_prob: Dropout probability for attention probabilities. Must be
        between 0.0 and 1.0. The default of 0.0 skips dropout.
      share_kv_projections: If True, key and value projections will be shared
        between main-to-main and main-to-side components. This results in 1 key
        projection per layer instead of 2 (and similarly for value projections).
      initializer: Initializer to use for non-bias variables other than the
        relative embedding table and persistent memory vectors. Bias variables
        will be initialized to 0.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(FusedSideAttention, self).__init__(name=name, **kwargs)

    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.att_dropout_prob = att_dropout_prob
    self.share_kv_projections = share_kv_projections
    self.initializer = initializer

    self._validate_init_parameters()

    def make_att_head_projection(name):
      return ProjectAttentionHeads(
          num_heads=num_heads,
          size_per_head=hidden_size // num_heads,
          use_bias=True,
          initializer=initializer,
          name=name)

    # TODO(urikz): Test if combining projections into one is more efficient
    self.main_query_projection = make_att_head_projection(
        'main_query_projection')
    self.main_key_projection = make_att_head_projection('main_key_projection')
    self.main_value_projection = make_att_head_projection(
        'main_value_projection')
    if self.share_kv_projections:
      self.side_key_projection = self.main_key_projection
      self.side_value_projection = self.main_value_projection
    else:
      self.side_key_projection = make_att_head_projection('side_key_projection')
      self.side_value_projection = make_att_head_projection(
          'side_value_projection')

    if self.att_dropout_prob != 0.0:
      self.att_dropout = recomputing_dropout.RecomputingDropout(
          rate=self.att_dropout_prob)

    self.output_projection = _make_output_projection(
        output_size=self.hidden_size,
        name='output_projection',
        kernel_initializer=initializer)

  def call(self,
           main_input,
           side_input,
           att_mask,
           training=None):
    """Calls the layer.

    Args:
      main_input: <float32>[batch_size, main_seq_len, hidden_size].
      side_input: <float32>[side_seq_len, hidden_size].
      att_mask: <int32>[batch_size, main_seq_len, main_seq_len + side_seq_len]
        Should have only 0 and 1 values, with 0 for entries that should be
        masked and 1 otherwise. Leave as None to skip masking.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
        main_output: <float32>[batch_size, main_seq_len, hidden_size]
    """
    # [batch_size, main_seq_len, num_heads, size_per_head]
    main_queries = self.main_query_projection(main_input, training=training)
    main_keys = self.main_key_projection(main_input, training=training)
    main_values = self.main_value_projection(main_input, training=training)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # [batch_size, num_heads, main_seq_len, main_seq_len]
    attention_scores = tf.einsum('bqhd,bkhd->bhqk', main_queries, main_keys)

    if side_input is not None:
      # [side_seq_len, num_heads, size_per_head]
      side_keys = self.side_key_projection(side_input, training=training)

      # [batch_size, num_heads, main_seq_len, side_seq_len]
      side_attention_scores = tf.einsum('bqhd,khd->bhqk', main_queries,
                                        side_keys)

      # [batch_size, num_heads, main_seq_len, seq_len]
      attention_scores = tf.concat([attention_scores, side_attention_scores],
                                   axis=3)

    size_per_head = tf.shape(main_keys)[-1]
    attention_scores /= tf.sqrt(tf.cast(size_per_head, attention_scores.dtype))

    if att_mask is not None:
      # [batch_size, 1, main_seq_len, seq_len]
      att_mask = tf.expand_dims(att_mask, axis=1)

      # Since `att_mask` is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and a large negative for masked positions.
      adder = _large_compatible_negative(attention_scores.dtype) * (
          1.0 - tf.cast(att_mask, attention_scores.dtype))

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += adder

    # [batch_size, num_heads, main_seq_len, seq_len]
    attention_probs = tf.nn.softmax(attention_scores)

    if self.att_dropout_prob != 0.0:
      attention_probs = self.att_dropout(attention_probs, training=training)

    if side_input is not None:
      side_seq_len = tf.shape(side_input)[0]

      # [side_seq_len, num_heads, size_per_head]
      side_values = self.side_value_projection(side_input, training=training)

      # [batch_size, main_seq_len, num_heads, size_per_head]
      main_att_output = tf.einsum('bhqk,bkhd->bqhd',
                                  attention_probs[:, :, :, :-side_seq_len],
                                  main_values)
      # [batch_size, main_seq_len, num_heads, size_per_head]
      side_att_output = tf.einsum('bhqk,khd->bqhd',
                                  attention_probs[:, :, :,
                                                  -side_seq_len:], side_values)

      att_output = main_att_output + side_att_output
    else:
      # [batch_size, main_seq_len, num_heads, size_per_head]
      att_output = tf.einsum('bhqk,bkhd->bqhd', attention_probs, main_values)

    # [batch_size, main_seq_len, num_heads * size_per_head]
    flat_att_output = tensor_utils.flatten_dims(att_output, first_dim=-2)

    # [batch_size, main_seq_len, hidden_size]
    return self.output_projection(flat_att_output, training=training)

  def _validate_init_parameters(self):
    if self.hidden_size % self.num_heads != 0:
      raise ValueError('`hidden_size` must be a multiple of `num_heads`.')


class SideAttention(tf.keras.layers.Layer):
  """Side attention layer for Read-It-Twice model.

  The layer resembles standard Transformer cross-attention layer typically used
  in the decoder. The only difference is that keys and values are shared for all
  of the queries.
  """

  def __init__(self,
               hidden_size,
               num_heads,
               att_dropout_prob = 0.0,
               enable_default_side_input = False,
               initializer=None,
               top_k_attention=None,
               pos_embed_mode = None,
               pos_embed_size = None,
               use_one_hot_embeddings = None,
               name = 'fused_side_attention',
               **kwargs):
    """Init.

    Args:
      hidden_size: Size of the main input hidden dimension. This will also be
        the size of the main output and intermediate queries/keys/values.
      num_heads: Number of attention heads. Must be greater or equal than 0,
        where 0 heads means that cross attention layer will have a single
        attention head WITHOUT projection matrices.
      att_dropout_prob: Dropout probability for attention probabilities. Must be
        between 0.0 and 1.0. The default of 0.0 skips dropout.
      enable_default_side_input: Add a default side input, which acts like a
        no-op attention, effective allowing attention weights to sum up
        to something less than 1.
      initializer: Initializer to use for non-bias variables other than the
        relative embedding table and persistent memory vectors. Bias variables
        will be initialized to 0.
      top_k_attention: Whether to restrict attention to the top K items only.
      pos_embed_mode: Whether and how to add positional information.
      pos_embed_size: Max position.
      use_one_hot_embeddings: Whether to use one hot embeddings.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(SideAttention, self).__init__(name=name, **kwargs)

    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.att_dropout_prob = att_dropout_prob
    self.initializer = initializer
    self.enable_default_side_input = enable_default_side_input
    self.top_k_attention = top_k_attention
    self.pos_embed_mode = pos_embed_mode

    self._validate_init_parameters()

    def make_att_head_projection(name):
      if num_heads > 0:
        return ProjectAttentionHeads(
            num_heads=num_heads,
            size_per_head=hidden_size // num_heads,
            use_bias=True,
            initializer=initializer,
            name=name)
      else:
        return None

    self.query_projection = make_att_head_projection('query_projection')
    self.key_projection = make_att_head_projection('key_projection')
    self.value_projection = make_att_head_projection('value_projection')

    if self.num_heads > 0:
      self.output_projection = _make_output_projection(
          output_size=self.hidden_size,
          name='output_projection',
          kernel_initializer=initializer)
    else:
      self.output_projection = tf.keras.layers.Layer()

    if self.att_dropout_prob != 0.0:
      self.att_dropout = recomputing_dropout.RecomputingDropout(
          rate=self.att_dropout_prob)

    if self.pos_embed_mode in [
        'absolute', 'absolute_add_ln', 'simple_relative', 'query_dot_relative'
    ]:
      if pos_embed_size is None:
        raise ValueError('pos_embed_size` must be not None when '
                         '`pos_embed_mode` is not None')
      if use_one_hot_embeddings is None:
        raise ValueError('use_one_hot_embeddings` must be not None when '
                         '`pos_embed_mode` is not None')
      self.pos_embed_size = pos_embed_size
      self.block_position_embedding = embedding.EmbeddingLookup(
          vocab_size=(pos_embed_size if self.pos_embed_mode == 'absolute' else
                      2 * pos_embed_size + 1),
          embedding_size=(max(self.num_heads, 1)
                          if self.pos_embed_mode == 'simple_relative' else
                          self.hidden_size),
          initializer_range=0.02,
          use_one_hot_lookup=use_one_hot_embeddings,
          name='block_position_emb_lookup')
      if self.pos_embed_mode == 'absolute_add_ln':
        self.block_position_embedding_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='block_position_emb_layer_norm')

    elif self.pos_embed_mode is None:
      self.block_position_embedding = None
    else:
      raise ValueError('Unknown position embeddings mode: ' +
                       self.pos_embed_mode)

  def build(self, input_shape):
    """Keras build function.

    Args:
      input_shape: TensorShape of the input; unused.
    """
    if self.enable_default_side_input:
      self.default_key = self.add_weight(
          name='default_side_input',
          shape=[self.hidden_size],
          initializer=self.initializer,
          trainable=True)

    super(SideAttention, self).build(input_shape)

  def call(
      self,
      main_input,
      side_input,
      att_mask,
      main_pos = None,
      side_pos = None,
      att_value_mask = None,
      training=None):
    """Calls the layer.

    Args:
      main_input: <float32>[batch_size, main_seq_len, hidden_size].
      side_input: <float32>[side_seq_len, hidden_size].
      att_mask: <int32>[batch_size, main_seq_len, side_seq_len] Should have only
        0 and 1 values, with 0 for entries that should be masked and 1
        otherwise. Leave as None to skip masking.
      main_pos: <float32>[batch_size].
      side_pos: <float32>[side_seq_len].
      att_value_mask: <int32>[batch_size, main_seq_len, side_seq_len] Should
        have only 0 and 1 values, with 0 for entries that should be masked and 1
        otherwise. Leave as None to skip masking.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
        main_output: <float32>[batch_size, main_seq_len, hidden_size]
    """
    if self.num_heads > 0:
      # [batch_size, main_seq_len, num_heads, size_per_head]
      queries = self.query_projection(main_input, training=training)
      # [side_seq_len, num_heads, size_per_head]
      keys = self.key_projection(side_input, training=training)
      # [side_seq_len, num_heads, size_per_head]
      values = self.value_projection(side_input, training=training)
    else:
      # [batch_size, main_seq_len, 1, hidden_size]
      queries = tf.expand_dims(main_input, 2)
      # [side_seq_len, 1, hidden_size]
      keys = tf.expand_dims(side_input, 1)
      # [side_seq_len, 1, hidden_size]
      values = tf.expand_dims(side_input, 1)

    batch_size = tf.shape(queries)[0]
    main_seq_length = tf.shape(queries)[1]
    side_seq_len = tf.shape(keys)[0]
    effective_num_heads = max(self.num_heads, 1)
    head_size = self.hidden_size // effective_num_heads

    if self.block_position_embedding is not None:
      assert main_pos is not None
      assert side_pos is not None

    if self.pos_embed_mode == 'absolute_add_ln':
      # [batch_size, hidden_size]
      main_pos_emb = self.block_position_embedding(main_pos)
      main_pos_emb = tf.reshape(main_pos_emb,
                                [batch_size, 1, effective_num_heads, head_size])
      queries = self.block_position_embedding_norm(queries + main_pos_emb)
      # [side_seq_len, hidden_size]
      side_pos_emb = self.block_position_embedding(side_pos)
      side_pos_emb = tf.reshape(side_pos_emb,
                                [side_seq_len, effective_num_heads, head_size])
      keys = keys + side_pos_emb
      # [1, side_seq_len, num_heads, head_size]
      keys = tf.expand_dims(keys, 0)
      keys = self.block_position_embedding_norm(keys)
      keys = tf.squeeze(keys, 0)

    if self.enable_default_side_input:
      # We are adding a default key element to act as a no-op for attention.
      # [side_seq_len + 1, num_heads, size_per_head]
      keys = tf.concat([
          keys,
          tf.reshape(self.default_key, [1, effective_num_heads, head_size])
      ],
                       axis=0)
      # [values + 1, num_heads, size_per_head]
      values = tf.concat(
          [values, tf.zeros((1, effective_num_heads, head_size))], axis=0)
      if att_mask is not None:
        # Even if `att_mask` was provided it doesn't take the default side input
        # into account. Thus, we have append 1 for the default attention.
        att_mask = tf.concat([
            att_mask,
            tf.cast(
                tf.fill(dims=[batch_size, main_seq_length, 1], value=1),
                dtype=att_mask.dtype)
        ],
                             axis=2)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # [batch_size, num_heads, main_seq_len, side_seq_len]
    attention_scores = tf.einsum('bqhd,khd->bhqk', queries, keys)

    if self.pos_embed_mode == 'absolute':
      # [batch_size, hidden_size]
      main_pos_emb = self.block_position_embedding(main_pos)
      # [batch_size, num_heads, head_size]
      main_pos_emb = tf.reshape(main_pos_emb,
                                [batch_size, effective_num_heads, head_size])
      # [side_seq_len, hidden_size]
      side_pos_emb = self.block_position_embedding(side_pos)
      # [side_seq_len, num_heads, head_size]
      side_pos_emb = tf.reshape(side_pos_emb,
                                [side_seq_len, effective_num_heads, head_size])
      # [batch_size, num_heads, side_seq_len]
      attention_pos_scores = tf.einsum('bhd,khd->bhk', main_pos_emb,
                                       side_pos_emb)
      if self.enable_default_side_input:
        attention_pos_scores = tf.concat([
            attention_pos_scores,
            tf.zeros((batch_size, effective_num_heads, 1))
        ],
                                         axis=2)
      # [batch_size, num_heads, main_seq_len, side_seq_len]
      attention_scores += tf.expand_dims(attention_pos_scores, 2)
    elif self.pos_embed_mode in ['simple_relative', 'query_dot_relative']:
      assert self.pos_embed_size is not None
      # [batch_size, side_seq_len]
      relative_pos = tf.expand_dims(main_pos, 1) - tf.expand_dims(side_pos, 0)
      # Calm down, linter
      # pylint: disable=invalid-unary-operand-type
      relative_pos = tf.math.maximum(relative_pos, -self.pos_embed_size)
      # pylint: enable=invalid-unary-operand-type
      relative_pos = tf.math.minimum(relative_pos, self.pos_embed_size)
      # [batch_size, side_seq_len, hidden_size or num_heads]
      relatibe_pos_emb = self.block_position_embedding(relative_pos +
                                                       self.pos_embed_size)
      if self.pos_embed_mode == 'simple_relative':
        # [batch_size, side_seq_len, num_heads, 1]
        relatibe_pos_emb = tf.reshape(
            relatibe_pos_emb,
            [batch_size, side_seq_len, effective_num_heads, 1])
        # [batch_size, num_heads, 1, side_seq_len]
        relatibe_pos_emb = tf.transpose(relatibe_pos_emb, [0, 2, 3, 1])
        if self.enable_default_side_input:
          relatibe_pos_emb = tf.concat([
              relatibe_pos_emb,
              tf.zeros((batch_size, effective_num_heads, 1, 1))
          ],
                                       axis=3)
        attention_scores += relatibe_pos_emb
      else:
        # [batch_size, side_seq_len, num_heads, head_size]
        relatibe_pos_emb = tf.reshape(
            relatibe_pos_emb,
            [batch_size, side_seq_len, effective_num_heads, head_size])
        # [batch_size, num_heads, main_seq_len, side_seq_len]
        relatibe_pos_scores = tf.einsum('bqhd,bkhd->bhqk', queries,
                                        relatibe_pos_emb)
        if self.enable_default_side_input:
          relatibe_pos_scores = tf.concat([
              relatibe_pos_scores,
              tf.zeros((batch_size, effective_num_heads, main_seq_length, 1))
          ],
                                          axis=3)
        attention_scores += relatibe_pos_scores

    size_per_head = tf.shape(keys)[-1]
    attention_scores /= tf.sqrt(tf.cast(size_per_head, attention_scores.dtype))

    if att_mask is not None:
      # [batch_size, 1, main_seq_len, seq_len]
      att_mask = tf.expand_dims(att_mask, axis=1)

      # Since `att_mask` is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and a large negative for masked positions.
      adder = _large_compatible_negative(attention_scores.dtype) * (
          1.0 - tf.cast(att_mask, attention_scores.dtype))

      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += tf.stop_gradient(adder)

    if self.top_k_attention is not None:
      # [batch_size, num_heads, main_seq_len, K]
      attention_scores, top_k_indices = tf.math.top_k(
          attention_scores, k=self.top_k_attention, sorted=False)
      # [batch_size, num_heads, main_seq_len, K]
      attention_probs = tf.nn.softmax(attention_scores)

      # [num_heads, side_seq_len, head_size]
      values = tf.transpose(values, [1, 0, 2])
      # [num_heads, batch_size, main_seq_len, K]
      top_k_indices = tf.transpose(top_k_indices, [1, 0, 2, 3])
      # [num_heads, batch_size, main_seq_len, K, head_size]
      values_for_top_k_indices = tf.gather(
          values, top_k_indices, batch_dims=1, axis=1)
      # [batch_size, main_seq_len, num_heads, size_per_head]
      att_output = tf.einsum('bhqt,hbqtd->bqhd', attention_probs,
                             values_for_top_k_indices)
    else:
      # [batch_size, num_heads, main_seq_len, seq_len]
      attention_probs = tf.nn.softmax(attention_scores)

      # [batch_size, main_seq_len, num_heads, size_per_head]
      att_output = tf.einsum('bhqk,khd->bqhd', attention_probs, values)

    # [batch_size, main_seq_len, num_heads * size_per_head]
    flat_att_output = tensor_utils.flatten_dims(att_output, first_dim=-2)

    # [batch_size, main_seq_len, hidden_size]
    output = self.output_projection(flat_att_output, training=training)

    if att_value_mask is not None:
      output = output * tf.expand_dims(tf.cast(att_value_mask, tf.float32), -1)

    return output

  def _validate_init_parameters(self):
    if self.num_heads < 0:
      raise ValueError('`num_heads` must be non-negative.')

    if self.num_heads > 0 and self.hidden_size % self.num_heads != 0:
      raise ValueError('`hidden_size` must be a multiple of `num_heads`.')


class ProjectAttentionHeads(tf.keras.layers.Layer):
  """Layer for projecting a sequence to multi-head queries/keys/values."""

  def __init__(self,
               num_heads,
               size_per_head,
               use_bias = True,
               initializer=None,
               name = 'attention_head_projection',
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

  def call(self, inputs):
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


def _make_output_projection(output_size, name, kernel_initializer):
  """Helper for output projection."""
  return tf.keras.layers.Dense(
      units=output_size,
      activation=None,
      use_bias=True,
      kernel_initializer=kernel_initializer,
      bias_initializer='zeros',
      name=name)


def _large_compatible_negative(tensor_type):
  """Large negative number as Tensor.

  This function is necessary because the standard value for epsilon
  in this module (-1e9) cannot be represented using tf.float16

  Args:
    tensor_type: a dtype to determine the type.

  Returns:
    a large negative number.
  """
  if tensor_type == tf.float16:
    return tf.float16.min
  return -1e9
