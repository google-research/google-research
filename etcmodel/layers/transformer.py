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

"""Transformer layers for ETC."""

import functools
from typing import List, Optional, Text

import tensorflow as tf

from etcmodel import tensor_utils
from etcmodel.layers import attention
from etcmodel.layers import recompute_grad as recompute_grad_lib
from etcmodel.layers import wrappers


class GlobalLocalTransformerLayers(tf.keras.layers.Layer):
  """A sequence of Transformer layers with factorized attention for long inputs.

  These layers should accommodate inputs much larger than standard Transformer
  (using full self-attention).  The input is divided between a large "long"
  input and small "global" input.  "Long-to-global", "global-to-global",
  and "global-to-long" attention are all full attention, while "long-to-long"
  attention uses local self-attention.  As a result, computational complexity
  scales linearly with the "long" input length rather than quadratically as in
  standard Transformer.

  The configuration is very similar to `RelativeTransformerLayers` apart
  from the separate "global" and "long" settings.

  See the ETC paper for more details: https://arxiv.org/abs/2004.08483
  """

  def __init__(self,
               long_hidden_size: int,
               global_hidden_size: int,
               num_hidden_layers: int,
               num_attention_heads: int,
               local_radius: int,
               att_size_per_head: Optional[int] = None,
               long_intermediate_size: Optional[int] = None,
               global_intermediate_size: Optional[int] = None,
               hidden_act=tensor_utils.get_activation('gelu'),
               hidden_dropout_prob: float = 0.1,
               attention_probs_dropout_prob: float = 0.1,
               initializer_range: float = 0.02,
               relative_vocab_size: Optional[int] = None,
               share_feed_forward_params: bool = True,
               share_kv_projections: bool = False,
               share_qkv_projections: bool = True,
               share_att_output_projection: bool = False,
               use_pre_activation_order: bool = False,
               use_one_hot_lookup: bool = False,
               grad_checkpointing_period: int = 0,
               name: Text = 'global_local_transformer_layers',
               **kwargs):
    """Init.

    Args:
      long_hidden_size: Size of the long input hidden dimension.
      global_hidden_size: Size of the global input hidden dimension. If this is
        different from `long_hidden_size`, you must turn off parameter sharing
        between long and global operations. In particular, the following
        sharing options which default to True must be set to False instead:
          `share_feed_forward_params`
          `share_qkv_projections`
      num_hidden_layers: Number of Transformer layers.  Each layer includes both
        an attention sublayer and a feed-forward sublayer.
      num_attention_heads: Number of attention heads for global-local attention.
        Must evenly divide both `global_hidden_size` and `long_hidden_size`
        unless `att_size_per_head` is specified.
      local_radius: How many tokens to the left/right for long input tokens to
        locally self-attend to. For example, a value of 1 would allow each token
        to only attend to 1 token to the left and 1 token to the right of it.
      att_size_per_head: Size of attention query/key/value vectors per head.
        By default this will be `long_hidden_size / num_attention_heads`, so
        `num_attention_heads` must evenly divide `long_hidden_size` in this
        case.
      long_intermediate_size: The size of the "intermediate" (i.e. feed-forward)
        layers for long input. Defaults to 4 * long_hidden_size.
      global_intermediate_size: The size of the "intermediate" (i.e.
        feed-forward) layers for global input. Defaults to 4 *
        global_hidden_size. Must not be different from `long_intermediate_size`
        if `share_feed_forward_params` is True (the default).
      hidden_act: The non-linear activation function in the intermediate layers.
      hidden_dropout_prob: The dropout probability for the attention and
        feed-forward residual blocks. Must be between 0.0 and 1.0.
      attention_probs_dropout_prob: Dropout probability for attention
        probabilities. Must be between 0.0 and 1.0.
      initializer_range: The standard deviation of the truncated normal
        initializer for initializing all weight matrices.
      relative_vocab_size: Size of relative position vocabulary. If left
        unspecified, relative positions will be ignored for attention.
      share_feed_forward_params: If True (the default), we share the same
        fully connected feed-forward parameters for the long and global inputs.
      share_kv_projections: If True, key and value projections will be shared
        between long-to-long and long-to-global components, as well as between
        global-to-global and global-to-long components. This results in 2 key
        projections per layer instead of 4 (and similarly for value
        projections). Note that if `share_qkv_projections` is True, then
        `share_kv_projections` is completely ignored since the former results
        in even more sharing.
      share_qkv_projections: If True (the default), all 4 attention operations
        (long-to-long, global-to-global, long-to-global, and global-to-long)
        will share the same query, key, and value projections. The 3 projections
        will still be different from each other and different per layer.
      share_att_output_projection: If True, all 4 attention operations
        (long-to-long, global-to-global, long-to-global, and global-to-long)
        will share the same output projection per layer.
      use_pre_activation_order: If True, use "pre-activation" order for residual
        blocks (see ResidualBlock docstring).
      use_one_hot_lookup: Whether to use tf.one_hot for embedding lookup instead
        of tf.gather. Default is False, but setting to True may be more
        efficient on TPUs for vocab sizes that aren't too large. Currently this
        is only used during lookup of relative position embeddings.
      grad_checkpointing_period: How often to checkpoint activations. The
        default of 0 stores all activations. If greater than 0, activations are
        recomputed as necessary when calculating gradients to save memory. As an
        optimization, we avoid recomputing the last `grad_checkpointing_period`
        layers, so larger values result in less computational overhead but
        reduced memory savings. Using a value of `1` results in potentially the
        greatest memory savings but with the highest recompute cost.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(GlobalLocalTransformerLayers, self).__init__(name=name, **kwargs)

    if long_intermediate_size is None:
      long_intermediate_size = 4 * long_hidden_size
    if global_intermediate_size is None:
      global_intermediate_size = 4 * global_hidden_size

    (att_size_per_head, long_total_att_size,
     global_total_att_size) = self._resolve_att_sizes(
         att_size_per_head=att_size_per_head,
         long_hidden_size=long_hidden_size,
         global_hidden_size=global_hidden_size,
         num_attention_heads=num_attention_heads)

    self.long_hidden_size = long_hidden_size
    self.global_hidden_size = global_hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.local_radius = local_radius
    self.att_size_per_head = att_size_per_head
    self.long_intermediate_size = long_intermediate_size
    self.global_intermediate_size = global_intermediate_size
    self.hidden_act = hidden_act
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.relative_vocab_size = relative_vocab_size
    self.share_feed_forward_params = share_feed_forward_params
    self.share_kv_projections = share_kv_projections
    self.share_qkv_projections = share_qkv_projections
    self.share_att_output_projection = share_att_output_projection
    self.use_pre_activation_order = use_pre_activation_order
    self.use_one_hot_lookup = use_one_hot_lookup
    self.grad_checkpointing_period = grad_checkpointing_period

    self._long_total_att_size = long_total_att_size
    self._global_total_att_size = global_total_att_size

    self._validate_init_parameters()

    # TODO(jainslie): When using pre-activation order, the recommendation
    # from https://arxiv.org/abs/1904.10509 is to scale some of the
    # initialization by 1 / sqrt(2 * num_hidden_layers).  Add logic
    # to do this scaling (maybe within ResidualBlock rather than through
    # initialization).
    self.initializer = tf.keras.initializers.TruncatedNormal(
        stddev=initializer_range)

    self.fused_att_layers = []
    self.long_feed_forward_layers = []
    self.global_feed_forward_layers = []

    for i in range(num_hidden_layers):
      normalization_layers = [
          tf.keras.layers.LayerNormalization(
              axis=-1, epsilon=1e-12, name='layer_norm_0'),
          tf.keras.layers.LayerNormalization(
              axis=-1, epsilon=1e-12, name='layer_norm_1')
      ]
      self.fused_att_layers.append(
          wrappers.ResidualBlock(
              inner_layer=attention.FusedGlobalLocalAttention(
                  long_hidden_size=long_hidden_size,
                  global_hidden_size=global_hidden_size,
                  num_heads=num_attention_heads,
                  local_radius=local_radius,
                  long_total_att_size=long_total_att_size,
                  global_total_att_size=global_total_att_size,
                  relative_vocab_size=relative_vocab_size,
                  att_dropout_prob=attention_probs_dropout_prob,
                  initializer=self.initializer,
                  share_kv_projections=share_kv_projections,
                  share_qkv_projections=share_qkv_projections,
                  share_att_output_projection=share_att_output_projection,
                  use_one_hot_lookup=use_one_hot_lookup),
              normalization_layer=normalization_layers,
              dropout_probability=self.hidden_dropout_prob,
              use_pre_activation_order=self.use_pre_activation_order,
              name='fused_att_layer_%d' % i))

      if share_feed_forward_params:
        feed_forward_layer = wrappers.ResidualBlock(
            dropout_probability=hidden_dropout_prob,
            use_pre_activation_order=use_pre_activation_order,
            inner_intermediate_size=long_intermediate_size,
            inner_activation=hidden_act,
            inner_kernel_initializer=self.initializer,
            name='feed_forward_layer_%d' % i)
        feed_forward_layer.build(tf.TensorShape([None, long_hidden_size]))
        self.long_feed_forward_layers.append(feed_forward_layer)
        # Create separate layer to generate a new dropout seed.
        self.global_feed_forward_layers.append(
            wrappers.ResidualBlock(
                dropout_probability=hidden_dropout_prob,
                use_pre_activation_order=use_pre_activation_order,
                inner_layer=feed_forward_layer.inner_layer,
                normalization_layer=feed_forward_layer.normalization_layers,
                name='global_feed_forward_layer_%d' % i))
      else:
        self.long_feed_forward_layers.append(
            wrappers.ResidualBlock(
                dropout_probability=hidden_dropout_prob,
                use_pre_activation_order=use_pre_activation_order,
                inner_intermediate_size=long_intermediate_size,
                inner_activation=hidden_act,
                inner_kernel_initializer=self.initializer,
                name='long_feed_forward_layer_%d' % i))
        self.global_feed_forward_layers.append(
            wrappers.ResidualBlock(
                dropout_probability=hidden_dropout_prob,
                use_pre_activation_order=use_pre_activation_order,
                inner_intermediate_size=global_intermediate_size,
                inner_activation=hidden_act,
                inner_kernel_initializer=self.initializer,
                name='global_feed_forward_layer_%d' % i))

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
    long_output = long_input
    global_output = global_input

    def make_layer_fn(index: int):
      """Makes a function that runs the entire `index` layer."""

      def layer_fn(long_input, global_input):
        """A function for an entire layer."""
        long_output = long_input
        global_output = global_input

        long_output, global_output = self.fused_att_layers[index](
            [long_output, global_output],
            l2l_att_mask=l2l_att_mask,
            g2g_att_mask=g2g_att_mask,
            l2g_att_mask=l2g_att_mask,
            g2l_att_mask=g2l_att_mask,
            l2l_relative_att_ids=l2l_relative_att_ids,
            g2g_relative_att_ids=g2g_relative_att_ids,
            l2g_relative_att_ids=l2g_relative_att_ids,
            g2l_relative_att_ids=g2l_relative_att_ids,
            att_implementation=att_implementation,
            training=training)

        # Long and global feed-forward
        long_output = self.long_feed_forward_layers[index](
            long_output, training=training)
        global_output = self.global_feed_forward_layers[index](
            global_output, training=training)

        return (long_output, global_output)

      return layer_fn

    # If `grad_checkpointing_period` is 0 or greater than or equal to the
    # number of layers, no checkpointing will be used.
    stride = (
        self.num_hidden_layers if self.grad_checkpointing_period <= 0 else min(
            self.grad_checkpointing_period, self.num_hidden_layers))
    # Split layers into chains of size `stride`. Put remainder at the beginning.
    for split in range(stride - (-self.num_hidden_layers % stride),
                       self.num_hidden_layers + 1, stride):
      # Chain layers together with max length `stride`.
      layer_fn = functools.partial(
          functools.reduce, lambda outputs, f: f(*outputs),
          list(map(make_layer_fn, range(max(0, split - stride), split))))
      # Destructure arguments for compatibility with `recompute_grad`.
      layer_fn = functools.partial(lambda f, *args: f(args), layer_fn)
      # Skip the last block. Store activations for gradient computation.
      if split < self.num_hidden_layers:
        layer_fn = recompute_grad_lib.recompute_grad(layer_fn)
      long_output, global_output = layer_fn(long_output, global_output)

    return [long_output, global_output]

  def _resolve_att_sizes(self, att_size_per_head, long_hidden_size,
                         global_hidden_size, num_attention_heads):
    if att_size_per_head is None:
      if long_hidden_size % num_attention_heads != 0:
        raise ValueError(
            '`long_hidden_size` must be a multiple of `num_attention_heads` '
            'when `att_size_per_head` is None.')
      if global_hidden_size % num_attention_heads != 0:
        raise ValueError(
            '`global_hidden_size` must be a multiple of `num_attention_heads` '
            'when `att_size_per_head` is None.')
      att_size_per_head = long_hidden_size // num_attention_heads
      long_total_att_size = long_hidden_size
      global_total_att_size = global_hidden_size
    else:
      long_total_att_size = att_size_per_head * num_attention_heads
      global_total_att_size = long_total_att_size

    return (att_size_per_head, long_total_att_size, global_total_att_size)

  def _validate_init_parameters(self) -> None:
    if self.share_feed_forward_params:
      if self.long_hidden_size != self.global_hidden_size:
        raise ValueError(
            '`long_hidden_size` must equal `global_hidden_size` when '
            '`share_feed_forward_params` is True.')
      if self.long_intermediate_size != self.global_intermediate_size:
        raise ValueError(
            '`long_intermediate_size` must equal `global_intermediate_size` '
            'when `share_feed_forward_params` is True.')
    if (self.share_qkv_projections and
        self.long_hidden_size != self.global_hidden_size):
      raise ValueError(
          '`long_hidden_size` must equal `global_hidden_size` when '
          '`share_qkv_projections` is True.')
    if (self.share_kv_projections and
        self.long_hidden_size != self.global_hidden_size):
      raise ValueError(
          '`long_hidden_size` must equal `global_hidden_size` when '
          '`share_kv_projections` is True.')
    if (self.share_att_output_projection and
        self.long_hidden_size != self.global_hidden_size):
      raise ValueError(
          '`long_hidden_size` must equal `global_hidden_size` when '
          '`share_att_output_projection` is True.')


class RelativeTransformerLayers(tf.keras.layers.Layer):
  """A sequence of Transformer encoder layers with optional relative attention.

  Just like the original Transformer, this layer uses full attention and scales
  quadratically with the input length.  To efficiently handle large inputs,
  ETC uses `GlobalLocalTransformerLayers` instead.  We just include this layer
  as a convenience since it contains the efficient relative attention
  implementation used by ETC and may be useful for applications with shorter
  graph-like inputs.

  See the ETC paper (https://arxiv.org/abs/2004.08483) Appendix A for a
  description of the relative attention implementation.
  """

  def __init__(self,
               hidden_size: int,
               num_hidden_layers: int,
               num_attention_heads: int,
               intermediate_size: Optional[int] = None,
               hidden_act=tensor_utils.get_activation('gelu'),
               hidden_dropout_prob: float = 0.1,
               attention_probs_dropout_prob: float = 0.1,
               initializer_range: float = 0.02,
               relative_vocab_size: Optional[int] = None,
               use_pre_activation_order: bool = False,
               use_one_hot_lookup: bool = False,
               name: Text = 'relative_transformer_layers',
               **kwargs):
    """Init.

    Args:
      hidden_size: Size of the output hidden dimension.  Must match the input
        hidden dimension size.
      num_hidden_layers: Number of Transformer layers.  Each layer includes both
        an attention sublayer and a feed-forward sublayer.
      num_attention_heads: Number of attention heads. Must evenly divide
        `hidden_size`.
      intermediate_size: The size of the "intermediate" (i.e. feed-forward)
        layers. Defaults to 4 * hidden_size.
      hidden_act: The non-linear activation function in the intermediate layers.
      hidden_dropout_prob: The dropout probability for the attention and
        feed-forward residual blocks. Must be between 0.0 and 1.0.
      attention_probs_dropout_prob: Dropout probability for attention
        probabilities. Must be between 0.0 and 1.0.
      initializer_range: The standard deviation of the truncated normal
        initializer for initializing weight matrices.
      relative_vocab_size: Size of relative position vocabulary. If left
        unspecified, relative positions will be ignored for attention.
      use_pre_activation_order: If True, use "pre-activation" order for residual
        blocks (see ResidualBlock docstring).
      use_one_hot_lookup: Whether to use tf.one_hot for embedding lookup instead
        of tf.gather. Default is False, but setting to True may be more
        efficient on TPUs for vocab sizes that aren't too large. Currently this
        is only used during lookup of relative position embeddings.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(RelativeTransformerLayers, self).__init__(name=name, **kwargs)

    if intermediate_size is None:
      intermediate_size = 4 * hidden_size

    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.hidden_act = hidden_act
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.relative_vocab_size = relative_vocab_size
    self.use_pre_activation_order = use_pre_activation_order
    self.use_one_hot_lookup = use_one_hot_lookup

    # TODO(jainslie): When using pre-activation order, the recommendation
    # from https://arxiv.org/abs/1904.10509 is to scale some of the
    # initialization by 1 / sqrt(2 * num_hidden_layers).  Add logic
    # to do this scaling (maybe within ResidualBlock rather than through
    # initialization).
    self.initializer = tf.keras.initializers.TruncatedNormal(
        stddev=initializer_range)

    self.attention_layers = []
    self.feed_forward_layers = []
    for i in range(num_hidden_layers):
      self.attention_layers.append(
          wrappers.ResidualBlock(
              inner_layer=attention.RelativeAttention(
                  hidden_size=hidden_size,
                  num_heads=num_attention_heads,
                  relative_vocab_size=relative_vocab_size,
                  att_dropout_prob=attention_probs_dropout_prob,
                  initializer=self.initializer,
                  use_one_hot_lookup=use_one_hot_lookup),
              dropout_probability=hidden_dropout_prob,
              use_pre_activation_order=use_pre_activation_order,
              name='attention_layer_%d' % i))
      self.feed_forward_layers.append(
          wrappers.ResidualBlock(
              dropout_probability=hidden_dropout_prob,
              use_pre_activation_order=use_pre_activation_order,
              inner_intermediate_size=intermediate_size,
              inner_activation=hidden_act,
              inner_kernel_initializer=self.initializer,
              name='feed_forward_layer_%d' % i))

  def call(self,
           inputs: tf.Tensor,
           att_mask: Optional[tf.Tensor] = None,
           relative_att_ids: Optional[tf.Tensor] = None,
           training=None) -> tf.Tensor:
    """Calls the layer.

    Args:
      inputs: <float32>[batch_size, seq_len, hidden_size].
      att_mask: <int32>[batch_size, seq_len, seq_len]. Should have only 0 and 1
        values, with 0 for entries that should be masked and 1 otherwise. Leave
        as None to allow all elements to attend to all other elements within
        each example.
      relative_att_ids: <int32>[batch_size, seq_len, seq_len]. Leave as None to
        skip the relative portion of attention.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
      <float32>[batch_size, seq_len, hidden_size].
    """
    output_tensor = inputs

    for i in range(self.num_hidden_layers):
      output_tensor = self.attention_layers[i](
          output_tensor,
          training=training,
          att_mask=att_mask,
          relative_att_ids=relative_att_ids)
      output_tensor = self.feed_forward_layers[i](
          output_tensor, training=training)

    return output_tensor
