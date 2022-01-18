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

"""Transformer layers for ReadTwice."""

from typing import List, Optional, Text, Union

import tensorflow as tf

from readtwice.layers import attention
from readtwice.layers import tensor_utils
from readtwice.layers import wrappers


class TransformerWithSideInputLayers(tf.keras.layers.Layer):
  """A sequence of Transformer layers (currently only the encoder side).

  The model follows the original Transformer model
  (https://arxiv.org/abs/1706.03762) while allowing additional side inputs.
  These inputs are only used during self-attention computations
  as extra keys and values.
  """

  def __init__(self,
               hidden_size,
               num_hidden_layers,
               num_attention_heads,
               intermediate_size = None,
               hidden_act=tensor_utils.get_activation('gelu'),
               hidden_dropout_prob = 0.1,
               attention_probs_dropout_prob = 0.1,
               initializer_range = 0.02,
               share_kv_projections = False,
               num_cross_attention_heads = None,
               enable_default_side_input = False,
               name = 'transformer_layers',
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
        initializer for initializing weight matrices not created by
        `linear_make_fn`. If zero, the scale of the truncated normal initializer
        will be tuned automatically according to the distribution of the inputs.
      share_kv_projections: If True, key and value projections will be shared
        between main-to-main and main-to-side components. This results in 1
        key projection per layer instead of 2 (and similarly for value
        projections). Only relevant for fused side attention,
        NOT cross attention over the side input (when num_cross_attention_heads
        is not None).
      num_cross_attention_heads: If it is not None, will add a cross-attention
        layer over side inputs. In this case, side inputs will NOT be used
        in the `FusedSideAttention`. Must be greater or equal than 0, where 0
        means that cross attention layer will have a single attention head
        WITHOUT projection matrices.
      enable_default_side_input: Add a default side input, which acts like a
        no-op attention, effective allowing attention weights to sum up
        to something less than 1.
        Currently, only available for the cross attention over side inputs.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(TransformerWithSideInputLayers, self).__init__(name=name, **kwargs)

    if intermediate_size is None:
      intermediate_size = 4 * hidden_size

    if num_cross_attention_heads is not None:
      # This will prevent from allocating extra parameters for
      # fused side attention since side input will not be used there anyway.
      share_kv_projections = True

    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.hidden_act = hidden_act
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.share_kv_projections = share_kv_projections
    self.num_cross_attention_heads = num_cross_attention_heads
    self.enable_default_side_input = enable_default_side_input

    if (self.enable_default_side_input and
        self.num_cross_attention_heads is None):
      raise ValueError('`enable_default_side_input` is only used when '
                       'num_cross_attention_heads is enabled.')
    if (self.num_cross_attention_heads is not None and
        self.num_cross_attention_heads < 0):
      raise ValueError('If `num_cross_attention_heads` is specified '
                       'it must be non-negative.')

    self.initializer = tf.keras.initializers.TruncatedNormal(
        stddev=initializer_range)

    self.attention_layers = []
    self.cross_attention_layers = []
    self.feed_forward_layers = []
    for i in range(num_hidden_layers):
      self.attention_layers.append(
          wrappers.ResidualBlock(
              inner_layer=attention.FusedSideAttention(
                  hidden_size=self.hidden_size,
                  num_heads=self.num_attention_heads,
                  att_dropout_prob=self.attention_probs_dropout_prob,
                  share_kv_projections=self.share_kv_projections,
                  initializer=self.initializer),
              dropout_probability=self.hidden_dropout_prob,
              use_pre_activation_order=False,
              name='attention_layer_%d' % i))

      if self.num_cross_attention_heads is not None:
        self.cross_attention_layers.append(
            wrappers.ResidualBlock(
                inner_layer=attention.SideAttention(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_cross_attention_heads,
                    att_dropout_prob=self.attention_probs_dropout_prob,
                    initializer=self.initializer,
                    enable_default_side_input=self.enable_default_side_input),
                dropout_probability=self.hidden_dropout_prob,
                use_pre_activation_order=False,
                name='cross_attention_layer_%d' % i))

      self.feed_forward_layers.append(
          wrappers.ResidualBlock(
              dropout_probability=self.hidden_dropout_prob,
              use_pre_activation_order=False,
              inner_intermediate_size=self.intermediate_size,
              inner_activation=self.hidden_act,
              inner_kernel_initializer=self.initializer,
              name='feed_forward_layer_%d' % i))

  # TODO(urikz): Add some way for the user to access activations from
  # each hidden layer.
  def call(self,
           main_input,
           side_input = None,
           att_mask = None,
           training=None):
    """Calls the layer.

    Args:
      main_input: <float32>[batch_size, main_seq_len, hidden_size].
      side_input: <float32>[batch_size, side_seq_len, hidden_size] or a list
      of tensors with this shape. The length of the list must be equal to
      `num_hidden_layers`.
      att_mask: <int32>[batch_size, main_seq_len, main_seq_len +
        side_seq_len]. Should have only 0 and 1 values, with 0 for entries
        that should be masked and 1 otherwise. Leave as None to allow all
        elements to attend to all other elements within each example.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.

    Returns:
      <float32>[batch_size, main_seq_len, hidden_size].

    Raises:
      ValueError if `side_input` is list and the length of `side_input`
      is not equal to `num_hidden_layers`.
    """
    output_tensor = main_input

    if side_input is not None and not isinstance(side_input, list):
      side_input = [side_input] * self.num_hidden_layers
    else:
      side_input = [None] * self.num_hidden_layers

    if len(side_input) != self.num_hidden_layers:
      raise ValueError('Length of side input ({}) is not equal to '
                       'the number of hidden layers ({})'.format(
                           len(side_input), self.num_hidden_layers))

    if att_mask is not None and self.num_cross_attention_heads is not None:
      main_seq_len = tf.shape(main_input)[1]
      att_mask_attention_layer = att_mask[:, :, :main_seq_len]
      att_mask_cross_attention_layer = att_mask[:, :, main_seq_len:]
    else:
      att_mask_attention_layer = None
      att_mask_cross_attention_layer = None

    for i in range(self.num_hidden_layers):
      if self.num_cross_attention_heads is not None:
        output_tensor = self.attention_layers[i](
            output_tensor,
            training=training,
            side_input=None,
            att_mask=att_mask_attention_layer)

        output_tensor = self.cross_attention_layers[i](
            output_tensor,
            side_input=side_input[i],
            att_mask=att_mask_cross_attention_layer,
            training=training)
      else:
        output_tensor = self.attention_layers[i](
            output_tensor,
            training=training,
            side_input=side_input[i],
            att_mask=att_mask)

      output_tensor = self.feed_forward_layers[i](
          output_tensor, training=training)

    return output_tensor
