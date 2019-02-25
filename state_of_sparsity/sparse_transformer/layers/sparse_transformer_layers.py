# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Sparse transformer layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log

import tensorflow as tf
from state_of_sparsity.sparse_transformer.layers import sparse_attention
from state_of_sparsity.sparse_transformer.layers import sparse_layers


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convolutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: a Tensors
  """
  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(
          encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):

      initial_sparsity = None
      if hparams.get("load_masks_from"):
        initial_sparsity = hparams.get("initial_sparsity")

      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = sparse_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              sparsity_technique=hparams.get("sparsity_technique"),
              threshold=hparams.get("log_alpha_threshold"),
              training=hparams.get("mode") == tf.estimator.ModeKeys.TRAIN,
              clip_alpha=hparams.get("clip_log_alpha"),
              initial_sparsity=initial_sparsity,
              split_heads=hparams.get("split_heads"))
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              pad_remover)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x, hparams, pad_remover=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparameters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.

  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]

  Raises:
    ValueError: If losses arg is None, but layer generates extra losses.
  """
  ffn_layer = hparams.ffn_layer
  if ffn_layer != "dense_relu_dense":
    raise ValueError("sparse transformer only supports dense_relu_dense ffn.")

  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))
  # In simple convolution mode, use `pad_remover` to speed up processing.
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
      value={
          "filter_size": hparams.filter_size,
          "use_bias": "True",
          "activation": mlperf_log.RELU
      })
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
      value={
          "hidden_size": hparams.hidden_size,
          "use_bias": "True",
      })
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
  if pad_remover:
    original_shape = common_layers.shape_list(x)
    # Collapse `x` across examples, and remove padding positions.
    x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
    x = tf.expand_dims(pad_remover.remove(x), axis=0)

  initial_sparsity = None
  if hparams.get("load_masks_from"):
    initial_sparsity = hparams.get("initial_sparsity")

  conv_output = sparse_layers.dense_relu_dense(
      x,
      hparams.filter_size,
      hparams.hidden_size,
      dropout=hparams.relu_dropout,
      dropout_broadcast_dims=relu_dropout_broadcast_dims,
      sparsity_technique=hparams.get("sparsity_technique"),
      threshold=hparams.get("log_alpha_threshold"),
      training=hparams.get("mode") == tf.estimator.ModeKeys.TRAIN,
      clip_alpha=hparams.get("clip_log_alpha"),
      initial_sparsity=initial_sparsity)
  if pad_remover:
    # Restore `conv_output` to the original shape of `x`, including padding.
    conv_output = tf.reshape(
        pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
  return conv_output
