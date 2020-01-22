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

"""Transformer layers.

From "Attention Is All You Need", https://arxiv.org/abs/1706.03762.

Implementation based on
  learning/brain/research/summarization/wikigen/layers/transformer.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers

# pylint: disable=invalid-name

_MIN_TIMESCALE = 1.0
_MAX_TIMESCALE = 1.0e4


class TimingSignal(object):
  """Layer that adds a transformer-style timing signal to inputs.

  Equivalent to tensor2tensor's 1d timing signal, generalized to allow each
  example in a batch to begin at a different index. See
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
  """

  def __call__(self, inputs, start_index=None):
    dtype = inputs.dtype
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    length = inputs_shape[1]
    channels = inputs_shape[2]
    if start_index is None:
      start_index = tf.zeros((batch_size, 1), tf.int32)

    position = tf.expand_dims(tf.range(length), 0)
    position = tf.tile(position, [batch_size, 1]) + start_index
    position = tf.cast(position, dtype)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(_MAX_TIMESCALE / _MIN_TIMESCALE) /
        tf.maximum(tf.cast(num_timescales, dtype) - 1, 1))
    inv_timescales = _MIN_TIMESCALE * tf.exp(
        tf.cast(tf.range(num_timescales), dtype) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 2) * tf.reshape(
        inv_timescales, [1, 1, -1])
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
    signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [-1, length, channels])
    return inputs + signal


class _Projection(object):
  """Projection layer, purposed for use in multi-head attention."""

  def __init__(self, hidden_size, num_heads=None, name=None):
    self._dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name=name)
    self._num_heads = num_heads
    self._hidden_size = hidden_size

  def __call__(self, x):
    x_shape = tf.shape(x)

    # If number of heads is specified, we're projecting up into attention head
    # space.
    if self._num_heads is not None:
      x = self._dense_layer(x)
      x = tf.reshape(x, [x_shape[0], x_shape[1], self._num_heads, -1])

    x = tf.transpose(x, [0, 2, 1, 3])

    # Otherwise, we're projecting down into output space.
    if self._num_heads is None:
      x = tf.reshape(x, [x_shape[0], -1, self._hidden_size])
      x = self._dense_layer(x)

    return x


class Attention(object):
  """Multihead scaled dot product attention."""

  def __init__(self, hidden_size, num_heads, attention_dropout):
    if hidden_size % num_heads != 0:
      raise ValueError("Number of attention heads must divide hidden size")

    self._q_projection = _Projection(
        hidden_size, num_heads=num_heads, name="q_proj")
    self._k_projection = _Projection(
        hidden_size, num_heads=num_heads, name="k_proj")
    self._v_projection = _Projection(
        hidden_size, num_heads=num_heads, name="v_proj")
    self._output_projection = _Projection(hidden_size, name="output_proj")
    self._attention_dropout = attention_dropout
    self._scale = (hidden_size // num_heads) ** -0.5

  def __call__(self, x, y, bias, training):
    q = self._q_projection(x) * self._scale
    k = self._k_projection(y)  # (B, nH, T, H // nH), nH stands for num_heads
    v = self._v_projection(y)

    logits = tf.matmul(q, k, transpose_b=True) + bias  # (B, nH, T, T)
    # TODO(peterjliu): May need to cast float16->float32 to avoid precision
    # issues in the softmax.
    outputs = tf.nn.softmax(logits)
    outputs_shape = tf.shape(outputs)
    outputs = tf.layers.dropout(
        outputs,
        rate=self._attention_dropout,
        noise_shape=[1, 1, outputs_shape[2], outputs_shape[3]],
        training=training)  # (B, nH, T, T)
    outputs = tf.matmul(outputs, v)  # (B, nH, T, H // nH)

    return self._output_projection(outputs)


class SelfAttention(Attention):
  """Multihead scaled dot product self-attention."""

  def __call__(self, x, bias, training):
    return super(SelfAttention, self).__call__(x, x, bias, training)


class _FeedForward(object):
  """FeedForward layer used inside a Transformer."""

  def __init__(self, output_size, filter_size, dropout):
    self._relu_layer = tf.layers.Dense(filter_size, activation=tf.nn.relu)
    self._output_layer = tf.layers.Dense(output_size)
    self._dropout = dropout

  def __call__(self, inputs, training):
    with tf.variable_scope("ffn"):
      x = self._relu_layer(inputs)
      x_shape = tf.shape(x)
      x = tf.layers.dropout(
          x,
          self._dropout,
          noise_shape=[x_shape[0], 1, x_shape[2]],
          training=training)
      x = self._output_layer(x)
    return x


class _PreProcess(object):
  """Preprocessing layer used inside Transformers."""

  def __call__(self, inputs):
    with tf.variable_scope("preprocess"):
      x = contrib_layers.layer_norm(inputs, begin_norm_axis=2)
    return x


class _PostProcess(object):
  """PostProcessing layer used inside Transformers."""

  def __init__(self, dropout):
    self._dropout = dropout

  def __call__(self, x, y, training):
    with tf.variable_scope("postprocess"):
      y_shape = tf.shape(y)
      y = tf.layers.dropout(
          y,
          self._dropout,
          noise_shape=[y_shape[0], 1, y_shape[2]],
          training=training)
      x += y
    return x


class _Encoder(object):
  """A single encoder layer, part of the stack of encoders in a Transformer."""

  def __init__(self, hidden_size, filter_size, num_heads, attention_dropout,
               relu_dropout, postprocess_dropout):
    self._self_attn_preprocess = _PreProcess()  # layer norm
    self._self_attn_layer = SelfAttention(hidden_size, num_heads,
                                          attention_dropout)
    self._self_attn_postprocess = _PostProcess(postprocess_dropout)  # dropout
    self._ffn_preprocess = _PreProcess()
    self._ffn_layer = _FeedForward(hidden_size, filter_size, relu_dropout)
    self._ffn_postprocess = _PostProcess(postprocess_dropout)

  def __call__(self, inputs, bias, training):
    x = inputs
    with tf.variable_scope("self_attention"):
      y = self._self_attn_layer(self._self_attn_preprocess(x), bias, training)
      x = self._self_attn_postprocess(x, y, training)
    with tf.variable_scope("ffn"):
      y = self._ffn_layer(self._ffn_preprocess(x), training)
      x = self._ffn_postprocess(x, y, training)
    return x


class _Decoder(object):
  """A single decoder layer, part of the stack of decoders in a Transformer."""

  def __init__(self, hidden_size, filter_size, num_heads, attention_dropout,
               relu_dropout, postprocess_dropout):
    self._self_attn_preprocess = _PreProcess()
    self._self_attn_layer = SelfAttention(hidden_size, num_heads,
                                          attention_dropout)
    self._self_attn_postprocess = _PostProcess(postprocess_dropout)
    self._encdec_attn_preprocess = _PreProcess()
    self._encdec_attn_layer = Attention(hidden_size, num_heads,
                                        attention_dropout)
    self._encdec_attn_postprocess = _PostProcess(postprocess_dropout)
    self._ffn_preprocess = _PreProcess()
    self._ffn_layer = _FeedForward(hidden_size, filter_size, relu_dropout)
    self._ffn_postprocess = _PostProcess(postprocess_dropout)

  def __call__(self, x, decoder_self_attention_bias, inputs,
               inputs_attention_bias, training):
    with tf.variable_scope("self_attention"):
      y = self._self_attn_layer(
          self._self_attn_preprocess(x), decoder_self_attention_bias, training)
      x = self._self_attn_postprocess(x, y, training)
    with tf.variable_scope("encdec_attention"):
      y = self._encdec_attn_layer(
          self._encdec_attn_preprocess(x), inputs, inputs_attention_bias,
          training)
      x = self._encdec_attn_postprocess(x, y, training)
    with tf.variable_scope("ffn"):
      y = self._ffn_layer(self._ffn_preprocess(x), training)
      x += self._ffn_postprocess(x, y, training)
    return x


class TransformerEncoderDecoder(object):
  """Transformer encoder-decoder layer.

  For tensors in this file, B, I, T, and H denote batch_size, input_len,
  target_len, and hidden_size, respectively.

  Inputs are:
    inputs_BxIxH: tensor of representing encoder inputs
    targets_BxTxH: tensor representing decoder targets that will be used as
      decoder inputs during teacher-forcing training
    padding_BxI: tensor with 1 indicating padded positions in encoder inputs
    training: bool indicating whether we are in training mode
    targets_start: starting index for adding position information in decoder
      inputs
  Outputs are:
    3-d tensor of shape [B, T, H] representing the decoder outputs
  """

  def __init__(self, hidden_size, filter_size, num_encoder_layers,
               num_decoder_layers, num_encoder_heads, num_decoder_heads,
               attention_dropout, relu_dropout, postprocess_dropout):
    self._encoder_layer = TransformerEncoder(hidden_size, filter_size,
                                             num_encoder_layers,
                                             num_encoder_heads,
                                             attention_dropout,
                                             relu_dropout, postprocess_dropout)
    self._decoder_layer = TransformerDecoder(hidden_size, filter_size,
                                             num_decoder_layers,
                                             num_decoder_heads,
                                             attention_dropout,
                                             relu_dropout, postprocess_dropout)

  def __call__(self,
               inputs_BxIxH,
               targets_BxTxH,
               padding_BxI,
               training,
               cache=None,
               targets_start=None):
    encoder_output_BxIxH = self._encoder_layer(inputs_BxIxH, padding_BxI,
                                               training, cache=cache)
    decoder_output_BxTxH = self._decoder_layer(
        encoder_output_BxIxH, padding_BxI, targets_BxTxH, training,
        targets_start=targets_start)
    return decoder_output_BxTxH


class TransformerEncoder(object):
  """Transformer encoder layer.

  Inputs are:
    inputs_BxIxH: tensor representing encoder inputs
    padding_BxI: tensor with 1 indicating padded positions in encoder inputs
    training: bool indicating whether we are in training mode
  Outputs are:
    3-d tensor of shape [B, I, H] representing the encoder outputs
  """

  def __init__(self, hidden_size, filter_size, num_layers, num_heads,
               attention_dropout, relu_dropout, postprocess_dropout):
    self._postprocess_dropout = postprocess_dropout
    self._preprocess_layer = _PreProcess()

    # A stack of encoder layers which are applied in sequence in the main body
    # of the encoder.
    self._encoder_layers = [
        _Encoder(hidden_size, filter_size, num_heads, attention_dropout,
                 relu_dropout, postprocess_dropout) for _ in range(num_layers)
    ]

  def __call__(self, inputs_BxIxH, padding_BxI, training, cache=None):
    if cache is not None and "encoder_output" in cache:
      return cache["encoder_output"]

    # Bias the attention to ignore padding in the inputs.
    attention_bias_Bx1x1xI = tf.expand_dims(
        tf.expand_dims(padding_BxI * padding_BxI.dtype.min, 1), 1)
    encoder_input_BxIxH = TimingSignal()(inputs_BxIxH)
    encoder_input_BxIxH = tf.layers.dropout(
        encoder_input_BxIxH, self._postprocess_dropout, training=training)

    x_BxIxH = encoder_input_BxIxH
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      for i, encoder_layer in enumerate(self._encoder_layers):
        with tf.variable_scope("layer_%d" % i):
          x_BxIxH = encoder_layer(x_BxIxH, attention_bias_Bx1x1xI, training)
      encoder_output_BxIxH = self._preprocess_layer(x_BxIxH)

    if cache is not None:
      cache["encoder_output"] = encoder_output_BxIxH

    return encoder_output_BxIxH


class TransformerDecoder(object):
  """Transformer decoder layer.

  Inputs are:
    inputs_BxIxH: tensor representing encoder outputs used for encoder-decoder
      attention
    inputs_padding_BxI: tensor representing encoder output paddings (same as
      encoder input paddings)
    targets_BxTxH: tensor representing decoder targets that will be used as
      decoder inputs during teacher-forcing training
    training: bool indicating whether we are in training mode
    targets_start: starting index for adding position information
  Outputs are:
    3-d tensor of shape [B, T, H] representing decoder outputs
  """

  def __init__(self, hidden_size, filter_size, num_layers, num_heads,
               attention_dropout, relu_dropout, postprocess_dropout):
    self._postprocess_dropout = postprocess_dropout
    self._preprocess_layer = _PreProcess()

    # A stack of decoder layers which are applied in sequence in the main body
    # of the decoder.
    self._decoder_layers = [
        _Decoder(hidden_size, filter_size, num_heads, attention_dropout,
                 relu_dropout, postprocess_dropout) for _ in range(num_layers)
    ]

  def __call__(self,
               inputs_BxIxH,
               inputs_padding_BxI,
               targets_BxTxH,
               training,
               targets_start=None):
    # Mask off padding in inputs when computing attention.
    inputs_attention_bias_Bx1x1xI = tf.expand_dims(
        tf.expand_dims(inputs_padding_BxI * inputs_padding_BxI.dtype.min, 1), 1)

    # Mask off "future" targets to avoid them creeping into predictions when
    # computing loss over an entire targets matrix.
    targets_len = tf.shape(targets_BxTxH)[1]
    upper_triangular_TxT = 1 - tf.matrix_band_part(
        tf.ones((targets_len, targets_len), dtype=inputs_BxIxH.dtype), -1, 0)
    decoder_self_attention_bias_1x1xTxT = tf.expand_dims(
        tf.expand_dims(upper_triangular_TxT, 0), 0) * inputs_BxIxH.dtype.min

    # Pad a zero on the LHS of targets.
    decoder_input_BxTxH = tf.pad(
        targets_BxTxH, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input_BxTxH = TimingSignal()(decoder_input_BxTxH, targets_start)
    decoder_input_BxTxH = tf.layers.dropout(
        decoder_input_BxTxH, self._postprocess_dropout, training=training)

    x_BxTxH = decoder_input_BxTxH
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for i, decoder_layer in enumerate(self._decoder_layers):
        with tf.variable_scope("layer_%d" % i):
          x_BxTxH = decoder_layer(x_BxTxH, decoder_self_attention_bias_1x1xTxT,
                                  inputs_BxIxH, inputs_attention_bias_Bx1x1xI,
                                  training)
      decoder_output_BxTxH = self._preprocess_layer(x_BxTxH)

    return decoder_output_BxTxH


class TransformerDecoderOnly(object):
  """Like encoder but with a masked self-attention, preventing attention to the future."""

  def __init__(self, hidden_size, filter_size, num_layers, num_heads,
               attention_dropout, relu_dropout, postprocess_dropout):
    self._postprocess_dropout = postprocess_dropout
    self._preprocess_layer = _PreProcess()
    # Use _Encoder rather than _Decoder since we don't need enc-dec attention.
    self._decoder_layers = [
        _Encoder(hidden_size, filter_size, num_heads, attention_dropout,
                 relu_dropout, postprocess_dropout) for _ in range(num_layers)
    ]

  def __call__(self, inputs_BxTxH, training, targets_start=None):
    """TransformerDecoderOnly call operator.

    Args:
      inputs_BxTxH: a 3d-tensor representing decoder inputs; during training
                    essentially the right-shifted decoder targets
      training: bool indicating whether we are in training mode
      targets_start: starting index for adding position information

    Returns:
      3d-tensor of shape [B, T, H] representing decoder outputs
    """
    # Mask off "future" targets to avoid them creeping into predictions when
    # computing loss over an entire targets matrix.
    targets_len = tf.shape(inputs_BxTxH)[1]
    upper_triangular_TxT = 1 - tf.matrix_band_part(
        tf.ones((targets_len, targets_len), dtype=tf.float32), -1, 0)
    # For example, when targets_len == 3, upper_triangular_TxT is:
    # [[0., 1., 1.],
    #  [0., 0., 1.],
    #  [0., 0., 0.]]
    masked_attention_bias_1x1xTxT = tf.expand_dims(
        tf.expand_dims(upper_triangular_TxT, 0), 0) * tf.float32.min
    # For example, masked_attention_bias_1x1xTxT is:
    # [[[[0., -inf, -inf],
    #    [0.,   0., -inf],
    #    [0.,   0.,   0.]]]]

    # No padding is needed here as inputs_BxTxH is already prepended with
    # special token.
    decoder_input_BxTxH = TimingSignal()(inputs_BxTxH, targets_start)
    decoder_input_BxTxH = tf.layers.dropout(
        decoder_input_BxTxH, self._postprocess_dropout, training=training)

    x_BxTxH = decoder_input_BxTxH
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for i, decoder_layer in enumerate(self._decoder_layers):
        with tf.variable_scope("layer_%d" % i):
          x_BxTxH = decoder_layer(x_BxTxH, masked_attention_bias_1x1xTxT,
                                  training)
      decoder_output_BxTxH = self._preprocess_layer(x_BxTxH)

    return decoder_output_BxTxH
