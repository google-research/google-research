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

"""Implements layers for BERT-like pretraining of (protein) language models."""

from typing import Optional

import gin
import tensorflow as tf

from dedal import vocabulary
from dedal.models import activations


@gin.configurable
class DensePerTokenOutputHead(tf.keras.layers.Layer):
  """Computes language model logits over vocabulary given embeddings.

  Logits are computed as

    logits[i][j] = inputs[i][j] W + b,

  where inputs[i][j] is the (contextual) embedding for the token at position j
  of sequence i, and W and b are the learnable weights and biases of the layer.
  """

  def __init__(self,
               vocab = None,
               kernel_init='GlorotUniform',
               bias_init='Zeros',
               **kwargs):
    super().__init__(**kwargs)
    self._vocab = vocabulary.get_default() if vocab is None else vocab
    self._kernel_init = tf.keras.initializers.get(kernel_init)
    self._bias_init = tf.keras.initializers.get(bias_init)

  def build(self, _):
    self.dense = tf.keras.layers.Dense(
        units=len(self._vocab),
        kernel_initializer=self._kernel_init,
        bias_initializer=self._bias_init,
        name='output_head/dense')

  def call(self,
           inputs,
           mask = None):
    """Computes output language model logits given input embeddings.

    Args:
      inputs: A tf.Tensor of shape (batch_size, padded_len, emb_dim) and type
        tf.float32 such that inputs[i][j] represents the (contextual) embedding
        of the token at position j in sequence i.
      mask: A tf.Tensor of shape (batch_size, padded_len) and type tf.float32.
        Indicates which tokens are "real" (mask[i][j] = 1.0) and which are
        padding (mask[i][j] = 0.0).

    Returns:
      A tf.Tensor of shape (batch_size, padded_len, n_tokens) and type
      tf.float32, where n_tokens is the number of tokens over the output
      vocabulary as given by the layer's config. Contains the output logits for
      each input token. If mask is not None, the logits corresponding to padding
      tokens will be zeroed-out.
    """
    logits = self.dense(inputs)
    if mask is not None:
      logits *= tf.cast(mask[:, :, None], logits.dtype)
    return logits


@gin.configurable
class BERTPerTokenOutputHead(tf.keras.layers.Layer):
  """Computes language model logits over vocabulary given embeddings.

  If use_hidden_layer is True, logits are computed as

    logits[i][j] = LayerNorm(activation(inputs[i][j] W + b)) W_{out} + b_{out},

  where inputs[i][j] is the (contextual) embedding for the token at position j
  of sequence i; W, b and b_{out} are the learnable weight matrix and bias
  vectors of the layer and W_{out} is an externally provided weight matrix,
  typically the transpose of the token embedding look-up matrix that is used to
  compute the input embeddings for the transformer layers.

  By setting use_hidden_layer to False, this can be simplified to

    logits[i][j] = inputs[i][j] W_{out} + b_{out}.

  In both cases, the output of the matrix multiplication by W_{out} can be
  optionally normalized by a factor sqrt(emb_dim) by setting norm_logits to True
  in the layer's config. That is, for example if use_hidden_layer is False and
  norm_logits is True, we would have

    logits[i][j] = (1 / sqrt(emb_dim)) inputs[i][j] W_{out} + b_{out}.
  """

  def __init__(self,
               vocab = None,
               use_hidden_layer = True,
               activation=activations.approximate_gelu,
               kernel_init='GlorotUniform',
               bias_init='Zeros',
               logits_bias_init='Zeros',
               norm_logits = True,
               **kwargs):
    super().__init__(**kwargs)
    self._vocab = vocabulary.get_default() if vocab is None else vocab
    self._use_hidden_layer = use_hidden_layer
    self._activation = activation
    self._kernel_init = tf.keras.initializers.get(kernel_init)
    self._bias_init = tf.keras.initializers.get(bias_init)
    self._logits_bias_init = tf.keras.initializers.get(logits_bias_init)
    self._norm_logits = norm_logits

  def build(self, input_shape):
    self._dense = None
    self._layer_norm = None
    if self._use_hidden_layer:
      self._dense = tf.keras.layers.Dense(
          units=input_shape[-1],  # emb_dim
          kernel_initializer=self._kernel_init,
          bias_initializer=self._bias_init,
          name='output_head/dense_proj')
      self._layer_norm = tf.keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, name='output_head/layer_norm')

    self._b_out = None
    if self._logits_bias_init is not None:
      self._b_out = self.add_weight(
          shape=(len(self._vocab),),
          initializer=self._logits_bias_init,
          trainable=True,
          name='output_head/bias')

  def call(self,
           inputs,
           shared_weights,
           mask = None,
           ):
    """Computes output language model logits given input embeddings.

    Args:
      inputs: A tf.Tensor of shape (batch_size, padded_len, emb_dim) and type
        tf.float32 such that inputs[i][j] represents the (contextual) embedding
        of the token at position j in sequence i.
      shared_weights: A tf.Tensor of shape (n_tokens, emb_dim) and type
        tf.float32, where n_tokens is the number of tokens over the output
        vocabulary as given by the layer's config. In BERT, this is typically
        the token embeddings look-up matrix used to obtain the input embeddings.
      mask: A tf.Tensor of shape (batch_size, padded_len) and type tf.float32.
        Indicates which tokens are "real" (mask[i][j] = 1.0) and which are
        padding (mask[i][j] = 0.0).

    Returns:
      A tf.Tensor of shape (batch_size, padded_len, n_tokens) and type
      tf.float32, where n_tokens is the number of tokens over the output
      vocabulary as given by the layer's config. Contains the output logits for
      each input token. If mask is not None, the logits corresponding to padding
      tokens will be zeroed-out.
    """
    norm_factor = tf.shape(inputs)[-1] if self._norm_logits else 1.0
    norm_factor = tf.sqrt(tf.cast(norm_factor, dtype=inputs.dtype))

    if self._layer_norm is not None:
      inputs = self._layer_norm(self._activation(self._dense(inputs)))

    logits = tf.matmul(inputs, shared_weights, transpose_b=True) / norm_factor
    logits = tf.nn.bias_add(logits, self._b_out)

    if mask is not None:
      logits *= tf.cast(mask[:, :, None], dtype=logits.dtype)
    return logits


@gin.configurable
class DenseEOSOutputHead(tf.keras.layers.Layer):
  """Computes end of sequence (EOS) logits given embeddings.

  Logits are computed as

    logits[i][j] = inputs[i][j] W + b,

  where inputs[i][j] is the (contextual) embedding for the token at position j
  of sequence i, and W and b are the learnable weights and biases of the layer.
  """

  def __init__(self,
               vocab = None,
               kernel_init='GlorotUniform',
               bias_init='Zeros',
               **kwargs):
    super().__init__(**kwargs)
    self._vocab = vocabulary.get_default() if vocab is None else vocab
    self._kernel_init = tf.keras.initializers.get(kernel_init)
    self._bias_init = tf.keras.initializers.get(bias_init)

  def build(self, _):
    self._dense = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=self._kernel_init,
        bias_initializer=self._bias_init,
        name='eos_head/dense')

  def call(self,
           inputs,
           mask = None):
    """Computes EOS logits given input embeddings.

    Args:
      inputs: A tf.Tensor<float>[batch, padded_len, emb_dim].
      mask: A tf.Tensor<bool>[batch, padded_len] to account for padding.

    Returns:
      A tf.Tensor<float>[batch, padded_len] that contains the logits for the EOS
      state. If mask is not None, the logits corresponding to padding tokens
      will be zeroed-out.
    """
    logits = self._dense(inputs)
    if mask is not None:
      logits *= tf.cast(mask[:, :, None], dtype=logits.dtype)
    return logits
