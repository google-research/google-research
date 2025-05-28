# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Transformer layers."""

from typing import Optional

import numpy as np
import tensorflow as tf


class WindowMultiHeadAttention(tf.Module):
  """Windowed multi-head attention."""

  def __init__(
      self,
      d_model,
      num_heads,
      attn_drop = 0.,
      proj_drop = 0.,
      name = None,
  ):
    super().__init__(name=name)
    self._d_model = d_model
    self._num_heads = num_heads
    if d_model % num_heads != 0:
      raise ValueError(
          f"Size of hidden units ({d_model}) not divisible by number "
          f"of head ({num_heads}).")
    head_dim = d_model // num_heads
    self._attn_scale = head_dim**(-0.5)

    def _dense(name):
      return tf.keras.layers.Dense(
          units=d_model,
          use_bias=True,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
          name=name)

    self._q, self._k, self._v = map(_dense, ("query", "key", "value"))
    self._attn_drop = tf.keras.layers.Dropout(
        rate=attn_drop, name="attention_dropout")
    self._proj = tf.keras.layers.Dense(
        units=d_model,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name="projection")
    self._proj_drop = tf.keras.layers.Dropout(
        rate=proj_drop, name="proj_dropout")
    self._softmax = tf.nn.softmax

  def __call__(self,
               v,
               k,
               q,
               *,
               mask,
               training = True):
    """Forward calls.

    Note that seq_len_kv must be an integer multiple of seq_len_q.

    Args:
      v: (*b, seq_len_kv, C) inputs.
      k: (*b, seq_len_kv, C) inputs.
      q: (*b, seq_len_q, C) inputs.
      mask: Optional mask, must broadcast to (*b, num_heads, seq_len_q,
        seq_len_q), and must be in {0, 1}, 1s will be masked.
      training: Whether training mode or not

    Returns:
      Output tensor of shape (*b, seq_len_q, C), as well as the
      attention matrix used, shape (*b, seq_len_q, seq_len_kv).
    """
    tf.debugging.assert_shapes([
        (v, ("*", "seq_len_kv", "dv_model")),
        (k, ("*", "seq_len_kv", "dv_model")),
        (q, ("*", "seq_len_q", "d_model")),
    ])
    *b, seq_len_q, c = q.shape
    assert c == self._d_model, (c, self._d_model)
    seq_len_kv = v.shape[-2]
    blowup = seq_len_kv // seq_len_q

    def _apply_and_reshape(network, inp):
      # *b, seq_len, d_model
      otp = network(inp)
      # *b, seq_len, num_heads, d_model//num_heads
      otp = tf.reshape(otp, (*b, -1, self._num_heads, c // self._num_heads))
      # *b, num_heads, seq_len, d_model//num_heads
      return tf.einsum("...nhd->...hnd", otp)

    q = _apply_and_reshape(self._q, q)
    k = _apply_and_reshape(self._k, k)
    v = _apply_and_reshape(self._v, v)

    q = q * self._attn_scale
    # *b, num_heads, seq_len_q, seq_len_kv
    attn = tf.matmul(q, k, transpose_b=True)

    if mask is not None:
      if mask.shape[-2:] != (seq_len_q, seq_len_q):
        # Note that we only mask for self-attention in the decoder,
        # where the attention matrix will have o
        # shape (..., seq_len_q, seq_len_q).
        raise ValueError(f"Invalid mask shape: {mask.shape}.")

      # Here, we add the mask to the attention with a large negative multiplier,
      # since this goes into a softmax and we want to disable the masked
      # entries.
      tile_pattern = [1] * mask.shape.rank
      tile_pattern[-1] = blowup
      attn = attn + tf.tile(mask, tile_pattern) * -1e6

    # *b, num_heads, seq_len, seq_len
    attn = self._softmax(attn)
    if mask is not None:
      # We use the mask again, to be double sure that no masked dimension
      # affects the output.
      keep = 1 - mask
      attn *= tf.tile(keep, tile_pattern)

    attn = self._attn_drop(attn, training)

    # *b, num_heads, seq_len_q, d_model//num_heads
    features = tf.matmul(attn, v)
    assert features.shape == (*b, self._num_heads, seq_len_q,
                              c // self._num_heads)
    # Switch num_heads, seq_len_q dimensions.
    features = tf.einsum("...hnd->...nhd", features)
    # Merge num_heads, seq_len_q dimensions
    features = tf.reshape(features, (*b, seq_len_q, self._d_model))
    features = self._proj(features)
    features = self._proj_drop(features, training)
    assert features.shape == (*b, seq_len_q, c)
    return features, attn


def create_look_ahead_mask(size):
  """Creates a lookahead mask for autoregressive masking."""
  mask = np.triu(np.ones((size, size), np.float32), 1)
  return tf.constant(mask)


_truncnormal_init = lambda: tf.keras.initializers.TruncatedNormal(stddev=0.02)


class StochasticDepth(tf.keras.layers.Layer):
  """Creates a stochastic depth layer."""

  def __init__(self, stochastic_depth_drop_rate, name):
    """Initializes a stochastic depth layer.

    Args:
      stochastic_depth_drop_rate: A `float` of drop rate.
      name: Name of the layer.

    Returns:
      A output `tf.Tensor` of which should have the same shape as input.
    """
    super().__init__(name=name)
    self._drop_rate = stochastic_depth_drop_rate

  def call(self, inputs, training):
    if not training or self._drop_rate == 0.:
      return inputs
    keep_prob = 1.0 - self._drop_rate
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(
        [batch_size] + [1] * (inputs.shape.rank - 1), dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class MLP(tf.keras.layers.Layer):
  """MLP head for transformer."""

  def __init__(self, expansion_rate, act_name, dropout_rate,
               **kwargs):
    super().__init__(**kwargs)
    self._expansion_rate = expansion_rate
    self._act_name = act_name
    self._dropout_rate = dropout_rate

  def build(self, input_shape):
    """Build the MLP head.

    Args:
      input_shape: Shape of the input tensor, with the last dimension as the
        size of channel -- [N1, ... Nn, C].
    """
    n_channel = input_shape[-1]
    self._fc1 = tf.keras.layers.Dense(
        units=self._expansion_rate * n_channel,
        activation=self._act_name,
        kernel_initializer=_truncnormal_init(),
        name="expansion")
    self._fc2 = tf.keras.layers.Dense(
        units=n_channel,
        kernel_initializer=_truncnormal_init(),
        name="compress")
    self._drop = tf.keras.layers.Dropout(
        rate=self._dropout_rate, name="dropout")

    super().build(input_shape)

  def call(self, features, training):
    """Forward pass."""
    features = self._fc1(features)
    features = self._drop(features, training)
    features = self._fc2(features)
    features = self._drop(features, training)
    return features


class TransformerBlock(tf.Module):
  """Transformer block that is similar to the Swin encoder block.

  However, an important difference is that we _do not_ shift the windows
  for the second Attention layer. Instead, we _feed the encoder outputs_
  as Keys and Values. This allows for autoregressive applications.

  If `style == "encoder"`, no autoregression is happening.

  Also, this class operates on windowed tensor, see `call` docstring.
  """

  def __init__(
      self,
      *,
      d_model,
      seq_len,
      num_head = 4,
      mlp_expansion = 4,
      mlp_act = "gelu",
      drop_out_rate = 0.1,
      drop_path_rate = 0.1,
      style = "decoder",
  ):
    super().__init__()
    self._style = style
    if style == "decoder":
      self.look_ahead_mask = create_look_ahead_mask(seq_len)
    elif style == "encoder":
      self.look_ahead_mask = None
    else:
      raise ValueError(f"Invalid style: {style}")

    self._norm1a = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-5, name="mhsa_normalization1")
    self._norm1b = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-5, name="ffn_normalization1")

    self._norm2a = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-5, name="mhsa_normalization2")
    self._norm2b = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-5, name="ffn_normalization2")

    self._attn1 = WindowMultiHeadAttention(
        d_model,
        num_head,
        attn_drop=drop_out_rate,
        proj_drop=drop_out_rate,
    )

    self._attn2 = WindowMultiHeadAttention(
        d_model,
        num_head,
        attn_drop=drop_out_rate,
        proj_drop=drop_out_rate,
    )

    self._mlp1 = MLP(
        expansion_rate=mlp_expansion,
        act_name=mlp_act,
        dropout_rate=drop_out_rate,
        name="ffn1")
    self._mlp2 = MLP(
        expansion_rate=mlp_expansion,
        act_name=mlp_act,
        dropout_rate=drop_out_rate,
        name="ffn2")

    # No weights, so we share for both blocks.
    self._drop_path = StochasticDepth(drop_path_rate, name="drop_path")

  def __call__(self, features, enc_output,
               training):
    if enc_output is None:
      if self._style == "decoder":
        raise ValueError("Need `enc_output` when running decoder.")
    else:
      tf.debugging.assert_shapes([
          (enc_output, ("Bp", "k_times_seq_len", "d_model")),
          (features, ("Bp", "seq_len", "d_model")),
      ])

    # First Block ---
    shortcut = features
    features = self._norm1a(features)
    # Masked self-attention.
    features, _ = self._attn1(
        v=features,
        k=features,
        q=features,
        mask=self.look_ahead_mask,
        training=training)
    assert features.shape == shortcut.shape
    features = shortcut + self._drop_path(features, training)

    features = features + self._drop_path(
        self._mlp1(self._norm1b(features), training), training)

    # Second Block ---
    shortcut = features
    features = self._norm2a(features)
    # Unmasked "lookup" into enc_output, no need for mask.
    features, _ = self._attn2(  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
        v=enc_output if enc_output is not None else features,
        k=enc_output if enc_output is not None else features,
        q=features,
        mask=None,
        training=training)
    features = shortcut + self._drop_path(features, training)
    output = features + self._drop_path(
        self._mlp2(self._norm2b(features), training), training)

    return output


class Transformer(tf.Module):
  """A stack of transformer blocks, useable for encoding or decoding."""

  def __init__(
      self,
      is_decoder,
      *,
      num_layers = 4,
      d_model = 192,
      seq_len = 16,
      num_head = 4,
      mlp_expansion = 4,
      drop_out = 0.1,
      name = None
  ):
    super().__init__(name=name)
    self.is_decoder = is_decoder

    # Use a plain list here since we have to pass the enc_output to each.
    self.layers = []
    for _ in range(num_layers):
      self.layers.append(
          TransformerBlock(
              d_model=d_model,
              seq_len=seq_len,
              num_head=num_head,
              mlp_expansion=mlp_expansion,
              drop_out_rate=drop_out,
              drop_path_rate=drop_out,
              style="decoder" if is_decoder else "encoder",
          ))

  def __call__(
      self, latent, enc_output, training
  ):
    """Forward pass.

    For decoder, this predicts distribution of `latent` given `enc_output`.

    We assume that `latent` has already been embedded in a d_model-dimensional
    space.

    Args:
      latent: (B', seq_len, C) latent.
      enc_output: (B', seq_len_enc, C) result of concatenated encode output.
      training: Whether we are training.

    Returns:
      Decoder output of shape (B', seq_len, C).
    """
    assert len(latent.shape) == 3, latent.shape
    if enc_output is not None:
      assert latent.shape[-1] == enc_output.shape[-1], (latent.shape,
                                                        enc_output.shape)
    for layer in self.layers:
      latent = layer(
          features=latent, enc_output=enc_output, training=training)
    return latent


class EncoderSection(Transformer):
  """N-layer encoder."""

  def __init__(
      self,
      num_layers,
      d_model,
      mlp_expansion,
      num_head,
      drop_out,
      name = None,
  ):
    super().__init__(
        is_decoder=False,
        num_layers=num_layers,
        d_model=d_model,
        seq_len=0,
        num_head=num_head,
        mlp_expansion=mlp_expansion,
        drop_out=drop_out,
        name=name)

  def __call__(self, latent, training):
    return super().__call__(latent, None, training)
