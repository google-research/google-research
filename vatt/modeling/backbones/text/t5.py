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

# Lint as: python3
"""T5 language modeling modules."""

import re

import numpy as np
import tensorflow as tf


def assign_weight_from_h5(w, h5_handle):
  """Restoring weights from Huggingface checkpoints."""

  success = False
  native_to_h5 = [
      [r"^transformer/(.*)/layer_([0-99]+)/(.*)",
       r"\1/tf_t5with_lm_head_model/\1/block_._\2/\3"],
      [r"^transformer/(.*)/final_layer_norm/(.*)",
       r"\1/tf_t5with_lm_head_model/\1/final_layer_norm/\2"],
      [r"^transformer/encoder/encoder_embeddings/(.*)",
       r"shared/tf_t5with_lm_head_model/shared/\1"],
      [r"^transformer/decoder/decoder_embeddings/(.*)",
       r"shared/tf_t5with_lm_head_model/shared/\1"],

      [r"^(.*)/self_attention/layer_norm/(.*)", r"\1/layer_._0/layer_norm/\2"],
      [r"^(.*)/cross_attention/layer_norm/(.*)", r"\1/layer_._1/layer_norm/\2"],

      [r"^(.*)/encoder/block_._([0-99]+)/feed_forward/layer_norm/(.*)",
       r"\1/encoder/block_._\2/layer_._1/layer_norm/\3"],
      [r"^(.*)/decoder/block_._([0-99]+)/feed_forward/layer_norm/(.*)",
       r"\1/decoder/block_._\2/layer_._2/layer_norm/\3"],

      [r"^(.*)/self_attention/(.*)", r"\1/layer_._0/SelfAttention/\2"],
      [r"^(.*)/cross_attention/(.*)", r"\1/layer_._1/EncDecAttention/\2"],

      [r"^(.*)/encoder/block_._([0-99]+)/feed_forward/dense_relu_dense/(.*)",
       r"\1/encoder/block_._\2/layer_._1/DenseReluDense/\3"],
      [r"^(.*)/decoder/block_._([0-99]+)/feed_forward/dense_relu_dense/(.*)",
       r"\1/decoder/block_._\2/layer_._2/DenseReluDense/\3"],
  ]

  h5_name = w.name
  for source, dest in native_to_h5:
    h5_name = re.sub(source, dest, h5_name)

  if h5_name in h5_handle:
    weights = h5_handle[h5_name][Ellipsis]
    if weights.shape != w.numpy().shape:
      success = False
    else:
      assigned = w.assign(weights)
      if np.linalg.norm(assigned-weights) == 0:
        success = True

  return success


def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.

  Args:
    initializer_range: float, initializer range for stddev.
  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class Embeddings(tf.keras.layers.Layer):
  """Construct token embeddings.

     Shared weights logic adapted from
     https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
  """

  def __init__(self,
               vocab_size,
               hidden_size,
               initializer_range=None,
               **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.initializer_range = (hidden_size ** -0.5 if initializer_range is None
                              else initializer_range)

  def build(self, input_shape):
    """Build shared token embedding layer.

    Args:
      input_shape: keras internal input_shape arg
    """
    self.weight = self.add_weight(
        "weight",
        shape=[self.vocab_size, self.hidden_size],
        initializer=get_initializer(self.initializer_range)
        )
    super().build(input_shape)

  def call(self, inputs, mode="embedding"):
    """Get token embeddings of inputs.

    Args:
      inputs: list of three int64 tensors with shape [batch_size, length]:
        (input_ids, position_ids, token_type_ids)
      mode: string, a valid value is one of "embedding" and "linear".
    Returns:
      outputs: (1) If mode == "embedding", output embedding tensor,
        float32 with shape [batch_size, length, embedding_size];
        (2) mode == "linear", output linear tensor, float32 with shape
        [batch_size, length, vocab_size].
    Raises:
      ValueError: if mode is not valid.
    """
    if mode == "embedding":
      return self._embedding(inputs)
    elif mode == "linear":
      return self._linear(inputs)
    else:
      raise ValueError("mode {} is not valid.".format(mode))

  def _embedding(self, input_ids):
    """Applies embedding based on inputs tensor."""
    return tf.gather(self.weight, input_ids)

  def _linear(self, inputs):
    """Computes logits by running inputs through a linear layer.

    Args:
      inputs: A float32 tensor with shape [..., hidden_size]
    Returns:
      float32 tensor with shape [..., vocab_size].
    """
    first_dims = get_shape(inputs)[:-1]

    x = tf.reshape(inputs, [-1, self.hidden_size])
    logits = tf.matmul(x, self.weight, transpose_b=True)

    return tf.reshape(logits, first_dims + [self.vocab_size])


class T5LayerNorm(tf.keras.layers.Layer):
  """Construct T5-specific layer norm."""

  def __init__(self,
               epsilon=1e-6,
               **kwargs):
    """Construct a layernorm module in the T5 style.

       No bias and no substraction of mean.

    Args:
      epsilon: default variance epsilon
      **kwargs: arbitrary layer args
    Returns:
      T5-specific normalized output
    """
    super().__init__(**kwargs)
    self.variance_epsilon = epsilon

  def build(self, input_shape):
    """Build shared word embedding layer."""
    self.weight = self.add_weight("weight",
                                  shape=(input_shape[-1],),
                                  initializer="ones")
    super().build(input_shape)

  def call(self, x):
    variance = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
    x = x * tf.math.rsqrt(variance + self.variance_epsilon)
    return self.weight * x


class DenseReLUDense(tf.keras.layers.Layer):
  """Construct Dense+ReLU+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               dropout_rate,
               name="dense_relu_dense"):
    super(DenseReLUDense, self).__init__(name=name)
    self.wi = tf.keras.layers.Dense(d_ff, use_bias=False, name="wi")
    self.wo = tf.keras.layers.Dense(d_model, use_bias=False, name="wo")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.act = tf.keras.activations.relu

  def call(self,
           hidden_states,
           training=False):

    h = self.wi(hidden_states)
    h = self.act(h)
    h = self.dropout(h, training=training)
    h = self.wo(h)
    return h


class DenseGeLUDense(DenseReLUDense):
  """Construct Dense+GeLU+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               dropout_rate,
               name="dense_gelu_dense"):
    super(DenseGeLUDense, self).__init__(d_ff=d_ff,
                                         d_model=d_model,
                                         dropout_rate=dropout_rate,
                                         name=name)
    self.act = tf.keras.activations.gelu


class DenseSwishDense(DenseReLUDense):
  """Construct Dense+Swish+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               dropout_rate,
               name="dense_swish_dense"):
    super(DenseSwishDense, self).__init__(d_ff=d_ff,
                                          d_model=d_model,
                                          dropout_rate=dropout_rate,
                                          name=name)
    self.act = tf.keras.activations.swish


class DenseGeGLUDense(tf.keras.layers.Layer):
  """Construct Dense+GeGLU+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               dropout_rate,
               name="dense_geglu_dense"):
    super(DenseGeGLUDense, self).__init__(name=name)
    self.wi_0 = tf.keras.layers.Dense(d_ff, use_bias=False, name="wi_0")
    self.wi_1 = tf.keras.layers.Dense(d_ff, use_bias=False, name="wi_1")
    self.wo = tf.keras.layers.Dense(d_model, use_bias=False, name="wo")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.act = tf.keras.activations.gelu

  def call(self,
           hidden_states,
           training=False):

    h_g = self.act(self.wi_0(hidden_states))
    h_l = self.wi_1(hidden_states)
    h = h_g * h_l
    h = self.dropout(h, training=training)
    h = self.wo(h)
    return h


class FeedForward(tf.keras.layers.Layer):
  """Construct FeedForward module used in Transformer layers."""

  def __init__(self,
               d_ff,
               d_model,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               **kwargs):

    super().__init__(**kwargs)
    if activation == "relu":
      self.mlp = DenseReLUDense(d_ff,
                                d_model,
                                dropout_rate,
                                name="dense_relu_dense")
    elif activation == "gelu":
      self.mlp = DenseGeLUDense(d_ff,
                                d_model,
                                dropout_rate,
                                name="dense_gelu_dense")
    elif activation == "swish":
      self.mlp = DenseSwishDense(d_ff,
                                 d_model,
                                 dropout_rate,
                                 name="dense_swish_dense")
    elif activation == "geglu":
      self.mlp = DenseGeGLUDense(d_ff,
                                 d_model,
                                 dropout_rate,
                                 name="dense_geglu_dense")

    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=layer_norm_epsilon,
        name="layer_norm"
        )
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self,
           hidden_states,
           training=False):

    norm_x = self.layer_norm(hidden_states)
    y = self.mlp(norm_x, training=training)
    layer_output = hidden_states + self.dropout(y, training=training)
    return layer_output


class MultiHeadAttention(tf.keras.layers.Layer):
  """Construct the main MHA module used in Transformer layers."""

  def __init__(self,
               d_model,
               d_kv,
               num_heads,
               dropout_rate,
               num_relative_buckets,
               max_relative_distance,
               **kwargs):
    super().__init__(**kwargs)
    self.num_relative_buckets = num_relative_buckets
    self.max_relative_distance = max_relative_distance
    self.d_model = d_model
    self.d_kv = d_kv
    self.n_heads = num_heads
    self.inner_dim = self.n_heads * self.d_kv

    # query, key, and value mapping
    self.q = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="q")
    self.k = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="k")
    self.v = tf.keras.layers.Dense(self.inner_dim, use_bias=False, name="v")
    self.o = tf.keras.layers.Dense(self.d_model, use_bias=False, name="o")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    if self.num_relative_buckets is not None:
      self.relative_attention_bias = tf.keras.layers.Embedding(
          self.num_relative_buckets,
          self.n_heads,
          name="relative_attention_bias"
          )

  def _relative_position_bucket(self,
                                relative_position,
                                bidirectional,
                                num_buckets,
                                max_distance):
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += tf.dtypes.cast(tf.math.less(n, 0), tf.int32) * num_buckets
      n = tf.math.abs(n)
    else:
      n = tf.math.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = tf.math.less(n, max_exact)
    val_if_large = max_exact + tf.dtypes.cast(
        tf.math.log(tf.dtypes.cast(n, tf.float32) / max_exact)
        / np.log(max_distance / max_exact)
        * (num_buckets - max_exact),
        tf.int32,
    )
    val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
    ret += tf.where(is_small, n, val_if_large)
    return ret

  def compute_bias(self,
                   qlen,
                   klen,
                   bidirectional):
    """Compute binned relative position bias."""

    context_position = tf.range(qlen)[:, None]
    memory_position = tf.range(klen)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self._relative_position_bucket(
        relative_position=relative_position,
        bidirectional=bidirectional,
        num_buckets=self.num_relative_buckets,
        max_distance=self.max_relative_distance
        )

    # shape (qlen, klen, num_heads)
    values = self.relative_attention_bias(rp_bucket)

    # shape (1, num_heads, qlen, klen)
    values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
    return values

  def _split_heads(self, x, bs):
    """Split heads and rearrange elements."""

    # output shape: (bs, n_heads, seq_len, d_kv)
    return tf.transpose(
        tf.reshape(x, (bs, -1, self.n_heads, self.d_kv)),
        perm=(0, 2, 1, 3)
        )

  def _join_heads(self, x, bs):
    """Join heads and rearrange elements."""

    # output shape: (bs, seq_len, inner_dim)
    return tf.reshape(
        tf.transpose(x, perm=(0, 2, 1, 3)),
        (bs, -1, self.inner_dim)
        )

  def self_attention(self,
                     inputs,
                     mask=None,
                     bidirectional=True,
                     position_bias=None,
                     position_embeddings=None,
                     past_key_value_state=None,
                     use_cache=False,
                     training=False):

    bs, qlen, _ = get_shape(inputs)
    if past_key_value_state is not None:
      error_message = (
          "past_key_value_state should have 2 past states: "
          "keys and values. Got {} past states".format(
              len(past_key_value_state)
              )
          )
      assert (len(past_key_value_state) == 2), error_message
      seq_len = qlen + get_shape(past_key_value_state[0])[2]
    else:
      seq_len = qlen

    q = self.q(inputs)  # (bs, seq_len, inner_dim)
    k = self.k(inputs)  # (bs, seq_len, inner_dim)
    v = self.v(inputs)  # (bs, seq_len, inner_dim)

    q = self._split_heads(q, bs)  # (bs, n_heads, seq_len, dim_per_head)
    k = self._split_heads(k, bs)  # (bs, n_heads, seq_len, dim_per_head)
    v = self._split_heads(v, bs)  # (bs, n_heads, seq_len, dim_per_head)

    if past_key_value_state is not None:
      k_, v_ = past_key_value_state
      k = tf.concat([k_, k], axis=2)  # (bs, n_heads, seq_len, dim_per_head)
      v = tf.concat([v_, v], axis=2)  # (bs, n_heads, seq_len, dim_per_head)

    if tf.is_tensor(use_cache):
      if hasattr(use_cache, "numpy"):
        use_cache = bool(use_cache.numpy())
      else:
        use_cache = True

    if use_cache:
      present_key_value_state = (k, v)
    else:
      present_key_value_state = None

    if position_bias is None:
      if self.num_relative_buckets is None:
        raise ValueError("No position_bias provided and no "
                         "weights to compute position_bias")
      position_bias = self.compute_bias(qlen=seq_len,
                                        klen=seq_len,
                                        bidirectional=bidirectional)

      # if key and values are already calculated
      # we want only the last query position bias
      if past_key_value_state is not None:
        position_bias = position_bias[:, :, -1:, :]

      if mask is not None:
        position_bias = position_bias + mask  # (bs, n_heads, seq_len, seq_len)

    # (bs, n_heads, seq_len, seq_len)
    scores = tf.einsum("bnqd,bnkd->bnqk", q, k)
    scores += position_bias

    # (bs, n_heads, seq_len, seq_len)
    attention_weights = tf.nn.softmax(scores, axis=-1)
    # (bs, n_heads, seq_len, seq_len)
    attention_weights = self.dropout(attention_weights, training=training)

    # (bs, n_heads, seq_len, dim_per_head)
    hidden_states = tf.matmul(attention_weights, v)
    # (bs, seq_len, dim)
    hidden_states = self._join_heads(hidden_states, bs)
    # (bs, seq_len, out_dim)
    hidden_states = self.o(hidden_states)

    outputs = {
        "hidden_states": hidden_states,
        "key_value_state": present_key_value_state,
        "attention_weights": attention_weights,
        "position_bias": position_bias,
    }

    return outputs

  def cross_attention(self,
                      query,
                      key,
                      value,
                      mask=None,
                      bidirectional=False,
                      use_cache=False,
                      position_bias=None,
                      position_embeddings=None,
                      kv_position_embeddings=None,
                      query_length=None,
                      past_key_value_state=None,
                      training=False):

    bs, qlen, _ = get_shape(query)
    klen = get_shape(key)[1]

    if past_key_value_state is not None:
      error_message = ("past_key_value_state should have 2 past states: "
                       "keys and values. Got {} past states".format(
                           len(past_key_value_state)
                           )
                       )
      assert (len(past_key_value_state) == 2), error_message
      real_qlen = (
          qlen + get_shape(past_key_value_state[0])[2] if query_length is None
          else query_length
          )
    else:
      real_qlen = qlen

    qlen = real_qlen

    q = self.q(query)  # (bs, qlen, inner_dim)
    q = self._split_heads(q, bs)  # (bs, n_heads, qlen, dim_per_head)

    if past_key_value_state is None:
      k = self.k(key)  # (bs, klen, inner_dim)
      v = self.v(value)  # (bs, klen, inner_dim)
      k = self._split_heads(k, bs)  # (bs, n_heads, klen, dim_per_head)
      v = self._split_heads(v, bs)  # (bs, n_heads, vlen, dim_per_head)
    else:
      k, v = past_key_value_state

    if tf.is_tensor(use_cache):
      if hasattr(use_cache, "numpy"):
        use_cache = bool(use_cache.numpy())
      else:
        use_cache = True

    if use_cache:
      present_key_value_state = (k, v)
    else:
      present_key_value_state = None

    if position_bias is None:
      if self.num_relative_buckets is None:
        raise ValueError(
            "No position_bias provided and no weights to compute position_bias"
            )
      position_bias = self.compute_bias(qlen=qlen,
                                        klen=klen,
                                        bidirectional=bidirectional)

      # if key and values are already calculated
      # we want only the last query position bias
      if past_key_value_state is not None:
        position_bias = position_bias[:, :, -1:, :]

      if mask is not None:
        position_bias = position_bias + mask  # (bs, n_heads, seq_len, seq_len)

    # (bs, n_heads, seq_len, seq_len)
    scores = tf.einsum("bnqd,bnkd->bnqk", q, k)
    scores += position_bias

    # (bs, n_heads, seq_len, seq_len)
    attention_weights = tf.nn.softmax(scores, axis=-1)
    # (bs, n_heads, seq_len, seq_len)
    attention_weights = self.dropout(attention_weights, training=training)
    # (bs, n_heads, seq_len, dim_per_head)
    hidden_states = tf.matmul(attention_weights, v)
    hidden_states = self._join_heads(hidden_states, bs)  # (bs, seq_len, dim)
    hidden_states = self.o(hidden_states)  # (bs, seq_len, out_dim)

    outputs = {
        "hidden_states": hidden_states,
        "key_value_state": present_key_value_state,
        "attention_weights": attention_weights,
        "position_bias": position_bias,
    }

    return outputs


class AttentionLayer(tf.keras.layers.Layer):
  """Construct the main Attention module which includes MHA + Res + Norm."""

  def __init__(self,
               d_model,
               d_kv,
               num_heads,
               dropout_rate,
               layer_norm_epsilon,
               attention_type,
               num_relative_buckets,
               max_relative_distance,
               **kwargs):
    super().__init__(**kwargs)
    attention_valid = attention_type in {"self_attention", "cross_attention"}
    assert attention_valid, "Attention type not supported!"

    self.is_self_attention = attention_type == "self_attention"
    self.attention = MultiHeadAttention(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        num_relative_buckets=num_relative_buckets,
        max_relative_distance=max_relative_distance,
        name="multi_head_attention"
        )
    self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon,
                                  name="layer_norm")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def self_attention(self,
                     inputs,
                     bidirectional=True,
                     attention_mask=None,
                     position_bias=None,
                     position_embeddings=None,
                     past_key_value_state=None,
                     use_cache=False,
                     training=False):

    norm_x = self.layer_norm(inputs)
    attention_outputs = self.attention.self_attention(
        inputs=norm_x,
        bidirectional=bidirectional,
        mask=attention_mask,
        position_bias=position_bias,
        position_embeddings=position_embeddings,
        past_key_value_state=past_key_value_state,
        use_cache=use_cache,
        training=training
        )

    hidden_states = attention_outputs["hidden_states"]
    hidden_states = inputs + self.dropout(hidden_states, training=training)

    # update hidden states
    attention_outputs["hidden_states"] = hidden_states

    return attention_outputs

  def cross_attention(self,
                      query,
                      key,
                      value,
                      bidirectional=False,
                      attention_mask=None,
                      position_bias=None,
                      position_embeddings=None,
                      kv_position_embeddings=None,
                      past_key_value_state=None,
                      query_length=None,
                      use_cache=False,
                      training=False):

    norm_query = self.layer_norm(query)
    attention_outputs = self.attention.cross_attention(
        query=norm_query,
        key=key,
        value=value,
        bidirectional=bidirectional,
        mask=attention_mask,
        position_bias=position_bias,
        position_embeddings=position_embeddings,
        kv_position_embeddings=kv_position_embeddings,
        past_key_value_state=past_key_value_state,
        query_length=query_length,
        use_cache=use_cache,
        training=training
        )

    hidden_states = attention_outputs["hidden_states"]
    hidden_states = query + self.dropout(hidden_states, training=training)

    # update hidden states
    attention_outputs["hidden_states"] = hidden_states

    return attention_outputs

  def call(self,
           inputs,
           key=None,
           value=None,
           bidirectional=False,
           attention_mask=None,
           position_bias=None,
           position_embeddings=None,
           kv_position_embeddings=None,
           past_key_value_state=None,
           query_length=None,
           use_cache=False,
           training=False):

    if self.is_self_attention:
      return self.self_attention(
          inputs=inputs,
          bidirectional=bidirectional,
          attention_mask=attention_mask,
          position_bias=position_bias,
          position_embeddings=position_embeddings,
          past_key_value_state=past_key_value_state,
          use_cache=use_cache,
          training=training
          )
    else:
      return self.cross_attention(
          query=inputs,
          key=key,
          value=value,
          bidirectional=bidirectional,
          attention_mask=attention_mask,
          position_bias=position_bias,
          position_embeddings=position_embeddings,
          kv_position_embeddings=kv_position_embeddings,
          past_key_value_state=past_key_value_state,
          query_length=query_length,
          use_cache=use_cache,
          training=training
          )


class EncoderLayer(tf.keras.layers.Layer):
  """Construct the main Encoder layer."""

  def __init__(self,
               d_model,
               d_kv,
               d_ff,
               num_heads,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               num_relative_buckets,
               max_relative_distance,
               **kwargs):
    super().__init__(**kwargs)
    self.self_attention = AttentionLayer(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        num_relative_buckets=num_relative_buckets,
        max_relative_distance=max_relative_distance,
        attention_type="self_attention",
        name="self_attention"
        )
    self.feed_forward = FeedForward(d_ff=d_ff,
                                    d_model=d_model,
                                    activation=activation,
                                    dropout_rate=dropout_rate,
                                    layer_norm_epsilon=layer_norm_epsilon,
                                    name="feed_forward")

  def call(self,
           inputs,
           attention_mask=None,
           position_bias=None,
           position_embeddings=None,
           training=False):

    attention_outputs = self.self_attention(
        inputs=inputs,
        bidirectional=True,
        attention_mask=attention_mask,
        position_bias=position_bias,
        position_embeddings=position_embeddings,
        training=training
        )

    hidden_states = attention_outputs["hidden_states"]

    # Apply Feed Forward layer
    hidden_states = self.feed_forward(hidden_states, training=training)

    outputs = {
        "hidden_states": hidden_states,
        "self_attention_weights": attention_outputs["attention_weights"],
        "self_position_bias": attention_outputs["position_bias"],
    }

    return outputs


class DecoderLayer(tf.keras.layers.Layer):
  """Construct the main Decoder layer."""

  def __init__(self,
               d_model,
               d_kv,
               d_ff,
               num_heads,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               num_self_relative_buckets,
               max_self_relative_distance,
               num_cross_relative_buckets,
               max_cross_relative_distance,
               **kwargs):
    super().__init__(**kwargs)
    self.self_attention = AttentionLayer(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        num_relative_buckets=num_self_relative_buckets,
        max_relative_distance=max_self_relative_distance,
        attention_type="self_attention",
        name="self_attention"
        )
    self.cross_attention = AttentionLayer(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        num_relative_buckets=num_cross_relative_buckets,
        max_relative_distance=max_cross_relative_distance,
        attention_type="cross_attention",
        name="cross_attention"
        )
    self.feed_forward = FeedForward(d_ff=d_ff,
                                    d_model=d_model,
                                    activation=activation,
                                    dropout_rate=dropout_rate,
                                    layer_norm_epsilon=layer_norm_epsilon,
                                    name="feed_forward")

  def call(self,
           inputs,
           encoder_hidden_states,
           encoder_position_embeddings=None,
           attention_mask=None,
           encoder_attention_mask=None,
           position_bias=None,
           position_embeddings=None,
           encoder_decoder_position_bias=None,
           past_key_value_state=None,
           use_cache=False,
           training=False):

    if past_key_value_state is not None:
      expected_num_past_key_value_states = (2 if encoder_hidden_states is None
                                            else 4)
      is_num_asserted = expected_num_past_key_value_states == 4
      crss_attn_message = (
          " and 2 (past / key) for cross attention" if is_num_asserted
          else ""
          )

      assrt_cond = (
          len(past_key_value_state) == expected_num_past_key_value_states
          )

      error_message = (
          "There should be {} past states. 2 (past / key) for "
          "self attention{}. Got {} past key / value states".format(
              expected_num_past_key_value_states,
              crss_attn_message,
              len(past_key_value_state)
              )
          )

      assert assrt_cond, error_message

      self_attn_past_key_value_state = past_key_value_state[:2]
      cross_attn_past_key_value_state = past_key_value_state[2:]
    else:
      self_attn_past_key_value_state = None
      cross_attn_past_key_value_state = None

    self_attention_outputs = self.self_attention(
        inputs=inputs,
        bidirectional=False,
        attention_mask=attention_mask,
        position_bias=position_bias,
        position_embeddings=position_embeddings,
        past_key_value_state=self_attn_past_key_value_state,
        use_cache=use_cache,
        training=training
        )

    decoder_hidden_states = self_attention_outputs["hidden_states"]
    # This is either None or (k_self,v_self)
    self_attn_key_value_state = self_attention_outputs["key_value_state"]

    # the actual query length is unknown for cross attention
    # if using past key value states. Need to inject it here
    if self_attn_key_value_state is not None:
      query_length = get_shape(self_attn_key_value_state[0])[2]
    else:
      query_length = None

    cross_attention_outputs = self.cross_attention(
        inputs=decoder_hidden_states,
        key=encoder_hidden_states,
        value=encoder_hidden_states,
        bidirectional=False,
        attention_mask=encoder_attention_mask,
        position_bias=encoder_decoder_position_bias,
        position_embeddings=position_embeddings,
        kv_position_embeddings=encoder_position_embeddings,
        past_key_value_state=cross_attn_past_key_value_state,
        query_length=query_length,
        use_cache=use_cache,
        training=training
        )

    decoder_hidden_states = cross_attention_outputs["hidden_states"]
    cross_attn_key_value_state = cross_attention_outputs["key_value_state"]

    # Combine self attn and cross attn key value states
    if self_attn_key_value_state is not None:
      # (k_self,v_self,k_cross,v_cross)
      present_key_value_state = (self_attn_key_value_state
                                 + cross_attn_key_value_state)
    else:
      present_key_value_state = None

    # Apply Feed Forward layer
    decoder_hidden_states = self.feed_forward(decoder_hidden_states,
                                              training=training)

    outputs = {
        "hidden_states": decoder_hidden_states,
        "key_value_state": present_key_value_state,
        "self_attention_weights": self_attention_outputs["attention_weights"],
        "self_position_bias": self_attention_outputs["position_bias"],
        "cross_attention_weights": cross_attention_outputs["attention_weights"],
        "cross_position_bias": cross_attention_outputs["position_bias"],
    }

    return outputs


class T5Encoder(tf.keras.layers.Layer):
  """Construct the final Encoder stack."""

  def __init__(self,
               d_model,
               d_kv,
               d_ff,
               num_layers,
               num_heads,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               num_relative_buckets,
               max_relative_distance,
               use_embeddings=False,
               vocab_size=None,
               name="t5_encoder",
               **kwargs):
    super(T5Encoder, self).__init__(name=name)

    if use_embeddings:
      self.embeddings = Embeddings(vocab_size=vocab_size,
                                   hidden_size=d_model,
                                   name="encoder_embeddings")
    else:
      self.embeddings = None

    self.d_model = d_model
    self.num_hidden_layers = num_layers
    self.relative_buckets = [num_relative_buckets]+[None]*(num_layers-1)
    self.relative_distances = [max_relative_distance]+[None]*(num_layers-1)

    self.layers = []
    for n in range(self.num_hidden_layers):
      self.layers.append(
          EncoderLayer(d_model=d_model,
                       d_kv=d_kv,
                       d_ff=d_ff,
                       num_heads=num_heads,
                       activation=activation,
                       dropout_rate=dropout_rate,
                       layer_norm_epsilon=layer_norm_epsilon,
                       num_relative_buckets=self.relative_buckets[n],
                       max_relative_distance=self.relative_distances[n],
                       name="layer_{}".format(n))
          )

    self.final_layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon,
                                        name="final_layer_norm")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(
      self,
      inputs,
      inputs_embeddings=None,
      attention_mask=None,
      training=False):

    if inputs is not None and inputs_embeddings is not None:
      raise ValueError("You cannot specify both inputs and "
                       "inputs_embeddings at the same time")
    elif inputs is not None:
      input_shape = get_shape(inputs)
      inputs = tf.reshape(inputs, (-1, input_shape[-1]))
    elif inputs_embeddings is not None:
      input_shape = get_shape(inputs_embeddings)[:-1]
    else:
      raise ValueError("You have to specify either inputs or inputs_embeddings")

    if inputs_embeddings is None:
      err_msg = "You have to intialize the model with valid token embeddings"
      assert self.embeddings is not None, err_msg
      inputs_embeddings = self.embeddings(inputs)

    batch_size, seq_length = input_shape
    mask_seq_length = seq_length
    if attention_mask is None:
      attention_mask = tf.fill((batch_size, mask_seq_length), 1)

    # Provided a padding mask of dimensions [batch_size, mask_seq_length]
    # make the mask broadcastable to
    #         [batch_size, num_heads, mask_seq_length, mask_seq_length]
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)
    attention_mask = attention_mask[:, None, None, :]
    attention_mask = (1.0 - attention_mask) * -1e9

    all_hidden_states = ()
    all_attentions = ()
    position_bias = None

    hidden_states = self.dropout(inputs_embeddings, training=training)

    for n, layer in enumerate(self.layers):
      # temporary --- to reduce memory consumption
      # all_hidden_states = all_hidden_states + (hidden_states,)

      layer_outputs = layer(inputs=hidden_states,
                            attention_mask=attention_mask,
                            position_bias=position_bias,
                            training=training)

      # layer_outputs is a dictionary with the following keys:
      # hidden_states, key_value_state,
      #                           self_attention_weights, self_position_bias

      hidden_states = layer_outputs["hidden_states"]
      if n == 0:
        position_bias = layer_outputs["self_position_bias"]

      all_attentions = all_attentions + (
          layer_outputs["self_attention_weights"],
          )

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states, training=training)

    # Add last layer
    all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = {
        "hidden_states": all_hidden_states,
        "attention_weights": all_attentions,
    }

    return outputs


class T5Decoder(tf.keras.layers.Layer):
  """Construct the final Decoder stack."""

  def __init__(self,
               d_model,
               d_kv,
               d_ff,
               num_layers,
               num_heads,
               activation,
               num_self_relative_buckets,
               max_self_relative_distance,
               num_cross_relative_buckets,
               max_cross_relative_distance,
               dropout_rate,
               layer_norm_epsilon,
               vocab_size,
               name="t5_decoder",
               **kwargs):
    super(T5Decoder, self).__init__(name=name)
    self.embeddings = Embeddings(vocab_size=vocab_size,
                                 hidden_size=d_model,
                                 name="decoder_embeddings")

    self.d_model = d_model
    self.num_hidden_layers = num_layers

    self.self_relative_buckets = [num_self_relative_buckets]
    self.self_relative_buckets += [None]*(num_layers-1)

    self.cross_relative_buckets = [num_cross_relative_buckets]
    self.cross_relative_buckets += [None]*(num_layers-1)

    self.self_relative_distances = [max_self_relative_distance]
    self.self_relative_distances += [None]*(num_layers-1)

    self.cross_relative_distances = [max_cross_relative_distance]
    self.cross_relative_distances += [None]*(num_layers-1)

    self.layers = []
    for n in range(self.num_hidden_layers):
      self.layers.append(
          DecoderLayer(
              d_model=d_model,
              d_kv=d_kv,
              d_ff=d_ff,
              num_heads=num_heads,
              activation=activation,
              dropout_rate=dropout_rate,
              layer_norm_epsilon=layer_norm_epsilon,
              num_self_relative_buckets=self.self_relative_buckets[n],
              max_self_relative_distance=self.self_relative_distances[n],
              num_cross_relative_buckets=self.cross_relative_buckets[n],
              max_cross_relative_distance=self.cross_relative_distances[n],
              name="layer_{}".format(n))
          )

    self.final_layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon,
                                        name="final_layer_norm")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self,
           inputs,
           encoder_hidden_states,
           encoder_position_embeddings=None,
           attention_mask=None,
           encoder_attention_mask=None,
           inputs_embeddings=None,
           past_key_value_states=None,
           use_cache=False,
           training=False):

    if inputs is not None and inputs_embeddings is not None:
      raise ValueError("You cannot specify both inputs and "
                       "inputs_embeddings at the same time")
    elif inputs is not None:
      input_shape = get_shape(inputs)
      inputs = tf.reshape(inputs, (-1, input_shape[-1]))
    elif inputs_embeddings is not None:
      input_shape = get_shape(inputs_embeddings)[:-1]
    else:
      raise ValueError("You have to specify either inputs or inputs_embeddings")

    batch_size, seq_length = input_shape

    if inputs_embeddings is None:
      err_msg = "You have to intialize the model with valid token embeddings"
      assert self.embeddings is not None, err_msg
      inputs_embeddings = self.embeddings(inputs)

    if past_key_value_states is not None:
      err_msg = ("Input shape is {}, but should be {} "
                 "when using past_key_value_sates".format(
                     input_shape,
                     (batch_size, 1)
                     )
                 )
      assert seq_length == 1, err_msg
      # required mask seq length can be calculated via length of past
      # key value states and seq_length = 1 for the last token
      mask_seq_length = get_shape(past_key_value_states[0][0])[2] + seq_length
    else:
      mask_seq_length = seq_length

    if attention_mask is None:
      attention_mask = tf.fill((batch_size, mask_seq_length), 1)

    if encoder_attention_mask is None and encoder_hidden_states is not None:
      encoder_seq_length = get_shape(encoder_hidden_states)[1]
      encoder_attention_mask = tf.fill((batch_size, encoder_seq_length), 1)

    # initialize past_key_value_states with `None` if past does not exist
    if past_key_value_states is None:
      past_key_value_states = [None] * self.num_hidden_layers

    # We can provide a self-attention mask of dimensions
    #   [batch_size, from_seq_length, to_seq_length] ourselves in which case we
    #   just need to make it broadcastable to all heads.
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)
    num_dims_attention_mask = len(get_shape(attention_mask))
    if num_dims_attention_mask == 3:
      extended_attention_mask = attention_mask[:, None, :, :]
    elif num_dims_attention_mask == 2:
      # Provided a padding mask of dimensions [batch_size, mask_seq_length]
      # - in a decoder, apply a causal mask in addition to the padding mask
      seq_ids = tf.range(mask_seq_length)
      causal_mask = tf.less_equal(
          tf.tile(seq_ids[None, None, :],
                  (batch_size, mask_seq_length, 1)),
          seq_ids[None, :, None],
          )
      causal_mask = tf.cast(causal_mask, dtype=tf.float32)
      extended_attention_mask = (
          causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
          )
      if past_key_value_states[0] is not None:
        extended_attention_mask = extended_attention_mask[:, :, -1:, :]

    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

    if encoder_attention_mask is not None:
      # If a 2D or 3D attention mask is provided for the cross-attention
      #   we need to make broadcastabe to
      #     [batch_size, num_heads, mask_seq_length, mask_seq_length]
      #   we need to make broadcastabe to
      #     [batch_size, num_heads, seq_length, seq_length]
      encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=tf.float32)
      num_dims_encoder_attention_mask = len(get_shape(encoder_attention_mask))
      if num_dims_encoder_attention_mask == 3:
        encoder_extended_attention_mask = (
            encoder_attention_mask[:, None, :, :]
            )
      if num_dims_encoder_attention_mask == 2:
        encoder_extended_attention_mask = (
            encoder_attention_mask[:, None, None, :]
            )

      encoder_extended_attention_mask = (
          (1.0 - encoder_extended_attention_mask) * -1e9
          )
    else:
      encoder_extended_attention_mask = None

    present_key_value_states = ()
    all_hidden_states = ()
    all_self_attentions = ()
    all_cross_attentions = ()
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeddings, training=training)

    for n, (layer, past_key_value_state) in enumerate(
        zip(self.layers, past_key_value_states)
        ):
      # temporary --- to reduce memory consumption
      # all_hidden_states = all_hidden_states + (hidden_states,)
      layer_outputs = layer(
          inputs=hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          attention_mask=extended_attention_mask,
          encoder_attention_mask=encoder_extended_attention_mask,
          position_bias=position_bias,
          encoder_decoder_position_bias=encoder_decoder_position_bias,
          past_key_value_state=past_key_value_state,
          use_cache=use_cache,
          training=training
          )

      # layer_outputs is a dictionary with the following keys:
      #   (hidden_states, key_value_state, self_attention_weights,
      #    self_position_bias, cross_attention_weights, cross_position_bias)
      hidden_states = layer_outputs["hidden_states"]
      present_key_value_state = layer_outputs["key_value_state"]

      if n == 0:
        # We share the position biases between the layers -
        #   the first layer store them
        position_bias = layer_outputs["self_position_bias"]
        encoder_decoder_position_bias = layer_outputs["cross_position_bias"]

      # append next layer key value states
      present_key_value_states = (
          present_key_value_states + (present_key_value_state,)
          )

      all_self_attentions = (
          all_self_attentions + (layer_outputs["self_attention_weights"],)
          )
      all_cross_attentions = (
          all_cross_attentions + (layer_outputs["cross_attention_weights"],)
          )

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states, training=training)

    # Add last layer
    all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = {
        "hidden_states": all_hidden_states,
        "key_value_states": present_key_value_states,
        "self_attention_weights": all_self_attentions,
        "cross_attention_weights": all_cross_attentions,
    }

    return outputs
