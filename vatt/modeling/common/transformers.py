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

# Lint as: python3
"""Transformer modules."""

import tensorflow as tf
import tensorflow_addons.image as tfa_image


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


class ExpandableEmbedding(tf.keras.layers.Embedding):
  """An embedding which supports differentiable expand beyond max_buckets."""

  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer="uniform",
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               **kwargs):
    self.max_buckets = input_dim
    super(ExpandableEmbedding, self).__init__(
        input_dim=input_dim,
        output_dim=output_dim,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer,
        activity_regularizer=activity_regularizer,
        embeddings_constraint=embeddings_constraint,
        mask_zero=mask_zero,
        input_length=input_length,
        **kwargs)

  def _get_expanded_embeddings(self, max_target_buckets, order=3):
    # define all currently possible buckets
    # shape = [1, max_buckets, 1]
    lookup_keys = tf.range(self.max_buckets)
    available_buckets = lookup_keys / self.max_buckets
    available_buckets = tf.cast(available_buckets, tf.float32)[None, :, None]

    # define all possible target buckets
    # shape = [1, max_target_buckets, 1]
    query_buckets = tf.range(max_target_buckets) / max_target_buckets
    query_buckets = tf.cast(query_buckets, tf.float32)[None, :, None]

    # fetch current available embeddings
    # shape = [1, max_buckets, embd_dim]
    available_embeddings = self.embeddings[None, Ellipsis]

    expanded_embeddings = tf.squeeze(tfa_image.interpolate_spline(
        train_points=available_buckets,
        train_values=available_embeddings,
        query_points=query_buckets,
        order=order), axis=0)

    return expanded_embeddings

  def call(self,
           inputs,
           interpolate=False,
           max_target_buckets=None):
    """If interpolate==True, first interpolates embeddings then looksup."""

    if interpolate:
      assert max_target_buckets is not None, (
          "max_target_buckets should be specified when interpolating"
          )
      expanded_embeddings = self._get_expanded_embeddings(max_target_buckets)
      return tf.nn.embedding_lookup(expanded_embeddings, inputs)

    else:
      return super(ExpandableEmbedding, self).call(inputs)


class TemporalEmbeddings(tf.keras.layers.Layer):
  """Construct the embeddings from temporal tokens."""

  def __init__(self,
               hidden_size,
               max_temporal_buckets,
               dropout_rate=None,
               initializer_range=None,
               layer_norm_epsilon=None,
               name="temporal_embeddings",
               **kwargs):
    super(TemporalEmbeddings, self).__init__(name=name)
    self.max_temporal_positions = max_temporal_buckets
    self.hidden_size = hidden_size
    self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate
    self.initializer_range = (
        hidden_size ** -0.5 if initializer_range is None else initializer_range
        )
    self.layer_norm_epsilon = (
        1e-6 if layer_norm_epsilon is None else layer_norm_epsilon
        )

    self.temporal_position_embeddings = ExpandableEmbedding(
        self.max_temporal_positions,
        self.hidden_size,
        embeddings_initializer=get_initializer(self.initializer_range),
        name="temporal_position_embeddings",
        )

    self.layernorm = tf.keras.layers.LayerNormalization(
        epsilon=self.layer_norm_epsilon,
        name="layer_norm"
        )
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def _embedding_lookup(self,
                        lookup_fn,
                        lookup_keys,
                        ref_buckets,
                        target_buckets):
    """Checks whether interpolation is necessary, then calls appropriate fn."""

    if target_buckets == ref_buckets:
      position_embeddings = lookup_fn(lookup_keys,
                                      interpolate=False,
                                      max_target_buckets=None)
    else:
      position_embeddings = lookup_fn(lookup_keys,
                                      interpolate=True,
                                      max_target_buckets=target_buckets)

    return position_embeddings

  def call(self, inputs, dimensions, training=False):
    """Get token embeddings of inputs.

    Args:
        inputs: input embeddings
        dimensions: a list of dimensions
        training: train flag
    Returns:
        position_embeddings: output embedding tensor, float32 with
          shape [batch_size, length, embedding_size]
    """
    _, t, _ = dimensions
    temporal_position_ids = tf.range(t)

    position_embeddings = self._embedding_lookup(
        lookup_fn=self.temporal_position_embeddings,
        lookup_keys=temporal_position_ids,
        ref_buckets=self.max_temporal_positions,
        target_buckets=t,
        )

    position_embeddings = self.layernorm(position_embeddings)
    embeddings = inputs + position_embeddings
    embeddings = self.dropout(embeddings, training=training)

    return embeddings


class SpectroTemporalEmbeddings(tf.keras.layers.Layer):
  """Construct the embeddings from spectro-temporal tokens."""

  def __init__(self,
               hidden_size,
               max_temporal_buckets,
               max_spectoral_buckets,
               dropout_rate=None,
               initializer_range=None,
               layer_norm_epsilon=None,
               name="spectro_temporal_embeddings",
               **kwargs):
    super(SpectroTemporalEmbeddings, self).__init__(name=name)
    self.max_temporal_positions = max_temporal_buckets
    self.max_spectoral_positions = max_spectoral_buckets
    self.hidden_size = hidden_size
    self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate
    self.initializer_range = (
        hidden_size ** -0.5 if initializer_range is None else initializer_range
        )
    self.layer_norm_epsilon = (
        1e-6 if layer_norm_epsilon is None else layer_norm_epsilon
        )

    self.temporal_position_embeddings = ExpandableEmbedding(
        self.max_temporal_positions,
        self.hidden_size,
        embeddings_initializer=get_initializer(self.initializer_range),
        name="temporal_position_embeddings",
        )

    self.spectoral_position_embeddings = ExpandableEmbedding(
        self.max_spectoral_positions,
        self.hidden_size,
        embeddings_initializer=get_initializer(self.initializer_range),
        name="spectoral_position_embeddings",
        )

    self.layernorm = tf.keras.layers.LayerNormalization(
        epsilon=self.layer_norm_epsilon,
        name="layer_norm"
        )
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def _build_aud_pos_ids(self, t, s):
    """Creates and returns 2d positional ids.

    Args:
      t: time length
      s: number of spectoral bins


    Returns:
      pos_ids: outputs with shape [t * s, 2],
        where 2 = 1 + 1 ; 1 for spectoral id and 1 for temporal id, with
        the following order: [t, s]
    """

    # define pos_ids - a fixed tensor which is a function of input shape
    temporal_ids = tf.range(t)[None, :]  # (1, t)
    spectoral_ids = tf.range(s)[:, None]  # (s, 1)

    temporal_ids = tf.tile(temporal_ids, [s, 1])  # (s, t)
    spectoral_ids = tf.tile(spectoral_ids, [1, t])  # (s, t)

    pos_ids = tf.stack([temporal_ids, spectoral_ids], axis=2)  # (t, s, 2)
    pos_ids = tf.reshape(pos_ids, [-1, 2])  # (t*s, 2)

    return pos_ids

  def _embedding_lookup(self,
                        lookup_fn,
                        lookup_keys,
                        ref_buckets,
                        target_buckets):
    """Checks whether interpolation is necessary, then calls appropriate fn."""

    if target_buckets == ref_buckets:
      position_embeddings = lookup_fn(lookup_keys,
                                      interpolate=False,
                                      max_target_buckets=None)
    else:
      position_embeddings = lookup_fn(lookup_keys,
                                      interpolate=True,
                                      max_target_buckets=target_buckets)

    return position_embeddings

  def call(self, inputs, dimensions, training=False):
    """Get token embeddings of inputs.

    Args:
        inputs: input embeddings
        dimensions: a list of dimensions
        training: train flag
    Returns:
        position_embeddings: output embedding tensor, float32 with
          shape [batch_size, length, embedding_size]
    """
    _, t, s, _ = dimensions
    pos_ids = self._build_aud_pos_ids(t, s)

    temporal_position_ids = pos_ids[None, :, 0]
    spectoral_position_ids = pos_ids[None, :, 1]

    temporal_position_embeddings = self._embedding_lookup(
        lookup_fn=self.temporal_position_embeddings,
        lookup_keys=temporal_position_ids,
        ref_buckets=self.max_temporal_positions,
        target_buckets=t,
        )

    spectoral_position_embeddings = self._embedding_lookup(
        lookup_fn=self.spectoral_position_embeddings,
        lookup_keys=spectoral_position_ids,
        ref_buckets=self.max_spectoral_positions,
        target_buckets=s,
        )

    position_embeddings = (
        spectoral_position_embeddings +
        temporal_position_embeddings
        )
    position_embeddings = self.layernorm(position_embeddings)
    embeddings = inputs + position_embeddings
    embeddings = self.dropout(embeddings, training=training)

    return embeddings


class SpatioTemporalEmbeddings(tf.keras.layers.Layer):
  """Construct the embeddings from spatio-temporal tokens."""

  def __init__(self,
               hidden_size,
               max_temporal_buckets,
               max_vertical_buckets,
               max_horizontal_buckets,
               dropout_rate=None,
               initializer_range=None,
               layer_norm_epsilon=None,
               name="spatio_temporal_embeddings",
               **kwargs):
    super(SpatioTemporalEmbeddings, self).__init__(name=name)
    self.max_temporal_positions = max_temporal_buckets
    self.max_vertical_positions = max_vertical_buckets
    self.max_horizontal_positions = max_horizontal_buckets
    self.hidden_size = hidden_size
    self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate
    self.initializer_range = (
        hidden_size ** -0.5 if initializer_range is None else initializer_range
        )
    self.layer_norm_epsilon = (
        1e-6 if layer_norm_epsilon is None else layer_norm_epsilon
        )

    self.temporal_position_embeddings = ExpandableEmbedding(
        self.max_temporal_positions,
        self.hidden_size,
        embeddings_initializer=get_initializer(self.initializer_range),
        name="temporal_position_embeddings",
        )

    self.vertical_position_embeddings = ExpandableEmbedding(
        self.max_vertical_positions,
        self.hidden_size,
        embeddings_initializer=get_initializer(self.initializer_range),
        name="vertical_position_embeddings",
        )

    self.horizontal_position_embeddings = ExpandableEmbedding(
        self.max_horizontal_positions,
        self.hidden_size,
        embeddings_initializer=get_initializer(self.initializer_range),
        name="horizontal_position_embeddings",
        )

    self.layernorm = tf.keras.layers.LayerNormalization(
        epsilon=self.layer_norm_epsilon,
        name="layer_norm"
        )
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

  def _build_vid_pos_ids(self, t, h, w):
    """Creates and returns 3d positional ids.

    Args:
      t: time length
      h: height
      w: width


    Returns:
      pos_ids: outputs with shape [t * h * w, 3],
        where 3 = 1 + 1 + 1; 1 for temporal id, 1 for vertical id, and 1 for
        horizontal id, with the following order: [t, h, w]
    """

    # define pos_ids - a fixed tensor which is a function of input shape
    temporal_ids = tf.range(t)[:, None, None]  # (t, 1, 1)
    vertical_ids = tf.range(h)[None, :, None]  # (1, h, 1)
    horizontal_ids = tf.range(w)[None, None, :]  # (1, 1, w)

    temporal_ids = tf.tile(temporal_ids, [1, h, w])  # (t, h, w)
    vertical_ids = tf.tile(vertical_ids, [t, 1, w])  # (t, h, w)
    horizontal_ids = tf.tile(horizontal_ids, [t, h, 1])  # (t, h, w)

    # (t, h, w, 3)
    pos_ids = tf.stack([temporal_ids, vertical_ids, horizontal_ids], axis=3)
    pos_ids = tf.reshape(pos_ids, [-1, 3])  # (t*h*w, 3)

    return pos_ids

  def _embedding_lookup(self,
                        lookup_fn,
                        lookup_keys,
                        ref_buckets,
                        target_buckets):
    """Checks whether interpolation is necessary, then calls appropriate fn."""

    if target_buckets == ref_buckets:
      position_embeddings = lookup_fn(lookup_keys,
                                      interpolate=False,
                                      max_target_buckets=None)
    else:
      position_embeddings = lookup_fn(lookup_keys,
                                      interpolate=True,
                                      max_target_buckets=target_buckets)

    return position_embeddings

  def call(self, inputs, dimensions, training=False):
    """Get token embeddings of inputs.

    Args:
        inputs: input embeddings
        dimensions: a list of dimensions
        training: train flag
    Returns:
        position_embeddings: output embedding tensor, float32 with
          shape [batch_size, length, embedding_size]
    """

    _, t, h, w, _ = dimensions
    pos_ids = self._build_vid_pos_ids(t, h, w)

    temporal_position_ids = pos_ids[None, :, 0]
    vertical_position_ids = pos_ids[None, :, 1]
    horizontal_position_ids = pos_ids[None, :, 2]

    temporal_position_embeddings = self._embedding_lookup(
        lookup_fn=self.temporal_position_embeddings,
        lookup_keys=temporal_position_ids,
        ref_buckets=self.max_temporal_positions,
        target_buckets=t,
        )

    vertical_position_embeddings = self._embedding_lookup(
        lookup_fn=self.vertical_position_embeddings,
        lookup_keys=vertical_position_ids,
        ref_buckets=self.max_vertical_positions,
        target_buckets=h,
        )

    horizontal_position_embeddings = self._embedding_lookup(
        lookup_fn=self.horizontal_position_embeddings,
        lookup_keys=horizontal_position_ids,
        ref_buckets=self.max_horizontal_positions,
        target_buckets=w,
        )

    position_embeddings = (
        temporal_position_embeddings +
        vertical_position_embeddings +
        horizontal_position_embeddings
        )

    position_embeddings = self.layernorm(position_embeddings)
    embeddings = inputs + position_embeddings
    embeddings = self.dropout(embeddings, training=training)

    return embeddings


class DenseReLUDense(tf.keras.layers.Layer):
  """Construct Dense+ReLU+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               use_bias,
               dropout_rate,
               name="dense_relu_dense"):
    super(DenseReLUDense, self).__init__(name=name)
    self.wi = tf.keras.layers.Dense(d_ff, use_bias=use_bias, name="wi")
    self.wo = tf.keras.layers.Dense(d_model, use_bias=use_bias, name="wo")
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
               use_bias,
               dropout_rate,
               name="dense_gelu_dense"):
    super(DenseGeLUDense, self).__init__(d_ff=d_ff,
                                         d_model=d_model,
                                         use_bias=use_bias,
                                         dropout_rate=dropout_rate,
                                         name=name)
    self.act = tf.keras.activations.gelu


class DenseSwishDense(DenseReLUDense):
  """Construct Dense+Swish+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               use_bias,
               dropout_rate,
               name="dense_swish_dense"):
    super(DenseSwishDense, self).__init__(d_ff=d_ff,
                                          d_model=d_model,
                                          use_bias=use_bias,
                                          dropout_rate=dropout_rate,
                                          name=name)
    self.act = tf.keras.activations.swish


class DenseGeGLUDense(tf.keras.layers.Layer):
  """Construct Dense+GeGLU+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               use_bias,
               dropout_rate,
               name="dense_geglu_dense"):
    super(DenseGeGLUDense, self).__init__(name=name)
    self.wi_0 = tf.keras.layers.Dense(d_ff, use_bias=use_bias, name="wi_0")
    self.wi_1 = tf.keras.layers.Dense(d_ff, use_bias=use_bias, name="wi_1")
    self.wo = tf.keras.layers.Dense(d_model, use_bias=use_bias, name="wo")
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
               pre_norm,
               use_bias,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               **kwargs):

    super().__init__(**kwargs)
    self.pre_norm = pre_norm
    if activation == "relu":
      self.mlp = DenseReLUDense(d_ff,
                                d_model,
                                use_bias,
                                dropout_rate,
                                name="dense_relu_dense")
    elif activation == "gelu":
      self.mlp = DenseGeLUDense(d_ff,
                                d_model,
                                use_bias,
                                dropout_rate,
                                name="dense_gelu_dense")
    elif activation == "swish":
      self.mlp = DenseSwishDense(d_ff,
                                 d_model,
                                 use_bias,
                                 dropout_rate,
                                 name="dense_swish_dense")
    elif activation == "geglu":
      self.mlp = DenseGeGLUDense(d_ff,
                                 d_model,
                                 use_bias,
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

    res_inputs = hidden_states
    if self.pre_norm:
      hidden_states = self.layer_norm(hidden_states)
    y = self.mlp(hidden_states, training=training)
    layer_output = res_inputs + self.dropout(y, training=training)
    if not self.pre_norm:
      layer_output = self.layer_norm(layer_output)
    return layer_output


class MultiHeadAttention(tf.keras.layers.Layer):
  """Construct the main MHA module used in Transformer layers."""

  def __init__(self,
               d_model,
               d_kv,
               num_heads,
               use_bias,
               dropout_rate,
               **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.d_kv = d_kv
    self.n_heads = num_heads
    self.inner_dim = self.n_heads * self.d_kv

    # query, key, and value mapping
    self.q = tf.keras.layers.Dense(self.inner_dim, use_bias=use_bias, name="q")
    self.k = tf.keras.layers.Dense(self.inner_dim, use_bias=use_bias, name="k")
    self.v = tf.keras.layers.Dense(self.inner_dim, use_bias=use_bias, name="v")
    self.o = tf.keras.layers.Dense(self.d_model, use_bias=use_bias, name="o")
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

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

  def call(self,
           query,
           key,
           value,
           mask=None,
           training=False):

    bs = get_shape(query)[0]

    q = self.q(query)  # (bs, qlen, inner_dim)
    k = self.k(key)  # (bs, klen, inner_dim)
    v = self.v(value)  # (bs, klen, inner_dim)

    q = self._split_heads(q, bs)  # (bs, n_heads, qlen, dim_per_head)
    k = self._split_heads(k, bs)  # (bs, n_heads, klen, dim_per_head)
    v = self._split_heads(v, bs)  # (bs, n_heads, vlen, dim_per_head)

    # (bs, n_heads, seq_len, seq_len)
    scores = tf.einsum("bnqd,bnkd->bnqk", q, k)

    # scale attention_scores
    dk = tf.cast(get_shape(k)[-1], dtype=scores.dtype)
    scores = scores / tf.math.sqrt(dk)

    if mask is not None:
      scores += mask

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
        "attention_weights": attention_weights,
    }

    return outputs


class TransformerEncoderLayer(tf.keras.layers.Layer):
  """Construct the main Transformer module which includes MHA + FeedForward."""

  def __init__(self,
               d_model,
               d_kv,
               d_ff,
               num_heads,
               pre_norm,
               use_bias,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               name="encoder_layer",
               **kwargs):
    super(TransformerEncoderLayer, self).__init__(name=name)

    self.pre_norm = pre_norm
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=layer_norm_epsilon,
        name="layer_norm"
        )

    self.mha = MultiHeadAttention(
        d_model=d_model,
        d_kv=d_kv,
        use_bias=use_bias,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        name="multi_head_attention"
        )

    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    self.feed_forward = FeedForward(d_ff=d_ff,
                                    d_model=d_model,
                                    pre_norm=pre_norm,
                                    use_bias=use_bias,
                                    activation=activation,
                                    dropout_rate=dropout_rate,
                                    layer_norm_epsilon=layer_norm_epsilon,
                                    name="feed_forward")

  def call(self,
           inputs,
           attention_mask=None,
           training=False):

    res_inputs = inputs

    # apply layer_norm on inputs if pre_norm
    if self.pre_norm:
      inputs = self.layer_norm(inputs)

    # apply multi-head attention module
    attention_outputs = self.mha(
        query=inputs,
        key=inputs,
        value=inputs,
        mask=attention_mask,
        training=training
        )

    hidden_states = attention_outputs["hidden_states"]

    # apply residual + dropout
    hidden_states = res_inputs + self.dropout(hidden_states, training=training)

    # apply layer_norm if not pre_norm
    if not self.pre_norm:
      hidden_states = self.layer_norm(hidden_states)

    # apply Feed Forward layer
    hidden_states = self.feed_forward(hidden_states, training=training)

    # update hidden states
    attention_outputs["hidden_states"] = hidden_states

    return attention_outputs


class TransformerEncoder(tf.keras.layers.Layer):
  """Construct the final Transformer stack."""

  def __init__(self,
               d_model,
               d_kv,
               d_ff,
               num_layers,
               num_heads,
               pre_norm,
               use_bias,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               name="transformer_encoder",
               **kwargs):
    super(TransformerEncoder, self).__init__(name=name)

    self.d_model = d_model
    self.pre_norm = pre_norm
    self.num_hidden_layers = num_layers

    self.layers = []
    for n in range(self.num_hidden_layers):
      self.layers.append(
          TransformerEncoderLayer(d_model=d_model,
                                  d_kv=d_kv,
                                  d_ff=d_ff,
                                  num_heads=num_heads,
                                  pre_norm=pre_norm,
                                  use_bias=use_bias,
                                  activation=activation,
                                  dropout_rate=dropout_rate,
                                  layer_norm_epsilon=layer_norm_epsilon,
                                  name="layer_{}".format(n))
          )

    if self.pre_norm:
      self.final_layer_norm = tf.keras.layers.LayerNormalization(
          epsilon=layer_norm_epsilon,
          name="final_layer_norm"
          )
      self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self,
           inputs,
           attention_mask=None,
           training=False):

    input_shape = get_shape(inputs)[:-1]
    batch_size, seq_length = input_shape

    if attention_mask is None:
      attention_mask = tf.fill((batch_size, seq_length), 1)

    # Provided a padding mask of dimensions [batch_size, seq_length]
    # make the mask broadcastable to
    #         [batch_size, num_heads, seq_length, seq_length]
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)
    attention_mask = attention_mask[:, None, None, :]
    attention_mask = (1.0 - attention_mask) * -1e9

    all_hidden_states = ()
    all_attentions = ()

    hidden_states = inputs

    for layer in self.layers:
      # temporary --- to reduce memory consumption
      # all_hidden_states = all_hidden_states + (hidden_states,)

      layer_outputs = layer(inputs=hidden_states,
                            attention_mask=attention_mask,
                            training=training)

      # layer_outputs is a dictionary with the following keys:
      # hidden_states, self_attention_weights
      hidden_states = layer_outputs["hidden_states"]
      all_attentions = all_attentions + (layer_outputs["attention_weights"],)

    if self.pre_norm:
      hidden_states = self.final_layer_norm(hidden_states)
      hidden_states = self.dropout(hidden_states, training=training)

    # Add last layer
    all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = {
        "hidden_states": all_hidden_states,
        "attention_weights": all_attentions,
    }

    return outputs
