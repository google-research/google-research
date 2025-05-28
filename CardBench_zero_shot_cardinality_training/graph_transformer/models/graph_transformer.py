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

"""Graph Transformer model implementation."""

from typing import Mapping, Optional, Tuple
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-Head Attention implementation from the original transformer paper."""

  def __init__(
      self,
      num_heads,
      input_dim,
      num_nodes,
      atten_dropout_prob = 0.0,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.input_dim = input_dim
    self.num_nodes = num_nodes
    if self.input_dim % self.num_heads != 0:
      raise ValueError(
          f'{self.input_dim=} must be a multiple of {self.num_heads=}'
      )
    self.atten_dropout_prob = atten_dropout_prob
    self.depth = self.input_dim // self.num_heads

    self.wq = tf.keras.layers.Dense(self.input_dim)
    self.wk = tf.keras.layers.Dense(self.input_dim)
    self.wv = tf.keras.layers.Dense(self.input_dim)
    self.dense = tf.keras.layers.Dense(self.input_dim)
    self.atten_dropout = tf.keras.layers.Dropout(self.atten_dropout_prob)

  def _compute_attention_mask(
      self, attention_mask, paddings
  ):
    """Convert padding mask from [B, S] to [B, H, S, S]."""

    padding_mask = 1 - tf.expand_dims(paddings, axis=1)
    padding_mask = tf.repeat(padding_mask, repeats=self.num_heads, axis=1)
    padding_mask = tf.einsum(
        'BHTI,BHIS->BHTS',
        tf.expand_dims(padding_mask, axis=3),
        tf.expand_dims(padding_mask, axis=2),
    )

    if attention_mask is not None:
      # Convert attention mask from [B, S, S] to [B, H, S, S].
      attention_mask = tf.expand_dims(attention_mask, axis=1)
      attention_mask = tf.repeat(attention_mask, repeats=self.num_heads, axis=1)
      return tf.multiply(attention_mask, padding_mask)
    else:
      return padding_mask

  def _split_heads(self, x):
    """Split and transpose a (B, S, HD) tensor into one with shape (B, H, S, D).

    Split the last dimension into (num_heads, depth), then transpose the inner
    two dimensions, returning a tensor with shape ((b)atch_size, num_(h)eads,
    (s)eq_len, (d)epth).

    Args:
      x: [B, S, HD]

    Returns:
      Tensor split by heads and transposed [B, H, S, D]
    """
    x = tf.reshape(x, [-1, self.num_nodes, self.num_heads, self.depth])
    return tf.transpose(x, [0, 2, 1, 3])

  def _compute_attention(
      self,
      query,
      key,
      value,
      attention_bias,
      attention_mask,
      training,
  ):
    _, _, _, d = key.shape

    logits = tf.einsum('BHTD,BHSD->BHTS', query, key)
    if attention_bias is not None:
      scaled_attention_logits = (
          logits / tf.sqrt(tf.constant(d, dtype=tf.float32)) + attention_bias
      )
    else:
      scaled_attention_logits = logits / tf.sqrt(
          tf.constant(d, dtype=tf.float32)
      )
    if attention_mask is not None:
      scaled_attention_logits += tf.scalar_mul(-1.0e9, 1 - attention_mask)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    attention_weights = self.atten_dropout(attention_weights, training)
    output = tf.einsum('BNTS,BNSH->BNTH', attention_weights, value)
    return output, attention_weights

  def call(
      self,
      query,
      key,
      value,
      attention_bias,
      attention_mask = None,
      paddings = None,
      training = None,
  ):
    """Multi-head attention implementation.

    Args:
      query: Query `Tensor` of shape `(B, T, dim)`.
      key: Optional key `Tensor` of shape `(B, S, dim)`
      value: Value `Tensor` of shape `(B, S, dim)`.
      attention_bias: [B, H, S, S]
      attention_mask: [B, S, S]
      paddings: [B, S]
      training: Whether is training or not.

    Returns:
      output: [B, S, D]
      attention_weights: [B, H, S, S]
    """

    attention_mask = self._compute_attention_mask(attention_mask, paddings)

    query = self._split_heads(self.wq(query))
    key = self._split_heads(self.wk(key))
    value = self._split_heads(self.wv(value))

    scaled_attention, attention_weights = self._compute_attention(
        query, key, value, attention_bias, attention_mask, training
    )

    concat_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
    concat_attention = tf.reshape(
        concat_attention, [-1, self.num_nodes, self.input_dim]
    )

    output = self.dense(concat_attention)
    return output, attention_weights


class TransformerFeedForwardLayer(tf.keras.layers.Layer):
  """Transformer feed forward layer implementation."""

  input_dim: int = 0
  hidden_dim: int = 0
  residual_dropout_prob: float = 0.0
  relu_dropout_prob: float = 0.0

  def __init__(
      self,
      input_dim,
      hidden_dim,
      residual_dropout_prob = 0.0,
      relu_dropout_prob = 0.0,
  ):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.residual_dropout_prob = residual_dropout_prob
    self.relu_dropout_prob = relu_dropout_prob

    self.layer_norm = tf.keras.layers.LayerNormalization()
    self.ffn_layer1 = tf.keras.layers.Dense(
        self.hidden_dim, activation='gelu', name='ffn_layer1'
    )
    self.relu_dropout = tf.keras.layers.Dropout(self.relu_dropout_prob)
    self.ffn_layer2 = tf.keras.layers.Dense(self.input_dim, name='ffn_layer2')
    self.residual_dropout = tf.keras.layers.Dropout(self.residual_dropout_prob)

  def call(
      self,
      x,
      paddings = None,
      training = None,
  ):
    """Computes transformer feedforward layer outputs given inputs and paddings.

    Args:
      x: tf.Tensor of shape [B, S, D].
      paddings: tf.Tensor of shape [B, S]. A binary tensor where 1 indicates a
        padded position (should be skipped).
      training: Whether is training or not.

    Returns:
      tf.Tensor of shape [B, S, D].
    """

    inputs_normalized = self.layer_norm(x)
    projected_inputs = self.ffn_layer1(inputs_normalized)
    projected_inputs = self.relu_dropout(projected_inputs, training)
    projected_inputs = self.ffn_layer2(projected_inputs)
    residual = self.residual_dropout(projected_inputs, training)
    out = x + residual
    if paddings is not None:
      out *= 1.0 - tf.expand_dims(paddings, axis=-1)

    return out


class TransformerLayer(tf.keras.layers.Layer):
  """Single transformer layer implementation."""

  model_dim: int = 0
  hidden_dim: int = 0
  num_heads: int = 0
  atten_dropout_prob: float = 0.0
  residual_dropout_prob: float = 0.0
  relu_dropout_prob: float = 0.0

  def __init__(
      self,
      model_dim,
      hidden_dim,
      num_heads,
      num_nodes,
      atten_dropout_prob = 0.0,
      residual_dropout_prob = 0.0,
      relu_dropout_prob = 0.0,
  ):
    super().__init__()
    self.model_dim = model_dim
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.atten_dropout_prob = atten_dropout_prob
    self.residual_dropout_prob = residual_dropout_prob
    self.relu_dropout_prob = relu_dropout_prob
    self.num_nodes = num_nodes

    self.atten_ln = tf.keras.layers.LayerNormalization()
    self.self_attention = MultiHeadAttention(
        input_dim=model_dim,
        num_heads=num_heads,
        num_nodes=num_nodes,
    )
    self.residual_dropout = tf.keras.layers.Dropout(residual_dropout_prob)
    self.ffn_layer = TransformerFeedForwardLayer(
        input_dim=model_dim,
        hidden_dim=hidden_dim,
        residual_dropout_prob=residual_dropout_prob,
        relu_dropout_prob=relu_dropout_prob,
    )

  def call(
      self,
      inputs,
      paddings,
      attention_bias,
      attention_mask,
      training,
  ):
    # Layer normalize input
    inputs_normalized = self.atten_ln(inputs)

    # Compute self-attention, query/key/value vectors are the input itself
    atten_output, _ = self.self_attention(
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        attention_mask=attention_mask,
        attention_bias=attention_bias,
        paddings=paddings,
    )

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output, training)
    atten_output += inputs

    # Apply FFN layer
    output = self.ffn_layer(atten_output, paddings=paddings, training=training)
    return output


class MLPLayer(tf.keras.layers.Layer):
  """Implementation of a simple MLP layer."""

  def __init__(self, num_nodes, model_dim, num_layer):
    super().__init__()
    self.model_dim = model_dim
    self.num_layer = num_layer

    self.dense_layers = tf.keras.Sequential()
    for _ in range(self.num_layer - 1):
      self.dense_layers.add(
          tf.keras.layers.Dense(
              model_dim, activation='gelu', bias_initializer='glorot_uniform'
          )
      )
    self.dense_layers.add(
        tf.keras.layers.Dense(model_dim, bias_initializer='glorot_uniform')
    )

  def call(self, inputs):
    """inputs: [B, S, feature_dim], outputs: [B, S, model_dim]."""

    _, s, feature_dim = inputs.shape
    x = tf.reshape(inputs, [-1, feature_dim])
    x = self.dense_layers(x)
    x = tf.reshape(x, [-1, s, self.model_dim])
    return x


class MultiplexNodeFeatureEncoder(tf.keras.layers.Layer):
  """Select node feature based on node type."""

  def __init__(
      self,
      model_dim,
      num_embedding_layer,
      num_nodes,
      num_node_types,
      feature_dim,
  ):
    super().__init__()
    self.node_feature_embedding_layer = MLPLayer(
        model_dim=model_dim * num_node_types,
        num_layer=num_embedding_layer,
        num_nodes=num_nodes,
    )

    self.num_node_types = num_node_types
    self.feature_dim = feature_dim
    self.model_dim = model_dim
    self.num_nodes = num_nodes
    self.node_type_indices = tf.range(self.num_node_types, dtype=tf.int32)
    self.node_feature_indices = tf.range(
        self.num_node_types, self.feature_dim, dtype=tf.int32
    )

  def call(self, node_feature):
    node_type_onehot = tf.gather(node_feature, self.node_type_indices, axis=-1)
    node_feature_tensor = tf.gather(
        node_feature, self.node_feature_indices, axis=-1
    )

    # [B, S, feature_dim] -> [B, S, model_dim * num_node_types]
    node_feature_embedding = self.node_feature_embedding_layer(
        node_feature_tensor
    )

    node_feature_embedding = tf.reshape(
        node_feature_embedding,
        [-1, self.num_nodes, self.num_node_types, self.model_dim],
    )

    node_type_mux = tf.argmax(node_type_onehot, axis=-1)
    node_feature_embedding = tf.gather(
        node_feature_embedding, node_type_mux, axis=-2, batch_dims=2
    )
    node_feature_embedding = tf.reshape(
        node_feature_embedding, [-1, self.num_nodes, self.model_dim]
    )
    return node_feature_embedding


class GraphTransformerEncoder(tf.keras.layers.Layer):
  """GraphTransformerEncoder implementation.

  node_feature: [B, S, feature_dim]
  edge_encoding: [B, S, S]
  spatial_encoding: [B, S, S]
  """

  num_encoder_layer: int = 0
  num_embedding_layer: int = 0
  model_dim: int = 0
  num_heads: int = 0
  hidden_dim: int = 0
  num_nodes: int = 0
  node_feature_dim: int = 0
  dropout_prob: float = 0.0
  num_edge_types: int = 0

  def __init__(
      self,
      num_encoder_layer,
      num_embedding_layer,
      num_output_layer,
      model_dim,
      num_heads,
      hidden_dim,
      output_dim,
      num_nodes,
      node_feature_dim,
      num_node_types,
      num_edge_types,
      dropout_prob=0.0,
  ):
    super().__init__()
    self.num_encoder_layer = num_encoder_layer
    self.model_dim = model_dim
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.num_nodes = num_nodes
    self.node_feature_dim = node_feature_dim
    self.dropout_prob = dropout_prob
    self.num_edge_types = num_edge_types
    self.num_embedding_layer = num_embedding_layer

    self.node_feature_embedding_layer = MultiplexNodeFeatureEncoder(
        model_dim,
        num_embedding_layer,
        num_nodes,
        num_node_types,
        node_feature_dim,
    )

    self.spatial_pos_encoder = tf.keras.layers.Embedding(
        input_dim=num_nodes + 1,
        output_dim=num_heads,
        name='spatial_pos_encoder',
    )

    self.encoder_dropout = tf.keras.layers.Dropout(rate=self.dropout_prob)
    self.embedding_dropout = tf.keras.layers.Dropout(rate=self.dropout_prob)

    self.encoder_layers = []
    for _ in range(self.num_encoder_layer):
      self.encoder_layers.append(
          TransformerLayer(
              model_dim=model_dim,
              hidden_dim=hidden_dim,
              num_heads=num_heads,
              num_nodes=num_nodes,
              atten_dropout_prob=dropout_prob,
              residual_dropout_prob=dropout_prob,
              relu_dropout_prob=dropout_prob,
          )
      )

    self.encoder_ln = tf.keras.layers.LayerNormalization()

  def _encoder_fprop(
      self,
      node_feature,
      causal_mask,
      spatial_encoding,
      paddings,
      training,
  ):
    """Forward pass of the encoder.

    Args:
      node_feature: [B, S, node_feature_dim]
      causal_mask: [B, S, S]
      spatial_encoding: [B, S, S]
      paddings: [B, S]
      training: Whether is training or not.

    Returns:
      encoder_out: [B, S, model_dim]
    """

    node_feature_embedding = self.node_feature_embedding_layer(
        node_feature
    )  # [b, s, d]

    x = node_feature_embedding
    x = self.embedding_dropout(x)

    # attention_mask = 1 - paddings  # [B, S]
    attention_mask = causal_mask

    # Attention bias
    spatial_embedding = self.spatial_pos_encoder(
        spatial_encoding
    )  # [b, s, s, h]

    attention_bias = spatial_embedding  # [B, S, S, H]

    # transpose to [B, H, S, S]
    # transpose dim 2 and 1 for the transformer setting
    attention_bias = tf.transpose(attention_bias, [0, 3, 2, 1])

    # Compute num_encoder_layer stacks of self attention + ffn
    for i in range(self.num_encoder_layer):
      x = self.encoder_layers[i](
          x, paddings, attention_bias, attention_mask, training
      )

    # Final layer norm
    x = self.encoder_ln(x)

    return x

  def call(
      self,
      node_feature,
      causal_mask,
      spatial_encoding,
      paddings,
      training,
  ):
    # [B, S, D]
    encoder_out = self._encoder_fprop(
        node_feature,
        causal_mask,
        spatial_encoding,
        paddings,
        training=training,
    )

    return encoder_out


class Predictor(tf.keras.Model):
  """Regression model based on GraphTransformerEncoder."""

  def __init__(
      self,
      num_encoder_layer,
      num_embedding_layer,
      num_output_layer,
      model_dim,
      num_heads,
      hidden_dim,
      output_dim,
      num_nodes,
      node_feature_dim,
      num_node_types,
      num_edge_types,
      mask_type,
      dropout_prob = 0.0,
  ):
    super().__init__()
    self.num_encoder_layer = num_encoder_layer
    self.num_embedding_layer = num_embedding_layer
    self.num_output_layer = num_output_layer
    self.model_dim = model_dim
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.num_nodes = num_nodes
    self.node_feature_dim = node_feature_dim
    self.num_edge_types = num_edge_types
    self.dropout_prob = dropout_prob
    self.mask_type = mask_type

    self.encoder = GraphTransformerEncoder(
        num_encoder_layer,
        num_embedding_layer,
        num_output_layer,
        model_dim,
        num_heads,
        hidden_dim,
        output_dim,
        num_nodes,
        node_feature_dim,
        num_node_types,
        num_edge_types,
        dropout_prob,
    )

    self.output_mlp = tf.keras.Sequential()
    self.output_ln = tf.keras.layers.LayerNormalization()
    for _ in range(num_output_layer - 1):
      self.output_mlp.add(tf.keras.layers.Dense(model_dim, activation='gelu'))
      self.output_mlp.add(tf.keras.layers.Dropout(rate=dropout_prob))

    self.output_mlp.add(tf.keras.layers.Dense(output_dim))

  def call(
      self,
      inputs,
      training = None,
      mask = None,
  ):
    node_feature = inputs['node']
    spatial_encoding = inputs['spatial_encoding']
    paddings = inputs['node_padding']
    parent_causal_mask = inputs['parent_causal_mask']
    ancestor_causal_mask = inputs['ancestor_causal_mask']

    if self.mask_type == 'parent_causal_mask':
      encoder_out = self.encoder(
          node_feature,
          parent_causal_mask,
          spatial_encoding,
          paddings,
          training,
      )
    elif self.mask_type == 'ancestor_causal_mask':
      encoder_out = self.encoder(
          node_feature,
          ancestor_causal_mask,
          spatial_encoding,
          paddings,
          training,
      )
    else:
      raise NotImplementedError('Other causal masks are not implemented')

    # Readout [VNODE] as the graph embedding [B, S, D] -> [B, D]
    v_node_embedding = tf.gather(encoder_out, indices=0, axis=1)

    v_node_embedding = self.output_ln(v_node_embedding)
    y = self.output_mlp(v_node_embedding)

    return y
