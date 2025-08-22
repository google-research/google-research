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

"""A Temporal Fusion Transformer (TFT) implementation for time series.

TFT is an attention-based architecture which combines high-performance
multi-horizon forecasting with interpretable insights into temporal dynamics.
Please see https://arxiv.org/pdf/1912.09363.pdf for details.

The code is adapted from:
https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
"""

import tensorflow as tf


def _dense_layer(size, activation=None, time_distributed=False, use_bias=True):
  """Returns a dense keras layer with activation.

  Args:
    size: The output size.
    activation: The activation to be applied to the linear layer output.
    time_distributed: If True, it applies the dense layer for every temporal
      slice of an input.
    use_bias: If True, it includes the bias to the dense layer.
  """
  dense = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
  if time_distributed:
    dense = tf.keras.layers.TimeDistributed(dense)
  return dense


def _apply_gating_layer(x,
                        hidden_layer_size,
                        dropout_rate=None,
                        time_distributed=True,
                        activation=None):
  """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: The input to gating layer.
    hidden_layer_size: The hidden layer size of GLU.
    dropout_rate: The dropout rate to be applied to the input.
    time_distributed: If True, it applies the dense layer for every temporal
      slice of an input.
    activation: The activation to be applied for the linear layer.

  Returns:
    Tuple of tensors for: (GLU output, gate).
  """

  if dropout_rate is not None:
    x = tf.keras.layers.Dropout(dropout_rate)(x)

  activation_layer = _dense_layer(hidden_layer_size, activation,
                                  time_distributed)(
                                      x)

  gated_layer = _dense_layer(hidden_layer_size, 'sigmoid', time_distributed)(x)

  return tf.keras.layers.Multiply()([activation_layer,
                                     gated_layer]), gated_layer


def _add_and_norm(x):
  """Applies skip connection followed by layer normalisation.

  Args:
    x: The list of inputs to sum for skip connection.

  Returns:
    A tf.tensor output from the skip and layer normalization layer.
  """
  return tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()(x))


def _gated_residual_network(x,
                            hidden_layer_size,
                            output_size=None,
                            dropout_rate=None,
                            time_distributed=True,
                            additional_context=None,
                            return_gate=False):
  """Applies the gated residual network (GRN) as defined in paper.

  Args:
    x: The input to the GRN.
    hidden_layer_size: The hidden layer size of GRN.
    output_size: The output layer size.
    dropout_rate: The dropout rate to be applied to the input.
    time_distributed: If True, it makes output layer apply for every temporal
      slice of an input.
    additional_context: The additional context vector to use if exists.
    return_gate: If True, the function returns GLU gate for diagnostic purposes.
      Otherwise, only the GRN output is returned.

  Returns:
    A tuple of tensors for (GRN output, GLU gate) when return_gate is True. If
    return_gate is False, it returns tf.Tensor of GRN output.
  """

  # Setup skip connection
  if output_size is None:
    output_size = hidden_layer_size
    skip = x
  else:
    skip = _dense_layer(output_size, None, time_distributed)(x)

  # Apply feedforward network
  hidden = _dense_layer(hidden_layer_size, None, time_distributed)(x)
  if additional_context is not None:
    context_layer = _dense_layer(
        hidden_layer_size,
        activation=None,
        time_distributed=time_distributed,
        use_bias=False)
    hidden = hidden + context_layer(additional_context)
  hidden = tf.keras.layers.Activation('elu')(hidden)
  hidden_layer = _dense_layer(
      hidden_layer_size, activation=None, time_distributed=time_distributed)

  hidden = hidden_layer(hidden)

  gating_layer, gate = _apply_gating_layer(
      hidden,
      output_size,
      dropout_rate=dropout_rate,
      time_distributed=time_distributed,
      activation=None)

  if return_gate:
    return _add_and_norm([skip, gating_layer]), gate
  else:
    return _add_and_norm([skip, gating_layer])


def _get_decoder_mask(self_attn_inputs, len_s):
  """Returns causal mask to apply for self-attention layer.

  Args:
    self_attn_inputs: The inputs to self attention layer to determine mask
      shape.
    len_s: Total length of the encoder and decoder sequences.

  Returns:
    A tf.tensor of causal mask to apply for the attention layer.
  """
  bs = tf.shape(self_attn_inputs)[:1]
  mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), 1)
  return mask


class ScaledDotProductAttention(object):
  """Defines scaled dot product attention layer.

  Attributes:
    attn_dropout_layer: The dropout layer for the attention output.
    activation: The activation for the scaled dot product attention. By default,
      it is set to softmax.
  """

  def __init__(self, activation='softmax'):
    self.activation = tf.keras.layers.Activation(activation)

  def __call__(self, q, k, v, mask):
    """Applies scaled dot product attention with softmax normalization.

    Args:
      q: The queries to the attention layer.
      k: The keys to the attention layer.
      v: The values to the attention layer.
      mask: The mask applied to the input to the softmax.

    Returns:
      A Tuple of layer outputs and attention weights.
    """
    normalization_constant = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
    q = tf.transpose(q, [1, 0, 2])
    k = tf.transpose(k, [1, 0, 2])
    v = tf.transpose(v, [1, 0, 2])
    mask = tf.transpose(mask, [1, 0, 2])

    attn = tf.einsum('fbd,tbd->fbt', q, k) / normalization_constant

    attn -= tf.reduce_max(
        attn + tf.math.log(mask + 1e-9), axis=2, keepdims=True)
    attn = mask * tf.exp(attn)
    attn = tf.math.divide(
        attn,
        tf.reduce_sum(attn, axis=2, keepdims=True) + 1e-9,
    )

    output = tf.einsum('fbt,tbd->fbd', attn, v)
    output = tf.transpose(output, [1, 0, 2])

    return output, attn


class InterpretableMultiHeadAttention(object):
  """Defines interpretable multi-head attention layer.

  Attributes:
    n_head: The number of heads for attention layer.
    d_k: The key and query dimensionality per head.
    d_v: The value dimensionality.
    dropout: The dropout rate to apply
    qs_layers: The list of query layers across heads.
    ks_layers: The list of key layers across heads.
    vs_layers: The list of value layers across heads.
    attention: The scaled dot product attention layer associated with the
      output.
    w_o: The output weight matrix to project internal state to the original TFT
      state size.
  """

  def __init__(self, n_head, d_model, dropout):
    """Initialises layer.

    Args:
      n_head: The number of heads.
      d_model: The dimensionality of TFT state.
      dropout: The dropout rate to be applied to the output.
    """
    self.n_head = n_head
    self.d_k = self.d_v = d_k = d_v = d_model // n_head
    self.dropout = dropout

    # Use same value layer to facilitate interp
    vs_layer = tf.keras.layers.Dense(d_v, use_bias=False)
    self.qs_layers = [_dense_layer(d_k, use_bias=False) for _ in range(n_head)]
    self.ks_layers = [_dense_layer(d_k, use_bias=False) for _ in range(n_head)]
    self.vs_layers = [vs_layer for _ in range(n_head)]

    self.attention = ScaledDotProductAttention()
    self.w_o = tf.keras.layers.Dense(d_model, use_bias=False)

  def __call__(self, q, k, v, mask=None):
    """Applies interpretable multihead attention.

    Using T to denote the number of time steps fed into the transformer.

    Args:
      q: The query of tf.tensor with shape=(?, T, d_model).
      k: The key of tf.tensor with shape=(?, T, d_model).
      v: The value of tf.tensor with shape=(?, T, d_model).
      mask: The optional mask of tf.tensor with shape=(?, T, T). If None,
        masking is not applied for the output.

    Returns:
      A Tuple of (layer outputs, attention weights).
    """
    n_head = self.n_head

    heads = []
    attns = []
    for i in range(n_head):
      qs = self.qs_layers[i](q)
      ks = self.ks_layers[i](k)
      vs = self.vs_layers[i](v)
      head, attn = self.attention(qs, ks, vs, mask)

      head_dropout = tf.keras.layers.Dropout(self.dropout)(head)
      heads.append(head_dropout)
      attns.append(attn)
    head = tf.stack(heads) if n_head > 1 else heads[0]
    attn = tf.stack(attns)

    outputs = tf.reduce_mean(head, axis=0) if n_head > 1 else head
    outputs = self.w_o(outputs)
    outputs = tf.keras.layers.Dropout(self.dropout)(outputs)

    return outputs, attn


# TFT model definitions.
class TFTModel(object):
  """Implements Temporal Fusion Transformer."""

  def __init__(self, hparams, quantile_targets=None):
    """Initializes TFT model."""

    if quantile_targets is None:
      quantile_targets = [0.5]

    # Consider point forecasting
    self.output_size = len(quantile_targets)

    self.use_cudnn = False
    self.hidden_layer_size = hparams['num_units']
    self.forecast_horizon = hparams['forecast_horizon']
    self.keep_prob = hparams['keep_prob']

    self.num_encode = hparams['num_encode']
    self.num_heads = hparams['num_heads']
    self.num_historical_features = hparams['num_historical_features']
    self.num_future_features = hparams['num_future_features']
    self.num_static_features = hparams['num_static_features']

  def _build_base_graph(self,
                        historical_inputs,
                        future_inputs,
                        static_inputs,
                        training=True):
    """Returns graph defining layers of the TFT."""

    if training:
      self.dropout_rate = 1.0 - self.keep_prob
    else:
      self.dropout_rate = 0.0

    def _static_combine_and_mask(embedding):
      """Applies variable selection network to static inputs.

      Args:
        embedding: Transformed static inputs.

      Returns:
        A tf.tensor for variable selection network.
      """

      mlp_outputs = _gated_residual_network(
          embedding,
          self.hidden_layer_size,
          output_size=self.num_static_features,
          dropout_rate=self.dropout_rate,
          time_distributed=False,
          additional_context=None)

      sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
      sparse_weights = tf.expand_dims(sparse_weights, axis=-1)

      trans_emb_list = []
      for i in range(self.num_static_features):
        e = _gated_residual_network(
            tf.expand_dims(embedding[:, i:i + 1], axis=-1),
            self.hidden_layer_size,
            output_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            time_distributed=False)
        trans_emb_list.append(e)

      transformed_embedding = tf.concat(trans_emb_list, axis=1)

      combined = sparse_weights * transformed_embedding

      static_vec = tf.reduce_sum(combined, axis=1)

      return static_vec, sparse_weights

    def _lstm_combine_and_mask(embedding, static_context_variable_selection,
                               num_features):
      """Applies temporal variable selection networks.

      Args:
        embedding: The inputs for temporal variable selection networks.
        static_context_variable_selection: The static context variable
          selection.
        num_features: Number of features.

      Returns:
        A Tuple of tensors that consts of temporal context, sparse weight, and
        static gate.
      """

      expanded_static_context = tf.expand_dims(
          static_context_variable_selection, axis=1)

      # Variable selection weights
      mlp_outputs, static_gate = _gated_residual_network(
          embedding,
          self.hidden_layer_size,
          output_size=num_features,
          dropout_rate=self.dropout_rate,
          time_distributed=True,
          additional_context=expanded_static_context,
          return_gate=True)

      sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)

      sparse_weights = tf.expand_dims(sparse_weights, axis=2)

      trans_emb_list = []
      for i in range(num_features):
        grn_output = _gated_residual_network(
            embedding[:, :, i:i + 1],
            self.hidden_layer_size,
            output_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            time_distributed=True)
        trans_emb_list.append(grn_output)

      transformed_embedding = tf.stack(trans_emb_list, axis=-1)
      combined = tf.keras.layers.Multiply()(
          [sparse_weights, transformed_embedding])
      temporal_context = tf.reduce_sum(combined, axis=-1)

      return temporal_context, sparse_weights, static_gate

    # LSTM layer
    def _get_lstm(return_state):
      """Returns LSTM cell initialized with default parameters.

      This function builds CuDNNLSTM or LSTM depending on the self.use_cudnn.

      Args:
        return_state: If True, the output LSTM layer returns output and state
          when called. Otherwise, only the output is returned when called.

      Returns:
        A tf.Tensor for LSTM layer.
      """
      if self.use_cudnn:
        lstm = tf.keras.layers.CuDNNLSTM(
            self.hidden_layer_size,
            return_sequences=True,
            return_state=return_state,
            stateful=False,
        )
      else:
        lstm = tf.keras.layers.LSTM(
            self.hidden_layer_size,
            return_sequences=True,
            return_state=return_state,
            stateful=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True)
      return lstm

    static_encoder, _ = _static_combine_and_mask(static_inputs)

    def _create_static_context():
      """Builds static contexts with the same structure and the same input."""
      return _gated_residual_network(
          static_encoder,
          self.hidden_layer_size,
          output_size=self.hidden_layer_size,
          dropout_rate=self.dropout_rate,
          time_distributed=False)

    static_context_variable_selection = _create_static_context()
    static_context_enrichment = _create_static_context()
    static_context_state_h = _create_static_context()
    static_context_state_c = _create_static_context()

    historical_features, _, _ = _lstm_combine_and_mask(
        historical_inputs, static_context_variable_selection,
        self.num_historical_features)
    future_features, _, _ = _lstm_combine_and_mask(
        future_inputs, static_context_variable_selection,
        self.num_future_features)

    history_lstm, state_h, state_c = _get_lstm(return_state=True)(
        historical_features,
        initial_state=[static_context_state_h, static_context_state_c])
    future_lstm = _get_lstm(return_state=False)(
        future_features, initial_state=[state_h, state_c])

    lstm_layer = tf.concat([history_lstm, future_lstm], axis=1)

    # Apply gated skip connection
    input_embeddings = tf.concat([historical_features, future_features], axis=1)

    lstm_layer, _ = _apply_gating_layer(
        lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
    temporal_feature_layer = _add_and_norm([lstm_layer, input_embeddings])

    # Static enrichment layers
    expanded_static_context = tf.expand_dims(static_context_enrichment, axis=1)
    enriched, _ = _gated_residual_network(
        temporal_feature_layer,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        time_distributed=True,
        additional_context=expanded_static_context,
        return_gate=True)

    # Decoder self attention
    self_attn_layer = InterpretableMultiHeadAttention(
        self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate)
    mask = _get_decoder_mask(enriched, self.num_encode + self.forecast_horizon)
    x, _ = self_attn_layer(enriched, enriched, enriched, mask=mask)
    x, _ = _apply_gating_layer(
        x,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        activation=None)
    x = _add_and_norm([x, enriched])

    # Nonlinear processing on outputs
    decoder = _gated_residual_network(
        x,
        self.hidden_layer_size,
        dropout_rate=self.dropout_rate,
        time_distributed=True)

    # Final skip connection
    decoder, _ = _apply_gating_layer(
        decoder, self.hidden_layer_size, activation=None)
    transformer_layer = _add_and_norm([decoder, temporal_feature_layer])

    return transformer_layer

  def return_baseline_model(self):
    """Returns the Keras model object for the TFT graph."""

    # Define the input features.
    past_features = tf.keras.Input(
        shape=(
            self.num_encode,
            self.num_historical_features,
        ))
    future_features = tf.keras.Input(
        shape=(
            self.forecast_horizon,
            self.num_future_features,
        ))
    static_features = tf.keras.Input(shape=(self.num_static_features,))

    transformer_layer = self._build_base_graph(past_features, future_features,
                                               static_features)

    # Get the future predictions from encoded attention representations.
    predictions = _dense_layer(
        self.output_size, time_distributed=True)(
            transformer_layer[:, -self.forecast_horizon:, :])

    # Define the Keras model.
    tft_model = tf.keras.Model(
        inputs=[past_features, future_features, static_features],
        outputs=predictions,
    )

    return tft_model

  def return_self_adapting_model(self):
    """Returns the Keras model object for the TFT graph."""

    # Define the input features.
    past_features = tf.keras.Input(
        shape=(
            self.num_encode,
            self.num_historical_features,
        ))
    future_features = tf.keras.Input(
        shape=(
            self.forecast_horizon,
            self.num_future_features,
        ))
    static_features = tf.keras.Input(shape=(self.num_static_features,))

    transformer_layer = self._build_base_graph(past_features, future_features,
                                               static_features)

    # Get the future predictions from encoded attention representations.
    predictions = _dense_layer(
        self.output_size, time_distributed=True)(
            transformer_layer[:, -self.forecast_horizon:, :])

    # Get the backcasts from encoded attention representations.
    backcasts = _dense_layer(
        self.num_historical_features, time_distributed=True)(
            transformer_layer[:, :self.num_encode, :])

    # Define the Keras model.
    tft_model = tf.keras.Model(
        inputs=[past_features, future_features, static_features],
        outputs=[backcasts, predictions])

    return tft_model
