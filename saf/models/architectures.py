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

"""Neural network architecture components."""

import tensorflow as tf


class MLPDecoder(tf.keras.Model):
  """MLP decoder class."""

  def __init__(self, hparams):

    super(MLPDecoder, self).__init__()

    # Model hyperparameters
    self.forecast_horizon = hparams["forecast_horizon"]
    self.representation_combination = hparams["representation_combination"]

    # Model layers
    self.forecast_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_forecast_n1 = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_forecast_n2 = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_forecast_f = tf.keras.layers.Dense(
        units=self.forecast_horizon, activation="linear")

  def forward(self, representation, static):

    static_mapped = self.forecast_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)

    if self.representation_combination == "multiplication":
      representation *= static_mapped
    elif self.representation_combination == "concatenation":
      representation = tf.concat([representation, static_mapped], axis=1)
    else:
      representation *= static_mapped

    output = self.mlp_forecast_f(
        self.mlp_forecast_n2(self.mlp_forecast_n1(representation)))

    return tf.reshape(output, [-1, self.forecast_horizon])


class MLPDecoderWithState(tf.keras.Model):
  """MLP decoder with state class."""

  def __init__(self, hparams):

    super(MLPDecoderWithState, self).__init__()

    # Model hyperparameters
    self.forecast_horizon = hparams["forecast_horizon"]
    self.representation_combination = hparams["representation_combination"]

    # Model layers
    self.forecast_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.forecast_state_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_forecast_n1 = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_forecast_n2 = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_forecast_f = tf.keras.layers.Dense(
        units=self.forecast_horizon, activation="linear")

  def forward(self, representation, static, state):

    state_mapped = self.forecast_state_mapper(state)
    state_mapped = tf.nn.relu(state_mapped)

    static_mapped = self.forecast_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)

    if self.representation_combination == "multiplication":
      representation *= state_mapped
      representation *= static_mapped
    elif self.representation_combination == "concatenation":
      representation = tf.concat([representation, state_mapped, static_mapped],
                                 axis=1)
    else:
      representation += state_mapped
      representation *= static_mapped

    output = self.mlp_forecast_f(
        self.mlp_forecast_n2(self.mlp_forecast_n1(representation)))

    return tf.reshape(output, [-1, self.forecast_horizon])


class MLPbackcast(tf.keras.Model):
  """MLP backcast architecture."""

  def __init__(self, hparams):

    super(MLPbackcast, self).__init__()

    # Model hyperparameters
    self.num_encode = hparams["num_encode"]
    self.num_features = hparams["num_features"]
    self.representation_combination = hparams["representation_combination"]

    # Model layers
    self.backcast_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_backcast_n1 = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_backcast_n2 = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.mlp_backcast_f = tf.keras.layers.Dense(
        units=hparams["num_encode"] * self.num_features, activation="linear")

  def forward(self, representation, static):

    static_mapped = self.backcast_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)

    if self.representation_combination == "multiplication":
      representation *= static_mapped
    elif self.representation_combination == "concatenation":
      representation = tf.concat([representation, static_mapped], axis=1)
    else:
      representation *= static_mapped

    output = self.mlp_backcast_f(
        self.mlp_backcast_n2(self.mlp_backcast_n1(representation)))
    return tf.reshape(output, [-1, self.num_encode, self.num_features])


class LSTMEncoder(tf.keras.Model):
  """LSTM encoder architecture."""

  def __init__(self,
               hparams,
               return_sequences=False,
               return_state=False,
               output_mapping=False):

    super(LSTMEncoder, self).__init__()

    # Model hyperparameters
    self.num_encode = hparams["num_encode"]
    self.representation_combination = hparams["representation_combination"]
    self.return_state = return_state
    self.output_mapping = output_mapping

    # Model layers
    self.lstm = tf.keras.layers.LSTM(
        units=hparams["num_units"],
        return_sequences=return_sequences,
        return_state=return_state)
    self.lstm_sequence_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.sequence_distributed_mapper_forecast_encoder = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_input_mapper)
    if self.output_mapping:
      self.output_dim = hparams["output_dim"]
      self.output_mapper = tf.keras.layers.Dense(units=self.output_dim)

  def forward(self, sequence, static):

    # sequence has dimensions of [batch, timesteps, features]
    # static has dimensions of [batch, features]

    lstm_input = self.sequence_distributed_mapper_forecast_encoder(sequence)

    static_mapped = self.lstm_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)
    static_mapped = tf.tile(
        tf.expand_dims(static_mapped, 1), [1, self.num_encode, 1])

    if self.representation_combination == "multiplication":
      lstm_input *= static_mapped
    elif self.representation_combination == "concatenation":
      lstm_input = tf.concat([lstm_input, static_mapped], axis=2)
    else:
      lstm_input += static_mapped

    if self.return_state:
      lstm_encoded, state_h, state_c = self.lstm(inputs=lstm_input)
      encoder_states = [state_h, state_c]
    else:
      lstm_encoded = self.lstm(inputs=lstm_input)
      encoder_states = None

    if self.output_mapping:
      lstm_encoded = self.output_mapper(lstm_encoded)

    return lstm_encoded, encoder_states


class LSTMEncoderWithState(tf.keras.Model):
  """LSTM encoder with state architecture."""

  def __init__(self, hparams, return_state=False):

    super(LSTMEncoderWithState, self).__init__()

    # Model hyperparameters
    self.num_encode = hparams["num_encode"]
    self.representation_combination = hparams["representation_combination"]
    self.return_state = return_state

    # Model layers
    self.lstm = tf.keras.layers.LSTM(
        units=hparams["num_units"], return_state=return_state)
    self.lstm_sequence_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm_state_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.sequence_distributed_mapper_forecast_encoder = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_input_mapper)

  def forward(self, sequence, static, state):

    # sequence has dimensions of [batch, timesteps, features]
    # static has dimensions of [batch, features]

    lstm_input = self.sequence_distributed_mapper_forecast_encoder(sequence)

    static_mapped = self.lstm_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)
    static_mapped = tf.tile(
        tf.expand_dims(static_mapped, 1), [1, self.num_encode, 1])

    state_mapped = self.lstm_state_input_mapper(state)
    state_mapped = tf.nn.relu(state_mapped)
    state_mapped = tf.tile(
        tf.expand_dims(state_mapped, 1), [1, self.num_encode, 1])

    if self.representation_combination == "multiplication":
      lstm_input *= static_mapped
      lstm_input *= state_mapped
    elif self.representation_combination == "concatenation":
      lstm_input = tf.concat([lstm_input, static_mapped, state_mapped], axis=2)
    else:
      lstm_input += static_mapped
      lstm_input += state_mapped

    if self.return_state:
      lstm_encoded, state_h, state_c = self.lstm(inputs=lstm_input)
      encoder_states = [state_h, state_c]
    else:
      lstm_encoded = self.lstm(inputs=lstm_input)
      encoder_states = None

    return lstm_encoded, encoder_states


class LSTMDecoder(tf.keras.Model):
  """LSTM decoder architecture."""

  def __init__(self, hparams):

    super(LSTMDecoder, self).__init__()

    # Model hyperparameters
    self.forecast_horizon = hparams["forecast_horizon"]
    self.representation_combination = hparams["representation_combination"]

    # Model layers
    self.forecast_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm = tf.keras.layers.LSTM(
        units=hparams["num_units"], return_sequences=True)
    self.lstm_sequence_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm_sequence_output_mapper = tf.keras.layers.Dense(units=1)
    self.input_sequence_distributed_mapper = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_input_mapper)
    self.output_sequence_distributed_mapper = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_output_mapper)
    self.mlp_forecast_f = tf.keras.layers.Dense(
        units=self.forecast_horizon, activation="linear")

  def forward(self, representation, static, future_features, initial_state):

    # Map static features
    static_mapped = self.forecast_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)

    if self.representation_combination == "multiplication":
      representation *= static_mapped
    elif self.representation_combination == "concatenation":
      representation = tf.concat([representation, static_mapped], axis=1)
    else:
      representation *= static_mapped

    # Combine with future features
    lstm_decoding_inputs = self.input_sequence_distributed_mapper(
        future_features)

    representation = tf.expand_dims(representation, 1)
    representation = tf.tile(representation, [1, self.forecast_horizon, 1])
    if self.representation_combination == "multiplication":
      lstm_decoding_inputs *= representation
    elif self.representation_combination == "concatenation":
      lstm_decoding_inputs = tf.concat([lstm_decoding_inputs, representation],
                                       axis=2)
    else:
      lstm_decoding_inputs *= representation

    output = self.lstm(inputs=lstm_decoding_inputs, initial_state=initial_state)
    output = self.output_sequence_distributed_mapper(output)
    output = tf.reshape(output, [-1, self.forecast_horizon])

    return output


class LSTMDecoderWithState(tf.keras.Model):
  """LSTM decoder with state architecture."""

  def __init__(self, hparams):

    super(LSTMDecoderWithState, self).__init__()

    # Model hyperparameters
    self.forecast_horizon = hparams["forecast_horizon"]
    self.representation_combination = hparams["representation_combination"]

    # Model layers
    self.forecast_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.forecast_state_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm = tf.keras.layers.LSTM(
        units=hparams["num_units"], return_sequences=True)
    self.lstm_sequence_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm_sequence_output_mapper = tf.keras.layers.Dense(units=1)
    self.input_sequence_distributed_mapper = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_input_mapper)
    self.output_sequence_distributed_mapper = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_output_mapper)
    self.mlp_forecast_f = tf.keras.layers.Dense(
        units=self.forecast_horizon, activation="linear")

  def forward(self, representation, static, state, future_features,
              initial_state):

    # Map static features
    static_mapped = self.forecast_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)

    state_mapped = self.forecast_state_mapper(state)
    state_mapped = tf.nn.relu(state_mapped)

    if self.representation_combination == "multiplication":
      representation *= state_mapped
      representation *= static_mapped
    elif self.representation_combination == "concatenation":
      representation = tf.concat([representation, state_mapped, static_mapped],
                                 axis=1)
    else:
      representation += state_mapped
      representation *= static_mapped

    # Combine with future features
    lstm_decoding_inputs = self.input_sequence_distributed_mapper(
        future_features)

    representation = tf.expand_dims(representation, 1)
    representation = tf.tile(representation, [1, self.forecast_horizon, 1])
    if self.representation_combination == "multiplication":
      lstm_decoding_inputs *= representation
    elif self.representation_combination == "concatenation":
      lstm_decoding_inputs = tf.concat([lstm_decoding_inputs, representation],
                                       axis=2)
    else:
      lstm_decoding_inputs *= representation

    output = self.lstm(inputs=lstm_decoding_inputs, initial_state=initial_state)
    output = self.output_sequence_distributed_mapper(output)
    output = tf.reshape(output, [-1, self.forecast_horizon])

    return output


class LSTMBackcast(tf.keras.Model):
  """LSTM backcast architecture."""

  def __init__(self, hparams):

    super(LSTMBackcast, self).__init__()

    # Model hyperparameters
    self.num_encode = hparams["num_encode"]
    self.num_features = hparams["num_features"]
    self.representation_combination = hparams["representation_combination"]

    # Model layers
    self.forecast_static_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm = tf.keras.layers.LSTM(
        units=hparams["num_units"], return_sequences=True)
    self.lstm_sequence_input_mapper = tf.keras.layers.Dense(
        units=hparams["num_units"], activation="relu")
    self.lstm_sequence_output_mapper = tf.keras.layers.Dense(
        units=self.num_features)
    self.input_sequence_distributed_mapper = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_input_mapper)
    self.output_sequence_distributed_mapper = tf.keras.layers.TimeDistributed(
        layer=self.lstm_sequence_output_mapper)

  def forward(self, representation, static, input_features, initial_state):

    # Map static features
    static_mapped = self.forecast_static_input_mapper(static)
    static_mapped = tf.nn.relu(static_mapped)

    if self.representation_combination == "multiplication":
      representation *= static_mapped
    elif self.representation_combination == "concatenation":
      representation = tf.concat([representation, static_mapped], axis=1)
    else:
      representation *= static_mapped

    # Combine with future features
    lstm_decoding_inputs = self.input_sequence_distributed_mapper(
        input_features)
    representation = tf.expand_dims(representation, 1)
    representation = tf.tile(representation, [1, self.num_encode, 1])

    if self.representation_combination == "multiplication":
      lstm_decoding_inputs *= representation
    elif self.representation_combination == "concatenation":
      lstm_decoding_inputs = tf.concat([lstm_decoding_inputs, representation],
                                       axis=2)
    else:
      lstm_decoding_inputs *= representation

    output = self.lstm(inputs=lstm_decoding_inputs, initial_state=initial_state)
    output = self.output_sequence_distributed_mapper(output)
    output = tf.reshape(output, [-1, self.num_encode, self.num_features])

    return output
