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
# pylint: disable=g-complex-comprehension
"""Encoder+LSTM network for use with MuZero."""

import os

import tensorflow as tf
import tensorflow_addons as tfa

from muzero import core as mzcore



class AbstractEncoderandLSTM(tf.Module):
  """Encoder+stacked LSTM Agent.

  When using this, implement the `_encode_observation` method.
  """

  def __init__(self,
               parametric_action_distribution,
               rnn_sizes,
               head_hidden_sizes,
               reward_encoder,
               value_encoder,
               normalize_hidden_state=False,
               rnn_cell_type='lstm_norm',
               recurrent_activation='sigmoid',
               head_relu_before_norm=False,
               nonlinear_to_hidden=False,
               embed_actions=True):
    """Creates an Encoder followed by a stacked LSTM agent.

    Args:
      parametric_action_distribution: an object of ParametricDistribution class
        specifing a parametric distribution over actions to be used
      rnn_sizes: list of integers with sizes of LSTM layers
      head_hidden_sizes: list of integers with sizes of head layers
      reward_encoder: a value encoder for the reward
      value_encoder: a value encoder for the value
      normalize_hidden_state: boolean to normalize the hidden state each step
      rnn_cell_type: 'gru', 'simple', 'lstm_norm' or 'lstm'
      recurrent_activation: a keras activation function
      head_relu_before_norm: put ReLU before the normalization
      nonlinear_to_hidden: appends recurrent_activation to the latent projection
      embed_actions: use action embeddings instead of one hot encodings
    """
    super().__init__(name='MuZeroAgent')
    self._parametric_action_distribution = parametric_action_distribution
    self.reward_encoder = reward_encoder
    self.value_encoder = value_encoder
    self.head_hidden_sizes = head_hidden_sizes
    self._normalize_hidden_state = normalize_hidden_state
    self._rnn_cell_type = rnn_cell_type
    self._head_relu_before_norm = head_relu_before_norm

    # LSTMs pass on 2x their state size
    self._rnn_sizes = rnn_sizes
    if rnn_cell_type in ('gru', 'simple'):
      self.hidden_state_size = sum(rnn_sizes)
    elif rnn_cell_type in ('lstm', 'lstm_norm'):
      self.hidden_state_size = sum(rnn_sizes) * 2

    self._to_hidden = tf.keras.Sequential(
        [
            # flattening the representation
            tf.keras.layers.Flatten(),
            # mapping it to the size and domain of the hidden state
            tf.keras.layers.Dense(
                self.hidden_state_size,
                activation=(recurrent_activation
                            if nonlinear_to_hidden else None),
                name='final')
        ],
        name='to_hidden')

    self._embed_actions = embed_actions
    if self._embed_actions:
      self._action_embeddings = tf.keras.layers.Dense(self.hidden_state_size)

    # RNNs are a convenient choice for muzero, because they can take the action
    # as input and compute the reward from the output, while computing
    # value and policy from the hidden states.

    rnn_cell_cls = {
        'gru': tf.keras.layers.GRUCell,
        'lstm': tf.keras.layers.LSTMCell,
        'lstm_norm': tfa.rnn.LayerNormLSTMCell,
        'simple': tf.keras.layers.SimpleRNNCell,
    }[rnn_cell_type]

    rnn_cells = [
        rnn_cell_cls(
            size,
            recurrent_activation=recurrent_activation,
            name='cell_{}'.format(idx)) for idx, size in enumerate(rnn_sizes)
    ]
    self._core = tf.keras.layers.StackedRNNCells(
        rnn_cells, name='recurrent_core')

    self._policy_head = tf.keras.Sequential(
        self._head_hidden_layers() + [
            tf.keras.layers.Dense(
                parametric_action_distribution.param_size, name='output')
        ],
        name='policy_logits')

    # Note that value and reward are logits, because their values are binned.
    # See utils.ValueEncoder for details.
    self._value_head = tf.keras.Sequential(
        self._head_hidden_layers() + [
            tf.keras.layers.Dense(self.value_encoder.num_steps, name='output'),
            tf.keras.layers.Softmax()
        ],
        name='value_logits')
    self._reward_head = tf.keras.Sequential(
        self._head_hidden_layers() + [
            tf.keras.layers.Dense(self.reward_encoder.num_steps, name='output'),
            tf.keras.layers.Softmax()
        ],
        name='reward_logits')

  # Each head can have its own hidden layers.
  def _head_hidden_layers(self):

    def _make_layer(size):
      if self._head_relu_before_norm:
        return [
            tf.keras.layers.Dense(size, 'relu'),
            tf.keras.layers.LayerNormalization(),
        ]
      else:
        return [
            tf.keras.layers.Dense(size, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU(),
        ]

    layers = [
        tf.keras.Sequential(
            _make_layer(size), name='intermediate_{}'.format(idx))
        for idx, size in enumerate(self.head_hidden_sizes)
    ]
    return layers

  @staticmethod
  def _rnn_to_flat(state):
    """Maps LSTM state to flat vector."""
    states = []
    for cell_state in state:
      if not (isinstance(cell_state, list) or isinstance(cell_state, tuple)):
        # This is a GRU or SimpleRNNCell
        cell_state = (cell_state,)
      states.extend(cell_state)
    return tf.concat(states, -1)

  def _flat_to_rnn(self, state):
    """Maps flat vector to LSTM state."""
    tensors = []
    cur_idx = 0
    for size in self._rnn_sizes:
      if self._rnn_cell_type in ('gru', 'simple'):
        states = (state[Ellipsis, cur_idx:cur_idx + size],)
        cur_idx += size
      elif self._rnn_cell_type in ('lstm', 'lstm_norm'):
        states = (state[Ellipsis, cur_idx:cur_idx + size],
                  state[Ellipsis, cur_idx + size:cur_idx + 2 * size])
        cur_idx += 2 * size
      tensors.append(states)
    assert cur_idx == state.shape[-1]
    return tensors

  def initial_state(self, batch_size):
    return tf.zeros((batch_size, self.hidden_state_size))

  def _encode_observation(self, observation, training=True):
    raise NotImplementedError()

  def pretraining_loss(self, sample, training=True):
    raise NotImplementedError()

  def get_pretraining_trainable_variables(self):
    return self.trainable_variables

  def get_rl_trainable_variables(self):
    return self.trainable_variables

  def get_trainable_variables(self, pretraining=False):
    if pretraining:
      return self.get_pretraining_trainable_variables()
    else:
      return self.get_rl_trainable_variables()

  def initial_inference(self, observation, training=True):
    encoded_observation = self._encode_observation(
        observation, training=training)
    hidden_state = self._to_hidden(encoded_observation, training=training)

    value_logits = self._value_head(hidden_state, training=training)
    value = self.value_encoder.decode(value_logits)

    # Rewards are only calculated in recurrent_inference.
    reward = tf.zeros_like(value)
    reward_logits = self.reward_encoder.encode(reward)

    policy_logits = self._policy_head(hidden_state, training=training)

    outputs = mzcore.NetworkOutput(
        value_logits=value_logits,
        value=value,
        reward_logits=reward_logits,
        reward=reward,
        policy_logits=policy_logits,
        hidden_state=hidden_state)
    return outputs

  def _maybe_normalize_hidden_state(self, hidden_state):
    if self._normalize_hidden_state:
      # This is in the paper, but probably unnecessary.
      max_hidden_state = tf.reduce_max(hidden_state, -1, keepdims=True)
      min_hidden_state = tf.reduce_min(hidden_state, -1, keepdims=True)
      hidden_state_range = max_hidden_state - min_hidden_state
      hidden_state = hidden_state - min_hidden_state
      hidden_state = tf.math.divide_no_nan(hidden_state, hidden_state_range)
      hidden_state = hidden_state * 2. - 1.
    return hidden_state

  def recurrent_inference(self, hidden_state, action, training=True):
    if self._embed_actions:
      one_hot_action = tf.one_hot(
          action, self._parametric_action_distribution.param_size)
      embedded_action = self._action_embeddings(one_hot_action)
    else:
      one_hot_action = tf.one_hot(
          action, self._parametric_action_distribution.param_size, 1., -1.)
      embedded_action = one_hot_action
    hidden_state = self._maybe_normalize_hidden_state(hidden_state)

    rnn_state = self._flat_to_rnn(hidden_state)
    rnn_output, next_rnn_state = self._core(embedded_action, rnn_state)
    next_hidden_state = self._rnn_to_flat(next_rnn_state)

    value_logits = self._value_head(next_hidden_state, training=training)
    value = self.value_encoder.decode(value_logits)

    reward_logits = self._reward_head(rnn_output, training=training)
    reward = self.reward_encoder.decode(reward_logits)

    policy_logits = self._policy_head(next_hidden_state, training=training)

    output = mzcore.NetworkOutput(
        value=value,
        value_logits=value_logits,
        reward=reward,
        reward_logits=reward_logits,
        policy_logits=policy_logits,
        hidden_state=next_hidden_state)
    return output


class ExportedAgent(tf.Module):
  """Wraps an Agent for export."""

  def __init__(self, agent_module):
    self._agent = agent_module

  def initial_inference(self, input_ids, segment_ids, features, action_history):
    output = self._agent.initial_inference(
        observation=(input_ids, segment_ids, features, action_history),
        training=False)
    return [
        output.value,
        output.value_logits,
        output.reward,
        output.reward_logits,
        output.policy_logits,
        output.hidden_state,
    ]

  def recurrent_inference(self, hidden_state, action):
    output = self._agent.recurrent_inference(
        hidden_state=hidden_state, action=action, training=False)
    return [
        output.value,
        output.value_logits,
        output.reward,
        output.reward_logits,
        output.policy_logits,
        output.hidden_state,
    ]


def export_agent_for_initial_inference(agent,
                                       model_dir):
  """Export `agent` as a TPU servable model for initial inference."""

  def get_initial_inference_fn(model):

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 512), dtype=tf.int32, name='input_ids'),
        tf.TensorSpec(shape=(None, 512, 1), dtype=tf.int32, name='segment_ids'),
        tf.TensorSpec(shape=(None, 512, 2), dtype=tf.float32, name='features'),
        tf.TensorSpec(shape=(None, 10), dtype=tf.int32, name='action_history'),
    ])
    def serve_fn(input_ids, segment_ids, features, action_history):
      return model.initial_inference(
          input_ids=input_ids,
          segment_ids=segment_ids,
          features=features,
          action_history=action_history)

    return serve_fn

  exported_agent = ExportedAgent(agent_module=agent)
  initial_fn = get_initial_inference_fn(exported_agent)

  # Export.
  save_options = tf.saved_model.SaveOptions(function_aliases={
      'initial_inference': initial_fn,
  })
  # Saves the CPU model, which will be rewritten to a TPU model.
  tf.saved_model.save(
      obj=exported_agent,
      export_dir=model_dir,
      signatures={
          'initial_inference': initial_fn,
      },
      options=save_options)



def export_agent_for_recurrent_inference(agent,
                                         model_dir):
  """Export `agent` as a TPU servable model for recurrent inference."""

  def get_recurrent_inference_fn(model):

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=(None, 1024), dtype=tf.float32, name='hidden_state'),
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='action')
    ])
    def serve_fn(hidden_state, action):
      return model.recurrent_inference(hidden_state=hidden_state, action=action)

    return serve_fn

  exported_agent = ExportedAgent(agent_module=agent)
  recurrent_fn = get_recurrent_inference_fn(exported_agent)

  # Export.
  save_options = tf.saved_model.SaveOptions(function_aliases={
      'recurrent_inference': recurrent_fn,
  })
  # Saves the CPU model, which will be rewritten to a TPU model.
  tf.saved_model.save(
      obj=exported_agent,
      export_dir=model_dir,
      signatures={
          'recurrent_inference': recurrent_fn,
      },
      options=save_options)

