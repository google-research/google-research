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

import tensorflow.compat.v2 as tf
import numpy as np
import typing


def create_mlp(
    input_dim,
    output_dim,
    hidden_dims = (256, 256),
    activation = tf.nn.relu,
    last_layer_activation = None,
    near_zero_last_layer = False,
):
  """Creates an MLP.

  Args:
    input_dim: input dimensionaloty.
    output_dim: output dimensionality.
    hidden_dims: hidden layers dimensionality.
    activation: activations after hidden units.

  Returns:
    An MLP model.
  """
  if not hasattr(input_dim, '__len__'):
    input_dim = [input_dim]
  initialization = tf.keras.initializers.VarianceScaling(
      scale=0.333, mode='fan_in', distribution='uniform')
  near_zero_initialization = tf.keras.initializers.VarianceScaling(
      scale=1e-2, mode='fan_in', distribution='uniform')

  layers = []
  for hidden_dim in hidden_dims:
    layers.append(
        tf.keras.layers.Dense(
            hidden_dim,
            activation=activation,
            #kernel_initializer=initialization
        ))
  layers += [
      tf.keras.layers.Dense(
          output_dim,
          activation=last_layer_activation,
          #kernel_initializer=near_zero_initialization
          #if near_zero_last_layer else initialization
      )
  ]

  inputs = tf.keras.Input(shape=input_dim)
  outputs = tf.keras.Sequential(layers)(inputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.call = tf.function(model.call)
  return model


def create_conv(
    input_shape,
    kernel_sizes = (2, 2, 3),
    stride_sizes = (1, 1, 2),
    pool_sizes = None,
    num_filters = 64,
    activation = tf.nn.relu,
    activation_last_layer = True,
    output_dim = None,
    padding = 'same',
    residual=False,
):
  """Creates an MLP.

  Args:
    input_shape: input shape.
    hidden_dims: hidden layers dimensionality.
    activation: activations after hidden units.

  Returns:
    An MLP model.
  """
  if not hasattr(num_filters, '__len__'):
    num_filters = (num_filters,) * len(kernel_sizes)
  if not hasattr(pool_sizes, '__len__'):
    pool_sizes = (pool_sizes,) * len(kernel_sizes)

  layers = []
  for i, (kernel_size, stride, filters, pool_size) in enumerate(
      zip(kernel_sizes, stride_sizes, num_filters, pool_sizes)):
    if i == len(kernel_sizes) - 1 and not output_dim:
      activation = activation if activation_last_layer else None
    layers.append(
        tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=(stride, stride),
            activation=activation,
            padding=padding))
    if pool_size:
      layers.append(
          tf.keras.layers.MaxPool2D(pool_size=pool_size, padding=padding))

  if output_dim:
    layers += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            output_dim,
            activation=activation if activation_last_layer else None)
    ]

  inputs = tf.keras.Input(shape=input_shape)
  if residual:
    outputs = layers[0](inputs)
    for layer in layers[1:-1]:
      outputs = outputs + layer(outputs)
    outputs = layers[-1](outputs)
  else:
    outputs = tf.keras.Sequential(layers)(inputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.call = tf.function(model.call)
  return model


def create_rnn(
    input_dims,
    output_dim = None,
    hidden_dims = (256,),
    activation = tf.nn.relu,
    bidirectional=True,
):
  inputs = tf.keras.Input(shape=input_dims)
  if bidirectional:
    layers = [
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True))
        for hidden_dim in hidden_dims
    ]
  else:
    layers = [
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        for hidden_dim in hidden_dims
    ]

  outputs = tf.keras.Sequential(layers)(inputs)
  if output_dim:
    outputs = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=1)),
        tf.keras.layers.Dense(output_dim)
    ])(
        outputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.call = tf.function(model.call)
  return model


def convert_to_tf_dataset(dataset,
                          max_trajectory_length,
                          batch_size,
                          pos_only=False):
  states, next_states, actions, rewards = [], [], [], []
  episodes, valid_steps = dataset.get_all_episodes()
  for episode_num in range(tf.shape(valid_steps)[0]):
    episode_states, episode_next_states, episode_actions, episode_rewards = [], [], [], []
    keep = False
    for step_num in range(tf.shape(valid_steps)[1] - 1):
      this_step = tf.nest.map_structure(lambda t: t[episode_num, step_num],
                                        episodes)
      next_step = tf.nest.map_structure(lambda t: t[episode_num, step_num + 1],
                                        episodes)
      if this_step.is_last() or not valid_steps[episode_num, step_num]:
        continue

      keep = keep or this_step.reward > 0
      episode_states.append(this_step.observation)
      episode_next_states.append(next_step.observation)
      episode_actions.append(this_step.action)
      episode_rewards.append(this_step.reward)
    if not keep and pos_only:
      continue
    states.append(episode_states)
    next_states.append(episode_next_states)
    actions.append(episode_actions)
    rewards.append(episode_rewards)

  states = tf.keras.preprocessing.sequence.pad_sequences(
      states,
      maxlen=max_trajectory_length,
      padding='post',
      dtype=np.int32,
      value=-1)
  next_states = tf.keras.preprocessing.sequence.pad_sequences(
      next_states,
      maxlen=max_trajectory_length,
      padding='post',
      dtype=np.int32,
      value=-1)
  actions = tf.keras.preprocessing.sequence.pad_sequences(
      actions,
      maxlen=max_trajectory_length,
      padding='post',
      dtype=np.int32,
      value=-1)
  rewards = tf.keras.preprocessing.sequence.pad_sequences(
      rewards,
      maxlen=max_trajectory_length,
      padding='post',
      dtype=np.float32,
      value=-1)
  mask = tf.cast(rewards >= 0, tf.float32)

  dataset = tf.data.Dataset.from_tensor_slices(
      (states, actions, rewards, next_states, mask)).cache().shuffle(
          states.shape[0], reshuffle_each_iteration=True).repeat().batch(
              batch_size,
              drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def convert_to_np_dataset(dataset, tabular_obs=False, tabular_act=False):
  dataset_spec = dataset.spec
  observation_spec = dataset_spec.observation
  action_spec = dataset_spec.action
  num_states = observation_spec.maximum + 1
  num_actions = action_spec.maximum + 1
  trajectories = []
  episodes, valid_steps = dataset.get_all_episodes()
  for episode_num in range(tf.shape(valid_steps)[0]):
    episode_states, episode_next_states, episode_actions, episode_rewards, episode_dones = [], [], [], [], []
    for step_num in range(tf.shape(valid_steps)[1] - 1):
      this_step = tf.nest.map_structure(lambda t: t[episode_num, step_num],
                                        episodes)
      next_step = tf.nest.map_structure(lambda t: t[episode_num, step_num + 1],
                                        episodes)
      episode_states.append(
          tf.one_hot(this_step.observation, num_states
                    ) if tabular_obs else this_step.observation)
      episode_next_states.append(
          tf.one_hot(next_step.observation, num_states
                    ) if tabular_obs else next_step.observation)
      episode_actions.append(
          tf.one_hot(this_step.action, num_actions
                    ) if tabular_act else this_step.action)
      reward = this_step.reward * tf.cast(
          not this_step.is_last() and valid_steps[episode_num, step_num],
          tf.float32)
      episode_rewards.append(reward)
      episode_dones.append(this_step.is_last() or
                           not valid_steps[episode_num, step_num])
    trajectory = {
        'observations': np.array(episode_states, dtype=np.float32),
        'actions': np.array(episode_actions, dtype=np.float32),
        'rewards': np.array(episode_rewards, dtype=np.float32),
        'dones': np.array(episode_dones, dtype=np.float32)
    }
    trajectories.append(trajectory)
  return trajectories


def transformer_module(query,
                       value,
                       key,
                       embedding_dim=256,
                       num_heads=4,
                       key_dim=128,
                       ff_dim=256,
                       rate=0.1,
                       output_dim=None,
                       last_layer=False,
                       attention_mask=None):
  """From https://keras.io/examples/nlp/masked_language_modeling/"""
  # Multi headed self-attention
  attention_output, attention_scores = tf.keras.layers.MultiHeadAttention(
      num_heads=num_heads, key_dim=key_dim)(
          query,
          value,
          key=key,
          attention_mask=attention_mask,
          return_attention_scores=True)
  attention_output = tf.keras.layers.Dropout(rate)(attention_output)
  attention_output = tf.keras.layers.LayerNormalization(
      epsilon=1e-6,)(
          query + attention_output)

  # Feed-forward layer
  ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(ff_dim, activation='relu'),
      tf.keras.layers.Dense(output_dim or embedding_dim),
  ],)
  ffn_output = ffn(attention_output)

  if last_layer:
    sequence_output = ffn_output
  else:
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    sequence_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention_output + ffn_output)

  return sequence_output, attention_scores


def transformer(embeddings,
                num_layers=4,
                embedding_dim=256,
                num_heads=4,
                key_dim=64,
                ff_dim=256,
                rate=0.,
                output_dim=None,
                attention_mask=None):
  output_dim = output_dim or embedding_dim
  encoder_output = embeddings

  for i in range(num_layers):
    last_layer = i == num_layers - 1
    encoder_output, attention_scores = transformer_module(
        encoder_output,
        encoder_output,
        encoder_output,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        key_dim=key_dim,
        ff_dim=ff_dim,
        output_dim=output_dim if last_layer else None,
        last_layer=last_layer,
        rate=rate,
        attention_mask=attention_mask)
  return encoder_output


def dense_gaussian_kl(mu_left, logvar_left, mu_right, logvar_right):
  gauss_klds = 0.5 * (
      logvar_right - logvar_left +
      (tf.exp(logvar_left) / tf.exp(logvar_right)) +
      ((mu_left - mu_right)**2.0 / tf.exp(logvar_right)) - 1.0)
  return gauss_klds


def dense_cross_entropy(logits, labels):
  """Applies sparse cross entropy loss between logits and target labels."""
  loss = -labels * tf.math.log(tf.nn.softmax(logits, axis=-1) + 1e-8)
  return loss


def accuracy(logits, labels):
  """Applies sparse cross entropy loss between logits and target labels."""
  predicted_label = tf.argmax(logits, axis=-1)
  acc = tf.cast(tf.equal(predicted_label, labels), tf.float32)
  return tf.reduce_mean(acc)


def to_categorical(obs, num_classes):
  b, w, _ = obs.shape
  flat_obs = obs.reshape(-1).copy()
  one_hot = np.zeros((b * w * w, num_classes))
  one_hot[np.arange(b * w * w), flat_obs] = 1
  return one_hot.reshape(
      (obs.shape[0], obs.shape[1], obs.shape[2], num_classes)).astype(np.uint8)
