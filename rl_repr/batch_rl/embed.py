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

"""Embedding for state representation learning."""

import typing

from dm_env import specs as dm_env_specs
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl_repr.batch_rl import keras_utils
from rl_repr.batch_rl import policies


def soft_update(net, target_net, tau=0.005):
  for var, target_var in zip(net.variables, target_net.variables):
    new_value = var * tau + target_var * (1 - tau)
    target_var.assign(new_value)


def huber(x, kappa=0.1):
  return (0.5 * tf.square(x) * tf.cast(tf.abs(x) <= kappa, x.dtype) +
          kappa * (tf.abs(x) - 0.5 * kappa) * tf.cast(tf.abs(x) > kappa, x.dtype)
          ) / kappa


def gaussian_kl(mean1, logvar1, mean2=None, logvar2=None):
  if mean2 is None:
    mean2 = tf.zeros_like(mean1)
  if logvar2 is None:
    logvar2 = tf.zeros_like(logvar1)

  kl = -0.5 * tf.reduce_sum(
      1.0 + logvar1 - logvar2
      - tf.exp(-1 * logvar2) * tf.pow(mean1 - mean2, 2)
      - tf.exp(logvar1 - logvar2), -1)

  return kl


def categorical_kl(probs1, probs2=None):
  if probs2 is None:
    probs2 = tf.ones_like(probs1) * tf.reduce_sum(probs1) / tf.reduce_sum(tf.ones_like(probs1))

  kl = tf.reduce_sum(
      probs1 * (-tf.math.log(1e-8 + probs2) + tf.math.log(1e-8 + probs1)), -1)
  return kl


def transformer_module(query,
                       key,
                       value,
                       embedding_dim=256,
                       num_heads=4,
                       key_dim=128,
                       ff_dim=256,
                       output_dim=None,
                       last_layer=False,
                       attention_mask=None):
  """From https://keras.io/examples/nlp/masked_language_modeling/"""
  # Multi headed self-attention
  attention_output = tf.keras.layers.MultiHeadAttention(
      num_heads=num_heads, key_dim=key_dim)(
          query, key, value, attention_mask=attention_mask)
  attention_output = tf.keras.layers.Dropout(0.1)(
      attention_output
  )
  attention_output = tf.keras.layers.LayerNormalization(
      epsilon=1e-6,
  )(query + attention_output)

  # Feed-forward layer
  ffn = tf.keras.Sequential(
      [
          tf.keras.layers.Dense(ff_dim, activation="relu"),
          tf.keras.layers.Dense(output_dim or embedding_dim),
      ],
  )
  ffn_output = ffn(attention_output)

  if last_layer:
    sequence_output = ffn_output
  else:
    ffn_output = tf.keras.layers.Dropout(0.1)(
        ffn_output
    )
    sequence_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(attention_output + ffn_output)

  return sequence_output


def transformer(embeddings,
                num_layers=1,
                embedding_dim=256,
                num_heads=4,
                key_dim=128,
                ff_dim=256,
                output_dim=None,
                attention_mask=None):
  output_dim = output_dim or embedding_dim
  encoder_output = embeddings

  for i in range(num_layers):
    last_layer = i == num_layers - 1
    encoder_output = transformer_module(
        encoder_output,
        encoder_output,
        encoder_output,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        key_dim=key_dim,
        ff_dim=ff_dim,
        output_dim=output_dim if last_layer else None,
        last_layer=last_layer,
        attention_mask=attention_mask)
  return encoder_output


def create_mlp(
    input_dim,
    output_dim,
    hidden_dims = (256, 256)):

  relu_gain = tf.math.sqrt(2.0)
  relu_orthogonal = tf.keras.initializers.Orthogonal(relu_gain)
  near_zero_orthogonal = tf.keras.initializers.Orthogonal(1e-2)

  layers = []
  for hidden_dim in hidden_dims:
    layers.append(
        tf.keras.layers.Dense(
            hidden_dim,
            activation=tf.nn.relu,
            kernel_initializer=relu_orthogonal))

  if isinstance(input_dim, int):
    input_shape = (input_dim,)
  else:
    input_shape = input_dim
  inputs = tf.keras.Input(shape=input_dim)
  outputs = tf.keras.Sequential(
      layers + [tf.keras.layers.Dense(
          output_dim - 1, kernel_initializer=near_zero_orthogonal),
                tf.keras.layers.Lambda(
                    lambda x: tf.concat([x, tf.ones_like(x[Ellipsis, :1])], -1)),
                tf.keras.layers.LayerNormalization(
                    epsilon=0.0, center=False, scale=False)]
      )(inputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


class EmbedNet(tf.keras.Model):
  """An embed network."""

  def __init__(self,
               state_dim,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256)):
    """Creates a neural net.

    Args:
      state_dim: State size.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()

    inputs = tf.keras.Input(shape=(state_dim,))
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0
    self.embedder = keras_utils.create_mlp(
        inputs.shape[-1], self.embedding_dim, hidden_dims=hidden_dims,
        activation=tf.nn.swish,
        near_zero_last_layer=bool(num_distributions))

  @tf.function
  def call(self,
           states,
           stop_gradient = True):
    """Returns embeddings of states.

    Args:
      states: A batch of states.
      stop_gradient: Whether to put a stop_gradient on embedding.

    Returns:
      Embeddings of states.
    """
    if not self.num_distributions:
      out = self.embedder(states)
    else:
      all_logits = self.embedder(states)
      all_logits = tf.split(all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [tfp.distributions.Categorical(logits=logits).sample()
                     for logits in all_logits]
      all_onehot_samples = [tf.one_hot(samples, self.embedding_dim // self.num_distributions)
                            for samples in all_samples]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      out = joined_onehot_samples + joined_probs - tf.stop_gradient(joined_probs)

    if stop_gradient:
      return tf.stop_gradient(out)
    return out


class RNNEmbedNet(tf.keras.Model):
  """An RNN embed network."""

  def __init__(self,
               input_dim,
               embedding_dim,
               num_distributions=None,
               return_sequences=False):
    """Creates a neural net.

    Args:
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      return_sequences: Whether to return the entire sequence embedding.
    """
    super().__init__()

    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0

    inputs = tf.keras.Input(shape=input_dim)
    outputs = tf.keras.layers.LSTM(
        embedding_dim, return_sequences=return_sequences)(
            inputs)
    self.embedder = tf.keras.Model(inputs=inputs, outputs=outputs)
    self.embedder.call = tf.function(self.embedder.call)

  @tf.function
  def call(self, states, stop_gradient = True):
    """Returns embeddings of states.

    Args:
      states: A batch of sequence of states].
      stop_gradient: Whether to put a stop_gradient on embedding.

    Returns:
      Auto-regressively computed Embeddings of the last states.
    """
    assert (len(states.shape) == 3)
    if not self.num_distributions:
      out = self.embedder(states)
    else:
      all_logits = self.embedder(states)
      all_logits = tf.split(all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [tfp.distributions.Categorical(logits=logits).sample()
                     for logits in all_logits]
      all_onehot_samples = [tf.one_hot(samples, self.embedding_dim // self.num_distributions)
                            for samples in all_samples]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      out = joined_onehot_samples + joined_probs - tf.stop_gradient(joined_probs)

    if stop_gradient:
      return tf.stop_gradient(out)
    return out


class StochasticEmbedNet(tf.keras.Model):
  """A stochastic embed network."""

  def __init__(self,
               state_dim,
               embedding_dim = 256,
               hidden_dims = (256, 256),
               num_distributions = None,
               logvar_min = -4.0,
               logvar_max = 15.0):
    """Creates a neural net.

    Args:
      state_dim: State size.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      logvar_min: Minimum allowed logvar.
      logvar_max: Maximum allowed logvar.
    """
    super().__init__()

    inputs = tf.keras.Input(shape=(state_dim,))
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0

    distribution_dim = (2 if not num_distributions else 1) * self.embedding_dim
    self.embedder = keras_utils.create_mlp(
        inputs.shape[-1], distribution_dim, hidden_dims=hidden_dims,
        activation=tf.nn.swish,
        near_zero_last_layer=False)
    self.logvar_min = logvar_min
    self.logvar_max = logvar_max

  @tf.function
  def call(self,
           states,
           stop_gradient = True,
           sample = True,
           sample_and_raw_output = False):
    """Returns embeddings of states.

    Args:
      states: A batch of states.
      stop_gradient: Whether to put a stop_gradient on embedding.
      sample: Whether to sample an embedding.
      sample_and_raw_output: Whether to return the original
        probability in addition to sampled embeddings.
    Returns:
      Embeddings of states.
    """
    if not self.num_distributions:
      mean_and_logvar = self.embedder(states)
      mean, logvar = tf.split(mean_and_logvar, 2, axis=-1)
      logvar = tf.clip_by_value(logvar, self.logvar_min, self.logvar_max)
      sample_out = mean + tf.random.normal(tf.shape(mean)) * tf.exp(0.5 * logvar)
      raw_out = tf.concat([mean, logvar], -1)
    else:
      all_logits = self.embedder(states)
      all_logits = tf.split(all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [tfp.distributions.Categorical(logits=logits).sample()
                     for logits in all_logits]
      all_onehot_samples = [tf.one_hot(samples, self.embedding_dim // self.num_distributions)
                            for samples in all_samples]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      sample_out = joined_onehot_samples + joined_probs - tf.stop_gradient(joined_probs)
      raw_out = joined_probs

    if sample_and_raw_output:
      out = (sample_out, raw_out)
    elif sample:
      out = sample_out
    else:
      out = raw_out

    if stop_gradient:
      if hasattr(out, '__len__'):
        return tuple(map(tf.stop_gradient, out))
      return tf.stop_gradient(out)
    return out


class TransformerNet(tf.keras.Model):
  """An embed network based on transformer."""

  def __init__(self,
               state_dim,
               embedding_dim = 256,
               num_distributions = None,
               input_embedding_dim = 256,
               num_heads = 4,
               key_dim = 256):
    """Creates a neural net.

    Args:
      state_dim: State size.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      input_embedding_dim: embedding dim for inputs to the transformer.
      hidden_dims: List of hidden dimensions.
    """
    super().__init__()

    self.state_dim = state_dim
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    assert not num_distributions or embedding_dim % num_distributions == 0
    self.input_embedding_dim = input_embedding_dim

    self.component_embedder = keras_utils.create_mlp(
        (state_dim, state_dim + 1),
        self.input_embedding_dim,
        hidden_dims=(256,),
        activation=tf.nn.swish,
        near_zero_last_layer=False)

    attention = tf.keras.layers.MultiHeadAttention(
        num_heads, key_dim=key_dim,
        output_shape=(self.embedding_dim,))
    inputs = tf.keras.Input(shape=(state_dim, self.input_embedding_dim))
    outputs = attention(inputs, inputs)
    self.transformer = tf.keras.Model(inputs=inputs, outputs=outputs)

    self.missing_x = tf.Variable(tf.zeros([self.input_embedding_dim]))

  def process_inputs(self,
                     states,
                     stop_gradient = True):
    one_hot_index = tf.zeros_like(states)[Ellipsis, None] + tf.eye(self.state_dim)
    state_inputs = tf.concat([one_hot_index, states[Ellipsis, None]], -1)
    components = self.component_embedder(state_inputs)
    return components

  @tf.function
  def call(self,
           states,
           stop_gradient = True,
           missing_mask = None):
    """Returns embeddings of states.

    Args:
      states: A batch of states.
      stop_gradient: Whether to put a stop_gradient on embedding.

    Returns:
      Embeddings of states.
    """
    processed_inputs = self.process_inputs(states, stop_gradient=stop_gradient)
    if missing_mask is not None:
      attention_inputs = tf.where(missing_mask[Ellipsis, None],
                                  tf.ones_like(states)[Ellipsis, None] * self.missing_x,
                                  processed_inputs)
    else:
      attention_inputs = processed_inputs

    attention_out = self.transformer(attention_inputs, training=not stop_gradient)

    if not self.num_distributions:
      out = tf.reduce_mean(attention_out, -2)
    else:
      all_logits = tf.reduce_mean(attention_out, -2)
      all_logits = tf.split(all_logits, num_or_size_splits=self.num_distributions, axis=-1)
      all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
      joined_probs = tf.concat(all_probs, -1)
      all_samples = [tfp.distributions.Categorical(logits=logits).sample()
                     for logits in all_logits]
      all_onehot_samples = [tf.one_hot(samples, self.embedding_dim // self.num_distributions)
                            for samples in all_samples]
      joined_onehot_samples = tf.concat(all_onehot_samples, -1)

      # Straight-through gradients.
      out = joined_onehot_samples + joined_probs - tf.stop_gradient(joined_probs)

    if stop_gradient:
      return tf.stop_gradient(out)
    return out


class CpcLearner(tf.keras.Model):
  """A learner for CPC."""

  def __init__(self,
               state_dim,
               action_dim,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               ctx_length = None,
               ctx_action = False,
               downstream_input_mode = 'embed',
               learning_rate = None):
    """Creates networks.

    Args:
      state_dim: State size.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      ctx_length: number of past steps to compute a context.
      ctx_action: Whether to include past actions as a part of the context.
      downstream_input_mode: Whether to use states, embedding, or context.
      learning_rate: Learning rate.
    """
    super().__init__()
    self.input_dim = state_dim
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    self.sequence_length = sequence_length
    self.ctx_length = ctx_length
    self.ctx_action = ctx_action
    self.downstream_input_mode = downstream_input_mode
    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=num_distributions,
        hidden_dims=hidden_dims)
    self.weight = tf.Variable(
        tf.eye(
            self.embedding_dim,
            batch_shape=[sequence_length - (ctx_length or 1)]))

    learning_rate = learning_rate or 3e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.all_variables = [self.weight] + self.embedder.variables
    if ctx_length:
      ctx_dim = embedding_dim + action_dim if ctx_action else embedding_dim
      self.ctx_embedder = RNNEmbedNet([ctx_length, ctx_dim],
                                      embedding_dim)
      self.all_variables += self.ctx_embedder.embedder.variables
    else:
      self.ctx_embedder = None

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: 2 or 3 dimensional state tensors.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    if not self.ctx_embedder:
      assert (len(states.shape) == 2)
      return self.embedder(states, stop_gradient=stop_gradient)

    outputs = []
    for mode in self.downstream_input_mode.split('-'):
      if mode == 'state':
        outputs.append(states[:, self.ctx_length, :])
      elif mode == 'embed':
        outputs.append(
            self.embedder(
                states[:, self.ctx_length, :], stop_gradient=stop_gradient))
      elif mode == 'ctx':
        embedding = tf.reshape(states[:, :self.ctx_length, :],
                               [-1, tf.shape(states)[-1]])
        embedding = self.embedder(embedding, stop_gradient=stop_gradient)
        embedding = tf.reshape(
            embedding, [-1, self.ctx_length, self.embedder.embedding_dim])
        if self.ctx_action:
          embedding = tf.concat([embedding, actions[:, :self.ctx_length, :]],
                                axis=-1)
        embedding = self.ctx_embedder(embedding, stop_gradient=stop_gradient)
        outputs.append(embedding)
    return tf.concat(outputs, axis=-1)

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes matrix of energies between every pair of (embedding, other_embedding)."""
    transformed_embeddings = tf.matmul(embeddings, self.weight)
    energies = tf.matmul(transformed_embeddings, other_embeddings, transpose_b=True)
    return energies

  def fit(self, states,
          actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.sequence_length, self.input_dim])
      all_embeddings = self.embedder(all_states, stop_gradient=False)
      all_embeddings = tf.reshape(
          all_embeddings, [batch_size, self.sequence_length, self.embedding_dim])

      if self.ctx_embedder:
        embeddings = all_embeddings[:, :self.ctx_length, :]
        if self.ctx_action:
          embeddings = tf.concat([embeddings, actions[:, :self.ctx_length, :]],
                                 axis=-1)
        embeddings = self.ctx_embedder(
            embeddings, stop_gradient=False)[None, Ellipsis]
        next_embeddings = tf.transpose(all_embeddings[:, self.ctx_length:, :],
                                       [1, 0, 2])
      else:
        embeddings = all_embeddings[None, :, 0, :]
        next_embeddings = tf.transpose(all_embeddings[:, 1:, :], [1, 0, 2])

      energies = self.compute_energy(embeddings, next_embeddings)
      positive_loss = tf.linalg.diag_part(energies)
      negative_loss = tf.reduce_logsumexp(energies, axis=-1)

      loss = tf.reduce_mean(-positive_loss + negative_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss,
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)

  def get_input_state_dim(self):
    if not self.ctx_embedder:
      return self.embedder.embedding_dim

    input_state_dim = 0
    for mode in self.downstream_input_mode.split('-'):
      if mode == 'state':
        input_state_dim += self.input_dim
      elif mode == 'embed':
        input_state_dim += self.embedder.embedding_dim
      elif mode == 'ctx':
        input_state_dim += self.ctx_embedder.embedding_dim
    return input_state_dim


class HiroLearner(tf.keras.Model):
  """A learner for HIRO."""

  def __init__(self,
               state_dim,
               action_dim,
               embedding_dim = 256,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
    """
    super().__init__()
    self.input_dim = state_dim
    self.embedding_dim = embedding_dim
    self.sequence_length = sequence_length
    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        hidden_dims=hidden_dims)
    self.action_embedder = EmbedNet(
        state_dim + action_dim * (self.sequence_length - 1),
        embedding_dim=self.embedding_dim,
        hidden_dims=hidden_dims)

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.embed_history = tf.Variable(tf.zeros([1024, self.embedding_dim]))
    self.all_variables = self.embedder.variables + self.action_embedder.variables

  @tf.function
  def call(self,
           states,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of states.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    return self.embedder(states, stop_gradient=stop_gradient)

  def _sample_next_states(self, states, discount = 0.99):
    """Given a sequence of states, samples the `next_states` for loss computation."""
    batch_size = tf.shape(states)[0]
    d = self.sequence_length - 1

    probs = discount ** tf.range(d, dtype=tf.float32)
    probs *= tf.constant([1.0] * (d - 1) + [1.0 / (1 - discount)],
                         dtype=tf.float32)
    probs /= tf.reduce_sum(probs)
    index_dist = tfp.distributions.Categorical(probs=probs, dtype=tf.int64)
    indices = index_dist.sample(batch_size)
    batch_size = tf.cast(batch_size, tf.int64)
    next_indices = tf.concat(
        [tf.range(batch_size, dtype=tf.int64)[:, None],
         1 + indices[:, None]], -1)
    next_states = tf.gather_nd(states, next_indices)
    return next_states

  def fit(self, states, actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      next_states = self._sample_next_states(states)
      cur_states = states[:, 0, :]

      cur_embed = self.embedder(cur_states, stop_gradient=False)
      next_embed = self.embedder(next_states, stop_gradient=False)

      # Update history of embeddings with this batch's next_embed.
      self.embed_history.assign(tf.concat([self.embed_history[batch_size:], next_embed], 0))

      action_embed_input = tf.concat(
          [cur_states, tf.reshape(actions[:, :-1, :], [batch_size, -1])], -1)
      action_embed = self.action_embedder(action_embed_input, stop_gradient=False)

      tau = 2.0
      energy_fn = lambda z: -tau * tf.reduce_sum(huber(z), -1)
      positive_loss = tf.reduce_mean(energy_fn(cur_embed + action_embed - next_embed))
      # Negative loss should just be a log-avg-exp, but we compute it in a more
      # numerically stable way below.
      prior_log_probs = tf.reduce_logsumexp(
          energy_fn((cur_embed + action_embed)[:, None, :]
                    - self.embed_history[None, :, :]),
          axis=-1) - tf.math.log(tf.cast(self.embed_history.shape[0], tf.float32))
      shifted_next_embed = tf.concat([next_embed[1:], next_embed[:1]], 0)
      negative_loss = tf.reduce_mean(
          tf.exp(energy_fn((cur_embed + action_embed) - shifted_next_embed)
                 - tf.stop_gradient(prior_log_probs)))

      loss = tf.reduce_mean(-positive_loss + negative_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)

  def get_input_state_dim(self, downstream_input_mode):
    return self.embedder.embedding_dim


class MomentumCpcLearner(CpcLearner):
  """A learner for momentum CPC."""

  def __init__(self,
               state_dim,
               action_dim,
               embedding_dim = 256,
               hidden_dims = (256, 256),
               residual_dims = (256,),
               sequence_length = 2,
               ctx_length = None,
               downstream_input_mode = 'embed',
               learning_rate = None,
               tau = 0.05,
               target_update_period = 1):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      residual_dims: hidden dims for the residual network.
      sequence_length: Expected length of sequences provided as input
      ctx_length: Number of past steps to compute a context.
      downstream_input_mode: Whether to use states, embedding, or context.
      learning_rate: Learning rate.
      tau: Rate for updating target network.
      target_update_period: Frequency for updating target network.
    """
    super().__init__(
        state_dim,
        action_dim,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        sequence_length=sequence_length,
        ctx_length=ctx_length,
        downstream_input_mode=downstream_input_mode,
        learning_rate=learning_rate)

    self.residual_mlp = EmbedNet(
        embedding_dim,
        embedding_dim=embedding_dim,
        hidden_dims=residual_dims)
    self.embedder_target = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        hidden_dims=hidden_dims)
    soft_update(self.embedder, self.embedder_target, tau=1.0)

    learning_rate = learning_rate or 3e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.tau = tau
    self.target_update_period = target_update_period

    self.all_variables += self.residual_mlp.variables

  def fit(self, states, actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      if self.ctx_length:
        cur_states = states[:, :self.ctx_length, :]
        cur_states = tf.reshape(cur_states, [-1, self.input_dim])
      else:
        cur_states = states[:, 0, :]
      all_next_states = tf.reshape(states[:, self.ctx_length or 1:, :], [
          batch_size * (self.sequence_length -
                        (self.ctx_length or 1)), self.input_dim
      ])

      embeddings = self.embedder(cur_states, stop_gradient=False)
      embeddings += self.residual_mlp(embeddings, stop_gradient=False)
      all_next_embeddings = self.embedder_target(
          all_next_states, stop_gradient=True)

      if self.ctx_length:
        embeddings = tf.reshape(embeddings,
                                [-1, self.ctx_length, self.embedding_dim])
        embeddings = self.ctx_embedder(embeddings, stop_gradient=False)

      next_embeddings = tf.reshape(all_next_embeddings, [
          batch_size, self.sequence_length -
          (self.ctx_length or 1), self.embedding_dim
      ])
      embeddings = embeddings[None, :, :]
      next_embeddings = tf.transpose(next_embeddings, [1, 0, 2])

      energies = self.compute_energy(embeddings, next_embeddings)
      positive_loss = tf.linalg.diag_part(energies)
      negative_loss = tf.reduce_logsumexp(energies, axis=-1)

      loss = tf.reduce_mean(-positive_loss + negative_loss)

    grads = tape.gradient(loss, self.all_variables)
    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))
    if self.optimizer.iterations % self.target_update_period == 0:
      soft_update(self.embedder, self.embedder_target, tau=self.tau)

    return {
        'embed_loss': loss
    }


class ActionVaeLearner(tf.keras.Model):
  """A learner for variational construction of action given state."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None,
               kl_weight = 0.02,
               trans_kl_weight = 0.0):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
      kl_weight: Weight on KL regularizer.
      trans_kl_weight: Weight on KL regularizer of transformer outputs.
    """
    super().__init__()
    self.input_dim = state_dim
    self.action_dim = action_spec.shape[0]
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    self.sequence_length = sequence_length
    self.kl_weight = kl_weight
    self.trans_kl_weight = trans_kl_weight

    self.embedder = StochasticEmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=self.num_distributions,
        hidden_dims=hidden_dims)

    self.transition = StochasticEmbedNet(
        self.embedding_dim + self.action_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=self.num_distributions,
        hidden_dims=hidden_dims)

    self.policy = policies.DiagGuassianPolicy(embedding_dim,
                                              action_spec,
                                              apply_tanh_squash=True)

    learning_rate = learning_rate or 1e-3
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.target_entropy = -action_spec.shape[0]

    self.all_variables = self.variables

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of states.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    return self.embedder(states, stop_gradient=stop_gradient)

  def fit(self, states, actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.sequence_length, -1])
      all_actions = tf.reshape(actions, [batch_size * self.sequence_length, -1])
      all_embed_sample, all_embed_raw = self.embedder(
          all_states, stop_gradient=False, sample_and_raw_output=True)

      embed_sample = tf.reshape(all_embed_sample,
                                [batch_size, self.sequence_length, self.embedding_dim])
      trans_input = tf.concat([embed_sample[:, 0, :], actions[:, 0, :]], -1)
      _, trans_output_raw = self.transition(trans_input, stop_gradient=False,
                                            sample_and_raw_output=True)
      trans_truth_raw = tf.reshape(all_embed_raw,
                                   [batch_size, self.sequence_length, -1])[:, 1, :]

      data_log_probs = self.policy.log_probs(all_embed_sample, all_actions)
      _, policy_log_probs = self.policy(all_embed_sample, sample=True, with_log_probs=True)

      if not self.num_distributions:
        all_embed_mean, all_embed_logvar = tf.split(all_embed_raw, 2, axis=-1)
        kl_loss = -0.5 * tf.reduce_sum(1.0 + all_embed_logvar - tf.pow(all_embed_mean, 2) -
                                       tf.exp(all_embed_logvar), -1)


        trans_output_mean, trans_output_logvar = tf.split(trans_output_raw, 2, axis=-1)
        trans_truth_mean, trans_truth_logvar = tf.split(trans_truth_raw, 2, axis=-1)
        trans_kl_loss = (gaussian_kl(trans_output_mean, trans_output_logvar,
                                     trans_truth_mean, trans_truth_logvar) +
                         gaussian_kl(trans_truth_mean, trans_truth_logvar,
                                     trans_output_mean, trans_output_logvar))
      else:
        all_embed_logprob = tf.math.log(1e-8 + all_embed_raw)
        kl_loss = tf.reduce_sum(
            all_embed_raw *
            (tf.math.log(float(self.embedding_dim // self.num_distributions))
             + all_embed_logprob), -1)

        trans_kl_loss = (categorical_kl(trans_output_raw, trans_truth_raw) +
                         categorical_kl(trans_truth_raw, trans_output_raw))

      alpha = tf.exp(self.log_alpha)
      alpha_loss = alpha * tf.stop_gradient(-policy_log_probs - self.target_entropy)
      reconstruct_loss = -data_log_probs + tf.stop_gradient(alpha) * policy_log_probs
      loss = (tf.reduce_mean(alpha_loss + reconstruct_loss + self.kl_weight * kl_loss) +
              self.trans_kl_weight * tf.reduce_mean(trans_kl_loss))

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss,
        'alpha': alpha,
        'alpha_loss': tf.reduce_mean(alpha_loss),
        'reconstruct_loss': tf.reduce_mean(reconstruct_loss),
        'embed_kl_loss': tf.reduce_mean(kl_loss),
        'trans_kl_loss': tf.reduce_mean(trans_kl_loss),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)

  def get_input_state_dim(self):
    return self.embedder.embedding_dim


class BertLearner(tf.keras.Model):
  """A learner for BERT."""

  def __init__(self,
               state_dim,
               action_dim,
               embedding_dim = 256,
               num_distributions = None,
               preprocess_dim = 256,
               hidden_dims = (256, 256),
               sequence_length = 2,
               ctx_length = None,
               downstream_input_mode = 'embed',
               learning_rate = None,
               num_heads = 4,
               drop_probability = 0.15,
               switch_probability = 0.05,
               keep_probability = 0.05,
               input_dimension_dropout = 0.2,
               modify_actions = True,
               embed_on_input = False,
               predict_actions = False):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      preprocess_dim: Dimension of input to transformer.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      ctx_length: Number of past steps to compute a context.
      downstream_input_mode: Whether to use states, embedding, or context.
      learning_rate: Learning rate.
      num_heads: Number of heads for transformer.
      drop_probability: Drop probability for input.
      switch_probability: Switch probability for input.
      keep_probability: Keep probability for input.
      input_dimension_dropout: Dropout probability on state inputs.
      modify_actions: Whether to drop/switch/keep actions as well as states.
      embed_on_input: Whether to use embedder on input in addition to output.
      predict_actions: Whether to predict actions.
    """
    super().__init__()
    self.input_dim = state_dim
    self.embedding_dim = embedding_dim
    self.sequence_length = sequence_length
    self.ctx_length = ctx_length
    self.attention_length = ctx_length or sequence_length
    if ctx_length:
      assert (ctx_length == sequence_length - 1)
    self.downstream_input_mode = downstream_input_mode
    self.drop_probability = drop_probability
    self.switch_probability = switch_probability
    self.keep_probability = keep_probability
    self.input_dimension_dropout = input_dimension_dropout
    self.modify_actions = modify_actions
    self.embed_on_input = embed_on_input
    self.predict_actions = predict_actions

    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=num_distributions,
        hidden_dims=hidden_dims)

    attention_input_dim = (
        self.attention_length + (0 if self.predict_actions else action_dim) +
        (self.embedding_dim if self.embed_on_input else state_dim))
    preprocess = tf.keras.layers.Dense(preprocess_dim, activation=tf.nn.relu)
    attention_output_dim = action_dim if self.predict_actions else self.embedding_dim
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads, key_dim=128,
        output_shape=(attention_output_dim,))

    attention_in = tf.keras.Input(
        shape=(self.attention_length, attention_input_dim))
    preprocessed_attention_in = preprocess(attention_in)
    attention_out = attention(preprocessed_attention_in, preprocessed_attention_in)
    self.transformer = tf.keras.Model(inputs=attention_in, outputs=attention_out)
    self.ctx_embedder = self.transformer if self.ctx_length else None
    if self.ctx_embedder:
      self.ctx_embedder.embedding_dim = attention_output_dim

    missing_x_dim = ((action_dim if self.modify_actions else 0) +
                     (self.embedding_dim if self.embed_on_input else state_dim))
    self.missing_x = tf.Variable(tf.zeros([missing_x_dim]))

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.all_variables = self.variables

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of potentially sequenced states.
      actions: A batch of potentially sequenced actions.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    if not self.ctx_embedder:
      assert (len(states.shape) == 2)
      return self.embedder(states, stop_gradient=stop_gradient)

    assert len(states.shape) == 3
    assert (actions is not None)

    outputs = []
    for mode in self.downstream_input_mode.split('-'):
      if mode == 'state':
        outputs.append(states[:, self.attention_length, :])
      elif mode == 'embed':
        outputs.append(
            self.embedder(
                states[:, self.attention_length, :],
                stop_gradient=stop_gradient))
      elif mode == 'ctx':
        embedding = tf.reshape(states[:, :self.attention_length, :],
                               [-1, tf.shape(states)[-1]])
        embedding = self.embedder(embedding, stop_gradient=stop_gradient)
        embedding = tf.reshape(
            embedding, [-1, self.attention_length, self.embedder.embedding_dim])
        states_in = embedding if self.embed_on_input else states[:, :self.
                                                                 attention_length, :]
        actions_in = actions[:, :self.ctx_length, :]

        time_encoding = tf.eye(
            self.attention_length, batch_shape=(tf.shape(states_in)[0],))
        attention_in = tf.concat([time_encoding, states_in, actions_in], -1)
        attention_out = self.ctx_embedder(attention_in, training=False)
        if stop_gradient:
          attention_out = tf.stop_gradient(attention_out)

        pred_embeddings = tf.reduce_max(attention_out, axis=1)
        outputs.append(pred_embeddings)
    outputs = tf.concat(outputs, axis=-1)
    return outputs

  def _prepare_input(self, x):
    """Prepares input for BERT training."""
    batch_size = tf.shape(x)[0]

    # Deal with probability of overlap in masks.
    keep_probability = self.keep_probability
    switch_probability = self.switch_probability / (1 - keep_probability)
    drop_probability = self.drop_probability / (1 - switch_probability) / (1 - keep_probability)
    drop = tf.random.uniform([batch_size, self.attention_length
                             ]) < drop_probability
    switch = tf.random.uniform([batch_size, self.attention_length
                               ]) < switch_probability
    keep = tf.random.uniform([batch_size, self.attention_length
                             ]) < keep_probability

    drop_mask = tf.cast(drop, tf.float32)
    masked_x = x * (1.0 - drop_mask)[Ellipsis, None] + self.missing_x * drop_mask[Ellipsis, None]

    shuffled_x = tf.concat([masked_x[1:], masked_x[:1]], 0)
    switch_mask = tf.cast(switch, tf.float32)
    masked_x = masked_x * (1.0 - switch_mask)[Ellipsis, None] + shuffled_x * switch_mask[Ellipsis, None]

    time_encoding = tf.eye(self.attention_length, batch_shape=(batch_size,))
    prepared_x = tf.concat([time_encoding, masked_x], -1)

    full_mask = tf.cast(drop | switch | keep, tf.float32)

    return prepared_x, full_mask

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes matrix of energies between every pair of (embedding, other_embedding)."""
    energies = tf.matmul(embeddings, other_embeddings, transpose_b=True)
    return energies

  def fit(self, states, actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    states = states[:, :self.attention_length, :]
    actions = actions[:, :self.attention_length, :]
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.attention_length, -1])
      all_embeddings = self.embedder(all_states, stop_gradient=False)
      embeddings = tf.reshape(
          all_embeddings,
          [batch_size, self.attention_length, self.embedding_dim])

      states_in = embeddings if self.embed_on_input else states
      actions_in = actions
      if self.input_dimension_dropout > 0:
        states_in *= tf.cast(
            tf.random.uniform(tf.shape(states_in)) < self.input_dimension_dropout,
            tf.float32)
        states_in *= 1 / (1 - self.input_dimension_dropout)
        actions_in *= tf.cast(
            tf.random.uniform(tf.shape(actions_in)) < self.input_dimension_dropout,
            tf.float32)
        actions_in *= 1 / (1 - self.input_dimension_dropout)
      if self.predict_actions:
        states_in, drop_mask = self._prepare_input(states_in)
        attention_in = states_in
      elif self.modify_actions:
        attention_in = tf.concat([states_in, actions_in], -1)
        attention_in, drop_mask = self._prepare_input(attention_in)
      else:
        states_in, drop_mask = self._prepare_input(states_in)
        attention_in = tf.concat([states_in, actions_in], -1)
      attention_out = self.transformer(attention_in, training=True)

      mask_indices = tf.where(drop_mask > 0.0)
      if self.predict_actions:
        pred_actions = tf.gather_nd(attention_out, mask_indices)
        true_actions = tf.gather_nd(actions, mask_indices)
        loss = tf.reduce_mean(tf.reduce_sum((pred_actions - true_actions) ** 2, -1))
      else:
        pred_embeddings = tf.gather_nd(attention_out, mask_indices)
        true_embeddings = tf.gather_nd(embeddings, mask_indices)

        energies = self.compute_energy(pred_embeddings, true_embeddings)
        positive_loss = tf.linalg.diag_part(energies)
        negative_loss = tf.reduce_logsumexp(energies, axis=-1)

        loss = tf.reduce_mean(-positive_loss + negative_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)

  def get_input_state_dim(self):
    if not self.ctx_embedder:
      return self.embedder.embedding_dim

    input_state_dim = 0
    for mode in self.downstream_input_mode.split('-'):
      if mode == 'state':
        input_state_dim += self.input_dim
      elif mode == 'embed':
        input_state_dim += self.embedder.embedding_dim
      elif mode == 'ctx':
        input_state_dim += self.ctx_embedder.embedding_dim
    return input_state_dim


class Bert2Learner(tf.keras.Model):
  """WORK IN PROGRESS.

    A learner for BERT, applied to individual dimensions of state.
  """

  def __init__(self,
               state_dim,
               action_dim,
               embedding_dim = 256,
               num_distributions = None,
               component_embedding_dim = None,
               sequence_length = 2,
               ctx_length = None,
               downstream_input_mode = 'embed',
               learning_rate = None,
               num_heads = 4,
               drop_probability = 0.15,
               switch_probability = 0.05,
               keep_probability = 0.05):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      sequence_length: Expected length of sequences provided as input
      ctx_length: Number of past steps to compute a context.
      downstream_input_mode: Whether to use states, embedding, or context.
      learning_rate: Learning rate.
      num_heads: Number of heads for transformer.
      drop_probability: Drop probability for input.
      switch_probability: Switch probability for input.
      keep_probability: Keep probability for input.
    """
    super().__init__()
    self.input_dim = state_dim
    self.state_dim = state_dim
    self.component_embedding_dim = component_embedding_dim or embedding_dim
    self.embedding_dim = embedding_dim
    self.sequence_length = sequence_length
    self.ctx_length = ctx_length
    if ctx_length:
      assert (ctx_length == sequence_length - 1)
    self.attention_length = ctx_length or sequence_length
    self.downstream_input_mode = downstream_input_mode
    self.drop_probability = drop_probability
    self.switch_probability = switch_probability
    self.keep_probability = keep_probability

    # Embedder is transformer on individual dimensions of state.
    self.embedder = TransformerNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=num_distributions,
        input_embedding_dim=self.component_embedding_dim)

    self.decoder = EmbedNet(
        action_dim * (self.attention_length - 1) +
        self.embedding_dim * self.attention_length,
        self.attention_length * self.component_embedding_dim * state_dim)

    self.ctx_embedder = self.embedder if self.ctx_length else None

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.all_variables = self.variables

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of potentially sequenced states.
      actions: A batch of potentially sequenced actions.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    if not self.ctx_embedder:
      assert (len(states.shape) == 2)
      return self.embedder(states, stop_gradient=stop_gradient)

    assert (actions is not None)
    batch_size = tf.shape(states)[0]
    outputs = []
    for mode in self.downstream_input_mode.split('-'):
      if mode == 'state':
        outputs.append(states[:, self.attention_length, :])
      elif mode == 'embed':
        outputs.append(
            self.embedder(
                states[:, self.attention_length, :],
                stop_gradient=stop_gradient))
      elif mode == 'ctx':
        all_states = tf.reshape(states[:, self.attention_length, :],
                                [batch_size * self.attention_length, -1])
        all_component_embeddings = self.embedder.process_inputs(
            all_states, stop_gradient=False)
        all_component_embeddings = tf.reshape(all_component_embeddings, [
            batch_size * self.attention_length * self.state_dim,
            self.component_embedding_dim
        ])
        all_embeddings = self.embedder(all_states, stop_gradient=False)
        embeddings = tf.reshape(
            all_embeddings,
            [batch_size, self.attention_length, self.embedding_dim])
        embeddings = tf.reduce_max(embeddings, axis=1)
        if stop_gradient:
          attention_out = tf.stop_gradient(embeddings)
        outputs.append(embeddings)
    outputs = tf.concat(outputs, axis=-1)
    return outputs

  def _prepare_input(self, x):
    """Prepares input for BERT training."""
    batch_size = tf.shape(x)[0]

    # Deal with probability of overlap in masks.
    keep_probability = self.keep_probability
    switch_probability = self.switch_probability / (1 - keep_probability)
    drop_probability = self.drop_probability / (1 - switch_probability) / (1 - keep_probability)
    drop = tf.random.uniform(tf.shape(x)) < drop_probability
    switch = tf.random.uniform(tf.shape(x)) < switch_probability
    keep = tf.random.uniform(tf.shape(x)) < keep_probability

    drop_mask = tf.cast(drop, tf.float32)
    masked_x = x * (1.0 - drop_mask)

    shuffled_x = tf.concat([masked_x[1:], masked_x[:1]], 0)
    switch_mask = tf.cast(switch, tf.float32)
    masked_x = masked_x * (1.0 - switch_mask) + shuffled_x * switch_mask

    full_mask = tf.cast(drop | switch | keep, tf.float32)

    return masked_x, full_mask, drop

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes matrix of energies between every pair of (embedding, other_embedding)."""
    energies = tf.matmul(embeddings, other_embeddings, transpose_b=True)
    return energies

  def fit(self, states, actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    states = states[:, :self.attention_length, :]
    actions = actions[:, :self.attention_length, :]
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.attention_length, -1])
      all_component_embeddings = self.embedder.process_inputs(all_states, stop_gradient=False)
      all_component_embeddings = tf.reshape(all_component_embeddings, [
          batch_size * self.attention_length * self.state_dim,
          self.component_embedding_dim
      ])

      modified_all_states, mask, missing = self._prepare_input(all_states)

      all_embeddings = self.embedder(modified_all_states,
                                     stop_gradient=False,
                                     missing_mask=missing)
      embeddings = tf.reshape(
          all_embeddings,
          [batch_size, self.attention_length, self.embedding_dim])
      attention_in = tf.concat([
          tf.reshape(embeddings,
                     [batch_size, self.attention_length * self.embedding_dim]),
          tf.reshape(actions[:, :self.attention_length - 1, :],
                     [batch_size, -1])
      ], -1)
      attention_out = self.decoder(attention_in, stop_gradient=False)
      attention_out = tf.reshape(attention_out, [
          batch_size * self.attention_length * self.state_dim,
          self.component_embedding_dim
      ])

      mask_indices = tf.where(
          tf.reshape(mask,
                     [batch_size * self.attention_length *
                      self.state_dim]) > 0.0)
      pred_embeddings = tf.gather_nd(attention_out, mask_indices)
      true_embeddings = tf.gather_nd(all_component_embeddings, mask_indices)

      energies = self.compute_energy(pred_embeddings, true_embeddings)
      positive_loss = tf.linalg.diag_part(energies)
      negative_loss = tf.reduce_logsumexp(energies, axis=-1)

      loss = tf.reduce_mean(-positive_loss + negative_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)

  def get_input_state_dim(self):
    if not self.ctx_embedder:
      return self.embedder.embedding_dim

    input_state_dim = 0
    for mode in self.downstream_input_mode.split('-'):
      if mode == 'state':
        input_state_dim += self.input_dim
      elif mode == 'embed':
        input_state_dim += self.embedder.embedding_dim
      elif mode == 'ctx':
        input_state_dim += self.ctx_embedder.embedding_dim
    return input_state_dim


class ACLLearner(tf.keras.Model):
  """Attentive contrastive learner."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               preprocess_dim = 256,
               hidden_dims = (256, 256),
               sequence_length = 2,
               ctx_length = None,
               downstream_input_mode = 'embed',
               learning_rate = None,
               num_heads = 4,
               drop_probability = 0.3,
               switch_probability = 0.15,
               keep_probability = 0.15,
               input_dimension_dropout = 0.0,
               input_actions = True,
               predict_actions = True,
               policy_decoder_on_embeddings = False,
               input_rewards = False,
               predict_rewards = False,
               reward_decoder_on_embeddings = False,
               embed_on_input = True,
               extra_embedder = True,
               positional_encoding_type = 'identity',
               direction = 'backward'):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      preprocess_dim: Dimension of input to transformer.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      ctx_length: Number of past steps to compute a context.
      downstream_input_mode: Whether to use states, embedding, or context.
      learning_rate: Learning rate.
      num_heads: Number of heads for transformer.
      drop_probability: Drop probability for input.
      switch_probability: Switch probability for input.
      keep_probability: Keep probability for input.
      input_dimension_dropout: Dropout probability on state inputs.
      input_actions: Whether to input actions to the transformer.
      predict_actions: Whether to predict actions.
      policy_decoder_on_embeddings: Whether to decode policy from
        state embeddings or transformer output.
      input_rewards: Whether to input rewards to the transformer.
      predict_rewards: Whether to predict rewards.
      reward_decoder_on_embeddings: Whether to decode reward from
        state embeddings or transformer output.
      embed_on_input: Whether to pass embedding or raw state to transformer.
      extra_embedder: Whether to use an extra embedder on input states.
      positional_encoding_type: One of [None, 'identity', 'sinusoid']
      direction: Direction of prediction.
    """
    super().__init__()
    self.input_dim = state_dim
    self.action_dim = action_spec.shape[0]
    self.embedding_dim = embedding_dim
    self.output_dim = self.embedding_dim
    self.num_distributions = num_distributions
    self.sequence_length = sequence_length
    self.ctx_length = ctx_length
    self.attention_length = ctx_length or sequence_length
    self.downstream_input_mode = downstream_input_mode
    self.drop_probability = drop_probability
    self.switch_probability = switch_probability
    self.keep_probability = keep_probability
    self.input_dimension_dropout = input_dimension_dropout
    self.input_actions = input_actions
    self.predict_actions = predict_actions
    self.policy_decoder_on_embeddings = policy_decoder_on_embeddings
    self.input_rewards = input_rewards
    self.predict_rewards = predict_rewards
    self.reward_decoder_on_embeddings = reward_decoder_on_embeddings
    self.embed_on_input = embed_on_input
    self.positional_encoding_type = positional_encoding_type

    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=num_distributions,
        hidden_dims=hidden_dims)

    if extra_embedder:
      self.extra_embedder = EmbedNet(
          state_dim, embedding_dim=self.output_dim, hidden_dims=hidden_dims)
    else:
      self.extra_embedder = None

    self.policy_decoder = policies.DiagGuassianPolicy(
        self.embedding_dim
        if self.policy_decoder_on_embeddings else self.output_dim,
        action_spec,
        apply_tanh_squash=True)
    self.reward_decoder = keras_utils.create_mlp(
        self.embedding_dim,
        1,
        hidden_dims=hidden_dims,
    )

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.target_entropy = -action_spec.shape[0]

    attention_input_dim = embedding_dim if embed_on_input else state_dim
    attention_output_dim = self.output_dim
    if input_actions:
      attention_input_dim += self.action_dim
    if predict_actions:
      attention_output_dim += self.output_dim
    if input_rewards:
      attention_input_dim += 1
    if predict_rewards:
      attention_output_dim += 1
    if positional_encoding_type == 'identity':
      attention_input_dim += self.attention_length
    self.attention_input_dim = attention_input_dim

    preprocess = tf.keras.layers.Dense(preprocess_dim, activation=tf.nn.relu)

    attention_in = tf.keras.Input(
        shape=(self.attention_length, attention_input_dim))
    preprocessed_attention_in = preprocess(attention_in)

    if direction == 'bidirectional':
      attention_mask = None
    else:
      # Create a look-ahead mask, e.g.,
      # [[0., 1., 1.],
      #  [0., 0., 1.],
      #  [0., 0., 0.]]
      attention_mask = tf.linalg.band_part(
          tf.ones((self.attention_length, self.attention_length)), -1, 0)[None,
                                                                          Ellipsis]
      attention_mask = attention_mask - tf.eye(self.attention_length)
      if direction == 'backward':
        attention_mask = 1 - attention_mask
    attention_out = transformer(
        preprocessed_attention_in,
        num_layers=1,
        embedding_dim=preprocess_dim,
        num_heads=num_heads,
        key_dim=128,
        ff_dim=preprocess_dim,
        output_dim=attention_output_dim,
        attention_mask=attention_mask)
    self.transformer = tf.keras.Model(
        inputs=attention_in, outputs=attention_out)
    self.ctx_embedder = self.transformer if self.ctx_length else None
    if self.ctx_embedder:
      self.ctx_embedder.embedding_dim = attention_output_dim

    self.missing_state = tf.Variable(
        tf.zeros([self.embedding_dim if self.embed_on_input else state_dim]))
    self.missing_action = tf.Variable(tf.zeros([self.action_dim]))
    self.missing_reward = tf.Variable(tf.zeros([1]))

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.all_variables = self.variables

  @tf.function
  def call(self,
           states,
           actions = None,
           rewards = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of potentially sequenced states.
      actions: A batch of potentially sequenced actions.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    if not self.ctx_embedder:
      return self.embedder(states, stop_gradient=stop_gradient)

    assert len(states.shape) == 3

    if self.embed_on_input:
      attention_in = tf.reshape(
          self.embedder(
              tf.reshape(states, [-1, tf.shape(states)[-1]]),
              stop_gradient=stop_gradient),
          [tf.shape(states)[0], -1, self.embedding_dim])
    else:
      attention_in = states
    if self.input_actions:
      if tf.shape(actions)[1] == tf.shape(states)[1] - 1:
        mask = tf.concat([
            tf.ones(tf.shape(actions)),
            tf.zeros([tf.shape(actions)[0], 1,
                      tf.shape(actions)[-1]],
                     dtype=tf.float32)
        ],
                         axis=1)
        actions = tf.concat(
            [actions, tf.zeros(tf.shape(actions[:, -1:, :]))], axis=1)
        actions = mask * actions + (1. - mask) * self.missing_action
      attention_in = tf.concat([attention_in, actions], axis=-1)
    if self.input_rewards:
      rewards = rewards[Ellipsis, None]
      if tf.shape(rewards)[1] == tf.shape(states)[1] - 1:
        mask = tf.concat([
            tf.ones([tf.shape(rewards)[0],
                     tf.shape(rewards)[1], 1],
                    dtype=tf.float32),
            tf.zeros([tf.shape(rewards)[0], 1, 1], dtype=tf.float32)
        ],
                         axis=1)
        rewards = tf.concat(
            [rewards, tf.zeros(tf.shape(rewards[:, -1:, :]))], axis=1)
        rewards = mask * rewards + (1. - mask) * self.missing_reward
      attention_in = tf.concat([attention_in, rewards], axis=-1)
    attention_in = self.add_positional_encoding(attention_in)
    attention_out = self.ctx_embedder(attention_in, training=False)
    if stop_gradient:
      attention_out = tf.stop_gradient(attention_out)

    return attention_out[:, -1, :self.embedding_dim]

  def _prepare_input(self, x, missing_x):
    """Prepares input for BERT training."""
    batch_size = tf.shape(x)[0]

    # Deal with probability of overlap in masks.
    keep_probability = self.keep_probability
    switch_probability = self.switch_probability / (1 - keep_probability)
    drop_probability = self.drop_probability / (1 - switch_probability) / (
        1 - keep_probability)
    drop = tf.random.uniform([batch_size, self.attention_length
                             ]) < drop_probability
    switch = tf.random.uniform([batch_size, self.attention_length
                               ]) < switch_probability
    keep = tf.random.uniform([batch_size, self.attention_length
                             ]) < keep_probability

    drop_mask = tf.cast(drop, tf.float32)
    masked_x = x * (1.0 - drop_mask)[Ellipsis, None] + missing_x * drop_mask[Ellipsis,
                                                                        None]

    shuffled_x = tf.concat([masked_x[1:], masked_x[:1]], 0)
    switch_mask = tf.cast(switch, tf.float32)
    masked_x = masked_x * (
        1.0 - switch_mask)[Ellipsis, None] + shuffled_x * switch_mask[Ellipsis, None]

    full_mask = tf.cast(drop | switch | keep, tf.float32)

    return masked_x, full_mask

  def add_positional_encoding(self, x):
    if self.positional_encoding_type == 'identity':
      time_encoding = tf.eye(
          self.attention_length, batch_shape=(tf.shape(x)[0],))
      x = tf.concat([time_encoding, x], -1)
    elif self.positional_encoding_type == 'sinusoid':

      def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

      angle_rads = get_angles(
          np.arange(self.attention_length)[:, np.newaxis],
          np.arange(self.attention_input_dim)[np.newaxis, :],
          self.attention_input_dim)
      # apply sin to even indices in the array; 2i
      angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
      # apply cos to odd indices in the array; 2i+1
      angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
      pos_encoding = angle_rads[np.newaxis, Ellipsis]
      x += pos_encoding
    elif self.positional_encoding_type == 'zero':
      x = x
    else:
      raise NotImplementedError
    return x

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes matrix of energies between every pair of (embedding, other_embedding)."""
    energies = tf.matmul(embeddings, other_embeddings, transpose_b=True)
    return energies

  def fit(self, states, actions,
          rewards):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.
      rewards: Batch of sequences of rewards.

    Returns:
      Dictionary with information to track.
    """
    states = states[:, :self.attention_length, :]
    actions = actions[:, :self.attention_length, :]
    rewards = rewards[:, :self.attention_length, None]
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.attention_length, -1])
      all_embeddings = self.embedder(all_states, stop_gradient=False)
      embeddings = tf.reshape(
          all_embeddings,
          [batch_size, self.attention_length, self.embedding_dim])

      states_in = embeddings if self.embed_on_input else states
      actions_in = actions
      rewards_in = rewards
      if self.input_dimension_dropout > 0:
        states_in *= tf.cast(
            tf.random.uniform(tf.shape(states_in)) <
            self.input_dimension_dropout, tf.float32)
        states_in *= 1 / (1 - self.input_dimension_dropout)
        actions_in *= tf.cast(
            tf.random.uniform(tf.shape(actions_in)) <
            self.input_dimension_dropout, tf.float32)
        actions_in *= 1 / (1 - self.input_dimension_dropout)
        rewards_in *= tf.cast(
            tf.random.uniform(tf.shape(rewards_in)) <
            self.input_dimension_dropout, tf.float32)
        rewards_in *= 1 / (1 - self.input_dimension_dropout)

      states_in, states_mask = self._prepare_input(states_in,
                                                   self.missing_state)
      actions_in, actions_mask = self._prepare_input(actions_in,
                                                     self.missing_action)
      rewards_in, rewards_mask = self._prepare_input(rewards_in,
                                                     self.missing_reward)

      attention_in = [states_in]
      if self.input_actions:
        attention_in.append(actions_in)
      if self.input_rewards:
        attention_in.append(rewards_in)
      attention_in = tf.concat(attention_in, -1)
      attention_in = self.add_positional_encoding(attention_in)

      attention_out = self.transformer(attention_in, training=True)

      # State prediction loss.
      states_mask_indices = tf.where(states_mask > 0.0)
      pred_embeddings = tf.gather_nd(attention_out[Ellipsis, :self.output_dim],
                                     states_mask_indices)
      if self.extra_embedder:
        true_states = tf.gather_nd(states, states_mask_indices)
        true_embeddings = self.extra_embedder(true_states, stop_gradient=False)
      else:
        true_embeddings = tf.gather_nd(embeddings, states_mask_indices)

      energies = self.compute_energy(pred_embeddings, true_embeddings)
      positive_loss = tf.linalg.diag_part(energies)
      negative_loss = tf.reduce_logsumexp(energies, axis=-1)
      state_loss = -positive_loss + negative_loss
      correct = tf.cast(positive_loss >= tf.reduce_max(energies, axis=-1),
                        tf.float32)

      if self.predict_actions or self.policy_decoder_on_embeddings:
        if self.policy_decoder_on_embeddings:
          policy_decoder_in = all_embeddings
          all_actions = tf.reshape(
              actions, [batch_size * self.attention_length, self.action_dim])
        else:
          actions_mask_indices = tf.where(actions_mask > 0.0)
          idx = -1 if self.predict_rewards else tf.shape(attention_out)[-1]
          policy_decoder_in = tf.gather_nd(
              attention_out[Ellipsis, self.output_dim:idx], actions_mask_indices)
          all_actions = tf.gather_nd(actions, actions_mask_indices)

        action_log_probs = self.policy_decoder.log_probs(
            policy_decoder_in, all_actions)
        _, policy_log_probs = self.policy_decoder(
            policy_decoder_in, sample=True, with_log_probs=True)

        alpha = tf.exp(self.log_alpha)
        alpha_loss = alpha * tf.stop_gradient(-policy_log_probs -
                                              self.target_entropy)
        reconstruct_loss = -action_log_probs + tf.stop_gradient(
            alpha) * policy_log_probs
        action_loss = alpha_loss + reconstruct_loss
      else:
        action_loss = 0.0

      if self.predict_rewards or self.reward_decoder_on_embeddings:
        if self.reward_decoder_on_embeddings:
          reward_decoder_in = all_embeddings
          pred_reward = self.reward_decoder(reward_decoder_in)
          pred_reward = tf.reshape(pred_reward,
                                   [batch_size, self.attention_length, 1])
          pred_reward = tf.gather(pred_reward, tf.where(rewards_mask > 0.0))
        else:
          pred_reward = tf.gather(attention_out[Ellipsis, -1:],
                                  tf.where(rewards_mask > 0.0))
        true_reward = tf.gather(rewards, tf.where(rewards_mask > 0.0))
        reward_loss = huber(pred_reward - true_reward)
      else:
        reward_loss = 0.0

      loss = tf.reduce_mean(state_loss) + tf.reduce_mean(
          action_loss) + tf.reduce_mean(reward_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(zip(grads, self.all_variables))

    return {
        'embed_loss': loss,
        'positive_loss': tf.reduce_mean(positive_loss),
        'negative_loss': tf.reduce_mean(negative_loss),
        'state_loss': tf.reduce_mean(state_loss),
        'state_correct': tf.reduce_mean(correct),
        'action_loss': tf.reduce_mean(action_loss),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, rewards, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions, rewards)

  def get_input_state_dim(self):
    return self.embedder.embedding_dim


class MomentumACLLearner(ACLLearner):
  """Extension of ACLLearner."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               preprocess_dim = 256,
               hidden_dims = (256, 256),
               sequence_length = 2,
               ctx_length = None,
               downstream_input_mode = 'embed',
               learning_rate = None,
               num_heads = 4,
               drop_probability = 0.3,
               switch_probability = 0.15,
               keep_probability = 0.15,
               input_dimension_dropout = 0.0,
               input_actions = True,
               predict_actions = True,
               policy_decoder_on_embeddings = False,
               input_rewards = True,
               predict_rewards = False,
               reward_decoder_on_embeddings = False,
               embed_on_input = True,
               extra_embedder = True,
               positional_encoding_type = 'identity',
               direction = 'backward',
               residual_dims = (256,),
               tau = 0.05,
               target_update_period = 1):

    super().__init__(
        state_dim,
        action_spec,
        embedding_dim=embedding_dim,
        num_distributions=num_distributions,
        preprocess_dim=preprocess_dim,
        hidden_dims=hidden_dims,
        sequence_length=sequence_length,
        ctx_length=ctx_length,
        downstream_input_mode=downstream_input_mode,
        learning_rate=learning_rate,
        num_heads=num_heads,
        drop_probability=drop_probability,
        switch_probability=switch_probability,
        keep_probability=keep_probability,
        input_dimension_dropout=input_dimension_dropout,
        input_actions=input_actions,
        predict_actions=predict_actions,
        policy_decoder_on_embeddings=policy_decoder_on_embeddings,
        input_rewards=input_rewards,
        predict_rewards=predict_rewards,
        reward_decoder_on_embeddings=reward_decoder_on_embeddings,
        embed_on_input=embed_on_input,
        extra_embedder=extra_embedder,
        positional_encoding_type=positional_encoding_type,
        direction=direction)

    self.residual_mlp = EmbedNet(
        embedding_dim, embedding_dim=embedding_dim, hidden_dims=residual_dims)
    self.embedder_target = EmbedNet(
        state_dim, embedding_dim=self.embedding_dim, hidden_dims=hidden_dims)
    soft_update(self.embedder, self.embedder_target, tau=1.0)

    learning_rate = learning_rate or 3e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.tau = tau
    self.target_update_period = target_update_period

    self.all_variables += self.residual_mlp.variables

  def fit(self, states, actions,
          rewards):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.
      rewards: Batch of sequences of rewards.

    Returns:
      Dictionary with information to track.
    """
    states = states[:, :self.attention_length, :]
    actions = actions[:, :self.attention_length, :]
    rewards = rewards[:, :self.attention_length, None]
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.attention_length, -1])
      all_embeddings = self.embedder(all_states, stop_gradient=False)
      all_embeddings += self.residual_mlp(all_embeddings, stop_gradient=False)
      embeddings = tf.reshape(
          all_embeddings,
          [batch_size, self.attention_length, self.embedding_dim])

      states_in = embeddings if self.embed_on_input else states
      actions_in = actions
      rewards_in = rewards
      if self.input_dimension_dropout > 0:
        states_in *= tf.cast(
            tf.random.uniform(tf.shape(states_in)) <
            self.input_dimension_dropout, tf.float32)
        states_in *= 1 / (1 - self.input_dimension_dropout)
        actions_in *= tf.cast(
            tf.random.uniform(tf.shape(actions_in)) <
            self.input_dimension_dropout, tf.float32)
        actions_in *= 1 / (1 - self.input_dimension_dropout)
        rewards_in *= tf.cast(
            tf.random.uniform(tf.shape(rewards_in)) <
            self.input_dimension_dropout, tf.float32)
        rewards_in *= 1 / (1 - self.input_dimension_dropout)

      states_in, states_mask = self._prepare_input(states_in,
                                                   self.missing_state)
      actions_in, actions_mask = self._prepare_input(actions_in,
                                                     self.missing_action)
      rewards_in, rewards_mask = self._prepare_input(rewards_in,
                                                     self.missing_reward)

      attention_in = [states_in]
      if self.input_actions:
        attention_in.append(actions_in)
      if self.input_rewards:
        attention_in.append(rewards_in)
      attention_in = tf.concat(attention_in, -1)
      attention_in = self.add_positional_encoding(attention_in)

      attention_out = self.transformer(attention_in, training=True)

      # State prediction loss.
      states_mask_indices = tf.where(states_mask > 0.0)
      pred_embeddings = tf.gather_nd(attention_out[Ellipsis, :self.output_dim],
                                     states_mask_indices)
      if self.extra_embedder:
        true_states = tf.gather_nd(states, states_mask_indices)
        true_embeddings = self.extra_embedder(true_states, stop_gradient=False)
      else:
        true_embeddings = tf.gather_nd(embeddings, states_mask_indices)

      true_embeddings = self.embedder_target(all_states, stop_gradient=True)
      true_embeddings = tf.reshape(
          true_embeddings,
          [batch_size, self.attention_length, self.embedding_dim])
      true_embeddings = tf.gather_nd(true_embeddings, states_mask_indices)

      energies = self.compute_energy(pred_embeddings, true_embeddings)
      positive_loss = tf.linalg.diag_part(energies)
      negative_loss = tf.reduce_logsumexp(energies, axis=-1)
      state_loss = -positive_loss + negative_loss
      correct = tf.cast(positive_loss >= tf.reduce_max(energies, axis=-1),
                        tf.float32)

      if self.predict_actions or self.policy_decoder_on_embeddings:
        if self.policy_decoder_on_embeddings:
          policy_decoder_in = all_embeddings
          all_actions = tf.reshape(
              actions, [batch_size * self.attention_length, self.action_dim])
        else:
          actions_mask_indices = tf.where(actions_mask > 0.0)
          idx = -1 if self.predict_rewards else tf.shape(attention_out)[-1]
          policy_decoder_in = tf.gather_nd(
              attention_out[Ellipsis, self.output_dim:idx], actions_mask_indices)
          all_actions = tf.gather_nd(actions, actions_mask_indices)

        action_log_probs = self.policy_decoder.log_probs(
            policy_decoder_in, all_actions)
        _, policy_log_probs = self.policy_decoder(
            policy_decoder_in, sample=True, with_log_probs=True)

        alpha = tf.exp(self.log_alpha)
        alpha_loss = alpha * tf.stop_gradient(-policy_log_probs -
                                              self.target_entropy)
        reconstruct_loss = -action_log_probs + tf.stop_gradient(
            alpha) * policy_log_probs
        action_loss = alpha_loss + reconstruct_loss
      else:
        action_loss = 0.0

      if self.predict_rewards or self.reward_decoder_on_embeddings:
        if self.reward_decoder_on_embeddings:
          reward_decoder_in = all_embeddings
          pred_reward = self.reward_decoder(reward_decoder_in)
          pred_reward = tf.reshape(pred_reward,
                                   [batch_size, self.attention_length, 1])
          pred_reward = tf.gather(pred_reward, tf.where(rewards_mask > 0.0))
        else:
          pred_reward = tf.gather(attention_out[Ellipsis, -1:],
                                  tf.where(rewards_mask > 0.0))
        true_reward = tf.gather(rewards, tf.where(rewards_mask > 0.0))
        reward_loss = huber(pred_reward - true_reward)
      else:
        reward_loss = 0.0

      loss = tf.reduce_mean(state_loss) + tf.reduce_mean(
          action_loss) + tf.reduce_mean(reward_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(zip(grads, self.all_variables))

    if self.optimizer.iterations % self.target_update_period == 0:
      soft_update(self.embedder, self.embedder_target, tau=self.tau)

    return {
        'embed_loss': loss,
        'positive_loss': tf.reduce_mean(positive_loss),
        'negative_loss': tf.reduce_mean(negative_loss),
        'state_loss': tf.reduce_mean(state_loss),
        'state_correct': tf.reduce_mean(correct),
        'action_loss': tf.reduce_mean(action_loss),
    }


class VpnLearner(tf.keras.Model):
  """A learner for value prediction network."""

  def __init__(self,
               state_dim,
               action_dim,
               embedding_dim = 256,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None,
               discount = 0.95,
               tau = 1.0,
               target_update_period = 1000):
    """Creates networks.

    Args:
      state_dim: State size.
      action_dim: Action size.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
    """
    super().__init__()
    self.input_dim = state_dim
    self.embedding_dim = embedding_dim
    self.sequence_length = sequence_length
    self.discount = discount
    self.tau = tau
    self.target_update_period = target_update_period

    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        hidden_dims=hidden_dims)
    self.f_value = keras_utils.create_mlp(
        self.embedding_dim, 1, hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_value_target = keras_utils.create_mlp(
        self.embedding_dim, 1, hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_trans = keras_utils.create_mlp(
        self.embedding_dim + action_dim, self.embedding_dim,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish)
    self.f_out = keras_utils.create_mlp(
        self.embedding_dim + action_dim, 2,
        hidden_dims=hidden_dims,
        activation=tf.nn.swish)

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.all_variables = self.variables
    soft_update(self.f_value, self.f_value_target, tau=1.0)

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: 2 or 3 dimensional state tensors.
      downstream_input_mode: mode of downstream inputs, e.g., state-ctx.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    assert (len(states.shape) == 2)
    return self.embedder(states, stop_gradient=stop_gradient)

  def fit(self, states, actions,
          rewards, discounts,
          next_states):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.
      rewards: Batch of sequences of rewards.
      next_states: Batch of sequences of next states.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      embeddings = self.embedder(states[:, 0, :], stop_gradient=False)
      all_pred_values = []
      all_pred_rewards = []
      all_pred_discounts = []
      for idx in range(self.sequence_length):
        pred_value = self.f_value(embeddings)[Ellipsis, 0]
        pred_reward, pred_discount = tf.unstack(
            self.f_out(tf.concat([embeddings, actions[:, idx, :]], -1)),
            axis=-1)
        pred_embeddings = embeddings + self.f_trans(
            tf.concat([embeddings, actions[:, idx, :]], -1))

        all_pred_values.append(pred_value)
        all_pred_rewards.append(pred_reward)
        all_pred_discounts.append(pred_discount)

        embeddings = pred_embeddings

      last_value = tf.stop_gradient(self.f_value_target(embeddings)[Ellipsis, 0]) / (1 - self.discount)
      all_true_values = []
      for idx in range(self.sequence_length - 1, -1, -1):
        value = self.discount * discounts[:, idx] * last_value + rewards[:, idx]
        all_true_values.append(value)
        last_value = value
      all_true_values = all_true_values[::-1]

      reward_error = tf.stack(all_pred_rewards, -1) - rewards
      value_error = tf.stack(all_pred_values, -1) - (1 - self.discount) * tf.stack(all_true_values, -1)
      reward_loss = tf.reduce_sum(tf.math.square(reward_error), -1)
      value_loss = tf.reduce_sum(tf.math.square(value_error), -1)

      loss = tf.reduce_mean(reward_loss + value_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))
    if self.optimizer.iterations % self.target_update_period == 0:
      soft_update(self.f_value, self.f_value_target, tau=self.tau)

    return {
        'embed_loss': loss,
        'reward_loss': tf.reduce_mean(reward_loss),
        'value_loss': tf.reduce_mean(value_loss),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, rewards, discounts, next_states = next(replay_buffer_iter)
    return self.fit(states, actions, rewards, discounts, next_states)

  def get_input_state_dim(self):
    return self.embedder.embedding_dim


class DiversePolicyLearner(tf.keras.Model):
  """WORK IN PROGRESS.

    A learner for expressing diverse policies.
  """

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               latent_dim = 64,
               latent_distributions = None,
               sequence_blocks = 1,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None,
               kl_weight = 0.1,
               perturbation_scale = 0.1,
               reg_weight = 0.):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      latent_dim: Dimension of the latent variable.
      latent_distributions: number of categorical distributions
        for the latent variable.
      sequence_blocks: Number of shifts applied to states and actions.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
      kl_weight: Weight on KL regularizer.
      perturbation_scale: Scale of perturbation.
      reg_weight: Weight on discrete embedding regularization.
    """
    super().__init__()
    self.input_dim = state_dim
    self.action_dim = action_spec.shape[0]
    self.action_spec = action_spec
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    self.latent_dim = latent_dim
    self.latent_distributions = latent_distributions
    assert not latent_distributions or latent_dim % latent_distributions == 0
    self.sequence_blocks = sequence_blocks
    self.sequence_length = sequence_length * self.sequence_blocks
    self.kl_weight = kl_weight
    self.reg_weight = reg_weight
    self.perturbation_scale = perturbation_scale

    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=self.num_distributions,
        hidden_dims=hidden_dims)

    policy_encoder_in = tf.keras.Input(
        shape=(self.sequence_length, self.input_dim + self.action_dim))
    preprocess = tf.keras.layers.Dense(256, activation=tf.nn.relu)
    transformer_output_dim = (1 if self.latent_distributions else 2) * self.latent_dim
    transformer_out = transformer(preprocess(policy_encoder_in),
                                     num_layers=1,
                                     embedding_dim=256,
                                     num_heads=4,
                                     key_dim=256,
                                     ff_dim=256,
                                     output_dim=transformer_output_dim)
    policy_encoder_out = tf.reduce_mean(transformer_out, axis=-2)
    self.policy_encoder = tf.keras.Model(
        inputs=policy_encoder_in, outputs=policy_encoder_out)

    self.policy_decoder = policies.DiagGuassianPolicy(
        self.embedding_dim + self.latent_dim,
        action_spec, apply_tanh_squash=True)

    learning_rate = learning_rate or 1e-3
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.target_entropy = -action_spec.shape[0]

    self.all_variables = self.variables
    self.average_embedding = tf.Variable(tf.zeros([self.embedding_dim]),
                                         trainable=False)

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of states.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    return self.embedder(states, stop_gradient=stop_gradient)

  def fit(self, states, actions):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      state_blocks = [states]
      action_blocks = [actions]
      shifted_states = states
      shifted_actions = actions
      for _ in range(self.sequence_blocks - 1):
        shifted_states = tf.concat([shifted_states[1:], shifted_states[:1]], 0)
        shifted_actions = tf.concat([shifted_actions[1:], shifted_actions[:1]], 0)
        state_blocks.append(shifted_states)
        action_blocks.append(shifted_actions)
      states = tf.concat(state_blocks, axis=-2)
      actions = tf.concat(action_blocks, axis=-2)

      noise = (self.perturbation_scale * tf.random.normal(tf.shape(actions)) *
               0.5 * (self.action_spec.maximum - self.action_spec.minimum))
      noisy_actions = tf.clip_by_value(actions + noise,
                                       self.action_spec.minimum + 1e-3,
                                       self.action_spec.maximum - 1e-3)

      policy_encoder_in = tf.concat([states, noisy_actions], -1)
      policy_encoder_out = self.policy_encoder(policy_encoder_in, training=True)
      if self.latent_distributions:
        all_logits = tf.split(policy_encoder_out,
                              num_or_size_splits=self.latent_distributions, axis=-1)
        all_probs = [tf.nn.softmax(logits, -1) for logits in all_logits]
        joined_probs = tf.concat(all_probs, -1)
        all_samples = [tfp.distributions.Categorical(logits=logits).sample()
                       for logits in all_logits]
        all_onehot_samples = [tf.one_hot(samples, self.latent_dim // self.latent_distributions)
                              for samples in all_samples]
        joined_onehot_samples = tf.concat(all_onehot_samples, -1)

        # Straight-through gradients.
        latent_sample = joined_onehot_samples + joined_probs - tf.stop_gradient(joined_probs)

        kl_loss = tf.reduce_sum(
            joined_probs *
            (tf.math.log(float(self.latent_dim // self.latent_distributions)) +
             tf.math.log(1e-6 + joined_probs)), -1)
      else:
        latent_mean, latent_logvar = tf.split(policy_encoder_out, 2, axis=-1)
        latent_sample = (latent_mean + tf.random.normal(tf.shape(latent_mean)) *
                         tf.exp(0.5 * latent_logvar))
        kl_loss = -0.5 * tf.reduce_sum(1.0 + latent_logvar - tf.pow(latent_mean, 2) -
                                       tf.exp(latent_logvar), -1)

      all_states = tf.reshape(states, [batch_size * self.sequence_length, -1])
      all_embed = self.embedder(all_states, stop_gradient=False)
      all_latents = tf.repeat(latent_sample, self.sequence_length, axis=0)
      policy_decoder_in = tf.concat([all_embed, all_latents], -1)
      all_noisy_actions = tf.reshape(noisy_actions, [batch_size * self.sequence_length, -1])
      action_log_probs = self.policy_decoder.log_probs(policy_decoder_in, all_noisy_actions)
      _, policy_log_probs = self.policy_decoder(policy_decoder_in, sample=True, with_log_probs=True)

      alpha = tf.exp(self.log_alpha)
      alpha_loss = alpha * tf.stop_gradient(-policy_log_probs - self.target_entropy)
      reconstruct_loss = -tf.reduce_sum(
          tf.reshape(action_log_probs - tf.stop_gradient(alpha) * policy_log_probs,
                     [batch_size, self.sequence_length]), -1)

      self.average_embedding.assign(0.99 * self.average_embedding +
                                    0.01 * tf.reduce_mean(all_embed, 0))
      if self.num_distributions:
        regularization = tf.reduce_sum(all_embed / (1e-6 + tf.stop_gradient(self.average_embedding)), -1)
        regularization = tf.reduce_sum(tf.reshape(regularization, [batch_size, self.sequence_length]), -1)
        entropy = -tf.reduce_sum(self.average_embedding * tf.math.log(1e-6 + self.average_embedding))
      else:
        regularization = 0.0
        entropy = 0.0

      loss = tf.reduce_mean(reconstruct_loss + self.kl_weight * kl_loss +
                            self.reg_weight * regularization) + tf.reduce_mean(alpha_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss,
        'alpha': alpha,
        'reconstruct_loss': tf.reduce_mean(reconstruct_loss),
        'latent_kl_loss': tf.reduce_mean(kl_loss),
        'regularization': tf.reduce_mean(regularization),
        'entropy': tf.reduce_mean(entropy),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, _, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions)

  def get_input_state_dim(self):
    return self.embedder.embedding_dim


class SuperModelLearner(tf.keras.Model):
  """A learner for model-based representation learning.

  Encompasses forward models, inverse models, as well as latent models like
  DeepMDP.
  """

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None,
               latent_dim = 256,
               reward_weight = 1.0,
               forward_weight = 1.0,  # Predict last state given prev actions/states.
               inverse_weight = 1.0,  # Predict last action given states.
               state_prediction_mode = 'energy'):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      num_distributions: Number of categorical distributions
        for discrete embedding.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
      latent_dim: Dimension of the latent variable.
      reward_weight: Weight on the reward loss.
      forward_weight: Weight on the forward loss.
      inverse_weight: Weight on the inverse loss.
      state_prediction_mode: One of ['latent', 'energy'].
    """
    super().__init__()
    self.input_dim = state_dim
    self.action_dim = action_spec.shape[0]
    self.action_spec = action_spec
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    self.sequence_length = sequence_length
    self.latent_dim = latent_dim
    self.reward_weight = reward_weight
    self.forward_weight = forward_weight
    self.inverse_weight = inverse_weight
    self.state_prediction_mode = state_prediction_mode

    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=self.num_distributions,
        hidden_dims=hidden_dims)

    if self.sequence_length > 2:
      self.latent_embedder = RNNEmbedNet(
          [self.sequence_length - 2, self.embedding_dim + self.action_dim],
          embedding_dim=self.latent_dim)

    self.reward_decoder = EmbedNet(
        self.latent_dim + self.embedding_dim + self.action_dim,
        embedding_dim=1,
        hidden_dims=hidden_dims)

    self.inverse_decoder = policies.DiagGuassianPolicy(
        2 * self.embedding_dim + self.latent_dim,
        action_spec, apply_tanh_squash=True)

    forward_decoder_out = (self.embedding_dim
                           if (self.state_prediction_mode in ['latent', 'energy']) else
                           self.input_dim)
    forward_decoder_dists = (self.num_distributions
                             if (self.state_prediction_mode in ['latent', 'energy']) else
                             None)
    self.forward_decoder = StochasticEmbedNet(
        self.latent_dim + self.embedding_dim + self.action_dim,
        embedding_dim=forward_decoder_out,
        num_distributions=forward_decoder_dists,
        hidden_dims=hidden_dims)

    self.weight = tf.Variable(tf.eye(self.embedding_dim))

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.target_entropy = -action_spec.shape[0]

    self.all_variables = self.variables

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of states.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    return self.embedder(states, stop_gradient=stop_gradient)

  def compute_energy(self, embeddings,
                     other_embeddings):
    """Computes matrix of energies between every pair of (embedding, other_embedding)."""
    transformed_embeddings = tf.matmul(embeddings, self.weight)
    energies = tf.matmul(transformed_embeddings, other_embeddings, transpose_b=True)
    return energies

  def fit(self, states, actions,
          rewards):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.sequence_length, self.input_dim])
      all_embeddings = self.embedder(all_states, stop_gradient=False)
      embeddings = tf.reshape(all_embeddings, [batch_size, self.sequence_length, self.embedding_dim])

      if self.sequence_length > 2:
        latent_embedder_in = tf.concat([embeddings[:, :-2, :], actions[:, :-2, :]], -1)
        latent = self.latent_embedder(latent_embedder_in, stop_gradient=False)
      else:
        latent = tf.zeros([batch_size, self.latent_dim])

      reward_decoder_in = tf.concat([latent, embeddings[:, -2, :], actions[:, -2, :]], -1)
      reward_pred = self.reward_decoder(reward_decoder_in, stop_gradient=False)
      reward_loss = tf.square(rewards[:, -2] - reward_pred[Ellipsis, 0])

      inverse_decoder_in = tf.concat([latent, embeddings[:, -2, :], embeddings[:, -1, :]], -1)
      action_log_probs = self.inverse_decoder.log_probs(inverse_decoder_in, actions[:, -2, :])
      _, policy_log_probs = self.inverse_decoder(inverse_decoder_in, sample=True, with_log_probs=True)

      alpha = tf.exp(self.log_alpha)
      alpha_loss = alpha * tf.stop_gradient(-policy_log_probs - self.target_entropy)
      inverse_loss = -action_log_probs + tf.stop_gradient(alpha) * policy_log_probs

      forward_decoder_in = tf.concat([latent, embeddings[:, -2, :], actions[:, -2, :]], -1)
      forward_pred_sample, forward_pred_raw = self.forward_decoder(
          forward_decoder_in, sample=True, sample_and_raw_output=True,
          stop_gradient=False)
      if self.state_prediction_mode in ['latent', 'energy']:
        true_sample = embeddings[:, -1, :]
      elif self.state_prediction_mode == 'raw':
        true_sample = states[:, -1, :]
      else:
        assert False, 'bad prediction mode'

      if self.state_prediction_mode in ['latent', 'raw']:
        if self.num_distributions and self.state_prediction_mode == 'latent':
          forward_loss = categorical_kl(true_sample, forward_pred_raw)
        else:
          forward_pred_mean, forward_pred_logvar = tf.split(forward_pred_raw, 2, axis=-1)
          forward_pred_dist = tfp.distributions.MultivariateNormalDiag(
              forward_pred_mean, tf.exp(0.5 * forward_pred_logvar))
          forward_loss = -forward_pred_dist.log_prob(true_sample)
      else:
        energies = self.compute_energy(forward_pred_sample, true_sample)
        positive_loss = tf.linalg.diag_part(energies)
        negative_loss = tf.reduce_logsumexp(energies, axis=-1)

        forward_loss = -positive_loss + negative_loss

      loss = tf.reduce_mean(alpha_loss +
                            self.reward_weight * reward_loss +
                            self.inverse_weight * inverse_loss +
                            self.forward_weight * forward_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss,
        'alpha': alpha,
        'alpha_loss': tf.reduce_mean(alpha_loss),
        'reward_loss': tf.reduce_mean(reward_loss),
        'inverse_loss': tf.reduce_mean(inverse_loss),
        'forward_loss': tf.reduce_mean(forward_loss),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, rewards, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions, rewards)

  def get_input_state_dim(self):
    return self.embedder.embedding_dim


class DeepMdpLearner(SuperModelLearner):
  """A learner for DeepMDP."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None):
    super().__init__(
        state_dim=state_dim,
        action_spec=action_spec,
        embedding_dim=embedding_dim,
        num_distributions=num_distributions,
        hidden_dims=hidden_dims,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        reward_weight=1.0,
        inverse_weight=0.0,
        forward_weight=1.0,
        state_prediction_mode='latent')


class ForwardModelLearner(SuperModelLearner):
  """A learner for forward model."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None):
    super().__init__(
        state_dim=state_dim,
        action_spec=action_spec,
        embedding_dim=embedding_dim,
        num_distributions=num_distributions,
        hidden_dims=hidden_dims,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        reward_weight=1.0,
        inverse_weight=0.0,
        forward_weight=1.0,
        state_prediction_mode='energy')


class InverseModelLearner(SuperModelLearner):
  """A learner for inverse model."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None):
    super().__init__(
        state_dim=state_dim,
        action_spec=action_spec,
        embedding_dim=embedding_dim,
        num_distributions=num_distributions,
        hidden_dims=hidden_dims,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        reward_weight=0.0,
        inverse_weight=1.0,
        forward_weight=0.0,
        state_prediction_mode='energy')


class BisimulationLearner(tf.keras.Model):
  """A learner for Deep Bisimulation for Control (DBC)."""

  def __init__(self,
               state_dim,
               action_spec,
               embedding_dim = 256,
               num_distributions = None,
               hidden_dims = (256, 256),
               sequence_length = 2,
               learning_rate = None,
               gamma = 0.99):
    """Creates networks.

    Args:
      state_dim: State size.
      action_spec: Action spec.
      embedding_dim: Embedding size.
      hidden_dims: List of hidden dimensions.
      sequence_length: Expected length of sequences provided as input
      learning_rate: Learning rate.
    """
    super().__init__()
    self.input_dim = state_dim
    self.action_dim = action_spec.shape[0]
    self.action_spec = action_spec
    self.embedding_dim = embedding_dim
    self.num_distributions = num_distributions
    self.sequence_length = sequence_length
    self.gamma = gamma

    self.embedder = EmbedNet(
        state_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=self.num_distributions,
        hidden_dims=hidden_dims)

    self.reward_pred = EmbedNet(
        self.embedding_dim,
        embedding_dim=1,
        hidden_dims=hidden_dims)
    self.trans_pred = StochasticEmbedNet(
        self.embedding_dim + self.action_dim,
        embedding_dim=self.embedding_dim,
        num_distributions=self.num_distributions,
        hidden_dims=hidden_dims)

    learning_rate = learning_rate or 1e-4
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    self.all_variables = self.variables

  @tf.function
  def call(self,
           states,
           actions = None,
           stop_gradient = True):
    """Returns embedding.

    Args:
      states: A batch of states.
      stop_gradient: Whether to stop_gradient.

    Returns:
      Embedding.
    """
    return self.embedder(states, stop_gradient=stop_gradient)

  def fit(self, states, actions,
          rewards):
    """Updates critic parameters.

    Args:
      states: Batch of sequences of states.
      actions: Batch of sequences of actions.

    Returns:
      Dictionary with information to track.
    """
    batch_size = tf.shape(states)[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.all_variables)

      all_states = tf.reshape(states, [batch_size * self.sequence_length, self.input_dim])
      all_embeddings = self.embedder(all_states, stop_gradient=False)
      embeddings = tf.reshape(all_embeddings, [batch_size, self.sequence_length, self.embedding_dim])

      embeddings, next_embeddings = embeddings[:, :-1, :], embeddings[:, 1:, :]
      actions = actions[:, :-1, :]
      rewards = rewards[:, :-1]

      trans_in = tf.reshape(
          tf.concat([embeddings, actions], -1),
          [batch_size * (self.sequence_length - 1), self.embedding_dim + self.action_dim])
      trans_sample, trans_raw = self.trans_pred(tf.stop_gradient(trans_in), stop_gradient=False,
                                                sample_and_raw_output=True)
      trans_true = tf.reshape(tf.stop_gradient(next_embeddings),
                              [batch_size * (self.sequence_length - 1), self.embedding_dim])
      trans_loss = tf.reduce_sum(tf.square(trans_sample - trans_true), -1)

      reward_sample = self.reward_pred(tf.stop_gradient(trans_sample), stop_gradient=False)[Ellipsis, 0]
      reward_true = tf.reshape(rewards, [batch_size * (self.sequence_length - 1)])
      reward_loss = tf.square(reward_sample - reward_true)

      p_in = tf.concat([embeddings[:, -1, :], actions[:, -1, :]], -1)
      p_sample, p_raw = self.trans_pred(tf.stop_gradient(p_in), stop_gradient=True,
                                        sample_and_raw_output=True)

      r_in = tf.reshape(embeddings,
                        [batch_size * (self.sequence_length - 1), self.embedding_dim])
      r_sample = self.reward_pred(tf.stop_gradient(r_in), stop_gradient=True)[Ellipsis, 0]
      r_sample = tf.reshape(r_sample, [batch_size, self.sequence_length - 1])

      indices = tf.range(batch_size)
      shuffled_indices = tf.concat([indices[1:], indices[:1]], 0)

      z_i = tf.gather(embeddings[:, 0, :], indices)
      r_i = tf.gather(r_sample, indices)
      p_i = tf.gather(p_raw, indices)

      z_j = tf.gather(embeddings[:, 0, :], shuffled_indices)
      r_j = tf.gather(r_sample, shuffled_indices)
      p_j = tf.gather(p_raw, shuffled_indices)

      if self.num_distributions:
        last_value = self.gamma ** (self.sequence_length - 1) * (
            tf.reduce_sum(tf.math.abs(p_i - p_j), -1))
      else:
        p_i_mean, p_i_logvar = tf.split(p_i, 2, axis=-1)
        p_j_mean, p_j_logvar = tf.split(p_j, 2, axis=-1)
        last_value = self.gamma ** (self.sequence_length - 1) * (
            tf.reduce_sum(tf.square(p_i_mean - p_j_mean), -1) +
            tf.reduce_sum(tf.square(tf.exp(0.5 * p_i_logvar) -
                                    tf.exp(0.5 * p_j_logvar)), -1))

      reward_sum = tf.reduce_sum(
          tf.math.abs(r_i - r_j) *
          tf.pow(self.gamma, tf.range(self.sequence_length - 1, dtype=tf.float32)), -1)

      z_loss = tf.square(
          tf.reduce_sum(tf.math.abs(z_i - z_j), -1) - reward_sum - last_value)

      loss = tf.reduce_mean(reward_loss + trans_loss) + tf.reduce_mean(z_loss)

    grads = tape.gradient(loss, self.all_variables)

    self.optimizer.apply_gradients(
        zip(grads, self.all_variables))

    return {
        'embed_loss': loss,
        'trans_loss': tf.reduce_mean(trans_loss),
        'reward_loss': tf.reduce_mean(reward_loss),
        'z_loss': tf.reduce_mean(z_loss),
    }

  @tf.function
  def update_step(self, replay_buffer_iter):
    states, actions, rewards, _, _ = next(replay_buffer_iter)
    return self.fit(states, actions, rewards)

  def get_input_state_dim(self):
    return self.embedder.embedding_dim
