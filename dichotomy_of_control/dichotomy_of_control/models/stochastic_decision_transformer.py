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

"""Stochastic Decision transformer architecture in Tensorflow."""
import tensorflow as tf
from transformers import GPT2Config
from transformers import modeling_tf_utils
from transformers.models.gpt2 import modeling_tf_gpt2
from dichotomy_of_control import utils


@modeling_tf_utils.keras_serializable
class PositionlessTFGPT2MainLayer(tf.keras.layers.Layer):
  """GPT2 transformer layer without position embeddings.

  Takes input embeddings and attention mask directly, as opposed to usual
  transformer layer, which takes in inputs and maps to embeddings internally.
  """
  config_class = GPT2Config

  def __init__(self, config, *inputs, **kwargs):
    super().__init__(*inputs, **kwargs)

    self._config = config
    self._drop = tf.keras.layers.Dropout(config.embd_pdrop)
    self._h = [
        modeling_tf_gpt2.TFBlock(
            config, scale=True, name="h_._{}".format(i))
        for i in range(config.n_layer)
    ]
    self._ln_f = tf.keras.layers.LayerNormalization(
        epsilon=config.layer_norm_epsilon, name="ln_f")

  def call(self, inputs_embeds, attention_mask, training=False):
    """Forward pass for the PositionlessTFGPT2MainLayer.

    Args:
      inputs_embeds: A tf.Tensor of input embeddings.
      attention_mask: A tf.Tensor of booleans indicating which inputs can be
        attended to by the layer.
      training: A bool for whether training or evaluating.

    Returns:
      hidden_states: A tf.Tensor representing the final hidden states.
    """
    attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_mask = tf.cast(attention_mask, tf.float32)
    attention_mask = (1.0 - attention_mask) * -10000.0

    hidden_states = inputs_embeds
    hidden_states = self._drop(hidden_states, training=training)

    input_shape = modeling_tf_utils.shape_list(inputs_embeds)[:-1]
    output_shape = input_shape + [
        modeling_tf_utils.shape_list(hidden_states)[-1]
    ]

    for block in self._h:
      outputs = block(
          hidden_states,
          layer_past=None,
          encoder_attention_mask=attention_mask,
          head_mask=None,
          encoder_hidden_states=None,
          encoder_attention_mask=None,
          use_cache=False,
          output_attentions=False,
          training=training)
      hidden_states = outputs[0]

    hidden_states = self._ln_f(hidden_states)
    hidden_states = tf.reshape(hidden_states, output_shape)

    return hidden_states


class StochasticDecisionTransformer(tf.keras.Model):
  """Uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)."""

  def __init__(self,
               state_dim,
               act_dim,
               hidden_size,
               context_len,
               max_ep_len,
               model_type='transformer',
               future_len=None,
               sample_per_step=True,
               normalize_action=True,
               normalize_return=True,
               latent_size=32,
               **kwargs):
    """Initializes a DecisionTransformer.

    Args:
      state_dim: An int dimension of observations.
      act_dim: An int dimension of actions.
      hidden_size: An int size of hidden layers in transformer.
      context_len: An int maximum context length.
      max_ep_len: An int maximum episode length.
      **kwargs: Additional kwargs passed onto GPT2Config.
    """
    super().__init__()
    self._state_dim = state_dim
    self._act_dim = act_dim
    self._hidden_size = hidden_size
    self._latent_size = latent_size or hidden_size
    self._context_len = context_len
    self._future_len = future_len or context_len
    self._model_type = model_type
    self._sample_per_step = sample_per_step
    config = GPT2Config(
        vocab_size=1,  # doesn't matter -- we don't use the vocab
        n_embd=hidden_size,
        **kwargs)
    # note: the only difference between this GPT2Model and the default
    # Huggingface version is that the positional embeddings are removed
    # (since we'll add those ourselves)
    self._transformer = PositionlessTFGPT2MainLayer(config)
    self._prior_seq_net = PositionlessTFGPT2MainLayer(config)
    if self._model_type == 'transformer':
      self._future_seq_net = PositionlessTFGPT2MainLayer(config)
    elif self._model_type == 'rnn':
      self._future_seq_net = utils.create_rnn(
          [3 * self._future_len, self._hidden_size],
          hidden_dims=(self._hidden_size, self._hidden_size))
    elif self._model_type == 'mlp':
      self._future_seq_net = utils.create_mlp(
          3 * self._future_len * self._hidden_size, self._hidden_size)
    elif self._model_type == 'deepset':
      self._future_seq_net = utils.create_mlp(
          [3 * self._future_len, self._hidden_size], self._hidden_size)
    else:
      raise NotImplementedError

    self._embed_timestep = tf.keras.layers.Embedding(max_ep_len, hidden_size)
    self._embed_return = tf.keras.layers.Dense(hidden_size)
    self._embed_state = tf.keras.layers.Dense(hidden_size)
    self._embed_action = tf.keras.layers.Dense(hidden_size)

    self._embed_ln = tf.keras.layers.LayerNormalization()

    self._predict_action = utils.create_mlp(
        [self._context_len, self._hidden_size + self._latent_size],
        self._act_dim,
        last_layer_activation='tanh' if normalize_action else None)
    self._predict_future = tf.keras.layers.Dense(self._latent_size * 2)
    self._predict_prior = tf.keras.layers.Dense(self._latent_size * 2)
    self._predict_value = utils.create_mlp(
        [self._context_len, self._hidden_size + self._latent_size],
        1,
        last_layer_activation='tanh' if normalize_return else None)

    self._embed_energy = utils.create_mlp(
        self._hidden_size * 3 + self._latent_size, self._hidden_size)

    # Keep track of sampling frequency of z during inference.
    self._z = None
    self._z_counter = 0

  @tf.function
  def call(self,
           states,
           actions,
           returns_to_go,
           timesteps,
           attention_mask,
           future_states,
           future_actions,
           future_returns_to_go,
           future_timesteps,
           future_attention_mask,
           future_samples=100,
           training=False,
           z=None):
    """Forward pass for DecisionTransformer.

    Args:
      states: A tf.Tensor representing a batch of state trajectories of shape
        `[B, T, ...]`.
      actions: A tf.Tensor representing a batch of action trajectories of shape
        `[B, T, ...]`.
      returns_to_go: A tf.Tensor representing a batch of returns-to-go of shape
        `[B, T]`.
      timesteps: A tf.Tensor representing a batch of timesteps of shape `[B,
        T]`.
      attention_mask: A tf.Tensor representing which parts of the trajectories
        can be attended to by the model. For example, it cannot use the future
        to predict a next action. Shape `[B, T]`.
      training: A bool representing whether we are training or evaluating.

    Returns:
      action_preds: A tf.Tensor representing predicted actions
        of shape `[B, T]`.
    """

    batch_size, seq_length = states.shape[0], states.shape[1]
    future_seq_length = future_states.shape[1]

    # embed each modality with a different head
    state_embeddings = tf.reshape(
        self._embed_state(
            tf.reshape(states,
                       [batch_size * seq_length] + states.shape[2:].as_list())),
        [batch_size, seq_length, -1])
    action_embeddings = self._embed_action(actions)
    returns_embeddings = self._embed_return(returns_to_go)
    time_embeddings = self._embed_timestep(timesteps)
    future_state_embeddings = tf.reshape(
        self._embed_state(
            tf.reshape(future_states, [batch_size * seq_length] +
                       future_states.shape[2:].as_list())),
        [batch_size, seq_length, -1])
    future_action_embeddings = self._embed_action(future_actions)
    future_returns_embeddings = self._embed_return(future_returns_to_go)
    future_time_embeddings = self._embed_timestep(future_timesteps)

    # time embeddings are treated similar to positional embeddings
    state_embeddings = state_embeddings + time_embeddings
    action_embeddings = action_embeddings + time_embeddings
    returns_embeddings = returns_embeddings + time_embeddings
    future_state_embeddings = future_state_embeddings + future_time_embeddings
    future_action_embeddings = future_action_embeddings + future_time_embeddings
    future_returns_embeddings = future_returns_embeddings + future_time_embeddings

    # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
    # which works nice in an autoregressive sense since states predict actions
    stacked_inputs = tf.stack(
        (returns_embeddings, state_embeddings, action_embeddings), axis=1)
    stacked_inputs = tf.transpose(stacked_inputs, perm=[0, 2, 1, 3])
    stacked_inputs = tf.reshape(stacked_inputs,
                                (batch_size, 3 * seq_length, self._hidden_size))
    stacked_inputs = self._embed_ln(stacked_inputs)
    future_stacked_inputs = tf.stack(
        (future_returns_embeddings, future_state_embeddings,
         future_action_embeddings),
        axis=1)
    future_stacked_inputs = tf.transpose(
        future_stacked_inputs, perm=[0, 2, 1, 3])
    future_stacked_inputs = tf.reshape(
        future_stacked_inputs,
        (batch_size, 3 * future_seq_length, self._hidden_size))
    future_stacked_inputs = self._embed_ln(future_stacked_inputs)
    if self._model_type == 'mlp':
      future_stacked_inputs = tf.reshape(future_stacked_inputs,
                                         [batch_size, -1])

    # to make the attention mask fit the stacked inputs, have to stack it too
    stacked_attention_mask = tf.stack(
        (attention_mask, attention_mask, attention_mask), axis=1)
    stacked_attention_mask = tf.transpose(stacked_attention_mask, [0, 2, 1])
    stacked_attention_mask = tf.reshape(stacked_attention_mask,
                                        (batch_size, 3 * seq_length))
    future_stacked_attention_mask = tf.stack(
        (future_attention_mask, future_attention_mask, future_attention_mask),
        axis=1)
    future_stacked_attention_mask = tf.transpose(future_stacked_attention_mask,
                                                 [0, 2, 1])
    future_stacked_attention_mask = tf.reshape(
        future_stacked_attention_mask, (batch_size, 3 * future_seq_length))

    # we feed in the input embeddings (not word indices as in NLP) to the model
    x = self._transformer(
        inputs_embeds=stacked_inputs,
        attention_mask=stacked_attention_mask,
        training=training,
    )
    embed = x[:, -1, :]
    if self._model_type == 'transformer':
      f = self._future_seq_net(
          inputs_embeds=future_stacked_inputs,
          attention_mask=future_stacked_attention_mask,
          training=training,
      )
      f = f[:, -1, :]
    elif self._model_type == 'rnn':
      f = self._future_seq_net(future_stacked_inputs)
      f = f[:, -1, :]
    elif self._model_type == 'mlp':
      f = self._future_seq_net(future_stacked_inputs)
    else:
      f = self._future_seq_net(future_stacked_inputs)
      f = tf.reduce_mean(f, axis=1)
    f = self._predict_future(f)
    f_mean, f_logvar = tf.split(f, 2, axis=-1)
    f_pred = f_mean + tf.random.normal(tf.shape(f_mean)) * tf.exp(
        0.5 * f_logvar)

    # Take the second to last as the last is the missing action at inference.
    prior = self._prior_seq_net(
        inputs_embeds=stacked_inputs,
        attention_mask=stacked_attention_mask,
        training=training,
    )[:, -2, :]
    prior = self._predict_prior(prior)
    prior_mean, prior_logvar = tf.split(prior, 2, axis=-1)
    prior_pred = prior_mean + tf.random.normal(tf.shape(prior_mean)) * tf.exp(
        0.5 * prior_logvar)

    # reshape x so that the second dimension corresponds to the original
    # returns (0), states (1), or actions (2); i.e. x[:,1,t] is for s_t
    x = tf.reshape(x, (batch_size, seq_length, 3, self._hidden_size))
    x = tf.transpose(x, (0, 2, 1, 3))

    # get value predictions
    if training:
      value_preds = self._predict_value(
          tf.concat(
              [x[:, 1],
               tf.repeat((f_pred)[:, None, :], seq_length, axis=1)],
              axis=-1))
    else:
      value_preds = None
      if z is not None:
        f_pred = z
      else:
        prior_mean = tf.repeat(prior_mean, future_samples, axis=0)
        prior_logvar = tf.repeat(prior_logvar, future_samples, axis=0)
        prior_pred = prior_mean + tf.random.normal(
            tf.shape(prior_mean)) * tf.exp(0.5 * prior_logvar)
        value_preds = self._predict_value(
            tf.concat([
                tf.repeat(x[:, 1], future_samples, axis=0),
                tf.repeat(
                    tf.repeat((f_pred)[:, None, :], seq_length, axis=1),
                    future_samples,
                    axis=0)
            ],
                      axis=-1))
        value_preds = tf.reshape(value_preds,
                                 [batch_size, future_samples, seq_length])[Ellipsis,
                                                                           -1]
        best_idx = tf.argmax(value_preds, axis=1)
        value_preds = tf.squeeze(
            tf.gather(value_preds, best_idx, axis=1), axis=1)
        f_pred = tf.squeeze(
            tf.gather(
                tf.reshape(prior_pred, [batch_size, future_samples, -1]),
                best_idx,
                axis=1),
            axis=1)

    # get predictions
    action_preds = self._predict_action(
        tf.concat(
            [x[:, 1],
             tf.repeat((f_pred)[:, None, :], seq_length, axis=1)],
            axis=-1))

    # get energies
    pos_embed = self._embed_energy(
        tf.concat([
            embed, f_pred, future_state_embeddings[:, 0],
            future_state_embeddings[:, 0]
        ],
                  axis=-1))
    neg_embed = self._embed_energy(
        tf.concat([
            embed, f_pred,
            tf.random.shuffle(future_state_embeddings[:, 0]),
            tf.random.shuffle(future_state_embeddings[:, 0])
        ],
                  axis=-1))
    energies = tf.matmul(pos_embed, neg_embed, transpose_b=True)

    return (action_preds, value_preds, f_pred, f_mean, f_logvar, prior_mean,
            prior_logvar, energies)

  def get_action(self, states, actions, returns_to_go, timesteps):
    """Predict a next action given a trajectory.

    Args:
      states: A tf.Tensor representing a sequence of states of shape `[T, ...]`.
      actions: A tf.Tensor representing a sequence of actions of shape `[T,
        ...]`.
      returns_to_go: A tf.Tensor representing a sequence of returns-to-go of
        shape `[T]`.
      timesteps: A tf.Tensor representing a sequence of timesteps of shape
        `[T]`.

    Returns:
      action: A tf.Tensor representing a single next action.
    """
    states = tf.reshape(states, [1, -1, self._state_dim])
    actions = tf.reshape(actions, (1, -1, self._act_dim))
    returns_to_go = tf.reshape(returns_to_go, (1, -1, 1))
    timesteps = tf.reshape(timesteps, (1, -1))

    states = states[:, -self._context_len:]
    actions = actions[:, -self._context_len:]
    returns_to_go = returns_to_go[:, -self._context_len:]
    timesteps = timesteps[:, -self._context_len:]

    # pad all tokens to sequence length
    attention_mask = tf.concat([
        tf.zeros(self._context_len - states.shape[1]),
        tf.ones(states.shape[1])
    ],
                               axis=0)
    attention_mask = tf.reshape(attention_mask, (1, -1))
    states = tf.concat([
        tf.zeros([
            states.shape[0], self._context_len - states.shape[1],
            self._state_dim
        ]), states
    ],
                       axis=1)
    actions = tf.concat([
        tf.zeros((actions.shape[0], self._context_len - actions.shape[1],
                  self._act_dim)), actions
    ],
                        axis=1)
    returns_to_go = tf.concat([
        tf.zeros((returns_to_go.shape[0],
                  self._context_len - returns_to_go.shape[1], 1)), returns_to_go
    ],
                              axis=1)
    timesteps = tf.concat([
        tf.zeros((timesteps.shape[0], self._context_len - timesteps.shape[1]),
                 dtype=tf.int64), timesteps
    ],
                          axis=1)
    future_states = tf.zeros(
        [states.shape[0], self._future_len] + list(states.shape[2:]),
        dtype=states.dtype)
    future_actions = tf.zeros(
        [actions.shape[0], self._future_len, actions.shape[-1]],
        dtype=actions.dtype)
    future_returns_to_go = tf.zeros(
        [returns_to_go.shape[0], self._future_len, returns_to_go.shape[-1]],
        dtype=returns_to_go.dtype)
    future_timesteps = tf.zeros([timesteps.shape[0], self._future_len],
                                dtype=timesteps.dtype)
    future_attention_mask = tf.zeros(
        [attention_mask.shape[0], self._future_len], dtype=attention_mask.dtype)

    action_preds, value_preds, self._z, _, _, _, _, _ = self(
        states,
        actions,
        returns_to_go,
        timesteps,
        attention_mask,
        future_states,
        future_actions,
        future_returns_to_go,
        future_timesteps,
        future_attention_mask,
        training=False,
        z=self._z if not self._sample_per_step else None,
    )

    self._z_counter += 1
    if self._z_counter == self._future_len:
      self._z = None
      self._z_counter = 0

    return action_preds[0, -1]
