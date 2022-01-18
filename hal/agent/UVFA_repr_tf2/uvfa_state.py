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
# pylint: disable=unused-argument
# pylint: disable=unused-variable
# pylint: disable=line-too-long
"""Build UVFA with state observation for state input."""

from __future__ import absolute_import
from __future__ import division

import random

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from hal.agent.tf2_utils import soft_variables_update
from hal.agent.tf2_utils import stack_conv_layer
from hal.agent.tf2_utils import stack_dense_layer
from hal.agent.tf2_utils import vector_tensor_product
import hal.utils.word_vectorization as wv

embedding_group_1 = ('swivel', 'nnlm', 'word2vec')


class StateUVFA2:
  """UVFA that uses the ground truth states.

  Attributes:
    cfg: configuration file
    tokenizer: tokenizer for the agent
    saver: saver in charge of creating checkpoints
    manager: tf2 checkpoint manager
    global_step: current learning step the agent is at
    vocab_list: a list of vocabulary the agent knows
    vocab_to_int: a table from vocabulary to integer
    int_to_vocab: a table from integer to vocabulary
    decode_fn: a function that converts tokens to text
    online_encoder: online model of the instruction encoder
    target_encoder: target model of the instruction encoder
    online_embedding_layer: online embedding matrix
    target_embedding_layer: target embedding matrix
    online_models: online model of the policy network
    target_models: target model of the policy network
    optimizer: optimizer the optimizes the agent
  """

  def __init__(self, cfg):
    self.cfg = cfg
    self.tokenizer = None  # some settings do not have a tokenizer
    self._build()
    if 'model_dir' in cfg.as_dict():
      self.saver = tf.train.Checkpoint(
          optimizer=self.optimizer,
          online_encoder=self.online_encoder,
          online_model=self.online_models,
          target_encoder=self.target_encoder,
          target_model=self.target_models,
          step=self.global_step)
      self.manager = tf.train.CheckpointManager(
          self.saver, cfg.model_dir, max_to_keep=5, checkpoint_name='model')

  def _build(self):
    self.global_step = tf.Variable(0, name='global_step')
    self.create_models(self.cfg)
    self.vocab_list = self.cfg.vocab_list
    self.vocab_to_int, self.int_to_vocab = wv.create_look_up_table(
        self.vocab_list)
    self.decode_fn = wv.decode_with_lookup_table(self.int_to_vocab)

  def create_models(self, cfg):
    """Build the computation graph for the agent."""

    online_encoder_out = create_instruction_encoder(cfg, name='online_encoder')
    target_encoder_out = create_instruction_encoder(cfg, name='target_encoder')
    self.online_encoder = online_encoder_out['encoder']
    self.target_encoder = target_encoder_out['encoder']
    self.online_embedding_layer = online_encoder_out['token_embedding']
    self.target_embedding_layer = target_encoder_out['token_embedding']

    embedding_length = online_encoder_out['instruction_embedding_length']
    if self.cfg.action_type == 'perfect':
      self.online_models = self.build_q_perfect(cfg, 'online_network',
                                                embedding_length)
      self.target_models = self.build_q_perfect(cfg, 'target_network',
                                                embedding_length)
    elif self.cfg.action_type == 'discrete':
      self.online_models = self.build_q_discrete(cfg, 'online_network',
                                                 embedding_length)
      self.target_models = self.build_q_discrete(cfg, 'target_network',
                                                 embedding_length)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

  def _preprocess_instruction(self, g):
    """Pre-process instructions for agent consumption."""
    if isinstance(g, str):
      return tf.convert_to_tensor(np.array(g.split()))
    if len(g.shape) < 2:  # should expand 0th axis
      g = np.expand_dims(np.array(g), 0)
    if self.cfg.instruction_repr != 'language':
      return tf.convert_to_tensor(g)
    if self.cfg.embedding_type in embedding_group_1:
      original_shape = g.shape
      g = np.reshape(g, -1)
      tokens = []
      for i in g:
        tokens.append(self.int_to_vocab[i])
      return tf.convert_to_tensor(np.array(tokens).reshape(original_shape))
    else:
      return g

  def step(self, obs, g, env, explore_prob=0.0, debug=False):
    """Take a step in the environment."""
    g = self._preprocess_instruction(g)
    obs = tf.convert_to_tensor(np.expand_dims(obs, axis=0))
    if explore_prob == 0.0:  # TODO(ydjiang): find better way to initialize
      self.online_encoder([g, obs])
      self.target_encoder([g, obs])
    if random.uniform(0, 1) < explore_prob:
      return np.int32(env.sample_random_action())
    else:
      predict_action = np.int32(np.squeeze(self._step(obs, g).numpy()))
      return predict_action

  @tf.function
  def _step(self, obs, g):
    """Take step using the tf models."""
    online_embedding = self.online_encoder([g, obs])
    online_values = self.online_models(
        inputs={
            'state_input': obs,
            'goal_embedding': online_embedding
        },
        training=False)
    if self.cfg.action_type == 'discrete':
      predict_action = tf.argmax(online_values, axis=1)
    elif self.cfg.action_type == 'perfect':
      input_max_q = tf.reduce_max(online_values, axis=2)
      input_select = tf.argmax(input_max_q, axis=1)
      action_max_q = tf.reduce_max(online_values, axis=1)
      action_select = tf.argmax(action_max_q, axis=1)
      predict_action = tf.stack([input_select, action_select], axis=1)
    return predict_action

  def train(self, batch):
    """Take a single gradient step on the batch."""

    if self.cfg.action_type == 'perfect':
      batch['action'] = np.stack(batch['action'])
    elif self.cfg.action_type == 'discrete':
      batch['action'] = list(batch['action'])

    # TODO(ydjiang): figure out why reward cannot be converted to tensor as is
    batch['reward'] = list(batch['reward'])
    batch['g'] = self._preprocess_instruction(batch['g'])

    # make everything tensor except for g
    for k in batch:
      if k != 'g':
        batch[k] = tf.convert_to_tensor(batch[k])

    loss = self._train(batch).numpy()

    return {'loss': loss}

  @tf.function
  def _train(self, batch):
    with tf.GradientTape() as tape:
      loss = self._loss(batch)
    variables = self.online_models.trainable_variables
    variables += self.online_encoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    return loss

  def _loss(self, batch):
    """Compute td loss on a batch."""
    obs, action, obs_next = batch['obs'], batch['action'], batch['obs_next']
    r, g = batch['reward'], batch['g']

    q = self._compute_q(obs, action, g)
    q_next = self._compute_q_next(obs_next, g)
    if self.cfg.masking_q:
      mask = tf.cast(
          tf.math.less_equal(r, self.cfg.reward_scale), dtype=tf.float32)
      q_next *= mask

    gamma = self.cfg.discount
    td_target = tf.clip_by_value(r + gamma * q_next, -1. / (1 - gamma), 10.0)
    loss = tf.compat.v1.losses.huber_loss(
        tf.stop_gradient(td_target), q, reduction=tf.losses.Reduction.MEAN)
    return loss

  def _compute_q(self, obs, action, g):
    """Compute q values of current observation."""
    online_embedding = self.online_encoder([g, obs])
    online_q_values = self.online_models(
        {
            'state_input': obs,
            'goal_embedding': online_embedding
        }, training=True)
    if self.cfg.action_type == 'perfect':
      stacked_indices = tf.concat(
          [tf.expand_dims(tf.range(tf.shape(action)[0]), axis=1), action],
          axis=1)
      predicted_q = tf.gather_nd(online_q_values, stacked_indices)
    elif self.cfg.action_type == 'discrete':
      ac_dim = self.cfg.ac_dim[0]
      action_onehot = tf.one_hot(action, ac_dim, dtype=tf.float32)
      predicted_q = tf.reduce_sum(
          tf.multiply(online_q_values, action_onehot), axis=1)
    else:
      raise ValueError('Unrecognized action type: {}'.format(
          self.cfg.action_type))
    return predicted_q

  def _compute_q_next(self, obs_next, g):
    """Compute q values of next observation."""
    online_embedding = self.online_encoder([g, obs_next])
    target_embedding = self.target_encoder([g, obs_next])

    online_values = self.online_models(
        {
            'state_input': obs_next,
            'goal_embedding': online_embedding
        },
        training=True)
    target_values = self.target_models(
        {
            'state_input': obs_next,
            'goal_embedding': target_embedding
        },
        training=True)
    if self.cfg.action_type == 'discrete':
      predict_action = tf.argmax(online_values, axis=1)
      double_q = tf.reduce_sum(
          tf.one_hot(predict_action, self.cfg.ac_dim[0]) * target_values,
          axis=1)
    elif self.cfg.action_type == 'perfect':
      input_max_q = tf.reduce_max(online_values, axis=2)
      input_select = tf.argmax(input_max_q, axis=1, output_type=tf.int32)
      action_max_q = tf.reduce_max(online_values, axis=1)
      action_select = tf.argmax(action_max_q, axis=1, output_type=tf.int32)
      indices = tf.range(input_max_q.shape[0], dtype=tf.int32)
      stacked_indices = tf.stack([indices, input_select, action_select], axis=1)
      double_q = tf.gather_nd(target_values, stacked_indices)
    else:
      raise ValueError('Unrecognized action type: {}'.format(
          self.cfg.action_type))
    return double_q

  def build_q_perfect(self, cfg, name, embedding_length):
    """Build the q function for perfect action space."""
    per_input_ac_dim = cfg.per_input_ac_dim
    des_len, inner_len = cfg.descriptor_length, cfg.inner_product_length

    inputs = tf.keras.layers.Input(shape=(cfg.obs_dim[0], cfg.obs_dim[1]))
    goal_embedding = tf.keras.layers.Input(shape=(embedding_length))

    shape_layer = tf.keras.layers.Lambda(tf.shape)
    num_object = tf.cast(shape_layer(inputs)[1], tf.int32)
    tp_concat = vector_tensor_product(inputs, inputs)
    conv_layer_cfg = [[des_len * 8, 1, 1], [des_len * 4, 1, 1], [des_len, 1, 1]]
    # [B, ?, ?, des_len]
    tp_concat = stack_conv_layer(conv_layer_cfg)(tp_concat)

    # similarity with goal
    goal_key = stack_dense_layer([inner_len * 2,
                                  inner_len])(goal_embedding)  # [B, d_inner]
    expand_dims_layer = tf.keras.layers.Lambda(
        lambda inputs: tf.expand_dims(inputs[0], axis=inputs[1]))
    goal_key = expand_dims_layer((goal_key, 1))  # [B, 1, d_inner]
    # [B, ?, ?, d_inner]
    obs_query_layer = tf.keras.layers.Conv2D(inner_len, 1, 1, padding='same')
    obs_query = obs_query_layer(tp_concat)
    # [B, ?*?, d_inner]
    obs_query = tf.keras.layers.Reshape((-1, inner_len))(obs_query)
    obs_query_t = tf.keras.layers.Permute((2, 1))(obs_query)  # [B,d_inner,?*?]
    matmul_layer = tf.keras.layers.Lambda(
        lambda inputs: tf.matmul(inputs[0], inputs[1]))
    inner = matmul_layer((goal_key, obs_query_t))  # [B, 1, ?*?]
    weight = tf.keras.layers.Softmax()(inner)  # [B, 1, ?*?]
    prod = matmul_layer((weight, tf.keras.layers.Reshape(
        (-1, des_len))(tp_concat)))  # [B, 1, des_len]

    goal_embedding_ = expand_dims_layer((goal_embedding, 1))  # [B, 1, dg]

    tile_layer = tf.keras.layers.Lambda(
        lambda inputs: tf.tile(inputs[0], multiples=inputs[1]))
    # [B, ?, dg]
    goal_embedding_ = tile_layer((goal_embedding_, [1, num_object, 1]))
    # [B, ?, des_len]
    pair_wise_summary = tile_layer((prod, [1, num_object, 1]))
    # [B, ?, des_len+di+dg]
    augemented_inputs = tf.keras.layers.Concatenate(axis=-1)(
        [inputs, pair_wise_summary, goal_embedding_])
    # [B, ?, 1, des_len+di+dg]
    augemented_inputs = expand_dims_layer((augemented_inputs, 2))
    conv_layer_cfg = [[per_input_ac_dim * 64, 1, 1],
                      [per_input_ac_dim * 64, 1, 1], [per_input_ac_dim, 1, 1]]
    # [B, ?, per_input_ac_dim]
    q_out = tf.keras.layers.Reshape((-1, per_input_ac_dim))(
        stack_conv_layer(conv_layer_cfg)(augemented_inputs))
    all_inputs = {'state_input': inputs, 'goal_embedding': goal_embedding}
    q_out_layer = tf.keras.Model(name=name, inputs=all_inputs, outputs=q_out)

    return q_out_layer

  def build_q_discrete(self, cfg, name, embedding_length):
    """Build the q function for discrete action space."""
    ac_dim = cfg.ac_dim[0]
    des_len, inner_len = cfg.descriptor_length, cfg.inner_product_length

    inputs = tf.keras.layers.Input(shape=(cfg.obs_dim[0], cfg.obs_dim[1]))
    goal_embedding = tf.keras.layers.Input(shape=(embedding_length))

    tp_concat = vector_tensor_product(inputs, inputs)
    conv_layer_cfg = [[des_len * 8, 3, 1], [des_len * 4, 3, 1],
                      [des_len * 4, 1, 1]]
    # [B, ?, ?, des_len]
    tp_concat = stack_conv_layer(conv_layer_cfg)(tp_concat)
    summary = tf.reduce_mean(tp_concat, axis=(1, 2))

    goal_projection_layer = tf.keras.layers.Dense(
        des_len * 4, activation='sigmoid')
    gating = goal_projection_layer(summary)

    gated_summary = summary * gating
    out_layer = tf.keras.Sequential(layers=[
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(ac_dim),
    ])
    out = out_layer(gated_summary)
    all_inputs = {'state_input': inputs, 'goal_embedding': goal_embedding}
    q_out_layer = tf.keras.Model(name=name, inputs=all_inputs, outputs=out)
    return q_out_layer

  def compute_q_over_all_g(self, obs, action, g):
    """Compute each q(s,a) under all goals (N^2 operations)."""
    if self.cfg.action_type == 'perfect':
      action = np.stack(action)
    elif self.cfg.action_type == 'discrete':
      action = list(action)
    obs = tf.convert_to_tensor(np.stack(obs))
    action = tf.convert_to_tensor(action)
    g = tf.convert_to_tensor(g)
    dummy_obs = tf.zeros([g.shape[0]] + obs.shape[1:])
    goal_embedding = self._compute_goal_embedding(dummy_obs, g)
    goal_embedding = tf.convert_to_tensor(goal_embedding)

    # number of batches processed in a single feed
    num_parallelize = self.cfg.irl_parallel_n
    obs_q_wrt_g = []
    goal_embedding = tf.concat([goal_embedding] * num_parallelize, 0)
    for i in range(obs.shape[0] // num_parallelize):
      chosen_obs = [[obs[num_parallelize * i + j]] * g.shape[0]
                    for j in range(num_parallelize)]
      obs_i = tf.stack(sum_list(chosen_obs))
      chosen_a = [[action[num_parallelize * i + j]] * g.shape[0]
                  for j in range(num_parallelize)]
      action_i = tf.stack(sum_list(chosen_a))
      q_i = self._compute_q_with_goal_embedding(obs_i, action_i, goal_embedding)
      split_q_i = np.split(q_i, num_parallelize)
      _ = [obs_q_wrt_g.append(q) for q in split_q_i]
    obs_q_wrt_g = np.array(obs_q_wrt_g)
    goal_partition = np.mean(np.exp(obs_q_wrt_g), axis=0)
    return obs_q_wrt_g - np.log(goal_partition)

  @tf.function
  def _compute_goal_embedding(self, obs, g):
    return self.target_encoder([g, obs])

  @tf.function
  def _compute_q_with_goal_embedding(self, obs, action, g_embeddings):
    """Compute q values of current observation."""
    target_q_values = self.target_models(
        {
            'state_input': obs,
            'goal_embedding': g_embeddings
        }, training=True)
    batch_size = tf.shape(action)[0]
    target_q_values_flat = tf.reshape(target_q_values, (batch_size, -1))
    softmax = tf.nn.softmax(target_q_values_flat, axis=-1)
    entropy = tf.reduce_sum(softmax * tf.log(softmax), axis=1)
    if self.cfg.action_type == 'perfect':
      stacked_indices = tf.concat(
          [tf.expand_dims(tf.range(batch_size), axis=1), action], axis=1)
      predicted_q = tf.gather_nd(target_q_values, stacked_indices)
    elif self.cfg.action_type == 'discrete':
      ac_dim = self.cfg.ac_dim[0]
      action_onehot = tf.one_hot(action, ac_dim, dtype=tf.float32)
      predicted_q = tf.reduce_sum(
          tf.multiply(target_q_values, action_onehot), axis=1)
    else:
      raise ValueError('Unrecognized action type: {}'.format(
          self.cfg.action_type))
    return predicted_q + self.cfg.entropy_alpha * entropy

  def update_target_network(self, polyak_rate=None):
    """"Update the moving average in the target networks."""
    online_var = self.online_models.trainable_variables
    online_var += self.online_encoder.trainable_variables
    target_var = self.target_models.trainable_variables
    target_var += self.target_encoder.trainable_variables
    target_update_op = soft_variables_update(online_var, target_var,
                                             self.cfg.polyak_rate)
    tf.group(*target_update_op)

  def init_networks(self):
    """Initialize the model weights."""
    print('No need to initialize in eager mode.')

  def save_model(self, model_dir=None):
    """Save the model weights to model_dir or with manager."""
    assert self.manager or model_dir, 'No manager and no model dir!'
    save_path = self.manager.save()
    print('Save model: step {} to {}'.format(int(self.global_step), save_path))

  def load_model(self, model_dir=None):
    """Load the model weights from model_dir or with manager."""
    assert self.manager or model_dir, 'No manager and no model dir!'
    if not model_dir:
      model_dir = self.manager.latest_checkpoint
    else:
      model_dir = tf.train.latest_checkpoint(model_dir)
    print('Checkpoint dir: {}'.format(model_dir))
    self.saver.restore(model_dir)

  def increase_global_step(self):
    """Increment the global step of the agent."""
    return self.global_step.assign_add(1).numpy()

  def get_global_step(self):
    """Returns the value of global step as python integer."""
    return int(self.global_step)

  def randomize_partial_word_embedding(self, num_word_change=4):
    """Setting a part of the word embedding to be random."""
    assert self.cfg.embedding_type == 'random', 'Only support random embedding'
    current_embedding = self.online_embedding_layer.trainable_variables[
        0].numpy()
    shape = current_embedding.shape
    new_embedding = current_embedding.copy()
    idx = np.random.randint(shape[0], size=num_word_change)
    for i in idx:
      new_embedding[i] = np.random.normal(size=list(shape)[1:])
    return current_embedding

  def set_embedding(self, embedding_value):
    """Set the word embedding to given values."""
    tf.assign(self.online_embedding_layer.trainable_variables[0],
              embedding_value)


################################################################################
############################# Helper functions #################################
################################################################################
def sum_list(list_of_list):
  """Concatenates a list of python list."""
  final_list = []
  for l in list_of_list:
    final_list += l
  return final_list


def create_instruction_encoder(cfg, name):
  """Helper for making the encoder.

  Args:
    cfg: configuration object
    name: name of the encoder

  Returns:
    return the created encoder
  """
  if cfg.instruction_repr == 'language' and cfg.encoder_type == 'vanilla_rnn':
    all_embedding_layer = get_embedding_layer(cfg)
    encoder = tf.keras.Sequential(layers=[
        all_embedding_layer['embedding_layer'],
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GRU(cfg.encoder_n_unit),
        tf.keras.layers.Dropout(0.5),
    ])
    text_input = all_embedding_layer['text_input_signature']
    obs_input = tf.keras.layers.Input(shape=(cfg.obs_dim[0], cfg.obs_dim[1]))
    embedded_instruction = encoder(text_input)
    full_encoder = tf.keras.Model(
        name=name, inputs=[text_input, obs_input], outputs=embedded_instruction)
    return {
        'encoder': full_encoder,
        'token_embedding': all_embedding_layer['language_model'],
        'text_input_signature': all_embedding_layer['text_input_signature'],
        'instruction_embedding_length': cfg.encoder_n_unit,
    }
  elif cfg.instruction_repr == 'language' and cfg.encoder_type == 'attention':
    all_embedding_layer = get_embedding_layer(cfg)
    embedding_layer = all_embedding_layer['embedding_layer']
    text_input = all_embedding_layer['text_input_signature']
    obs_input = tf.keras.layers.Input(shape=(5, cfg.obs_dim[1]))

    num_attention_head = 5
    flat_obs_input = tf.keras.layers.Flatten()(obs_input)
    out = tf.keras.layers.Dense(256, activation='relu')(flat_obs_input)
    total_attention_vector = tf.keras.layers.Dense(num_attention_head * 32)(out)
    # output of shape [B x T x E]
    gru = tf.keras.layers.GRU(cfg.encoder_n_unit, return_sequences=True)
    gru_out = gru(embedding_layer(text_input))
    # projected output of shape [B x T x 32]
    projected_gru_out = tf.keras.layers.Dense(32)(gru_out)
    attention_vector = tf.split(
        total_attention_vector, num_attention_head, axis=-1)
    # 5 tensor of shape B x 16
    attention_vector = [tf.nn.softmax(v, -1) for v in attention_vector]
    unnormalized_inner_product = []
    for v in attention_vector:
      unnormalized_inner_product.append(
          tf.einsum('...ij,...j->...i', projected_gru_out, v))
    # [B x T]
    weights = [tf.nn.softmax(w) for w in unnormalized_inner_product]
    # [B x T x E]
    projected_gru_out_2 = tf.keras.layers.Dense(16)(gru_out)
    out = [
        tf.einsum('...i,...ij->...j', w, projected_gru_out_2) for w in weights
    ]
    out = tf.concat(out, axis=1)
    full_encoder = tf.keras.Model(
        name=name, inputs=[text_input, obs_input], outputs=out)
    return {
        'encoder': full_encoder,
        'token_embedding': all_embedding_layer['language_model'],
        'text_input_signature': all_embedding_layer['text_input_signature'],
        'instruction_embedding_length': 16 * num_attention_head,
    }
  elif cfg.instruction_repr == 'one_hot':
    embedding_layer = tf.keras.layers.Embedding(cfg.max_sequence_length,
                                                cfg.max_sequence_length // 8)
    encoder = tf.keras.Sequential(
        name=name,
        layers=[embedding_layer,
                tf.keras.layers.Dense(cfg.encoder_n_unit)])
    text_input = tf.keras.layers.Input(shape=(cfg.max_sequence_length))
    obs_input = tf.keras.layers.Input(shape=(cfg.obs_dim[0], cfg.obs_dim[1]))
    embedded_instruction = encoder(text_input)
    full_encoder = tf.keras.Model(
        name=name, inputs=[text_input, obs_input], outputs=embedded_instruction)
    return {
        'encoder': full_encoder,
        'token_embedding': embedding_layer,
        'text_input_signature': tf.keras.layers.Input(shape=(), dtype=tf.int32),
        'instruction_embedding_length': cfg.encoder_n_unit,
    }
  else:
    raise ValueError('Unrecognized configurations for encoder')


def get_embedding_layer(cfg):
  """Helper for getting the embedding layer of the agenet."""
  if cfg.embedding_type == 'random':
    embedding_layer = tf.keras.layers.Embedding(cfg.vocab_size,
                                                cfg.embedding_size)
    embedding_layer = {
        'embedding_layer':
            embedding_layer,
        'language_model':
            embedding_layer,
        'text_input_signature':
            tf.keras.layers.Input(
                shape=(cfg.max_sequence_length), dtype=tf.int32)
    }
  elif cfg.embedding_type == 'swivel':
    model_name = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
    layer = hub.KerasLayer(
        model_name,
        output_shape=[20],
        input_shape=[],
        dtype=tf.string,
        trainable=False)
    embedding_layer = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1])), layer,
        tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [-1, cfg.max_sequence_length, 20]))
    ])
    embedding_layer = {
        'embedding_layer': embedding_layer,
        'language_model': layer,
        'text_input_signature': tf.keras.layers.Input(
            shape=(), dtype=tf.string)
    }
  elif cfg.embedding_type == 'nnlm':
    model_name = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1'
    layer = hub.KerasLayer(
        model_name,
        output_shape=[50],
        input_shape=[],
        dtype=tf.string,
        trainable=False)
    embedding_layer = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1])), layer,
        tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [-1, cfg.max_sequence_length, 50]))
    ])
    embedding_layer = {
        'embedding_layer': embedding_layer,
        'language_model': layer,
        'text_input_signature': tf.keras.layers.Input(
            shape=(), dtype=tf.string)
    }
  elif cfg.embedding_type == 'word2vec':
    model_name = 'https://tfhub.dev/google/Wiki-words-250/2'
    layer = hub.KerasLayer(
        model_name,
        output_shape=[250],
        input_shape=[],
        dtype=tf.string,
        trainable=False)
    embedding_layer = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1])), layer,
        tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [-1, cfg.max_sequence_length, 250]))
    ])
    embedding_layer = {
        'embedding_layer': embedding_layer,
        'language_model': layer,
        'text_input_signature': tf.keras.layers.Input(
            shape=(), dtype=tf.string)
    }
  else:
    raise ValueError('Unrecognized embedding type: {}'.format(
        cfg.embedding_type))
  return embedding_layer
