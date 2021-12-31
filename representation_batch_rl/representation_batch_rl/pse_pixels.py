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

# python3
"""Implementation of PSEs (based on https://github.com/google-research/google-research/blob/master/pse/dm_control/agents/pse_drq_agent.py)."""

import typing

from dm_env import specs as dm_env_specs
import numpy as np
from seed_rl.agents.policy_gradient.modules import popart
from seed_rl.agents.policy_gradient.modules import running_statistics
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import tf_utils

EPS = 1e-8


@tf.function
def cosine_similarity(x, y):
  """Computes cosine similarity between all pairs of vectors in x and y."""
  x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
  similarity_matrix = tf.reduce_sum(x_expanded * y_expanded, axis=-1)
  similarity_matrix /= (
      tf.norm(x_expanded, axis=-1) * tf.norm(y_expanded, axis=-1) + EPS)
  return similarity_matrix


@tf.function
def sample_indices(dim_x, size=128, sort=False):
  dim_x = tf.cast(dim_x, tf.int32)
  indices = tf.range(0, dim_x, dtype=tf.int32)
  indices = tf.random.shuffle(indices)[:size]
  if sort:
    indices = tf.sort(indices)
  return indices


@tf.function
def representation_alignment_loss(representation_1,
                                  representation_2,
                                  metric_vals,
                                  use_coupling_weights=False,
                                  coupling_temperature=0.1,
                                  return_representation=False,
                                  temperature=1.0):
  """PSE loss. Refer to https://github.com/google-research/google-research/blob/master/pse/dm_control/utils/helper_utils.py#L54 ."""  # pylint: disable=line-too-long
  if np.random.randint(2) == 1:
    representation_1, representation_2 = representation_2, representation_1
    metric_vals = tf.transpose(metric_vals)

  indices = sample_indices(tf.shape(metric_vals)[0], sort=return_representation)
  metric_vals = tf.gather(metric_vals, indices, axis=0)

  similarity_matrix = cosine_similarity(representation_1, representation_2)
  alignment_loss = contrastive_loss(
      similarity_matrix,
      metric_vals,
      temperature,
      coupling_temperature=coupling_temperature,
      use_coupling_weights=use_coupling_weights)

  if return_representation:
    return alignment_loss, similarity_matrix
  else:
    return alignment_loss


@tf.function
def contrastive_loss(similarity_matrix,
                     metric_values,
                     temperature,
                     coupling_temperature=1.0,
                     use_coupling_weights=True):
  """Contrative Loss with soft coupling."""
  assert temperature > 0.
  metric_shape = tf.shape(metric_values)
  similarity_matrix /= temperature
  neg_logits1 = similarity_matrix

  col_indices = tf.cast(tf.argmin(metric_values, axis=1), dtype=tf.int32)
  pos_indices1 = tf.stack(
      (tf.range(metric_shape[0], dtype=tf.int32), col_indices), axis=1)
  pos_logits1 = tf.gather_nd(similarity_matrix, pos_indices1)

  if use_coupling_weights:
    metric_values /= coupling_temperature
    coupling = tf.exp(-metric_values)
    pos_weights1 = -tf.gather_nd(metric_values, pos_indices1)
    pos_logits1 += pos_weights1
    negative_weights = tf.math.log((1.0 - coupling) + EPS)
    neg_logits1 += tf.tensor_scatter_nd_update(negative_weights, pos_indices1,
                                               pos_weights1)
  neg_logits1 = tf.math.reduce_logsumexp(neg_logits1, axis=1)
  return tf.reduce_mean(neg_logits1 - pos_logits1)


def _get_action(replay):
  if isinstance(replay, list):
    return np.array([x.action for x in replay])
  else:
    return replay.action


def _calculate_action_cost_matrix(ac1, ac2):
  diff = tf.expand_dims(ac1, axis=1) - tf.expand_dims(ac2, axis=0)
  return tf.cast(tf.reduce_mean(tf.abs(diff), axis=-1), dtype=tf.float32)


def metric_fixed_point_fast(cost_matrix, gamma=0.99, eps=1e-7):
  """Dynamic prograaming for calculating PSM."""
  d = np.zeros_like(cost_matrix)
  def operator(d_cur):
    d_new = 1 * cost_matrix
    discounted_d_cur = gamma * d_cur
    d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
    d_new[:-1, -1] += discounted_d_cur[1:, -1]
    d_new[-1, :-1] += discounted_d_cur[-1, 1:]
    return d_new

  while True:
    d_new = operator(d)
    if np.sum(np.abs(d - d_new)) < eps:
      break
    else:
      d = d_new[:]
  return d


def compute_metric(actions1, actions2, gamma):
  action_cost = _calculate_action_cost_matrix(actions1, actions2)
  return tf_metric_fixed_point(action_cost, gamma=gamma)


@tf.function
def tf_metric_fixed_point(action_cost_matrix, gamma):
  return tf.numpy_function(
      metric_fixed_point_fast, [action_cost_matrix, gamma], Tout=tf.float32)


class PSE(object):
  """Class performing CQL training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 1e-4,
               critic_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_entropy = 0.0,
               reg = 0.0,
               num_cql_actions = 10,
               embedding_dim = 512,
               bc_pretraining_steps = 40_000,
               min_q_weight = 10.0,
               num_augmentations = 1,
               rep_learn_keywords = 'outer',
               batch_size = 256,
               temperature = 1.):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_entropy: Target entropy.
      reg: Coefficient for out of distribution regularization.
      num_cql_actions: Number of actions to sample for CQL loss.
      embedding_dim: Size of embedding (now hardcoded)
      bc_pretraining_steps: Use BC loss instead of CQL loss for N steps.
      min_q_weight: CQL alpha.
      num_augmentations: Num of random crops
      rep_learn_keywords: Representation learning loss to add.
      batch_size: Batch size
      temperature: NCE softmax temperature
    """
    del embedding_dim
    self.num_augmentations = num_augmentations
    self.batch_size = batch_size
    self.rep_learn_keywords = rep_learn_keywords.split('__')
    self.temperature = temperature

    actor_kwargs = {}
    critic_kwargs = {}

    if observation_spec.shape == (64, 64, 3):
      # IMPALA for Procgen
      def conv_stack():
        return make_impala_cnn_network(
            depths=[16, 32, 32], use_batch_norm=False, dropout_rate=0.)

      state_dim = 256
    else:
      # Reduced architecture for DMC
      def conv_stack():
        return ConvStack(observation_spec.shape)
      state_dim = 50

    conv_stack_bc = conv_stack()
    conv_stack_actor = conv_stack()
    conv_stack_critic = conv_stack()
    conv_target_stack_critic = conv_stack()

    if observation_spec.shape == (64, 64, 3):
      conv_stack_bc.output_size = state_dim
      conv_stack_actor.output_size = state_dim
      conv_stack_critic.output_size = state_dim
      conv_target_stack_critic.output_size = state_dim
    # Combine and stop_grad some of the above conv stacks
    actor_kwargs['encoder_bc'] = ImageEncoder(
        conv_stack_bc, feature_dim=state_dim, bprop_conv_stack=True)
    actor_kwargs['encoder'] = ImageEncoder(
        conv_stack_critic, feature_dim=state_dim, bprop_conv_stack=False)
    critic_kwargs['encoder'] = ImageEncoder(
        conv_stack_critic, feature_dim=state_dim, bprop_conv_stack=True)
    # Note: the target critic does not share any weights.
    critic_kwargs['encoder_target'] = ImageEncoder(
        conv_target_stack_critic, feature_dim=state_dim, bprop_conv_stack=True)

    conv_stack_critic_per_level = conv_stack()
    conv_target_stack_critic_per_level = conv_stack()
    if observation_spec.shape == (64, 64, 3):
      conv_stack_critic_per_level.output_size = state_dim
      conv_target_stack_critic_per_level.output_size = state_dim

    self.encoder_per_level = ImageEncoder(
        conv_stack_critic_per_level,
        feature_dim=state_dim,
        bprop_conv_stack=True)
    self.encoder_per_level_target = ImageEncoder(
        conv_target_stack_critic_per_level,
        feature_dim=state_dim,
        bprop_conv_stack=True)

    critic.soft_update(
        self.encoder_per_level, self.encoder_per_level_target, tau=1.0)

    if self.num_augmentations == 0:
      dummy_state = tf.constant(
          np.zeros(shape=[1] + list(observation_spec.shape)))
    else:  # account for padding of +4 everywhere and then cropping out 68
      dummy_state = tf.constant(np.zeros(shape=[1, 68, 68, 3]))

    @tf.function
    def init_models():
      actor_kwargs['encoder_bc'](dummy_state)
      actor_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder_target'](dummy_state)
      self.encoder_per_level(dummy_state)
      self.encoder_per_level_target(dummy_state)

    init_models()

    hidden_dims = (256, 256)
    # self.actor = policies.CategoricalPolicy(state_dim, action_spec,
    #               hidden_dims=hidden_dims, encoder=actor_kwargs['encoder'])
    action_dim = action_spec.maximum.item() + 1

    self.action_dim = action_dim
    self.output_dim_level = action_dim

    self.log_alpha = tf.Variable(tf.math.log(1.0), trainable=True)
    self.log_cql_alpha = self.log_alpha
    self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    self.critic = critic.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder'],
        discrete_actions=True,
        linear='linear_Q' in self.rep_learn_keywords)
    self.critic_target = critic.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder_target'],
        discrete_actions=True,
        linear='linear_Q' in self.rep_learn_keywords)

    self.latent_dim = 256
    self.embedding = tf.keras.Sequential([
        tf.keras.layers.Dense(self.latent_dim, use_bias=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(self.latent_dim, use_bias=True)
    ],
                                         name='embedding')

    dummy_enc = critic_kwargs['encoder'](dummy_state)
    hidden_dims_per_level = (256, 256)
    self.task_critic_one = critic.Critic(
        state_dim,
        action_dim * 200,
        hidden_dims=hidden_dims_per_level,
        encoder=None,  # critic_kwargs['encoder'],
        discrete_actions=True,
        cross_norm=False)
    self.task_critic_target_one = critic.Critic(
        state_dim,
        action_dim * 200,
        hidden_dims=hidden_dims_per_level,
        encoder=None,  # critic_kwargs['encoder'],
        discrete_actions=True,
        cross_norm=False)
    self.task_critic_one(
        dummy_enc,
        actions=None,
        training=False,
        return_features=False,
        stop_grad_features=False)
    self.task_critic_target_one(
        dummy_enc,
        actions=None,
        training=False,
        return_features=False,
        stop_grad_features=False)

    @tf.function
    def init_models2():
      dummy_state = tf.zeros((1, 68, 68, 3), dtype=tf.float32)
      phi_s = self.critic.encoder(dummy_state)
      phi_a = tf.eye(15, dtype=tf.float32)
      if 'linear_Q' in self.rep_learn_keywords:
        phi2_s = self.critic.critic1.state_encoder(phi_s)
        _ = self.critic.critic2.state_encoder(phi_s)
        _ = self.critic.critic1.action_encoder(phi_a)
        _ = self.critic.critic2.action_encoder(phi_a)
        _ = self.critic_target.critic1.state_encoder(phi_s)
        _ = self.critic_target.critic2.state_encoder(phi_s)
        _ = self.critic_target.critic1.action_encoder(phi_a)
        _ = self.critic_target.critic2.action_encoder(phi_a)
        self.embedding(phi2_s)

    init_models2()

    norm_beta = 0.1
    self.reward_normalizer = popart.PopArt(
        running_statistics.EMAMeanStd(norm_beta))
    self.reward_normalizer.init()

    critic.soft_update(
        self.critic, self.critic_target, tau=1.0)
    critic.soft_update(
        self.task_critic_one, self.task_critic_target_one, tau=1.0)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    self.task_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.br_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    self.tau = tau

    self.reg = reg
    self.target_entropy = target_entropy
    self.discount = discount

    self.num_cql_actions = num_cql_actions
    self.bc_pretraining_steps = bc_pretraining_steps
    self.min_q_weight = min_q_weight

    self.bc = None

    self.model_dict = {
        'critic': self.critic,
        'critic_target': self.critic_target,
        'critic_optimizer': self.critic_optimizer,
        'alpha_optimizer': self.alpha_optimizer
    }

  @tf.function
  def fit_task_critics(self, mb_states, mb_actions,
                       mb_next_states, mb_next_actions,
                       mb_rewards, mb_discounts,
                       level_ids):
    """Updates per-level critic parameters.

    Args:
      mb_states: Batch of states.
      mb_actions: Batch of actions.
      mb_next_states: Batch of next states.
      mb_next_actions: Batch of next actions from training policy.
      mb_rewards: Batch of rewards.
      mb_discounts: Batch of masks indicating the end of the episodes.
      level_ids: Batch of level ids

    Returns:
      Dictionary with information to track.
    """
    mb_rewards = self.reward_normalizer.normalize_target(mb_rewards)
    trainable_variables = (self.encoder_per_level.trainable_variables
                           + self.task_critic_one.trainable_variables)

    next_action_indices = tf.stack([
        tf.range(tf.shape(mb_next_actions)[0],
                 dtype=tf.int32), level_ids * self.output_dim_level +
        tf.cast(mb_next_actions, dtype=tf.int32)
    ],
                                   axis=-1)

    action_indices = tf.stack([
        tf.range(tf.shape(mb_actions)[0], dtype=tf.int32),
        level_ids * self.output_dim_level + tf.cast(mb_actions, dtype=tf.int32)
    ],
                              axis=-1)
    level_ids = tf.stack([
        tf.range(tf.shape(mb_next_actions)[0],
                 dtype=tf.int32), tf.cast(level_ids, dtype=tf.int32)
    ],
                         axis=-1)

    if 'parallelPerLevel' in self.rep_learn_keywords:
      next_states = [self.encoder_per_level_target(mb_next_states[0])]
      next_q1, next_q2 = self.task_critic_target_one(
          next_states[0], actions=None)
      target_q = tf.expand_dims(
          mb_rewards, 1) + self.discount * tf.expand_dims(
              mb_discounts, 1) * tf.minimum(next_q1, next_q2)

    target_q = tf.gather_nd(target_q, indices=next_action_indices)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(trainable_variables)

      states = [self.encoder_per_level(mb_states[0])]
      q1_all, q2_all = self.task_critic_one(states[0], actions=None)

      q = tf.minimum(q1_all, q2_all)

      critic_loss = (
          tf.losses.mean_squared_error(
              target_q, tf.gather_nd(q1_all, indices=action_indices)) +
          tf.losses.mean_squared_error(
              target_q, tf.gather_nd(q2_all, indices=action_indices)))

    critic_grads = tape.gradient(critic_loss, trainable_variables)

    self.task_critic_optimizer.apply_gradients(zip(critic_grads,
                                                   trainable_variables))

    critic.soft_update(
        self.encoder_per_level, self.encoder_per_level_target, tau=self.tau)
    critic.soft_update(
        self.task_critic_one, self.task_critic_target_one, tau=self.tau)

    gn = tf.reduce_mean(
        [tf.linalg.norm(v) for v in critic_grads if v is not None])

    return {
        'avg_level_critic_loss': tf.reduce_mean(critic_loss),
        'avg_q': tf.reduce_mean(q),
        'level_critic_grad_norm': gn
    }

  def fit_critic(self, states, actions,
                 next_states, next_actions, rewards,
                 discounts):
    """Updates critic parameters.

    Args:
      states: Batch of states.
      actions: Batch of actions.
      next_states: Batch of next states.
      next_actions: Batch of next actions from training policy.
      rewards: Batch of rewards.
      discounts: Batch of masks indicating the end of the episodes.

    Returns:
      Dictionary with information to track.
    """
    action_indices = tf.stack(
        [tf.range(tf.shape(actions)[0], dtype=tf.int64), actions], axis=-1)
    next_action_indices = tf.stack(
        [tf.range(tf.shape(next_actions)[0], dtype=tf.int64), next_actions],
        axis=-1)

    if self.num_augmentations > 0:
      target_q = 0.
      for i in range(self.num_augmentations):
        next_q1_i, next_q2_i = self.critic_target(next_states[i], actions=None)
        target_q_i = tf.expand_dims(
            rewards, 1) + self.discount * tf.expand_dims(
                discounts, 1) * tf.minimum(next_q1_i, next_q2_i)
        target_q += target_q_i
      target_q /= self.num_augmentations
    else:
      next_q1, next_q2 = self.critic_target(next_states, actions=None)
      target_q = tf.expand_dims(rewards, 1) + self.discount * tf.expand_dims(
          discounts, 1) * tf.minimum(next_q1, next_q2)

    target_q = tf.gather_nd(target_q, indices=next_action_indices)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.critic.trainable_variables)

      if self.num_augmentations > 0:
        critic_loss = 0.
        q1 = 0.
        q2 = 0.
        for i in range(self.num_augmentations):
          q1_i, q2_i = self.critic(states[i], actions=None)
          critic_loss_i = (
              tf.losses.mean_squared_error(
                  target_q, tf.gather_nd(q1_i, indices=action_indices)) +
              tf.losses.mean_squared_error(
                  target_q, tf.gather_nd(q2_i, indices=action_indices)))
          q1 += q1_i
          q2 += q2_i
          critic_loss += critic_loss_i
        q1 /= self.num_augmentations
        q2 /= self.num_augmentations
        critic_loss /= self.num_augmentations
      else:
        q1, q2 = self.critic(states, actions=None)
      q = tf.minimum(q1, q2)
      critic_loss = (
          tf.losses.mean_squared_error(
              target_q, tf.gather_nd(q1, indices=action_indices)) +
          tf.losses.mean_squared_error(
              target_q, tf.gather_nd(q2, indices=action_indices)))

      cql_logsumexp = tf.reduce_logsumexp(q, 1)
      cql_loss = tf.reduce_mean(cql_logsumexp -
                                tf.gather_nd(q, indices=action_indices))

      critic_loss += (self.reg * cql_loss)

    critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.critic.trainable_variables))

    critic.soft_update(self.critic, self.critic_target, tau=self.tau)

    return {
        'q1': tf.reduce_mean(q1),
        'q2': tf.reduce_mean(q2),
        'critic_loss': critic_loss,
        'cql_loss': cql_loss
    }

  # @tf.function
  def fit_embedding(self, states, actions,
                    next_states, next_actions, rewards,
                    discounts, level_ids):
    """Fit representation (pixel encoder).

    Args:
      states: tf.tensor
      actions: tf.tensor
      next_states: tf.tensor
      next_actions: tf.tensor
      rewards: tf.tensor
      discounts: tf.tensor
      level_ids: tf.tensor, contains level id labels

    Returns:
      embedding_dict: dict

    """
    del next_actions, discounts, rewards, next_states
    ssl_variables = (self.critic.trainable_variables +
                     self.embedding.trainable_variables)

    with tf.GradientTape(watch_accessed_variables=True) as tape:
      uniques, _, counts = tf.unique_with_counts(level_ids)
      uniques = tf.cast(uniques, dtype=tf.int32)
      level_order = tf.argsort(counts, direction='DESCENDING')
      # Take two most frequent levels in the batch
      lvls = tf.gather(uniques, level_order)[:2]
      lvl_1, lvl_2 = lvls[0], lvls[1]
      min_count = tf.reduce_min(tf.gather(counts, level_order)[:2])

      idx_1 = tf.math.equal(level_ids, lvl_1)
      idx_2 = tf.math.equal(level_ids, lvl_2)

      act1 = tf.one_hot(actions[idx_1][:min_count], 15)
      act2 = tf.one_hot(actions[idx_2][:min_count], 15)

      representation = self.embedding(self.critic.encoder(states[0]))
      representation_1 = representation[idx_1][:min_count]
      representation_2 = representation[idx_2][:min_count]

      metric_vals = compute_metric(act1, act2, self.discount)

      embedding_loss = representation_alignment_loss(
          representation_1,
          representation_2,
          metric_vals,
          use_coupling_weights=False,
          temperature=self.temperature,
          return_representation=False)

    br_grads = tape.gradient(embedding_loss, ssl_variables)
    self.br_optimizer.apply_gradients(zip(br_grads, ssl_variables))

    gn = tf.reduce_mean([tf.linalg.norm(v) for v in br_grads if v is not None])

    return {
        'embedding_loss': embedding_loss,
        'embedding_grad_norm': gn
    }

  # @tf.function
  def update_step(self, replay_buffer_iter, train_target='both'):
    """Performs a single training step for critic and embedding.

    Args:
      replay_buffer_iter: A tensorflow graph iteratable object.
      train_target: string specifying whether update RL and or representation

    Returns:
      Dictionary with losses to track.
    """

    transition = next(replay_buffer_iter)
    numpy_dataset = isinstance(replay_buffer_iter, np.ndarray)
    # observation: n_batch x n_timesteps x 1 x H*W*3*n_frames x 1
    # -> n_batch x H x W x 3*n_frames
    if not numpy_dataset:
      states = transition.observation[:, 0]
      next_states = transition.observation[:, 1]
      actions = transition.action[:, 0]
      rewards = transition.reward
      level_ids = transition.policy_info[:, 0]
      if tf.shape(transition.reward)[1] > 2:
        rewards = tf.einsum(
            'ij,j->i', rewards,
            self.discount**tf.range(
                0, tf.shape(transition.reward)[1], dtype=tf.float32))
        self.n_step_rewards = tf.shape(transition.reward)[1]
      else:
        rewards = transition.reward[:, 0]
        self.n_step_rewards = 1
      discounts = transition.discount[:, 0]

      if transition.observation.dtype == tf.uint8:
        states = tf.cast(states, tf.float32) / 255.
        next_states = tf.cast(next_states, tf.float32) / 255.
    else:
      states, actions, rewards, next_states, discounts = transition

    self.reward_normalizer.update_normalization_statistics(rewards)

    if self.num_augmentations > 0:
      states, next_states = tf_utils.image_aug(
          states,
          next_states,
          img_pad=4,
          num_augmentations=self.num_augmentations,
          obs_dim=64,
          channels=3,
          cropped_shape=[self.batch_size, 68, 68, 3])

    next_actions_pi = self.act(next_states, data_aug=True)
    next_actions_mu = transition.action[:, 1]  # pylint: disable=unused-variable
    next_actions_pi_per_level = next_actions_mu

    states_b1 = states
    next_states_b1 = next_states
    actions_b1 = actions
    next_actions_b1 = next_actions_pi
    rewards_b1 = rewards
    discounts_b1 = discounts
    level_ids_b1 = level_ids

    states_b2 = states
    next_states_b2 = next_states
    actions_b2 = actions
    next_actions_b2 = next_actions_pi
    rewards_b2 = rewards
    discounts_b2 = discounts

    if train_target == 'encoder':
      print('Updating per-task critics')
      critic_distillation_dict = self.fit_task_critics(
          states_b1, actions_b1, next_states_b1, next_actions_pi_per_level,
          rewards_b1,
          discounts_b1, level_ids_b1)
      print('Done updating per-task critics')
      ssl_dict = {}
      critic_dict = {}
      return {**ssl_dict, **critic_distillation_dict}

    elif train_target == 'rl':
      critic_distillation_dict = {}
      print('Updating critic')
      critic_dict = self.fit_critic(states_b2, actions_b2, next_states_b2,
                                    next_actions_b2, rewards_b2, discounts_b2)
      print('Updating embedding')
      ssl_dict = self.fit_embedding(states_b1, actions_b1, next_states_b1,
                                    next_actions_b1, rewards_b1, discounts_b1,
                                    level_ids)
      print('Done')

    return {**ssl_dict, **critic_dict, **critic_distillation_dict}

  @tf.function
  def act(self, states, data_aug=False):
    """Act with batch of states.

    Args:
      states: tf.tensor n_batch x 64 x 64 x 3
      data_aug: bool, whether to use stochastic data aug (else deterministic)

    Returns:
      action: tf.tensor
    """
    if data_aug and self.num_augmentations > 0:
      states = states[0]
    if self.num_augmentations > 0:
      # use pad of 2 to bump 64 to 68 with 2 + 64 + 2 on each side
      img_pad = 2
      paddings = tf.constant(
          [[0, 0], [img_pad, img_pad], [img_pad, img_pad], [0, 0]],
          dtype=tf.int32)
      states = tf.cast(
          tf.pad(tf.cast(states * 255., tf.int32), paddings, 'SYMMETRIC'),
          tf.float32) / 255.

    q1, q2 = self.critic(states, actions=None)
    q = tf.minimum(q1, q2)
    actions = tf.argmax(q, -1)
    return actions
