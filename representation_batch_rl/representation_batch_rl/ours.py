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

"""Implementation of Fisher-BRAC from pixels."""

import typing

from dm_env import specs as dm_env_specs
import numpy as np
from seed_rl.agents.policy_gradient.modules import popart
from seed_rl.agents.policy_gradient.modules import running_statistics
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs.tensor_spec import TensorSpec

from representation_batch_rl.batch_rl import critic as criticCL
from representation_batch_rl.batch_rl.encoders import ConvStack
from representation_batch_rl.batch_rl.encoders import ImageEncoder
from representation_batch_rl.batch_rl.encoders import make_impala_cnn_network
from representation_batch_rl.representation_batch_rl import tf_utils

tfd = tfp.distributions

# Lower-bound on possible returns for 200 easy levels of ProcGen based on PPO.
# See https://arxiv.org/abs/1912.01588 appendix.
reward_lowerbound_procgen = {
    'bigfish': 1.,
    'bossfight': 0.5,
    'caveflyer': 3.5,
    'chaser': 0.5,
    'climber': 2.,
    'coinrun': 5.,
    'dodgeball': 1.5,
    'fruitbot': -1.5,
    'heist': 3.5,
    'jumper': 3.,
    'leaper': 10.,
    'maze': 5.,
    'miner': 1.5,
    'ninja': 3.5,
    'plunder': 4.5,
    'starpilot': 2.5
}

# Upper-bound on possible returns for 200 easy levels of ProcGen based on PPO.
# See https://arxiv.org/abs/1912.01588 appendix.
reward_upperbound_procgen = {
    'bigfish': 40.,
    'bossfight': 13.,
    'caveflyer': 12.,
    'chaser': 13,
    'climber': 12.6,
    'coinrun': 10.,
    'dodgeball': 19.,
    'fruitbot': 32.4,
    'heist': 10,
    'jumper': 10.,
    'leaper': 10.,
    'maze': 10.,
    'miner': 13.,
    'ninja': 10.,
    'plunder': 30.,
    'starpilot': 64.
}


class OURS(object):
  """Class performing F-BRAC + SSL training."""

  def __init__(self,
               observation_spec,
               action_spec,
               actor_lr = 3e-4,
               critic_lr = 3e-4,
               alpha_lr = 3e-4,
               discount = 0.99,
               tau = 0.005,
               target_entropy = 0.0,
               f_reg = 1.0,
               reward_bonus = 5.0,
               num_augmentations = 1,
               rep_learn_keywords = 'outer',
               env_name = '',
               batch_size = 256,
               n_quantiles = 5,
               temp = 0.1,
               num_training_levels = 200,
               latent_dim = 256,
               n_levels_nce = 5,
               popart_norm_beta = 0.1):
    """Creates networks.

    Args:
      observation_spec: environment observation spec.
      action_spec: Action spec.
      actor_lr: Actor learning rate.
      critic_lr: Critic learning rate.
      alpha_lr: Temperature learning rate.
      discount: MDP discount.
      tau: Soft target update parameter.
      target_entropy: Target entropy.
      f_reg: Critic regularization weight.
      reward_bonus: Bonus added to the rewards.
      num_augmentations: Number of DrQ augmentations (crops)
      rep_learn_keywords: Representation learning loss to add (see below)
      env_name: Env name
      batch_size: Batch size
      n_quantiles: Number of GVF quantiles
      temp: Temperature of NCE softmax
      num_training_levels: Number of training MDPs (Procgen=200)
      latent_dim: Latent dimensions of auxiliary MLPs
      n_levels_nce: Number of MDPs to use contrastive loss on
      popart_norm_beta: PopArt normalization constant

    For `rep_learn_keywords`, pick from:
      stop_grad_FQI: whether to stop_grad TD/FQI critic updates?
      linear_Q: use a linear critic?

      successor_features: uses ||SF|| as cumulant
      gvf_termination: uses +1 if done else 0 as cumulant
      gvf_action_count: uses state-cond. action counts as cumulant

      nce: uses the multi-class dot-product InfoNCE objective
      cce: uses MoCo Categorical CrossEntropy objective
      energy: uses SimCLR + pairwise GVF distance (not fully tested)

    If no cumulant is specified, the reward will be taken as default one.
    """
    del actor_lr, critic_lr, alpha_lr, target_entropy
    self.action_spec = action_spec
    self.num_augmentations = num_augmentations
    self.rep_learn_keywords = rep_learn_keywords.split('__')
    self.batch_size = batch_size
    self.env_name = env_name
    self.stop_grad_fqi = 'stop_grad_FQI' in self.rep_learn_keywords
    critic_kwargs = {'hidden_dims': (1024, 1024)}
    self.latent_dim = latent_dim
    self.n_levels_nce = n_levels_nce
    hidden_dims = hidden_dims_per_level = (self.latent_dim, self.latent_dim)
    self.num_training_levels = int(num_training_levels)
    self.n_quantiles = n_quantiles
    self.temp = temp

    # Make 2 sets of weights:
    # - Critic
    # - Critic (target)
    # Optionally, make a 3rd set for per-level critics

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

    conv_stack_critic = conv_stack()
    conv_target_stack_critic = conv_stack()

    if observation_spec.shape == (64, 64, 3):
      conv_stack_critic.output_size = state_dim
      conv_target_stack_critic.output_size = state_dim
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

    criticCL.soft_update(
        self.encoder_per_level, self.encoder_per_level_target, tau=1.0)

    if self.num_augmentations == 0:
      dummy_state = tf.constant(np.zeros([1] + list(observation_spec.shape)))
    else:  # account for padding of +4 everywhere and then cropping out 68
      dummy_state = tf.constant(np.zeros(shape=[1, 68, 68, 3]))
    dummy_enc = critic_kwargs['encoder'](dummy_state)

    @tf.function
    def init_models():
      """This function initializes all auxiliary networks (state and action encoders) with dummy input (Procgen-specific, 68x68x3, 15 actions).
      """
      critic_kwargs['encoder'](dummy_state)
      critic_kwargs['encoder_target'](dummy_state)
      self.encoder_per_level(dummy_state)
      self.encoder_per_level_target(dummy_state)

    init_models()

    action_dim = action_spec.maximum.item() + 1

    self.action_dim = action_dim
    self.discount = discount
    self.tau = tau
    self.reg = f_reg
    self.reward_bonus = reward_bonus

    self.critic = criticCL.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder'],
        discrete_actions=True,
        linear='linear_Q' in self.rep_learn_keywords)
    self.critic_target = criticCL.Critic(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        encoder=critic_kwargs['encoder_target'],
        discrete_actions=True,
        linear='linear_Q' in self.rep_learn_keywords)

    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    self.task_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    self.br_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    if 'cce' in self.rep_learn_keywords:
      self.classifier = tf.keras.Sequential(
          [
              tf.keras.layers.Dense(self.latent_dim, use_bias=True),
              tf.keras.layers.ReLU(),
              tf.keras.layers.Dense(self.n_quantiles, use_bias=True)
          ],
          name='classifier')
    elif 'nce' in self.rep_learn_keywords:
      self.embedding = tf.keras.Sequential(
          [
              tf.keras.layers.Dense(self.latent_dim, use_bias=True),
              tf.keras.layers.ReLU(),
              tf.keras.layers.Dense(self.latent_dim, use_bias=True)
          ],
          name='embedding')

    # This snipet initializes all auxiliary networks (state and action encoders)
    # with dummy input (Procgen-specific, 68x68x3, 15 actions).
    dummy_state = tf.zeros((1, 68, 68, 3), dtype=tf.float32)
    phi_s = self.critic.encoder(dummy_state)
    phi_a = tf.eye(action_dim, dtype=tf.float32)
    if 'linear_Q' in self.rep_learn_keywords:
      _ = self.critic.critic1.state_encoder(phi_s)
      _ = self.critic.critic2.state_encoder(phi_s)
      _ = self.critic.critic1.action_encoder(phi_a)
      _ = self.critic.critic2.action_encoder(phi_a)
      _ = self.critic_target.critic1.state_encoder(phi_s)
      _ = self.critic_target.critic2.state_encoder(phi_s)
      _ = self.critic_target.critic1.action_encoder(phi_a)
      _ = self.critic_target.critic2.action_encoder(phi_a)
    if 'cce' in self.rep_learn_keywords:
      self.classifier(phi_s)
    elif 'nce' in self.rep_learn_keywords:
      self.embedding(phi_s)

    self.target_critic_to_use = self.critic_target
    self.critic_to_use = self.critic

    criticCL.soft_update(self.critic, self.critic_target, tau=1.0)

    self.cce = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE, from_logits=True)

    self.bc = None

    if 'successor_features' in self.rep_learn_keywords:
      self.output_dim_level = self.latent_dim
    elif 'gvf_termination' in self.rep_learn_keywords:
      self.output_dim_level = 1
    elif 'gvf_action_count' in self.rep_learn_keywords:
      self.output_dim_level = action_dim
    else:
      self.output_dim_level = action_dim

    self.task_critic_one = criticCL.Critic(
        state_dim,
        self.output_dim_level * self.num_training_levels,
        hidden_dims=hidden_dims_per_level,
        encoder=None,  # critic_kwargs['encoder'],
        discrete_actions=True,
        cross_norm=False)
    self.task_critic_target_one = criticCL.Critic(
        state_dim,
        self.output_dim_level * 200,
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
    criticCL.soft_update(
        self.task_critic_one, self.task_critic_target_one, tau=1.0)

    # Normalization constant beta, set to best default value as per PopArt paper
    self.reward_normalizer = popart.PopArt(
        running_statistics.EMAMeanStd(popart_norm_beta))
    self.reward_normalizer.init()

    if 'CLIP' in self.rep_learn_keywords or 'clip' in self.rep_learn_keywords:
      self.loss_temp = tf.Variable(
          tf.constant(0.0, dtype=tf.float32), name='loss_temp', trainable=True)

    self.model_dict = {
        'critic': self.critic,
        'critic_target': self.critic_target,
        'critic_optimizer': self.critic_optimizer,
        'br_optimizer': self.br_optimizer
    }

    self.model_dict['encoder_perLevel'] = self.encoder_per_level
    self.model_dict['encoder_perLevel_target'] = self.encoder_per_level_target
    self.model_dict['task_critic'] = self.task_critic_one
    self.model_dict['task_critic_target'] = self.task_critic_target_one

  @tf.function
  def infonce_by_class(self,
                       features,
                       classes,
                       target_features=None,
                       temp=1.,
                       n_batch=None):
    """InfoNCE between features of a given class vs other clases.

    Args:
      features: n_batch x n_features
      classes: n_batch x n_features
      target_features: optional target features for dot product
      temp: temperature parameter for softmax
      n_batch: int, optional dimension param
    Returns:
      nce_scores
    """
    if n_batch is None:
      n_batch = self.batch_size
    # \sum_ij A_i:A_:j
    # Picks elements of A which are the same class as A_i
    class_mapping = tf.einsum('ik,jk->ij', classes, classes)
    # outer_prod: n_batch x n_batch
    if target_features is None:
      outer_prod = tf.einsum('ik,jk->ij', features, features)
    else:
      outer_prod = tf.einsum('ik,jk->ij', features, target_features)

    scores = tf.nn.softmax(outer_prod / temp, -1)
    # Add all instances with class=i to numerator by summing over axis 1
    scores = tf.reduce_mean(class_mapping * scores, -1)
    # Apply log after softmax
    scores = tf.math.log(scores)

    return tf.reduce_mean(scores)

  @tf.function
  def infonce_by_class_level(self,
                             features,
                             classes,
                             target_features=None,
                             levels=None,
                             temp=1.,
                             n_batch=None):
    """InfoNCE between features of a given class vs other classes.

    Args:
      features: n_batch x n_features
      classes: n_batch x n_features
      target_features: optional target features for dot product
      levels: n_batch x n_levels, optional level ids
      temp: temperature parameter
      n_batch: int, optional dimension param
    Returns:
      nce_scores
    """
    assert temp > 0.
    if levels is None:
      return self.infonce_by_class(features, classes, target_features, temp,
                                   n_batch)
    if n_batch is None:
      n_batch = self.batch_size
    # \sum_ij A_i:A_:j
    # Picks elements of A which are the same class as A_i
    class_mapping = tf.einsum('ik,jk->ij', classes, classes)
    # outer_prod: n_batch x n_batch
    if target_features is None:
      outer_prod = tf.einsum('ik,jk->ij', features, features)
    else:
      outer_prod = tf.einsum('ik,jk->ij', features, target_features)

    level_mapping = tf.einsum('ik,jk->ij', (1.-levels), levels)

    scores = tf.nn.softmax(outer_prod / temp, -1)
    # Add all instances with class=i to numerator by summing over axis 1
    scores = tf.reduce_mean(level_mapping * class_mapping * scores, -1)
    # Apply log after softmax
    scores = tf.math.log(scores)

    return tf.reduce_mean(scores)

  @tf.function
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

    if self.num_augmentations > 1:
      target_q = 0.
      for i in range(self.num_augmentations):
        next_q1_i, next_q2_i = self.critic_target(
            next_states[i], actions=None, stop_grad_features=self.stop_grad_fqi)
        target_q_i = tf.expand_dims(
            rewards, 1) + self.discount * tf.expand_dims(
                discounts, 1) * tf.minimum(next_q1_i, next_q2_i)
        target_q += target_q_i
      target_q /= self.num_augmentations
    elif self.num_augmentations == 1:
      next_q1, next_q2 = self.critic_target(
          next_states[0], actions=None, stop_grad_features=self.stop_grad_fqi)
      target_q = tf.expand_dims(
          rewards, 1) + self.discount * tf.expand_dims(
              discounts, 1) * tf.minimum(next_q1, next_q2)
    else:
      next_q1, next_q2 = self.target_critic_to_use(
          next_states, actions=None, stop_grad_features=self.stop_grad_fqi)
      target_q = tf.expand_dims(rewards, 1) + self.discount * tf.expand_dims(
          discounts, 1) * tf.minimum(next_q1, next_q2)

    target_q = tf.gather_nd(target_q, indices=next_action_indices)
    trainable_variables = self.critic.trainable_variables

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(trainable_variables)

      if self.num_augmentations > 1:
        critic_loss = 0.
        q1 = 0.
        q2 = 0.
        for i in range(self.num_augmentations):
          q1_i, q2_i = self.critic_to_use(
              states[i], actions=None, stop_grad_features=self.stop_grad_fqi)
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
      elif self.num_augmentations == 1:
        q1, q2 = self.critic_to_use(
            states[0], actions=None, stop_grad_features=self.stop_grad_fqi)
        q = tf.minimum(q1, q2)
        critic_loss = (
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q1, indices=action_indices)) +
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q2, indices=action_indices)))
      else:
        q1, q2 = self.critic_to_use(
            states, actions=None, stop_grad_features=self.stop_grad_fqi)
        q = tf.minimum(q1, q2)
        critic_loss = (
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q1, indices=action_indices)) +
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q2, indices=action_indices)))

      # LSE from CQL
      cql_logsumexp = tf.reduce_logsumexp(q, 1)
      cql_loss = tf.reduce_mean(cql_logsumexp -
                                tf.gather_nd(q, indices=action_indices))
      # Jointly optimize both losses
      critic_loss = critic_loss + cql_loss

    critic_grads = tape.gradient(critic_loss,
                                 trainable_variables)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, trainable_variables))

    criticCL.soft_update(
        self.critic, self.critic_target, tau=self.tau)

    gn = tf.reduce_mean(
        [tf.linalg.norm(v) for v in critic_grads if v is not None])

    return {
        'q1': tf.reduce_mean(q1),
        'q2': tf.reduce_mean(q2),
        'critic_loss': critic_loss,
        'cql_loss': cql_loss,
        'critic_grad_norm': gn
    }

  @tf.function
  def fit_embedding(self, states, actions,
                    next_states, next_actions, rewards,
                    discounts, level_ids):
    """Fit embedding using contrastive objectives.

    Args:
      states: batch of states
      actions: batch of actions
      next_states: batch of next states
      next_actions: batch of next actions
      rewards: batch of next rewards
      discounts: batch of discounts
      level_ids: batch of level ids
    Returns:
      Dictionary with losses
    """
    del next_actions, discounts, next_states, rewards
    ssl_variables = self.critic.trainable_variables
    # Number of MDPs for which to compute quantiles
    n_levels = self.n_levels_nce
    if 'cce' in self.rep_learn_keywords or 'clip' in self.rep_learn_keywords:
      ssl_variables = ssl_variables + self.classifier.trainable_variables
    if 'nce' in self.rep_learn_keywords:
      ssl_variables = ssl_variables + self.embedding.trainable_variables
    # Track whether need to backprop over representation
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(ssl_variables)
      # Compute Q(s,a) as well as phi(s)
      q1, q2 = self.critic(
          states[0],
          actions=None,
          return_features=False,
          stop_grad_features=self.stop_grad_fqi)
      q = tf.minimum(q1, q2)

      rep_loss = tf.constant(0., dtype=tf.float32)

      n_quantiles = self.n_quantiles
      states_enc = [self.critic.encoder(states[0])]

      # Use encoder_per_level (NOT CQL encoder)
      states_level_enc = [tf.stop_gradient(self.encoder_per_level(states[0]))]
      q_level = tf.minimum(
          *self.task_critic_one(states_level_enc[0], actions=None))

      actions_argmax = tf.argmax(q, 1)
      action_indices = tf.stack([
          tf.range(tf.shape(actions)[0],
                   dtype=tf.int32), tf.cast(actions_argmax, dtype=tf.int32)
      ],
                                axis=-1)
      level_indices = tf.stack([
          tf.range(tf.shape(actions)[0],
                   dtype=tf.int32), tf.cast(level_ids, dtype=tf.int32)
      ],
                               axis=-1)

      q_level = tf.gather_nd(
          tf.reshape(q_level, (-1, 200, self.output_dim_level)),
          indices=level_indices)

      if ('successor_features' in self.rep_learn_keywords or
          'gvf_termination' in self.rep_learn_keywords):
        q_level_gathered = tf.norm(q_level, ord=1, axis=1)
      else:
        q_level_gathered = tf.gather_nd(q_level, indices=action_indices)

      if 'cce' in self.rep_learn_keywords:
        states_psi = states_enc[0]
        states_psi_target = tf.stop_gradient(states_enc[0])
      elif 'nce' in self.rep_learn_keywords:
        states_psi = self.embedding(states_enc[0])
        states_psi_target = tf.stop_gradient(
            self.embedding(states_level_enc[0]))

      uniques, _, counts = tf.unique_with_counts(level_ids)
      uniques = tf.cast(uniques, dtype=tf.int32)

      def compute_quantile_bins(level):
        idx = tf.math.equal(level_ids, level)
        quantiles_q_level = tfp.stats.quantiles(q_level_gathered[idx],
                                                n_quantiles)
        quantile_labels = tf.cast(
            tf.one_hot(
                tf.cast(
                    tfp.stats.find_bins(q_level_gathered[idx],
                                        quantiles_q_level),
                    dtype=tf.int32),
                depth=n_quantiles), tf.float32)
        return quantile_labels

      def compute_quantile_features(level):
        idx = tf.math.equal(level_ids, level)
        return states_psi[idx]

      def compute_target_quantile_features(level):
        idx = tf.math.equal(level_ids, level)
        return states_psi_target[idx]

      def rec_compute_quantile_levels(levels, ctr):
        if ctr <= 0:
          return tf.reshape(tf.one_hot(0, depth=200), (1, -1))
        else:
          return tf.concat([
              tf.one_hot(
                  level_ids[tf.math.equal(level_ids, levels[0])], depth=200),
              rec_compute_quantile_levels(levels[1:], ctr - 1)
          ], 0)

      def rec_compute_quantile_bins(levels, ctr):
        if ctr <= 0:
          return tf.zeros(shape=(1, n_quantiles))
        else:
          return tf.concat([
              compute_quantile_bins(levels[0]),
              rec_compute_quantile_bins(levels[1:], ctr - 1)
          ], 0)

      def rec_compute_quantile_features(levels, ctr):
        if ctr <= 0:
          return tf.zeros(shape=(1, self.latent_dim))
        else:
          return tf.concat([
              compute_quantile_features(levels[0]),
              rec_compute_quantile_features(levels[1:], ctr - 1)
          ], 0)

      def rec_compute_target_quantile_features(levels, ctr):
        if ctr <= 0:
          return tf.zeros(shape=(1, self.latent_dim))
        else:
          return tf.concat([
              compute_target_quantile_features(levels[0]),
              rec_compute_target_quantile_features(levels[1:], ctr - 1)
          ], 0)

      sorted_unique_levels = tf.gather(
          uniques, tf.argsort(counts, direction='DESCENDING'))
      quantile_bins = rec_compute_quantile_bins(sorted_unique_levels,
                                                n_levels)[:-1]
      quantile_features = rec_compute_quantile_features(
          sorted_unique_levels, n_levels)[:-1]

      quantile_levels = None
      if 'nce' in self.rep_learn_keywords:
        quantile_target_features = rec_compute_target_quantile_features(
            sorted_unique_levels, n_levels)[:-1]
        quantile_features = tf.linalg.l2_normalize(quantile_features, 1)
        quantile_target_features = tf.linalg.l2_normalize(
            quantile_target_features, 1)
        rep_loss += -self.infonce_by_class_level(
            features=quantile_features,
            target_features=quantile_target_features,
            classes=quantile_bins,
            levels=quantile_levels,
            temp=self.temp,
            n_batch=tf.shape(quantile_bins)[0])
      elif 'cce' in self.rep_learn_keywords:
        quantile_features = quantile_features / self.temp
        logits = self.classifier(quantile_features)
        rep_loss += tf.reduce_mean(
            self.cce(tf.argmax(quantile_bins, 1), logits))
      elif 'energy' in self.rep_learn_keywords:
        energy = tf.exp(
            -tf.reduce_sum(tf.abs(tf.expand_dims(q_level, 1) - q_level), -1))
        outer_prod = tf.einsum('ik,jk->ij', states_enc[0],
                               states_level_enc[0])
        scores = tf.nn.log_softmax(outer_prod / self.temp, -1)
        rep_loss += -tf.reduce_mean(tf.reduce_mean(energy * scores, -1))

      embedding_loss = self.reg * (rep_loss)

    br_grads = tape.gradient(embedding_loss, ssl_variables)
    self.br_optimizer.apply_gradients(zip(br_grads, ssl_variables))

    gn = tf.reduce_mean([tf.linalg.norm(v)
                         for v in br_grads if v is not None])

    metrics_dict = {
        'embedding_loss': embedding_loss,
        'embedding_grad_norm': gn
    }
    return metrics_dict

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
    if 'popart' in self.rep_learn_keywords:
      # The PopArt normalization normalizes the GVF's cumulant signal so that
      # it's not affected by the difference in scales across MDPs.
      mb_rewards = self.reward_normalizer.normalize_target(mb_rewards)

    trainable_variables = self.encoder_per_level.trainable_variables + self.task_critic_one.trainable_variables

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

    next_states = [self.encoder_per_level_target(mb_next_states[0])]
    next_q1, next_q2 = self.task_critic_target_one(
        next_states[0], actions=None)
    # Learn d-dimensional successor features
    if 'successor_features' in self.rep_learn_keywords:
      target_q = tf.concat(
          [next_states[0]] * 200, 1) + self.discount * tf.expand_dims(
              mb_discounts, 1) * tf.minimum(next_q1, next_q2)
    # Learn discounted episode termination
    elif 'gvf_termination' in self.rep_learn_keywords:
      target_q = tf.expand_dims(
          mb_discounts, 1) + self.discount * tf.expand_dims(
              mb_discounts, 1) * tf.minimum(next_q1, next_q2)
    # Learn discounted future action counts
    elif 'gvf_action_count' in self.rep_learn_keywords:
      target_q = tf.concat(
          [tf.one_hot(mb_actions, depth=self.action_dim)] * 200,
          1) + self.discount * tf.expand_dims(mb_discounts, 1) * tf.minimum(
              next_q1, next_q2)
    else:
      target_q = tf.expand_dims(
          mb_rewards, 1) + self.discount * tf.expand_dims(
              mb_discounts, 1) * tf.minimum(next_q1, next_q2)

    if ('successor_features' in self.rep_learn_keywords or
        'gvf_termination' in self.rep_learn_keywords or
        'gvf_action_count' in self.rep_learn_keywords):
      target_q = tf.reshape(target_q, (-1, 200, self.output_dim_level))
      target_q = tf.gather_nd(target_q, indices=level_ids)
    else:
      target_q = tf.gather_nd(target_q, indices=next_action_indices)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(trainable_variables)

      states = [self.encoder_per_level(mb_states[0])]
      q1_all, q2_all = self.task_critic_one(states[0], actions=None)

      q = tf.minimum(q1_all, q2_all)
      if ('successor_features' in self.rep_learn_keywords or
          'gvf_termination' in self.rep_learn_keywords or
          'gvf_action_count' in self.rep_learn_keywords):
        q1_all = tf.reshape(q1_all, (-1, 200, self.output_dim_level))
        q2_all = tf.reshape(q2_all, (-1, 200, self.output_dim_level))
        critic_loss = (
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q1_all, indices=level_ids)) +
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q2_all, indices=level_ids)))
      else:
        critic_loss = (
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q1_all, indices=action_indices)) +
            tf.losses.mean_squared_error(
                target_q, tf.gather_nd(q2_all, indices=action_indices)))

    critic_grads = tape.gradient(critic_loss, trainable_variables)

    self.task_critic_optimizer.apply_gradients(zip(critic_grads,
                                                   trainable_variables))

    criticCL.soft_update(
        self.encoder_per_level, self.encoder_per_level_target, tau=self.tau)
    criticCL.soft_update(
        self.task_critic_one, self.task_critic_target_one, tau=self.tau)

    gn = tf.reduce_mean(
        [tf.linalg.norm(v) for v in critic_grads if v is not None])

    return {
        'avg_level_critic_loss': tf.reduce_mean(critic_loss),
        'avg_q': tf.reduce_mean(q),
        'level_critic_grad_norm': gn
    }

  @tf.function
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
    next_actions_mu = transition.action[:, 1]
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

    if train_target == 'both':
      critic_dict = self.fit_critic(states_b2, actions_b2, next_states_b2,
                                    next_actions_b2, rewards_b2, discounts_b2)
      print('Updating per-task critics')
      ssl_dict = {}
      critic_distillation_dict = self.fit_task_critics(
          states_b1, actions_b1, next_states_b1, next_actions_pi_per_level,
          rewards_b1,
          discounts_b1, level_ids_b1)
      print('Done updating per-task critics')
      return {**ssl_dict, **critic_dict, **critic_distillation_dict}
    elif train_target == 'encoder':
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
      critic_dict = self.fit_critic(states_b2, actions_b2, next_states_b2,
                                    next_actions_b2, rewards_b2, discounts_b2)
      ssl_dict = self.fit_embedding(states_b1, actions_b1, next_states_b1,
                                    next_actions_b1, rewards_b1, discounts_b1,
                                    level_ids)

    return {**ssl_dict, **critic_dict, **critic_distillation_dict}

  @tf.function
  def act(self, states, data_aug=False):
    """Act from a batch of states.

    Args:
      states: batch of states
      data_aug: optional flag
    Returns:
      actions
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
    q1, q2 = self.critic_to_use(states, actions=None)
    q = tf.minimum(q1, q2)
    actions = tf.argmax(q, -1)
    return actions

  @tf.function
  def act_per_level(self, states, level_ids, data_aug=False):
    """Act from a batch of states, but with per-level critics.

    Args:
      states: batch of states
      level_ids: batch of level ids
      data_aug: optional flag
    Returns:
      actions
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
    features = self.encoder_per_level(states)
    # n_batch x 200 x 15
    q1, q2 = self.task_critic_one(features, actions=None)

    # n_batch x 200 x 15
    q = tf.minimum(q1, q2)
    # n_batch x 15
    level_ids = tf.stack([
        tf.range(tf.shape(q)[0], dtype=tf.int32),
        tf.cast(level_ids, dtype=tf.int32)
    ],
                         axis=-1)

    actions = tf.gather_nd(
        tf.argmax(tf.reshape(q, (-1, 200, self.action_dim)), -1), level_ids)
    return actions

  def save(self, path, step, overwrite_latest=True):
    """Saves all submodels into pre-defined directory.

    Args:
      path: str specifying model save path
      step: which iteration to save from
      overwrite_latest: depreceated, now handled via tf Checkpoints
    Returns:
      None
    """
    del overwrite_latest
    dir_list = tf.io.gfile.glob(path)
    if dir_list:
      for file in dir_list:
        # Delete all files from previous epoch
        tf.io.gfile.remove(file)
    for model_name, model in self.model_dict.items():
      model.save_weights(path + '/%s_%d' % (model_name, step))

    print('[Step %d] Saved model to %s' % (step, path))
