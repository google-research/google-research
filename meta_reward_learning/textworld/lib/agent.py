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
"""Helper functions for agent class."""

import sys
import numpy as np
import tensorflow.compat.v1 as tf
from meta_reward_learning.textworld.lib import model as nn_model
from meta_reward_learning.textworld.lib.helpers import create_joint_features
from meta_reward_learning.textworld.lib.helpers import pad_sequences
from meta_reward_learning.textworld.lib.replay_buffer import Sample
from meta_reward_learning.textworld.lib.replay_buffer import Traj
from tensorflow.contrib import optimizer_v2 as contrib_optimizer_v2
from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib.layers.python.layers import optimizers as optimizers_lib


class Agent(object):
  """Agent class defining the functions each agent should have."""

  def __init__(self,
               num_actions=4,
               gamma=1.0,
               eps=0.0,
               seed=42,
               debug=False,
               log_every=100):
    self._num_actions = num_actions
    # Discount factor
    self._gamma = gamma
    # noise for exploration
    self._eps = eps
    self._debug = debug

    # Set random seed for reproducibility
    self._seed = seed
    tf.set_random_seed(self._seed)

    # Visualization related
    self.log_every = log_every
    self._meta_train = False
    self._counter = 0

    self.pi = None
    self.trainable_variables = []

  def _discount_rewards(self, rews):
    """Compute discounted rewards recursively."""
    discounted_rews = []
    last_r = 0.
    for r in reversed(rews):
      last_r = last_r * self._gamma + r
      discounted_rews.append(last_r)
    discounted_rews.reverse()
    return discounted_rews

  def _sample_action(self, logprobs, greedy=False):
    if greedy:
      actions = tf.argmax(logprobs, axis=-1).numpy()
    else:
      probs = tf.nn.softmax(logprobs, axis=-1).numpy()
      probs = (1 - self._eps) * probs + self._eps / self._num_actions
      actions = [np.random.choice(self._num_actions, p=p) for p in probs]
    return actions

  def generate_samples(self, envs, greedy=False):
    trajs = self.sample_trajs(envs, greedy=greedy)
    return [Sample(traj=traj, prob=1.0) for traj in trajs]

  def sample_trajs(self, envs, greedy=False):
    raise NotImplementedError

  def play(self, env, render=False, greedy=False):
    """Run the trained policy on the env."""
    env.reset(render=render)
    done = False
    kwargs = {'return_state': True}
    context = env.context + 1  # Zero is not a valid index
    enc_context, initial_state = self.pi.encode_context(context[np.newaxis, :])
    kwargs.update(enc_output=enc_context)
    # pylint: disable=protected-access
    model_fn = self.pi._call
    # pylint: enable=protected-access
    rews = []
    while not done:
      # We need to convert ob to a tensor, and add a batch dimension...
      logprobs, next_state = model_fn(
          num_inputs=1, initial_state=initial_state, **kwargs)
      logprobs = logprobs[:, 0]  # Remove the time dimension
      initial_state = next_state
      # ... and remove it again once we are done with TF ops
      ac = self._sample_action(logprobs, greedy=greedy)[0]
      rew, done = env.step(ac)
      rews.append(rew)
    return rews

  def update_eps(self, ep_count, num_episodes):
    self._eps *= 1.0 - (ep_count / num_episodes)

  def compute_logits(self,
                     contexts,
                     maxlen,
                     context_lengths,
                     return_state=False):
    raise NotImplementedError

  def compute_probs(self, trajs, env_dict):
    """Compute the probability of the trajs for contexts provided."""
    contexts = [env_dict[t.env_name].context + 1 for t in trajs]
    contexts, context_lengths, _ = pad_sequences(contexts, 0)
    contexts = tf.stack(contexts, axis=0)
    context_lengths = np.array(context_lengths, dtype=np.int32)

    all_actions, sequence_length, maxlen = pad_sequences(
        [t.actions for t in trajs], 0)
    batch_actions = tf.stack(all_actions, axis=0)
    logits = self.compute_logits(
        contexts, maxlen, context_lengths, return_state=False)
    seq_neg_logprobs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=batch_actions)
    seq_mask = tf.sequence_mask(sequence_length, dtype=seq_neg_logprobs.dtype)
    logprobs = -tf.reduce_sum(seq_neg_logprobs * seq_mask, axis=-1)
    probs = (tf.exp(logprobs)).numpy()
    return probs


class RLAgent(Agent):
  """Reinforcement Learning agent."""

  def __init__(self,
               units=16,
               objective='mapo',
               learning_rate=1e-3,
               entropy_reg_coeff=0.0,
               max_grad_norm=1.0,
               use_critic=False,
               log_summaries=False,
               **kwargs):

    super(RLAgent, self).__init__(num_actions=4, **kwargs)
    self._use_critic = use_critic
    self.max_grad_norm = max_grad_norm
    self.pi = nn_model.EncoderDecoder(
        embedding_dim=4, units=units, num_outputs=self._num_actions)
    if self._use_critic:
      self.value_fn = nn_model.EncoderDecoder(
          embedding_dim=4, units=units, num_outputs=1)
    self._objective = objective
    self._entropy_reg_coeff = entropy_reg_coeff  # Entropy regularization
    self.global_step = tf.train.get_or_create_global_step()
    self.optimizer = contrib_optimizer_v2.AdamOptimizer(
        learning_rate=learning_rate)
    # This is need so that the product with IndexedSlices object is defined
    self.learning_rate = tf.constant(learning_rate)
    self.log_summaries = log_summaries
    self._init_models()

  def _init_models(self):
    """Initialize all the neural networks.

    TF Eager will not instantiate the NN until it is called for the first time.
    This can results in errors, e.g. when loading weights.
    """
    ctxt = np.array([[0, 1, 2, 3]], dtype=np.int8) + 1
    self.pi(ctxt)
    self.trainable_variables = self.pi.trainable_variables
    tf.logging.info('Policy function:')
    tf.logging.info(self.pi.summary())
    if self._use_critic:
      self.value_fn(ctxt)
      tf.logging.info('\nValue function:')
      tf.logging.info(self.value_fn.summary())
      self.trainable_variables += self.value_fn.trainable_variables

  def compute_logits(self,
                     contexts,
                     maxlen,
                     context_lengths,
                     return_state=False):
    return self.pi(contexts, maxlen, context_lengths, return_state=return_state)

  def sample_trajs(self, envs, greedy=False):
    env_names = [env.name for env in envs]
    env_dict = {env.name: env for env in envs}

    for env in envs:
      env.reset()
    rews = {env.name: [] for env in envs}
    actions = {env.name: [] for env in envs}

    kwargs = {'return_state': True}
    contexts = [env.context + 1 for env in envs]  # Zero is not a valid index
    contexts, context_lengths, _ = pad_sequences(contexts, 0)
    contexts = tf.stack(contexts, axis=0)
    context_lengths = np.array(context_lengths, dtype=np.int32)
    encoded_context, initial_state = self.pi.encode_context(
        contexts, context_lengths=context_lengths)
    kwargs.update(enc_output=encoded_context)
    # pylint: disable=protected-access
    model_fn = self.pi._call
    # pylint: enable=protected-access

    while env_names:
      logprobs, next_state = model_fn(
          num_inputs=1, initial_state=initial_state, **kwargs)
      logprobs = logprobs[:, 0]
      acs = self._sample_action(logprobs, greedy=greedy)
      dones = []
      new_env_names = []
      for ac, name in zip(acs, env_names):
        rew, done = env_dict[name].step(ac)
        actions[name].append(ac)
        rews[name].append(rew)
        dones.append(done)
        if not done:
          new_env_names.append(name)
      env_names = new_env_names

      if env_names:
        # Remove the states of `done` environments from recurrent_state only if
        # at least one of them is not `done`
        if isinstance(next_state, tf.Tensor):
          next_state = [
              next_state[i] for i, done in enumerate(dones) if not done
          ]
          initial_state = tf.stack(next_state, axis=0)
        else:  # LSTM Tuple
          raise NotImplementedError

        enc_output = kwargs['enc_output']
        enc_output = [enc_output[i] for i, done in enumerate(dones) if not done]
        kwargs['enc_output'] = tf.stack(enc_output, axis=0)

    env_names = set([env.name for env in envs])
    features = [
        create_joint_features(actions[name], env_dict[name].context)
        for name in env_names
    ]
    # pylint: disable=g-complex-comprehension
    trajs = [
        Traj(
            env_name=name,
            actions=actions[name],
            features=f,
            rewards=rews[name]) for name, f in zip(env_names, features)
    ]
    # pylint: enable=g-complex-comprehension
    return trajs

  def _compute_value_loss(self,
                          discounted_rewards,
                          seq_mask=None,
                          contexts=None,
                          **kwargs):
    """Defines the value loss for the RL agent."""
    if not self._use_critic:
      return 0.0
    num_inputs = discounted_rewards.shape[1]
    values = self.value_fn(contexts, num_inputs, **kwargs)  # [batch_size]
    advantages = discounted_rewards - values
    value_loss = 0.5 * (advantages**2)
    if seq_mask is not None:
      value_loss *= seq_mask
    value_loss = tf.reduce_sum(value_loss, axis=-1)
    return value_loss

  def _compute_policy_loss(self,
                           discounted_rewards,
                           actions,
                           seq_mask=None,
                           weights=None,
                           contexts=None,
                           use_entropy_regularization=True,
                           **kwargs):
    """Defines the policy loss for the RL agent."""
    num_inputs = discounted_rewards.shape[1]
    if self._use_critic:
      values = self.value_fn(num_inputs, **kwargs)
      advantages = discounted_rewards - tf.stop_gradient(values)
    else:
      advantages = discounted_rewards
    logits = self.pi(contexts, num_inputs=num_inputs, **kwargs)
    # Compute PG surrogate and  flip sign for gradient ascent
    neg_logprobs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=actions)
    # Scale PG by the advantages
    policy_loss = advantages * neg_logprobs
    if weights is not None:
      policy_loss *= weights
    if use_entropy_regularization:
      seq_entropy = nn_model.entropy_from_logits(logits)
      policy_loss -= self._entropy_reg_coeff * seq_entropy
    if seq_mask is not None:
      policy_loss *= seq_mask
    policy_loss = tf.reduce_sum(policy_loss, axis=-1)
    return policy_loss

  def _compute_gradients(self,
                         actions,
                         discounted_rewards,
                         weights=None,
                         sequence_length=None,
                         loss_str='train',
                         use_entropy_regularization=True,
                         **kwargs):
    """Implement the policy gradient in TF."""
    if sequence_length is not None:
      seq_mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
    else:
      seq_mask = None
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.trainable_variables)
      # Returns 0.0 if critic is not being used
      value_loss = self._compute_value_loss(
          discounted_rewards, seq_mask=seq_mask, **kwargs)
      policy_loss = self._compute_policy_loss(
          discounted_rewards,
          actions,
          seq_mask=seq_mask,
          weights=weights,
          use_entropy_regularization=use_entropy_regularization,
          **kwargs)
      loss = tf.reduce_mean(policy_loss + value_loss)
    if self.log_summaries and (self._counter % self.log_every == 0):
      contrib_summary.scalar('{}_loss'.format(loss_str), loss)
    return tape.gradient(loss, self.trainable_variables)

  def create_batch(self, samples, contexts=None):
    """Helper method for creating batches of data."""
    batch_actions, batch_rews, batch_weights = [], [], []
    kwargs = {}
    # Padding required for recurrent policies
    trajs = [s.traj for s in samples]
    batch_actions, sequence_length, maxlen = pad_sequences(
        [t.actions for t in trajs], 0)
    batch_rews, _, _ = pad_sequences(
        [self._discount_rewards(t.rewards) for t in trajs], 0, maxlen)
    batch_weights = [[s.prob] * maxlen for s in samples]
    kwargs.update(sequence_length=np.array(sequence_length, dtype=np.int32))
    if contexts is None:
      raise ValueError('No Contexts passed.')
    contexts, context_lengths, _ = pad_sequences(contexts, 0)
    contexts = tf.stack(contexts, axis=0)
    context_lengths = np.array(context_lengths, dtype=np.int32)
    kwargs.update(contexts=contexts, context_lengths=context_lengths)
    batch_rews = np.array(batch_rews, dtype=np.float32)
    batch_actions = np.array(batch_actions, dtype=np.int32)  # [batch_size]
    return batch_actions, batch_rews, batch_weights, kwargs

  def update(self, samples, contexts):
    """Update the policy based on the training samples and contexts."""
    # To prevent memory leaks in tf eager
    if self._counter % self.log_every == 0:
      tf.set_random_seed(self._seed)
    batch_actions, batch_rews, batch_weights, kwargs = \
      self.create_batch(samples, contexts=contexts)
    grads = self._compute_gradients(batch_actions, batch_rews, batch_weights,
                                    **kwargs)
    grads, norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
    if self._debug and self._counter % self.log_every == 0:
      tf.print(
          'Epoch {}: Grad norm='.format(self._counter),
          norm,
          output_stream=sys.stdout)
    self.optimizer.apply_gradients(
        zip(grads, self.trainable_variables), global_step=self.global_step)
    self._counter += 1


class MetaRLAgent(RLAgent):
  """Agent for meta learning the score function."""

  def __init__(self, meta_lr, score_fn, **kwargs):
    super(MetaRLAgent, self).__init__(**kwargs)
    if score_fn == 'simple_linear':
      tf.logging.info('Using simple linear score function.')
      self.score_fn = nn_model.SimpleLinearNN()
    elif score_fn == 'linear':
      tf.logging.info('Using linear score function with priors.')
      self.score_fn = nn_model.LinearNN()
    else:
      raise NotImplementedError
    self._init_score_fn()
    self.score_optimizer = contrib_optimizer_v2.AdamOptimizer(
        learning_rate=meta_lr)
    self._meta_train = True
    # Adaptive gradient clipping
    self._score_grad_clipping = optimizers_lib.adaptive_clipping_fn(
        decay=0.9,
        report_summary=self.log_summaries,
        static_max_norm=self.max_grad_norm / 2.0,
        global_step=self.global_step)

  def _init_score_fn(self):
    self.score_fn(np.ones([1, 16 * 17], dtype=np.float32))
    tf.logging.info('Score function:')
    tf.logging.info(self.score_fn.summary())
    self._score_vars = self.score_fn.trainable_variables

  def compute_scores(self, trajs, return_tensors=False):
    features = np.array([t.features for t in trajs], dtype=np.float32)
    scores = self.score_fn(features)
    if return_tensors:
      return scores
    else:
      return scores.numpy()

  def update(self, samples, contexts, dev_samples, dev_contexts):
    if self._counter % 20 == 0:
      # To prevent memory leaks in tf eager
      tf.set_random_seed(self._seed)
    actions, rews, weights, kwargs = self.create_batch(
        samples, contexts=contexts)
    dev_actions, dev_rews, dev_weights, dev_kwargs = self.create_batch(
        dev_samples, contexts=dev_contexts)
    trajs = (s.traj for s in samples)
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape0:
      tape0.watch(self._score_vars)
      scores = self.compute_scores(trajs, return_tensors=True)
      scores = [
          tf.nn.softmax(x)
          for x in tf.split(scores, len(actions) // 10, axis=0)
      ]
      scores = tf.concat(scores, axis=0)
      rews = rews * tf.expand_dims(scores, axis=-1)
      grads = self._compute_gradients(actions, rews, weights, **kwargs)
      grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
      grads_and_vars = zip(grads, self.trainable_variables)
      new_vars = [v - self.learning_rate * g for g, v in grads_and_vars]
    self.optimizer.apply_gradients(grads_and_vars)
    grads_loss = self._compute_gradients(
        dev_actions,
        dev_rews,
        dev_weights,
        loss_str='dev',
        use_entropy_regularization=False,
        **dev_kwargs)
    score_grads = tape0.gradient(
        new_vars, self._score_vars, output_gradients=grads_loss)
    del tape0
    score_grads_and_vars = self._score_grad_clipping(
        zip(score_grads, self._score_vars))
    self.score_optimizer.apply_gradients(
        score_grads_and_vars, global_step=self.global_step)
    if self.log_summaries:
      grads = list(zip(*grads_and_vars)[0])
      score_grads = list(zip(*score_grads_and_vars)[0])
      contrib_summary.scalar('global_norm/train_grad', tf.global_norm(grads))
      contrib_summary.scalar('global_norm/meta_grad',
                             tf.global_norm(score_grads))
    if self._debug and (self._counter % self.log_every == 0):
      tf.print(
          'Epoch {} scores='.format(self._counter),
          scores[:20],
          summarize=10,
          output_stream=sys.stdout)
    self._counter += 1
