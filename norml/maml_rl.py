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

r"""NoRML (No-Reward Meta Learning) Implementation.

See the original paper at:
https://arxiv.org/pdf/1903.01063.pdf
See documentation in train_maml.py for how to run
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import time
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from norml import rollout_service
from norml.tools import utility

flags.DEFINE_string('logs', '/tmp/', 'Logging (tensorboard) path')
FLAGS = flags.FLAGS


class MAMLReinforcementLearning(object):
  """Reinforcement Learning MAML implementation.

  Initialize this class using a configuration (see config_maml.py).
  """

  def __init__(self, config, reporter=None, logdir=None, save_config=True):
    """Creates a new MAML Reinforcement Learning instance.

    Args:
      config: MAML configuration object (see config_maml.py).
      reporter: a function(step, current_avg_test_return) (see vizier_maml_rl).
      logdir: a directory to save logs/checkpoints to. If None, set FLAGS.logs.
      save_config: save the configuration object to the logdir?
    """
    self.reporter = reporter
    self.config = config
    self.logdir = logdir
    self.save_config = save_config
    self._configure()
    self._construct_network()
    self._create_summaries()
    self._create_savers()
    self.avg_test_reward_log = []
    self.rollout_service = rollout_service.RolloutServiceLocal(config)

  def _configure(self):
    """Initializes variables."""
    self.writer = None
    self.current_step = 0  # Outer loop iteration
    if self.config.random_seed is not None:
      self.random_seed = self.config.random_seed
    else:
      self.random_seed = random.randint(0, 1000000)
    tf.set_random_seed(self.random_seed)
    np.random.seed(self.random_seed)
    self.inner_lr_init = self.config.inner_lr_init
    self.outer_lr_init = self.config.outer_lr_init
    self.outer_lr_decay = bool(self.config.outer_lr_decay)
    self.num_inner_rollouts = self.config.num_inner_rollouts
    self.log_every = self.config.log_every
    self.num_outer_iterations = self.config.num_outer_iterations
    if self.logdir is None:
      self.logdir = FLAGS.logs
    self.tensorboard_dir = os.path.join(
        self.logdir, 'train', '%d_%d' % (time.time(), self.random_seed))
    self.network_generator = self.config.network_generator
    self.value_network_generator = self.config.value_network_generator
    self.task_generator = self.config.task_generator
    self.task_env_modifiers = self.config.task_env_modifiers
    self.loss_generator = self.config.loss_generator
    self.first_order = self.config.first_order
    self.learn_inner_lr = self.config.learn_inner_lr
    self.learn_inner_lr_tensor = self.config.learn_inner_lr_tensor
    self.fixed_tasks = self.config.fixed_tasks
    self.outer_optimizer_algo = self.config.outer_optimizer_algo
    self.tasks_batch_size = self.config.tasks_batch_size
    self.input_dims = self.config.input_dims
    self.output_dims = self.config.output_dims
    self.policy = self.config.policy
    self.learn_offset = self.config.learn_offset
    self.reward_disc = self.config.reward_disc
    self.whiten_values = self.config.whiten_values
    self.always_full_rollouts = self.config.always_full_rollouts
    self.advantage_function = self.config.advantage_function
    self.learn_advantage_function_inner = self.config.learn_advantage_function_inner
    if self.config.advantage_generator:
      self.advantage_generator = self.config.advantage_generator
    else:
      self.advantage_generator = None
    self.max_rollout_len = self.config.max_rollout_len
    if self.config.ppo:
      self.ppo = self.config.ppo
      self.ppo_clip_value = self.config.ppo_clip_value
    else:
      self.ppo = False
    if self.config.early_termination:
      self.early_termination = self.config.early_termination
    else:
      self.early_termination = None
    if self.save_config:
      utility.save_config(self.config, self.tensorboard_dir)

  def _construct_network(self):
    self._create_weights()
    self._construct_inner_networks()
    self._construct_outer_network()
    self._lin_reg_weights = {}

  def _create_weights(self):
    """Initializes network weights."""
    self.weights = self.network_generator.construct_network_weights()
    self.weights['policy_logstd'] = tf.Variable(
        self.config.pol_log_std_init, name='log_std')
    if self.learn_offset:
      with tf.variable_scope('offsets'):
        # TODO(kencaluwaerts) should this be initialized to 0 or randomly?
        self.e_weights = {}
        for weight_key in self.weights:
          self.e_weights[weight_key] = tf.Variable(
              tf.zeros(self.weights[weight_key].shape),
              name='offset_%s' % weight_key)
        self.e_weights['policy_logstd'] = tf.Variable(0.0, name='log_std')
    if not self.learn_inner_lr_tensor:
      self.inner_lr = tf.Variable(self.inner_lr_init, name='inner_lr')
    else:
      with tf.variable_scope('inner_lr'):
        self.inner_lr = {}
        for weight_key in self.weights:
          self.inner_lr[weight_key] = tf.Variable(
              self.inner_lr_init * tf.ones(self.weights[weight_key].shape),
              name='inner_lr_%s' % weight_key)
    if self.learn_advantage_function_inner:
      self.adv_weights = self.advantage_generator.construct_network_weights(
          'advantage')

  def _create_savers(self):
    self.all_saver = tf.train.Saver()

  def _save_variables(self, session):
    self.all_saver.save(
        session,
        os.path.join(self.tensorboard_dir, 'all_weights.ckpt'),
        global_step=self.current_step)

  def restore(self, session, path):
    if '.ckpt-' in path:
      self.all_saver.restore(session, path)
    else:
      self.all_saver.restore(session, tf.train.latest_checkpoint(path))

  def _compute_returns_and_values(self, task_idx, rollouts):
    """Computes discounted returns and estimate values function."""
    val_targets = []
    ml = self.max_rollout_len
    for rollout in rollouts:
      val_inputs = np.hstack((
          rollout['states'],
          rollout['states']**2,
          rollout['states']**3,
          (1. * rollout['timesteps'][:, :1]) / ml,
          ((1. * rollout['timesteps'][:, :1]) / ml)**2,
          ((1. * rollout['timesteps'][:, :1]) / ml)**3,
      ))
      input_data = np.hstack((val_inputs, np.ones((val_inputs.shape[0], 1))))
      rollout['values'] = input_data.dot(self._lin_reg_weights[task_idx])

      val_targets.append(rollout['returns'])

    mean_return = np.mean(np.vstack(val_targets))
    std_return = np.std(np.vstack(val_targets))
    for rollout in rollouts:
      if self.whiten_values:
        rollout['values'] = rollout['values'] * std_return + mean_return
      else:
        rollout['values'] = rollout['values']

  def _update_value_function(self, task_idx, rollouts):
    """Update the value function estimator. Uses the baseline from RLLab."""
    for rollout in rollouts:
      rollout['returns'] = np.copy(rollout['rewards'])
      # Need to set the stop value of range to -1 since the array is 0-indexed
      for i in range(len(rollout['returns']) - 2, -1, -1):
        rollout['returns'][i] += rollout['returns'][i + 1] * self.reward_disc

    inps = []
    val_targets = []
    ml = self.max_rollout_len
    for rollout in rollouts:
      inp = np.hstack((
          rollout['states'][:-1],
          rollout['states'][:-1]**2,
          rollout['states'][:-1]**3,
          (1. * rollout['timesteps'][:-1, :1]) / ml,
          ((1. * rollout['timesteps'][:-1, :1]) / ml)**2,
          ((1. * rollout['timesteps'][:-1, :1]) / ml)**3,
      ))
      inps.append(inp)
      val_targets.append(rollout['returns'])

    inputs = np.vstack(inps)
    targets = np.vstack(val_targets)
    if self.whiten_values:
      targets -= np.mean(targets)
      targets /= np.std(targets)

    input_data = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
    linreg_weights = np.linalg.inv(
        input_data.T.dot(input_data) + np.eye(input_data.shape[1]) * 1e-5).dot(
            input_data.T.dot(targets))
    test_values = input_data.dot(linreg_weights)
    self._lin_reg_weights[task_idx] = linreg_weights
    tf.logging.debug('mean abs error value: %f',
                     np.mean(np.abs(targets - test_values)))

  def _construct_inner_networks(self):
    """Creates the Tensorflow subgraph for the inner optimization loop."""
    self.inner_train_inputs = []
    self.inner_train_outputs = []  # for debugging
    self.inner_train_next_inputs = []
    self.inner_train_actions = []
    self.inner_train_advantages = []
    self.inner_test_inputs = []
    self.inner_test_outputs = []
    self.inner_test_actions = []
    self.inner_test_advantages = []
    self.inner_train_losses = []
    self.inner_test_losses = []
    self.train_policies = []
    self.test_policies = []

    self.all_test_weights = []

    # inner "train" networks, 1 per task
    # technically, all these networks do the same,
    # just makes the code easier to maintain.
    for idx in range(self.tasks_batch_size):
      tf.logging.info('creating task train network: %d', idx)
      with tf.name_scope('task_%d' % idx):
        with tf.name_scope('train'):
          # Inner network: train
          network_input_train = tf.placeholder(
              tf.float32,
              shape=(None, self.input_dims),
              name='network_input_train_%d' % idx)
          network_output_inner_train = self.network_generator.construct_network(
              network_input_train,
              self.weights,
              scope='network_inner_train_%d' % idx)
          network_next_input_train = tf.placeholder(
              tf.float32,
              shape=(None, self.input_dims),
              name='network_next_input_train_%d' % idx)

          # Slap a policy on top of the network
          train_policy = self.policy(network_input_train,
                                     network_output_inner_train,
                                     self.output_dims,
                                     self.weights['policy_logstd'])

          self.train_policies.append(train_policy)
          self.inner_train_inputs.append(network_input_train)
          self.inner_train_outputs.append(network_output_inner_train)
          self.inner_train_next_inputs.append(network_next_input_train)

          # Compute policy gradient for this task
          # == gradient of expected reward wrt weights
          # We need a batch of rollouts for this.
          train_actions = tf.placeholder(
              tf.float32,
              shape=(None, self.output_dims),
              name='network_actions_train_%d' % idx)
          if not self.learn_advantage_function_inner:
            train_advantages = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='network_advantages_train_%d' % idx)
          else:
            adv_input = tf.concat(
                [network_next_input_train, network_input_train, train_actions],
                1)
            train_advantages = self.advantage_generator.construct_network(
                adv_input,
                self.adv_weights,
                scope='network_advantages_train_%d' % idx)

          train_policy_log_prob = train_policy.log_likelihood_op(train_actions)

          if self.ppo and (not self.learn_advantage_function_inner):
            # use PPO only if the advantage function is not learned
            old_train_policy_log_prob = tf.stop_gradient(train_policy_log_prob)
            ratio = tf.exp(train_policy_log_prob - old_train_policy_log_prob)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.ppo_clip_value,
                                             1 + self.ppo_clip_value)
            loss_inner_train = -tf.reduce_mean(
                tf.minimum(clipped_ratio * train_advantages,
                           ratio * train_advantages))
          else:
            loss_inner_train = -tf.reduce_mean(
                train_policy_log_prob * train_advantages)

          self.inner_train_actions.append(train_actions)
          self.inner_train_advantages.append(train_advantages)
          self.inner_train_losses.append(loss_inner_train)

          grad_inner_train = {}
          for weight_key in self.weights:
            grad_inner_train[weight_key] = tf.gradients(
                loss_inner_train,
                self.weights[weight_key],
                name='%s_inner_%d' % (weight_key, idx))[0]

          test_weights = {}
          for weight_key in self.weights:
            theta = self.weights[weight_key]
            if self.first_order:
              grad = tf.stop_gradient(grad_inner_train[weight_key])
            else:
              grad = grad_inner_train[weight_key]

            if not self.learn_inner_lr_tensor:
              a = self.inner_lr
            else:
              a = self.inner_lr[weight_key]

            if self.learn_offset:
              e = self.e_weights[weight_key]
              test_weights[weight_key] = theta - a * grad + e
            else:
              test_weights[weight_key] = theta - a * grad

    # inner "test" networks, 1 per task, weights = 1 gradient step of
    # corresponding "train" network
      with tf.name_scope('test'):
        # Inner network: test
        network_input_test = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dims),
            name='network_input_test_%d' % idx)
        network_output_inner_test = self.network_generator.construct_network(
            network_input_test,
            test_weights,
            scope='network_inner_test_%d' % idx)

        # Slap a policy on top of the network
        test_policy = self.policy(network_input_test, network_output_inner_test,
                                  self.output_dims,
                                  test_weights['policy_logstd'])
        self.test_policies.append(test_policy)

        test_actions = tf.placeholder(
            tf.float32,
            shape=(None, self.output_dims),
            name='network_actions_test_%d' % idx)
        test_advantages = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='network_advantages_test_%d' % idx)
        test_policy_log_prob = test_policy.log_likelihood_op(test_actions)

        if not self.ppo:
          loss_inner_test = -tf.reduce_mean(test_policy_log_prob *
                                            (test_advantages))
        else:
          old_test_policy_log_prob = tf.stop_gradient(test_policy_log_prob)
          ratio = tf.exp(test_policy_log_prob - old_test_policy_log_prob)
          clipped_ratio = tf.clip_by_value(ratio, 1 - self.ppo_clip_value,
                                           1 + self.ppo_clip_value)
          loss_inner_test = -tf.reduce_mean(
              tf.minimum(clipped_ratio * test_advantages,
                         ratio * test_advantages))
        # sum up all loss_inner_test variables to compute outer loss
        self.inner_test_losses.append(loss_inner_test)
        self.inner_test_inputs.append(network_input_test)
        self.inner_test_outputs.append(network_output_inner_test)
        self.inner_test_actions.append(test_actions)
        self.inner_test_advantages.append(test_advantages)
        self.all_test_weights.append(test_weights)

  def _construct_outer_network(self):
    """Creates the Tensorflow subgraph for the outer loop (meta-learning)."""
    with tf.name_scope('outer'):
      self.global_loss = tf.add_n(
          self.inner_test_losses, name='outer_loss') / self.tasks_batch_size
      if self.outer_lr_decay:
        self.outer_lr_ph = tf.placeholder(tf.float32, [])
        self.outer_optimizer = self.outer_optimizer_algo(self.outer_lr_ph)
      else:
        self.outer_optimizer = self.outer_optimizer_algo(self.outer_lr_init)
      self.outer_weights = []
      self.outer_weights.extend(self.weights.values())
      if self.learn_advantage_function_inner:
        self.outer_weights.extend(self.adv_weights.values())
      if self.learn_inner_lr:
        if self.learn_inner_lr_tensor:
          self.outer_weights.extend(self.inner_lr.values())
        else:
          self.outer_weights.append(self.inner_lr)
      if self.learn_offset:
        self.outer_weights.extend(self.e_weights.values())
      self.outer_grads_vars = self.outer_optimizer.compute_gradients(
          self.global_loss, self.outer_weights)
      self.apply_grads_outer = self.outer_optimizer.apply_gradients(
          self.outer_grads_vars)

  def _create_summaries(self):
    """Initializes learning summaries."""
    arr_summaries = []

    self.avg_train_reward = tf.placeholder(tf.float32, name='avg_train_reward')
    self.avg_test_reward = tf.placeholder(tf.float32, name='avg_test_reward')
    arr_summaries.append(tf.summary.scalar('outer_loss', self.global_loss))
    arr_summaries.append(
        tf.summary.scalar('avg_test_reward', self.avg_test_reward))
    arr_summaries.append(
        tf.summary.scalar('avg_train_reward', self.avg_train_reward))

    if self.learn_inner_lr_tensor:
      for weight_key in self.inner_lr:
        arr_summaries.append(
            tf.summary.scalar('inner_lr_norm_%s' % weight_key,
                              tf.norm(self.inner_lr[weight_key])))
    else:
      arr_summaries.append(tf.summary.scalar('inner_lr', self.inner_lr))

    self.summaries = tf.summary.merge(arr_summaries)

  def _flatten_rollouts(self, rollouts, no_reward=False):
    """Flattens the sample data from multiple rollouts into a single matrix.

    Args:
      rollouts: a list of trial data. Each trial contains states, actions,
        rewards, advantages, values, timesteps, and returns.
      no_reward: whether the reward, advantage and value estimation should be
        retained.

    Returns:
      samples: a dict with one matrix for states, actions, rewards, timesteps,
      values, and advantages.
        Note: The last index in states is discarded (final state).
    """
    states = np.vstack([rollout['states'][:-1] for rollout in rollouts])
    next_states = np.vstack([rollout['states'][1:] for rollout in rollouts])
    actions = np.vstack([rollout['actions'] for rollout in rollouts])
    rewards = np.vstack([rollout['rewards'] for rollout in rollouts])
    advantages = np.vstack([rollout['advantages'] for rollout in rollouts])
    returns = np.vstack([rollout['returns'] for rollout in rollouts])
    values = np.vstack([rollout['values'] for rollout in rollouts])
    timesteps = np.vstack([rollout['timesteps'][:-1] for rollout in rollouts])

    if no_reward:
      rewards *= 0.
      advantages *= 0.
      values *= 0.
    samples = {
        'timesteps': timesteps,
        'states': states,
        'next_states': next_states,
        'actions': actions,
        'rewards': rewards,
        'advantages': advantages,
        'returns': returns,
        'values': values,
    }
    return samples

  def _compute_advantages(self, rollouts):
    """Computes advantage values for a list of rollouts.

    Args:
      rollouts: list of rollouts. Each rollout is a dict with 'reward' and
        'state' arrays.

    Returns:
      the average return value of all rollouts
    """
    if self.advantage_function == 'returns-values':
      for rollout in rollouts:
        rollout['advantages'] = rollout['returns'] - rollout['values'][:-1]
    elif self.advantage_function == 'reward-values':
      for rollout in rollouts:
        rollout['advantages'] = rollout['rewards'] + (
            self.reward_disc * rollout['values'][1:] - rollout['values'][:-1])
    else:
      for rollout in rollouts:
        rollout['advantages'] = rollout['returns']

    avg_return = np.mean([np.sum(rollout['rewards']) for rollout in rollouts])

    return avg_return

  def train(self,
            session,
            num_outer_iterations=1,
            dont_update_weights=False,
            ignore_termination=False):
    """Performs one or multiple training steps.

    Per task: rollout train samples, rollout test samples
    Outer: computer outer loss gradient update

    Args:
      session: TF session.
      num_outer_iterations: Number of outer loop steps (gradient steps).
      dont_update_weights: Run the algorithm, but don't update any parameters.
      ignore_termination: Ignore early termination and max iterations condition.

    Returns:
      Objective value after optimization last step.

    Raises:
      ValueError: if the loss is NaN
    """
    inner_tasks = random.sample(self.task_env_modifiers, self.tasks_batch_size)
    done = False
    avg_test_reward = np.NaN
    for step in range(self.current_step,
                      self.current_step + num_outer_iterations):
      if ignore_termination:
        done = False
      elif self.current_step >= self.num_outer_iterations:
        done = True
      elif self.early_termination is not None:
        done = self.early_termination(self.avg_test_reward_log)
      if done:
        break
      self.current_step = step + 1
      tf.logging.info('iteration: %d', self.current_step)
      print('iteration: %d' % self.current_step)
      if not self.fixed_tasks:
        inner_tasks = random.sample(self.task_env_modifiers,
                                    self.tasks_batch_size)

      # If we do rollouts locally, don't parallelize inner train loops
      results = []
      for task_idx in range(self.tasks_batch_size):
        results.append(self.train_inner((session, inner_tasks, task_idx)))

      samples = {}
      avg_train_reward = 0.
      avg_test_reward = 0.

      for task_idx in range(self.tasks_batch_size):
        # training rollouts
        train_rollouts, train_reward, test_rollouts, test_reward = results[
            task_idx]
        avg_train_reward += train_reward
        avg_test_reward += test_reward

        samples[self.inner_train_inputs[task_idx]] = train_rollouts['states']
        samples[self.inner_train_next_inputs[task_idx]] = train_rollouts[
            'next_states']
        samples[self.inner_train_actions[task_idx]] = train_rollouts['actions']
        if not self.learn_advantage_function_inner:
          samples[self.inner_train_advantages[task_idx]] = train_rollouts[
              'advantages']
        samples[self.inner_test_inputs[task_idx]] = test_rollouts['states']
        samples[self.inner_test_actions[task_idx]] = test_rollouts['actions']
        samples[
            self.inner_test_advantages[task_idx]] = test_rollouts['advantages']

      # Normalize advantage for easier parameter tuning
      samples[self.inner_test_advantages[task_idx]] -= np.mean(
          samples[self.inner_test_advantages[task_idx]])
      samples[self.inner_test_advantages[task_idx]] /= np.std(
          samples[self.inner_test_advantages[task_idx]])

      avg_test_reward /= self.tasks_batch_size
      avg_train_reward /= self.tasks_batch_size
      self.avg_test_reward_log.append(avg_test_reward)

      if not dont_update_weights:
        if self.outer_lr_decay:
          samples[self.outer_lr_ph] = self.outer_lr_init * (
              1. - float(step) / self.num_outer_iterations)
        session.run(self.apply_grads_outer, samples)
      print('avg train reward: %f' % avg_train_reward)
      print('avg test reward: %f' % avg_test_reward)
      tf.logging.info('avg train reward: %f', avg_train_reward)
      tf.logging.info('avg test reward: %f', avg_test_reward)
      samples[self.avg_test_reward] = avg_test_reward
      samples[self.avg_train_reward] = avg_train_reward
      eval_summaries = session.run(self.summaries, samples)
      self.writer.add_summary(eval_summaries, self.current_step)

      if self.reporter is not None:
        eval_global_loss = session.run(self.global_loss, samples)
        if np.isnan(eval_global_loss):
          print('Loss is NaN')
          tf.logging.info('Loss is NaN')
          raise ValueError('Loss is NaN')
        else:
          self.reporter(self.current_step, avg_test_reward)

      if step % self.log_every == 0:
        print('Saving (%d) to: %s' % (self.current_step, self.tensorboard_dir))
        tf.logging.info('Saving (%d) to: %s', self.current_step,
                        self.tensorboard_dir)
        self._save_variables(session)

    return done, avg_test_reward

  def train_inner(self, args):
    """Performs inner (task-specific) training of MAML.

    This method rollouts out the meta policy for the specified task, computes
    the advantage and fine-tuned policy, and rollouts out the fine-tuned policy
    to compute the performance of fine-tuned policy.

    Args:
      args: a tuple containing the following three objects:
        -session: a TF session for all neural-net related stuff.
        -inner_tasks: the list of inner tasks.
        -task_idx: the task index to perform inner-train on.

    Returns:
      train_rollouts_flat: flattened rollouts using the meta policy.
      train_reward: reward using the meta policy.
      test_rollouts_flat: flattened rollouts using the inner-test (finetuned)
                          policy.
      test_reward: reward using the inner-test (finetuned) policy.
    """
    session, inner_tasks, task_idx = args
    feed_dict = {}
    task = inner_tasks[task_idx]
    train_rollouts = self.rollout_service.perform_rollouts(
        session, self.num_inner_rollouts, self.train_policies[task_idx], task,
        feed_dict)
    # note: this modifies the train_rollouts variable to include advantages
    self._update_value_function(task_idx, train_rollouts)
    self._compute_returns_and_values(task_idx, train_rollouts)
    train_reward = self._compute_advantages(train_rollouts)
    train_rollouts_flat = self._flatten_rollouts(
        train_rollouts, no_reward=self.learn_advantage_function_inner)

    feed_dict[
        self.inner_train_actions[task_idx]] = train_rollouts_flat['actions']
    feed_dict[self.inner_train_inputs[task_idx]] = train_rollouts_flat['states']
    feed_dict[self.inner_train_next_inputs[task_idx]] = train_rollouts_flat[
        'next_states']
    if not self.learn_advantage_function_inner:
      feed_dict[self.inner_train_advantages[task_idx]] = train_rollouts_flat[
          'advantages']

    test_rollouts = self.rollout_service.perform_rollouts(
        session, self.num_inner_rollouts, self.test_policies[task_idx], task,
        feed_dict)
    self._update_value_function(task_idx, test_rollouts)
    self._compute_returns_and_values(task_idx, test_rollouts)
    test_reward = self._compute_advantages(test_rollouts)
    test_rollouts_flat = self._flatten_rollouts(test_rollouts)
    print('Task {}: train_rewards: {}, test_reward: {}'.format(
        task_idx, train_reward, test_reward))
    return train_rollouts_flat, train_reward, test_rollouts_flat, test_reward

  def finetune(self, session, task_modifier):
    """Performs MAML fine-tuning at test-time.

    In order to perform fine-tuning, this function rolls out the meta policy and
    computes the fine-tuned policy as usual. Then, it updates the weights of the
    meta policy with the fine-tuned weights so that multiple gradient steps
    could be performed.

    Args:
      session: tf session.
      task_modifier: gym env modifier function
    """
    task_idx = 0  # use the fine-tune network constructed before
    train_rollouts = self.rollout_service.perform_rollouts(
        session, self.num_inner_rollouts, self.train_policies[task_idx],
        task_modifier)
    self._update_value_function(task_idx, train_rollouts)
    self._compute_returns_and_values(task_idx, train_rollouts)
    self._compute_advantages(train_rollouts)
    train_rollouts_flat = self._flatten_rollouts(
        train_rollouts, self.learn_advantage_function_inner)

    feed_dict = {}
    feed_dict[
        self.inner_train_actions[task_idx]] = train_rollouts_flat['actions']
    feed_dict[self.inner_train_inputs[task_idx]] = train_rollouts_flat['states']
    feed_dict[self.inner_train_next_inputs[task_idx]] = train_rollouts_flat[
        'next_states']
    if not self.learn_advantage_function_inner:
      feed_dict[self.inner_train_advantages[task_idx]] = train_rollouts_flat[
          'advantages']

    finetuned_weights_val = {}
    for name in self.all_test_weights[task_idx]:
      finetuned_weights_val[name] = session.run(
          self.all_test_weights[task_idx][name], feed_dict=feed_dict)
    for name in self.weights:
      self.weights[name].load(finetuned_weights_val[name], session)

  def get_parameters(self, session):
    if self.learn_offset:
      return session.run([self.weights, self.e_weights, self.inner_lr])
    else:
      return session.run([self.weights, self.inner_lr])

  def set_parameters(self, network_params, session):
    """Sets neural network parameters directly.

    Args:
      network_params: dict variable->values.
      session: TF session.
    """
    ops = []
    for weight_key in network_params:
      ops.append(tf.assign(weight_key, network_params[weight_key]))
    session.run(ops, network_params)

  def init_logging(self, session):
    self.writer = tf.summary.FileWriter(self.tensorboard_dir, session.graph)

  def stop_logging(self):
    if self.writer is not None:
      self.writer.close()
      self.writer = None
