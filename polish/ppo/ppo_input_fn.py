# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Input function for RL TF estimator.

Most RL algorithms require re-sampling the environment after
multiple training steps. This input_fn is a subclass of tf.train.SessionRunHook
that enables us to extend calls to MonitoredSession.run() with
user-defined functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import functools
import os
from absl import logging
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from polish.utils import math_utils
from polish.utils import tf_utils

# Set logging verbosity
logging.set_verbosity(logging.INFO)


@gin.configurable
class PpoInputFn(tf.train.SessionRunHook):
  """Input function used for tf.estimator."""

  def __init__(self,
               env_wrapper,
               model_fn,
               train_batch_size=64,
               iterations_per_loop=320,
               max_horizon=2048,
               env_state_space=2,
               env_action_space=2,
               use_tpu=False,
               mcts_enable=False,
               checkpoint_dir=gin.REQUIRED,
               summary_dir=gin.REQUIRED,
               train_state_file=gin.REQUIRED):
    """Creates an input function that prepares and feeds data into the model.

    Args:
      env_wrapper: Environment wrapper instantiated from src.env.env. The
        `PPOInputFn` interact with the environment throuh this object. The main
        function called on this object is `play` that roll out a trajectory and
        returns a set of state-action items.
      model_fn: Model function (policy and value network) used in tf.estimator.
      train_batch_size: The size of minibatch for training.
      iterations_per_loop: The number of iterations for each call to session.
        That generally is number of batches multiplied by number of epochs.
      max_horizon: The maximum number of steps for the trajectory.
      env_state_space: The size of environment's state space.
      env_action_space: The size of environment's action space.
      use_tpu: If True, TPU is used during training.
      mcts_enable: If True, input_fn uses Monte-Carlo Tree Search (MCTS) for
        performing trajectory rollout.
      checkpoint_dir: Specifies the checkpoint directory.
      summary_dir: Specifies the directory for storing TF summaries. Used by the
        Tensorboard to visualize training statistics.
      train_state_file: The filename used to store the most recent model
        version.

    Raises:
      IOError: If checkpoint_dir, summary_dir or train_state_file is not
        specified.
    """
    logging.debug('Created the estimator hook...')

    if not checkpoint_dir:
      raise IOError('Please specify a directory for checkpointing.')
    if not summary_dir:
      raise IOError('Please specify a directory for TF summary.')
    if not train_state_file:
      raise IOError('Please specify a file to save/restore the model '
                    'version.')

    # Instantiated from src.env.env
    self._env_wrapper = env_wrapper
    self._model_fn = model_fn
    self._train_batch_size = train_batch_size
    self._iterations_per_loop = iterations_per_loop
    self._max_horizon = max_horizon
    self._env_state_space = env_state_space
    self._env_action_space = env_action_space
    self._use_tpu = use_tpu
    self._mcts_enable = mcts_enable
    self._checkpoint_dir = checkpoint_dir
    self._summary_dir = summary_dir
    self._train_state_file = train_state_file

    # Summary to write training data into summary dir
    self.summary_writer = tf.summary.FileWriter(self._summary_dir)

    # Queues to calculate statistics for the latest 100 episodes
    # Stores the total reward of the last 100 episodes
    self._episode_reward_buf = collections.deque(maxlen=100)
    # Stores the length of last 100 episodes
    self._episode_length_buf = collections.deque(maxlen=100)

    self._mcts_episode_reward_buf = collections.deque(maxlen=20)
    # Stores the length of last 100 episodes
    self._mcts_episode_length_buf = collections.deque(maxlen=20)

    # Store the training global step
    self._global_step_value = 0

  def __call__(self, params):
    """Call function used to create dataset using initializable iterator.

    Args:
      params: parameters sent by estimator.

    Returns:
      An initializable iterator.
    """
    logging.info('Running __call__ function...')
    batch_size = self._train_batch_size
    # For MCTS, the number of features for each trajecotry is unknown beforehand
    num_features = None

    if self._global_step_value % self._iterations_per_loop == 0:
      logging.info('Update iterator (gs=%d)...', self._global_step_value)
      # Feature/Labels Placeholders
      self.features_ph = {
          'mcts_features':
              tf.placeholder(
                  tf.float32,
                  shape=[num_features, self._env_state_space],
                  name='mcts_state_ph'),
          'policy_features':
              tf.placeholder(
                  tf.float32,
                  shape=[num_features, self._env_state_space],
                  name='policy_state_ph'),
      }
      self.labels_ph = {
          'action_tensor':
              tf.placeholder(
                  tf.float32,
                  shape=[num_features, self._env_action_space],
                  name='action_ph'),
          'value_tensor':
              tf.placeholder(
                  tf.float32, shape=[num_features], name='value_ph'),
          'return_tensor':
              tf.placeholder(
                  tf.float32, shape=[num_features], name='return_ph'),
          'old_neg_logprob_tensor':
              tf.placeholder(
                  tf.float32, shape=[num_features], name='old_neg'),
          'mean_tensor':
              tf.placeholder(
                  tf.float32,
                  shape=[num_features, self._env_action_space],
                  name='mean_ph'),
          'logstd_tensor':
              tf.placeholder(
                  tf.float32,
                  shape=[num_features, self._env_action_space],
                  name='logstd_ph'),
          'mcts_enable_tensor':
              tf.placeholder(
                  tf.bool, shape=[num_features], name='mcts_enable_ph'),
          'policy_action_tensor':
              tf.placeholder(
                  tf.float32,
                  shape=[num_features, self._env_action_space],
                  name='policy_action_ph'),
          'policy_value_tensor':
              tf.placeholder(
                  tf.float32, shape=[num_features], name='policy_value_ph'),
          'policy_return_tensor':
              tf.placeholder(
                  tf.float32, shape=[num_features], name='policy_return_ph'),
          'policy_old_neg_logprob_tensor':
              tf.placeholder(
                  tf.float32, shape=[num_features], name='policy_old_neg'),
      }
      # Create the dataset
      dataset = tf.data.Dataset.from_tensor_slices(
          (self.features_ph, self.labels_ph))
      dataset = dataset.shuffle(buffer_size=self._max_horizon)
      dataset = dataset.batch(batch_size, drop_remainder=True)

      # repeat until the loop is done
      dataset = dataset.repeat()
      if self._use_tpu:
        dataset = dataset.map(functools.partial(self._set_shapes, batch_size))
        dataset = dataset.prefetch(2)
      self._iterator = dataset.make_initializable_iterator()
      return self._iterator.get_next()
    else:
      return self._iterator.get_next()

  def _set_shapes(self, batch_size, features_in, labels_in):
    """Reshape the tensors to fixed sizes.

    Args:
      batch_size: Batch size.
      features_in: Features of the model.
      labels_in: Labels of the model.

    Returns:
      Reshaped features and labels.
    """
    features_in['mcts_features'] = tf.reshape(
        features_in['mcts_features'], [batch_size, self._env_state_space],
        name='mcts_feature_reshape')

    features_in['policy_features'] = tf.reshape(
        features_in['policy_features'], [batch_size, self._env_state_space],
        name='policy_feature_reshape')

    labels_in['action_tensor'] = tf.reshape(
        labels_in['action_tensor'], [batch_size, self._env_action_space],
        name='action_reshape')

    labels_in['mean_tensor'] = tf.reshape(
        labels_in['mean_tensor'], [batch_size, self._env_action_space],
        name='mean_reshape')

    labels_in['logstd_tensor'] = tf.reshape(
        labels_in['logstd_tensor'], [batch_size, self._env_action_space],
        name='logstd_reshape')

    labels_in['value_tensor'] = tf.reshape(
        labels_in['value_tensor'], [batch_size], name='value_reshape')

    labels_in['return_tensor'] = tf.reshape(
        labels_in['return_tensor'], [batch_size], name='return_reshape')

    labels_in['old_neg_logprob_tensor'] = tf.reshape(
        labels_in['old_neg_logprob_tensor'], [batch_size], name='log_reshape')

    labels_in['mcts_enable_tensor'] = tf.reshape(
        labels_in['mcts_enable_tensor'], [batch_size], name='mcts_reshape')

    labels_in['policy_action_tensor'] = tf.reshape(
        labels_in['policy_action_tensor'], [batch_size, self._env_action_space],
        name='policy_action_reshape')

    labels_in['policy_value_tensor'] = tf.reshape(
        labels_in['policy_value_tensor'], [batch_size],
        name='policy_value_reshape')

    labels_in['policy_return_tensor'] = tf.reshape(
        labels_in['policy_return_tensor'], [batch_size],
        name='policy_return_reshape')

    labels_in['policy_old_neg_logprob_tensor'] = tf.reshape(
        labels_in['policy_old_neg_logprob_tensor'], [batch_size],
        name='log_reshape')

    return features_in, labels_in

  def begin(self):
    self._global_step = tf.train.get_or_create_global_step()

  def after_create_session(self, session, coord):
    """Handles which is called after creating a TF session.

    Args:
      session: the current TF session.
      coord: the current TF thread coordinator.
    """

    logging.debug('After creating the session...')

    # We MUST have some data for initializable iterator to avoid runtime error.
    # This data is not used for training.
    # Instead, the collected data in `before_run` function is used for training.
    # As such, we just pass an array of ones to the initializer.

    dummy_states = np.ones((self._max_horizon, self._env_state_space))
    dummy_actions = np.ones((self._max_horizon, self._env_action_space))
    dummy_values = np.ones((self._max_horizon,))
    dummy_returns = np.ones((self._max_horizon,))
    dummy_negative_log_prob = np.ones((self._max_horizon,))
    dummy_means = np.ones((self._max_horizon, self._env_action_space))
    dummy_logstds = np.ones((self._max_horizon, self._env_action_space))
    mcts_tensor = np.full(dummy_values.shape, True)
    session.run(
        self._iterator.initializer,
        feed_dict={
            self.features_ph['mcts_features']:
                dummy_states,
            self.features_ph['policy_features']:
                dummy_states,
            self.labels_ph['action_tensor']:
                dummy_actions,
            self.labels_ph['value_tensor']:
                dummy_values,
            self.labels_ph['return_tensor']:
                dummy_returns,
            self.labels_ph['old_neg_logprob_tensor']:
                dummy_negative_log_prob,
            self.labels_ph['mean_tensor']:
                dummy_means,
            self.labels_ph['logstd_tensor']:
                dummy_logstds,
            self.labels_ph['mcts_enable_tensor']:
                mcts_tensor,
            self.labels_ph['policy_action_tensor']:
                dummy_actions,
            self.labels_ph['policy_value_tensor']:
                dummy_values,
            self.labels_ph['policy_return_tensor']:
                dummy_returns,
            self.labels_ph['policy_old_neg_logprob_tensor']:
                dummy_negative_log_prob,
        })

  def before_run(self, run_context):
    """A handle called before running each session.

    Args:
      run_context: the running TF context manager.
    """
    logging.info('Before creating the session...')

    self._global_step_value = run_context.session.run(self._global_step)
    if self._global_step_value % self._iterations_per_loop == 0:

      # Calling `play` the environment roll out a trajectory of length
      #  `self._max_horizon`. Currently, we support two modes for play:
      # (1) stochastic play (similar to PPO)
      # (2) Monte-Carlo Tree Search (MCTS) play
      self._env_wrapper.play(self._max_horizon)

      # Computes explained variance between predicted values (from network)
      # and computed return values from environment.
      ev = math_utils.explained_variance(
          np.asarray(self._env_wrapper.trajectory_values),
          np.asarray(self._env_wrapper.trajectory_returns))
      tf_utils.add_summary(
          float(ev), 'Variation/explained_variance', self._global_step_value,
          self.summary_writer)

      if type(self._env_wrapper).__name__ == 'Env':
        # Update queues for episode data
        # (length of episodes and episode rewards)
        self._episode_reward_buf.extend(
            self._env_wrapper.trajectory_per_episode_rewards)
        self._episode_length_buf.extend(
            self._env_wrapper.trajectory_per_episode_lengths)
      else:
        self._episode_reward_buf.extend(
            self._env_wrapper.master_game.trajectory_per_episode_rewards)
        self._episode_length_buf.extend(
            self._env_wrapper.master_game.trajectory_per_episode_lengths)

      # Summaries for the current trajectory
      tf_utils.summary_stats(self._episode_reward_buf, 'Reward',
                             'Episode Rewards', self._global_step_value,
                             self.summary_writer, False)
      tf_utils.summary_stats(self._episode_length_buf, 'Reward',
                             'Episode Length', self._global_step_value,
                             self.summary_writer, False)

      mcts_tensor = np.full(
          np.asarray(self._env_wrapper.trajectory_values).shape,
          self._env_wrapper.mcts_sampling)

      run_context.session.run(
          self._iterator.initializer,
          feed_dict={
              self.features_ph['mcts_features']:
                  self._env_wrapper.trajectory_states,
              self.features_ph['policy_features']:
                  self._env_wrapper.policy_trajectory_states,
              self.labels_ph['action_tensor']:
                  self._env_wrapper.trajectory_actions,
              self.labels_ph['value_tensor']:
                  self._env_wrapper.trajectory_values,
              self.labels_ph['return_tensor']:
                  self._env_wrapper.trajectory_returns,
              self.labels_ph['old_neg_logprob_tensor']:
                  self._env_wrapper.trajectory_neg_logprobs,
              self.labels_ph['mean_tensor']:
                  self._env_wrapper.trajectory_means,
              self.labels_ph['logstd_tensor']:
                  self._env_wrapper.trajectory_logstds,
              self.labels_ph['mcts_enable_tensor']:
                  mcts_tensor,
              self.labels_ph['policy_action_tensor']:
                  self._env_wrapper.policy_trajectory_actions,
              self.labels_ph['policy_value_tensor']:
                  self._env_wrapper.policy_trajectory_values,
              self.labels_ph['policy_return_tensor']:
                  self._env_wrapper.policy_trajectory_returns,
              self.labels_ph['policy_old_neg_logprob_tensor']:
                  self._env_wrapper.policy_trajectory_neg_logprobs,
          })

  def after_run(self, run_context, run_values):
    """Runs after each session run.

    Each run may contain multiple iterations.

    Args:
      run_context: the running TF context manager.
      run_values: Contains results of requested ops/tensors by before_run().
    """
    tf.logging.info('After session run...')
    self._global_step_value = run_context.session.run(self._global_step)

  def bootstrap(self, working_dir):
    """Initialize a tf.Estimator run with random initial weights.

    This bootstrap is used when a random checkpoint is required.

    Args:
      working_dir: The directory where tf.estimator will drop logs, checkpoints.
    """
    estimator_initial_checkpoint_name = 'model.ckpt-1'
    save_file = os.path.join(working_dir, estimator_initial_checkpoint_name)
    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
      features = self.get_inference_input_ppo_bootstrap()
      self._model_fn(
          features=features,
          labels=None,
          mode=tf.estimator.ModeKeys.PREDICT,
          params=None)
      sess.run(tf.global_variables_initializer())
      tf.train.Saver().save(sess, save_file)

  def get_inference_input_ppo_bootstrap(self):
    """Set up placeholders for input features.

    Returns:
      the feature tensors that get passed into model_fn.
    """
    features = {
        'mcts_features':
            tf.placeholder(tf.float32, shape=[None, self._env_state_space]),
        'policy_features':
            tf.placeholder(tf.float32, shape=[None, self._env_state_space]),
    }
    return features

  def serving_input_receiver(self):
    """Defines the format of input data to the model."""
    features = {
        'mcts_features':
            tf.placeholder(tf.float32, shape=[None, self._env_state_space]),
        'policy_features':
            tf.placeholder(tf.float32, shape=[None, self._env_state_space]),
    }
    return tf.estimator.export.ServingInputReceiver(features, features)
