# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Implementations of imitation learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tqdm import tqdm
from value_dice import data_utils
from value_dice import gail
from value_dice import twin_sac
from value_dice import value_dice
from value_dice import wrappers

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2',
                    'Environment for training/evaluation.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
flags.DEFINE_integer('actor_update_freq', 1, 'Update actor every N steps.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('replay_regularization', 0.1, 'Amount of replay mixing.')
flags.DEFINE_float('nu_lr', 1e-3, 'nu network learning rate.')
flags.DEFINE_float('actor_lr', 1e-5, 'Actor learning rate.')
flags.DEFINE_float('critic_lr', 1e-3, 'Critic learning rate.')
flags.DEFINE_float('sac_alpha', 0.1, 'SAC temperature.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_integer('hidden_size', 256, 'Hidden size.')
flags.DEFINE_integer('updates_per_step', 5, 'Updates per time step.')
flags.DEFINE_integer('max_timesteps', int(1e5), 'Max timesteps to train.')
flags.DEFINE_integer('num_trajectories', 1, 'Number of trajectories to use.')
flags.DEFINE_integer('num_random_actions', int(2e3),
                     'Fill replay buffer with N random actions.')
flags.DEFINE_integer('start_training_timesteps', int(1e3),
                     'Start training when replay buffer contains N timesteps.')
flags.DEFINE_string('expert_dir', None, 'Directory to load expert demos.')
flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_boolean('learn_alpha', True,
                     'Whether to learn temperature for SAC.')
flags.DEFINE_boolean('normalize_states', True,
                     'Normalize states using expert stats.')
flags.DEFINE_integer('log_interval', int(1e3), 'Log every N timesteps.')
flags.DEFINE_integer('eval_interval', int(1e3), 'Evaluate every N timesteps.')
flags.DEFINE_enum('algo', 'value_dice', ['bc', 'dac', 'value_dice'],
                  'Algorithm to use to compute occupancy ration.')
flags.DEFINE_integer('absorbing_per_episode', 10,
                     'A number of absorbing states per episode to add.')


def _update_pbar_msg(pbar, total_timesteps):
  """Update the progress bar with the current training phase."""
  if total_timesteps < FLAGS.start_training_timesteps:
    msg = 'not training'
  else:
    msg = 'training'
  if total_timesteps < FLAGS.num_random_actions:
    msg += ' rand acts'
  else:
    msg += ' policy acts'
  if pbar.desc != msg:
    pbar.set_description(msg)


def add_samples_to_replay_buffer(replay_buffer, obs, action, next_obs):
  """Add a transition to a replay buffer.

  Args:
    replay_buffer: a replay buffer to add samples to.
    obs: observation.
    action: action.
    next_obs: next observation.
  """
  replay_buffer.add_batch((np.array([obs.astype(np.float32)]),
                           np.array([action.astype(np.float32)]),
                           np.array([next_obs.astype(np.float32)]),
                           np.array([[0]]).astype(np.float32),
                           np.array([[1.0]]).astype(np.float32)))


def evaluate(actor, env, num_episodes=10):
  """Evaluates the policy.

  Args:
    actor: A policy to evaluate.
    env: Environment to evaluate the policy on.
    num_episodes: A number of episodes to average the policy on.

  Returns:
    Averaged reward and a total number of steps.
  """
  total_timesteps = 0
  total_returns = 0

  for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
      action, _, _ = actor(np.array([state]))
      action = action[0].numpy()

      next_state, reward, done, _ = env.step(action)

      total_returns += reward
      total_timesteps += 1
      state = next_state

  return total_returns / num_episodes, total_timesteps / num_episodes


def main(_):
  tf.enable_v2_behavior()

  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  filename = os.path.join(FLAGS.expert_dir, FLAGS.env_name + '.npz')
  (expert_states, expert_actions, expert_next_states,
   expert_dones) = data_utils.load_expert_data(filename)

  (expert_states, expert_actions, expert_next_states,
   expert_dones) = data_utils.subsample_trajectories(expert_states,
                                                     expert_actions,
                                                     expert_next_states,
                                                     expert_dones,
                                                     FLAGS.num_trajectories)
  print('# of demonstraions: {}'.format(expert_states.shape[0]))

  if FLAGS.normalize_states:
    shift = -np.mean(expert_states, 0)
    scale = 1.0 / (np.std(expert_states, 0) + 1e-3)
    expert_states = (expert_states + shift) * scale
    expert_next_states = (expert_next_states + shift) * scale
  else:
    shift = None
    scale = None

  env = wrappers.create_il_env(FLAGS.env_name, FLAGS.seed, shift, scale)

  eval_env = wrappers.create_il_env(FLAGS.env_name, FLAGS.seed + 1, shift,
                                    scale)

  unwrap_env = env

  while hasattr(unwrap_env, 'env'):
    if isinstance(unwrap_env, wrappers.NormalizeBoxActionWrapper):
      expert_actions = unwrap_env.reverse_action(expert_actions)
      break
    unwrap_env = unwrap_env.env

  (expert_states, expert_actions, expert_next_states,
   expert_dones) = data_utils.add_absorbing_states(expert_states,
                                                   expert_actions,
                                                   expert_next_states,
                                                   expert_dones, env)

  spec = (
      tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32,
                             'observation'),
      tensor_spec.TensorSpec([env.action_space.shape[0]], tf.float32, 'action'),
      tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32,
                             'next_observation'),
      tensor_spec.TensorSpec([1], tf.float32, 'reward'),
      tensor_spec.TensorSpec([1], tf.float32, 'mask'),
  )

  # We need to store at most twice more transition due to
  # an extra absorbing to itself transition.
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      spec, batch_size=1, max_length=FLAGS.max_timesteps * 2)

  for i in range(expert_states.shape[0]):
    # Overwrite rewards for safety. We still have to add them to the replay
    # buffer to maintain the same interface. Also always use a zero mask
    # since we need to always bootstrap for imitation learning.
    add_samples_to_replay_buffer(replay_buffer, expert_states[i],
                                 expert_actions[i], expert_next_states[i])

  replay_buffer_iter = iter(
      replay_buffer.as_dataset(sample_batch_size=FLAGS.sample_batch_size))

  policy_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      spec, batch_size=1, max_length=FLAGS.max_timesteps * 2)

  policy_replay_buffer_iter = iter(
      policy_replay_buffer.as_dataset(
          sample_batch_size=FLAGS.sample_batch_size))

  expert_states = tf.Variable(expert_states, dtype=tf.float32)
  expert_actions = tf.Variable(expert_actions, dtype=tf.float32)
  expert_next_states = tf.Variable(expert_next_states, dtype=tf.float32)
  expert_dones = tf.Variable(expert_dones, dtype=tf.float32)

  expert_dataset = tf.data.Dataset.from_tensor_slices(
      (expert_states, expert_actions, expert_next_states))
  expert_dataset = expert_dataset.repeat().shuffle(
      expert_states.shape[0]).batch(
          FLAGS.sample_batch_size, drop_remainder=True)

  expert_dataset_iter = iter(expert_dataset)

  hparam_str_dict = dict(
      seed=FLAGS.seed, algo=FLAGS.algo, env_name=FLAGS.env_name)
  hparam_str = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in
                         sorted(hparam_str_dict.keys())])

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))

  log_dir = os.path.join(FLAGS.save_dir, 'logs')
  log_filename = os.path.join(log_dir, hparam_str)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  if 'dac' in FLAGS.algo:
    imitator = gail.RatioGANGP(env.observation_space.shape[0],
                               env.action_space.shape[0], FLAGS.log_interval)
  elif 'value_dice' in FLAGS.algo:
    imitator = value_dice.ValueDICE(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        nu_lr=FLAGS.nu_lr,
        actor_lr=FLAGS.actor_lr,
        alpha_init=FLAGS.sac_alpha,
        hidden_size=FLAGS.hidden_size,
        log_interval=FLAGS.log_interval)

  def get_imitation_learning_rewards(states, actions, _):
    return imitator.get_log_occupancy_ratio(states, actions)

  if 'value_dice' in FLAGS.algo:
    sac = imitator
  else:
    sac = twin_sac.SAC(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        FLAGS.log_interval,
        actor_lr=FLAGS.actor_lr,
        critic_lr=FLAGS.critic_lr,
        learn_alpha=FLAGS.learn_alpha,
        alpha_init=FLAGS.sac_alpha,
        rewards_fn=get_imitation_learning_rewards)

  episode_return = 0
  episode_timesteps = 0
  done = True

  total_timesteps = 0
  previous_time = time.time()

  eval_returns = []
  with tqdm(total=FLAGS.max_timesteps, desc='') as pbar:
    while total_timesteps < FLAGS.max_timesteps:
      _update_pbar_msg(pbar, total_timesteps)

      if total_timesteps % FLAGS.eval_interval == 0:
        logging.info('Performing policy eval.')
        average_returns, evaluation_timesteps = evaluate(sac.actor, eval_env)

        eval_returns.append(average_returns)
        np.save(log_filename, np.array(eval_returns))

        with summary_writer.as_default():
          tf.summary.scalar(
              'eval gym/average returns', average_returns, step=total_timesteps)
        with summary_writer.as_default():
          tf.summary.scalar(
              'eval gym/average episode length',
              evaluation_timesteps,
              step=total_timesteps)
        logging.info('Eval: ave returns=%f, ave episode length=%f',
                     average_returns, evaluation_timesteps)

      if done:
        if episode_timesteps > 0:
          current_time = time.time()
          with summary_writer.as_default():
            tf.summary.scalar(
                'train gym/returns', episode_return, step=total_timesteps)
            tf.summary.scalar(
                'train gym/FPS',
                episode_timesteps / (current_time - previous_time),
                step=total_timesteps)

        obs = env.reset()
        episode_return = 0
        episode_timesteps = 0
        previous_time = time.time()

      if total_timesteps < FLAGS.num_random_actions:
        action = env.action_space.sample()
      else:
        if 'dac' in FLAGS.algo:
          _, sampled_action, _ = sac.actor(np.array([obs]))
          action = sampled_action[0].numpy()
        else:
          mean_action, _, _ = sac.actor(np.array([obs]))
          action = mean_action[0].numpy()
          action = (action + np.random.normal(
              0, 0.1, size=action.shape)).clip(-1, 1)

      next_obs, reward, done, _ = env.step(action)

      # done caused by episode truncation.
      truncated_done = done and episode_timesteps + 1 == env._max_episode_steps  # pylint: disable=protected-access

      if done and not truncated_done:
        next_obs = env.get_absorbing_state()

      # Overwrite rewards for safety. We still have to add them to the replay
      # buffer to maintain the same interface. Also always use a zero mask
      # since we need to always bootstrap for imitation learning.
      add_samples_to_replay_buffer(replay_buffer, obs, action, next_obs)

      add_samples_to_replay_buffer(policy_replay_buffer, obs, action, next_obs)
      if done and not truncated_done:
        # Add several absobrsing states to absorbing states transitions.
        for abs_i in range(FLAGS.absorbing_per_episode):
          if abs_i + episode_timesteps < env._max_episode_steps:  # pylint: disable=protected-access
            obs = env.get_absorbing_state()
            action = env.action_space.sample()
            next_obs = env.get_absorbing_state()

            add_samples_to_replay_buffer(replay_buffer, obs, action, next_obs)
            add_samples_to_replay_buffer(policy_replay_buffer, obs, action,
                                         next_obs)

      episode_return += reward
      episode_timesteps += 1
      total_timesteps += 1
      pbar.update(1)

      obs = next_obs

      if total_timesteps >= FLAGS.start_training_timesteps:
        with summary_writer.as_default():
          for _ in range(FLAGS.updates_per_step):
            if 'dac' in FLAGS.algo:
              imitator.update(expert_dataset_iter, policy_replay_buffer_iter)
            elif 'value_dice' in FLAGS.algo:
              imitator.update(
                  expert_dataset_iter,
                  policy_replay_buffer_iter,
                  FLAGS.discount,
                  replay_regularization=FLAGS.replay_regularization)

            if 'bc' in FLAGS.algo:
              sac.train_bc(expert_dataset_iter)
            elif 'dac' in FLAGS.algo:
              sac.train(
                  replay_buffer_iter,
                  discount=FLAGS.discount,
                  tau=FLAGS.tau,
                  target_entropy=-env.action_space.shape[0],
                  actor_update_freq=FLAGS.actor_update_freq)


if __name__ == '__main__':
  app.run(main)
