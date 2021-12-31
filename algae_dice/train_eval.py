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

"""Run training loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.environments import suite_mujoco
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tqdm import tqdm
from algae_dice import algae
from algae_dice import wrappers

gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'DM-HalfCheetah-v2',
                    'Environment for training/evaluation.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
flags.DEFINE_integer('actor_update_freq', 2, 'Update actor every N steps.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('critic_lr', 1e-3, 'Critic learning rate.')
flags.DEFINE_float('actor_lr', 1e-3, 'Actor learning rate.')
flags.DEFINE_float('algae_alpha', 0.01, 'ALGAE alpha.')
flags.DEFINE_boolean('use_dqn', True, 'Use double q learning.')
flags.DEFINE_boolean('use_init_states', True, 'Use init states.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_integer('num_updates_per_env_step', 1,
                     'How many train steps per env step.')
flags.DEFINE_float('f_exponent', 1.5, 'Exponent for f.')
flags.DEFINE_integer('max_timesteps', int(5e5), 'Max timesteps to train.')
flags.DEFINE_integer('num_random_actions', int(1e4),
                     'Fill replay buffer with N random actions.')
flags.DEFINE_integer('start_training_timesteps', int(1e3),
                     'Start training when replay buffer contains N timesteps.')
flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_integer('log_interval', int(1e3), 'Log every N timesteps.')
flags.DEFINE_integer('eval_interval', int(5e3), 'Evaluate every N timesteps.')
flags.DEFINE_float(
    'target_entropy', None,
    '(optional) target_entropy for training actor. If None, '
    '-env.action_space.shape[0] is used.')
flags.DEFINE_integer('num_stack_frames', 1,
                     '(optional) wrap env to stack frames (use 1 to disable).')


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


def _write_measurements(summary_writer, labels_and_values, step):
  """Write all the measurements."""

  # Write TF Summaries Measurements.
  with summary_writer.as_default():
    for (label, value) in labels_and_values:
      tf.summary.scalar(label, value, step=step)



def main(_):
  tf.enable_v2_behavior()
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  base_env = suite_mujoco.load(FLAGS.env_name)
  if hasattr(base_env, 'max_episode_steps'):
    max_episode_steps = base_env.max_episode_steps
  else:
    logging.info('Unknown max episode steps. Setting to 1000.')
    max_episode_steps = 1000
  env = base_env.gym
  env = wrappers.check_and_normalize_box_actions(env)
  env.seed(FLAGS.seed)

  eval_env = suite_mujoco.load(FLAGS.env_name).gym
  eval_env = wrappers.check_and_normalize_box_actions(eval_env)
  eval_env.seed(FLAGS.seed + 1)

  spec = (
      tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32,
                             'observation'),
      tensor_spec.TensorSpec([env.action_space.shape[0]], tf.float32, 'action'),
      tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32,
                             'next_observation'),
      tensor_spec.TensorSpec([1], tf.float32, 'reward'),
      tensor_spec.TensorSpec([1], tf.float32, 'mask'),
  )
  init_spec = tensor_spec.TensorSpec([env.observation_space.shape[0]],
                                     tf.float32, 'observation')

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      spec, batch_size=1, max_length=FLAGS.max_timesteps)
  init_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      init_spec, batch_size=1, max_length=FLAGS.max_timesteps)

  hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
  hparam_str = ','.join([
      '%s=%s' % (k, str(hparam_str_dict[k]))
      for k in sorted(hparam_str_dict.keys())
  ])
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))

  rl_algo = algae.ALGAE(
      env.observation_space.shape[0],
      env.action_space.shape[0],
      FLAGS.log_interval,
      critic_lr=FLAGS.critic_lr,
      actor_lr=FLAGS.actor_lr,
      use_dqn=FLAGS.use_dqn,
      use_init_states=FLAGS.use_init_states,
      algae_alpha=FLAGS.algae_alpha,
      exponent=FLAGS.f_exponent)

  episode_return = 0
  episode_timesteps = 0
  done = True

  total_timesteps = 0
  previous_time = time.time()

  replay_buffer_iter = iter(
      replay_buffer.as_dataset(sample_batch_size=FLAGS.sample_batch_size))
  init_replay_buffer_iter = iter(
      init_replay_buffer.as_dataset(sample_batch_size=FLAGS.sample_batch_size))

  log_dir = os.path.join(FLAGS.save_dir, 'logs')
  log_filename = os.path.join(log_dir, hparam_str)
  if not gfile.isdir(log_dir):
    gfile.mkdir(log_dir)

  eval_returns = []

  with tqdm(total=FLAGS.max_timesteps, desc='') as pbar:
    # Final return is the average of the last 10 measurmenets.
    final_returns = collections.deque(maxlen=10)
    final_timesteps = 0
    while total_timesteps < FLAGS.max_timesteps:
      _update_pbar_msg(pbar, total_timesteps)
      if done:

        if episode_timesteps > 0:
          current_time = time.time()

          train_measurements = [
              ('train/returns', episode_return),
              ('train/FPS', episode_timesteps / (current_time - previous_time)),
          ]
          _write_measurements(summary_writer, train_measurements,
                              total_timesteps)
        obs = env.reset()
        episode_return = 0
        episode_timesteps = 0
        previous_time = time.time()

        init_replay_buffer.add_batch(np.array([obs.astype(np.float32)]))

      if total_timesteps < FLAGS.num_random_actions:
        action = env.action_space.sample()
      else:
        _, action, _ = rl_algo.actor(np.array([obs]))
        action = action[0].numpy()

      if total_timesteps >= FLAGS.start_training_timesteps:
        with summary_writer.as_default():
          target_entropy = (-env.action_space.shape[0]
                            if FLAGS.target_entropy is None else
                            FLAGS.target_entropy)
          for _ in range(FLAGS.num_updates_per_env_step):
            rl_algo.train(
                replay_buffer_iter,
                init_replay_buffer_iter,
                discount=FLAGS.discount,
                tau=FLAGS.tau,
                target_entropy=target_entropy,
                actor_update_freq=FLAGS.actor_update_freq)

      next_obs, reward, done, _ = env.step(action)
      if (max_episode_steps is not None and
          episode_timesteps + 1 == max_episode_steps):
        done = True

      if not done or episode_timesteps + 1 == max_episode_steps:  # pylint: disable=protected-access
        mask = 1.0
      else:
        mask = 0.0

      replay_buffer.add_batch((np.array([obs.astype(np.float32)]),
                               np.array([action.astype(np.float32)]),
                               np.array([next_obs.astype(np.float32)]),
                               np.array([[reward]]).astype(np.float32),
                               np.array([[mask]]).astype(np.float32)))

      episode_return += reward
      episode_timesteps += 1
      total_timesteps += 1
      pbar.update(1)

      obs = next_obs

      if total_timesteps % FLAGS.eval_interval == 0:
        logging.info('Performing policy eval.')
        average_returns, evaluation_timesteps = rl_algo.evaluate(
            eval_env, max_episode_steps=max_episode_steps)

        eval_returns.append(average_returns)
        fin = gfile.GFile(log_filename, 'w')
        np.save(fin, np.array(eval_returns))
        fin.close()

        eval_measurements = [
            ('eval/average returns', average_returns),
            ('eval/average episode length', evaluation_timesteps),
        ]
        # TODO(sandrafaust) Make this average of the last N.
        final_returns.append(average_returns)
        final_timesteps = evaluation_timesteps

        _write_measurements(summary_writer, eval_measurements, total_timesteps)

        logging.info('Eval: ave returns=%f, ave episode length=%f',
                     average_returns, evaluation_timesteps)
    # Final measurement.
    final_measurements = [
        ('final/average returns', sum(final_returns) / len(final_returns)),
        ('final/average episode length', final_timesteps),
    ]
    _write_measurements(summary_writer, final_measurements, total_timesteps)


if __name__ == '__main__':
  app.run(main)
