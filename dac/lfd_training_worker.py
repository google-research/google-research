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

"""Training worker for DAC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-import-not-at-top,g-bad-import-order
import platform
if int(platform.python_version_tuple()[0]) < 3:
  import cPickle as pickle
else:
  import _pickle as pickle

import os
import random
import zlib
from absl import app
from absl import flags
from absl import logging
import ddpg_td3
import gail
import gym
import lfd_envs
import numpy as np
from replay_buffer import ReplayBuffer
from replay_buffer import TimeStep
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from utils import do_rollout
# pylint: enable=g-import-not-at-top,g-bad-import-order


FLAGS = flags.FLAGS
flags.DEFINE_float('exploration_noise', 0.1,
                   'Scale of noise used for exploration.')
flags.DEFINE_float('actor_lr', 1e-3, 'Initial actor learning rate.')
flags.DEFINE_integer('random_actions', int(1e4),
                     'Number of random actions to sample to replay buffer '
                     'before sampling policy actions.')
flags.DEFINE_integer('training_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('policy_updates_delay', int(1e3),
                     'Starts policy updates after N critic updates.')
flags.DEFINE_string('env', 'Hopper-v1',
                    'Environment for training/evaluation.')
flags.DEFINE_string('expert_dir', '', 'Directory to load the expert demos.')
flags.DEFINE_integer('num_expert_trajectories', 11,
                     'Number of trajectories taken from the expert.')
flags.DEFINE_integer('trajectory_size', 50,
                     'Size of every trajectory after subsampling.')
flags.DEFINE_integer('updates_per_step', 1, 'Number of updates per step.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('min_samples_to_start', 1000,
                     'Minimal number of samples in replay buffer to start '
                     'training.')
flags.DEFINE_string('save_dir', '', 'Directory to save models.')
flags.DEFINE_string('eval_save_dir', '',
                    'Directory to save policy for evaluation.')
flags.DEFINE_string('algo', 'td3', 'Algorithm to use for training: ddpg | td3.')
flags.DEFINE_string('gail_loss', 'airl',
                    'GAIL loss to use, gail is -log(1-sigm(D)), airl is D : '
                    'gail | airl.')
flags.DEFINE_integer('save_interval', int(1e5), 'Save every N timesteps.')
flags.DEFINE_integer('eval_save_interval', int(5e3),
                     'Save for evaluation every N timesteps.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_boolean('use_gpu', False,
                     'Directory to write TensorBoard summaries.')
flags.DEFINE_boolean('learn_absorbing', True,
                     'Whether to learn the reward for absorbing states or not.')
flags.DEFINE_string('master', 'local', 'Location of the session.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of Parameter Server tasks.')
flags.DEFINE_integer('task_id', 0, 'Id of the current TF task.')


def main(_):
  """Run td3/ddpg training."""
  tfe.enable_eager_execution()

  if FLAGS.use_gpu:
    tf.device('/device:GPU:0').__enter__()

  tf.gfile.MakeDirs(FLAGS.log_dir)
  summary_writer = tf.contrib.summary.create_file_writer(
      FLAGS.log_dir, flush_millis=10000)

  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  env = gym.make(FLAGS.env)
  env.seed(FLAGS.seed)
  if FLAGS.learn_absorbing:
    env = lfd_envs.AbsorbingWrapper(env)

  if FLAGS.env in ['HalfCheetah-v2', 'Ant-v1']:
    rand_actions = int(1e4)
  else:
    rand_actions = int(1e3)

  obs_shape = env.observation_space.shape
  act_shape = env.action_space.shape

  subsampling_rate = env._max_episode_steps // FLAGS.trajectory_size  # pylint: disable=protected-access
  lfd = gail.GAIL(
      obs_shape[0] + act_shape[0],
      subsampling_rate=subsampling_rate,
      gail_loss=FLAGS.gail_loss)

  if FLAGS.algo == 'td3':
    model = ddpg_td3.DDPG(
        obs_shape[0],
        act_shape[0],
        use_td3=True,
        policy_update_freq=2,
        actor_lr=FLAGS.actor_lr,
        get_reward=lfd.get_reward,
        use_absorbing_state=FLAGS.learn_absorbing)
  else:
    model = ddpg_td3.DDPG(
        obs_shape[0],
        act_shape[0],
        use_td3=False,
        policy_update_freq=1,
        actor_lr=FLAGS.actor_lr,
        get_reward=lfd.get_reward,
        use_absorbing_state=FLAGS.learn_absorbing)

  random_reward, _ = do_rollout(
      env, model.actor, None, num_trajectories=10, sample_random=True)

  replay_buffer_var = tfe.Variable('', name='replay_buffer')
  expert_replay_buffer_var = tfe.Variable('', name='expert_replay_buffer')

  # Save and restore random states of gym/numpy/python.
  # If the job is preempted, it guarantees that it won't affect the results.
  # And the results will be deterministic (on CPU) and reproducible.
  gym_random_state_var = tfe.Variable('', name='gym_random_state')
  np_random_state_var = tfe.Variable('', name='np_random_state')
  py_random_state_var = tfe.Variable('', name='py_random_state')

  reward_scale = tfe.Variable(1, name='reward_scale')

  saver = tfe.Saver(
      model.variables + lfd.variables +
      [replay_buffer_var, expert_replay_buffer_var, reward_scale] +
      [gym_random_state_var, np_random_state_var, py_random_state_var])

  tf.gfile.MakeDirs(FLAGS.save_dir)

  eval_saver = tfe.Saver(model.actor.variables + [reward_scale])
  tf.gfile.MakeDirs(FLAGS.eval_save_dir)

  last_checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
  if last_checkpoint is None:
    expert_saver = tfe.Saver([expert_replay_buffer_var])
    last_checkpoint = os.path.join(FLAGS.expert_dir, 'expert_replay_buffer')
    expert_saver.restore(last_checkpoint)
    expert_replay_buffer = pickle.loads(expert_replay_buffer_var.numpy())
    expert_reward = expert_replay_buffer.get_average_reward()

    logging.info('Expert reward %f', expert_reward)
    print('Expert reward {}'.format(expert_reward))

    reward_scale.assign(expert_reward)
    expert_replay_buffer.subsample_trajectories(FLAGS.num_expert_trajectories)
    if FLAGS.learn_absorbing:
      expert_replay_buffer.add_absorbing_states(env)

    # Subsample after adding absorbing states, because otherwise we can lose
    # final states.

    print('Original dataset size {}'.format(len(expert_replay_buffer)))
    expert_replay_buffer.subsample_transitions(subsampling_rate)
    print('Subsampled dataset size {}'.format(len(expert_replay_buffer)))
    replay_buffer = ReplayBuffer()
    total_numsteps = 0
    prev_save_timestep = 0
    prev_eval_save_timestep = 0
  else:
    saver.restore(last_checkpoint)
    replay_buffer = pickle.loads(zlib.decompress(replay_buffer_var.numpy()))
    expert_replay_buffer = pickle.loads(
        zlib.decompress(expert_replay_buffer_var.numpy()))
    total_numsteps = int(last_checkpoint.split('-')[-1])
    prev_save_timestep = total_numsteps
    prev_eval_save_timestep = total_numsteps
    env.unwrapped.np_random.set_state(
        pickle.loads(gym_random_state_var.numpy()))
    np.random.set_state(pickle.loads(np_random_state_var.numpy()))
    random.setstate(pickle.loads(py_random_state_var.numpy()))

  with summary_writer.as_default():
    while total_numsteps < FLAGS.training_steps:
      # Decay helps to make the model more stable.
      # TODO(agrawalk): Use tf.train.exponential_decay
      model.actor_lr.assign(
          model.initial_actor_lr * pow(0.5, total_numsteps // 100000))
      logging.info('Learning rate %f', model.actor_lr.numpy())
      rollout_reward, rollout_timesteps = do_rollout(
          env,
          model.actor,
          replay_buffer,
          noise_scale=FLAGS.exploration_noise,
          rand_actions=rand_actions,
          sample_random=(model.actor_step.numpy() == 0),
          add_absorbing_state=FLAGS.learn_absorbing)
      total_numsteps += rollout_timesteps

      logging.info('Training: total timesteps %d, episode reward %f',
                   total_numsteps, rollout_reward)

      print('Training: total timesteps {}, episode reward {}'.format(
          total_numsteps, rollout_reward))

      with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar(
            'reward/scaled', (rollout_reward - random_reward) /
            (reward_scale.numpy() - random_reward),
            step=total_numsteps)
        tf.contrib.summary.scalar('reward', rollout_reward, step=total_numsteps)
        tf.contrib.summary.scalar(
            'length', rollout_timesteps, step=total_numsteps)

      if len(replay_buffer) >= FLAGS.min_samples_to_start:
        for _ in range(rollout_timesteps):
          time_step = replay_buffer.sample(batch_size=FLAGS.batch_size)
          batch = TimeStep(*zip(*time_step))

          time_step = expert_replay_buffer.sample(batch_size=FLAGS.batch_size)
          expert_batch = TimeStep(*zip(*time_step))

          lfd.update(batch, expert_batch)

        for _ in range(FLAGS.updates_per_step * rollout_timesteps):
          time_step = replay_buffer.sample(batch_size=FLAGS.batch_size)
          batch = TimeStep(*zip(*time_step))
          model.update(
              batch,
              update_actor=model.critic_step.numpy() >=
              FLAGS.policy_updates_delay)

        if total_numsteps - prev_save_timestep >= FLAGS.save_interval:
          replay_buffer_var.assign(zlib.compress(pickle.dumps(replay_buffer)))
          expert_replay_buffer_var.assign(
              zlib.compress(pickle.dumps(expert_replay_buffer)))
          gym_random_state_var.assign(
              pickle.dumps(env.unwrapped.np_random.get_state()))
          np_random_state_var.assign(pickle.dumps(np.random.get_state()))
          py_random_state_var.assign(pickle.dumps(random.getstate()))
          saver.save(
              os.path.join(FLAGS.save_dir, 'checkpoint'),
              global_step=total_numsteps)
          prev_save_timestep = total_numsteps

        if total_numsteps - prev_eval_save_timestep >= FLAGS.eval_save_interval:
          eval_saver.save(
              os.path.join(FLAGS.eval_save_dir, 'checkpoint'),
              global_step=total_numsteps)
          prev_eval_save_timestep = total_numsteps

if __name__ == '__main__':
  app.run(main)
