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

"""Implementations for DDPG in Tensorflow Eager.
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
import gym
import numpy as np
from replay_buffer import ReplayBuffer
from replay_buffer import TimeStep
import tensorflow.compat.v1 as tf
from utils import do_rollout
from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe
# pylint: enable=g-import-not-at-top,g-bad-import-order

FLAGS = flags.FLAGS
flags.DEFINE_float('exploration_noise', 0.1,
                   'Scale of noise used for exploration.')
flags.DEFINE_integer('training_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer(
    'min_samples_to_start', 1000,
    'Minimal number of samples in replay buffer to start training.')
flags.DEFINE_string('env', 'Hopper-v1',
                    'Environment for training/evaluation.')
flags.DEFINE_string('save_dir', '', 'Directory to save models.')
flags.DEFINE_string('eval_save_dir', '',
                    'Directory to save policy for evaluation.')
flags.DEFINE_string('algo', 'td3', 'Algorithm to use for training: ddpg | td3.')
flags.DEFINE_integer('save_interval', int(1e5), 'Save every N timesteps.')
flags.DEFINE_integer('eval_save_interval', int(5e3),
                     'Save for evaluation N timesteps.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_boolean('use_gpu', False,
                     'Directory to write TensorBoard summaries.')
flags.DEFINE_string('master', 'local', 'Location of the session.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of Parameter Server tasks.')
flags.DEFINE_integer('task_id', 0, 'Id of the current TF task.')


def main(_):
  """Run td3/ddpg training."""
  contrib_eager_python_tfe.enable_eager_execution()

  if FLAGS.use_gpu:
    tf.device('/device:GPU:0').__enter__()

  tf.gfile.MakeDirs(FLAGS.log_dir)
  summary_writer = contrib_summary.create_file_writer(
      FLAGS.log_dir, flush_millis=10000)

  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  env = gym.make(FLAGS.env)
  env.seed(FLAGS.seed)

  if FLAGS.env in ['HalfCheetah-v2', 'Ant-v1']:
    rand_actions = int(1e4)
  else:
    rand_actions = int(1e3)

  obs_shape = env.observation_space.shape
  act_shape = env.action_space.shape

  if FLAGS.algo == 'td3':
    model = ddpg_td3.DDPG(
        obs_shape[0],
        act_shape[0],
        use_td3=True,
        policy_update_freq=2,
        actor_lr=1e-3)
  else:
    model = ddpg_td3.DDPG(
        obs_shape[0],
        act_shape[0],
        use_td3=False,
        policy_update_freq=1,
        actor_lr=1e-4)

  replay_buffer_var = contrib_eager_python_tfe.Variable(
      '', name='replay_buffer')
  gym_random_state_var = contrib_eager_python_tfe.Variable(
      '', name='gym_random_state')
  np_random_state_var = contrib_eager_python_tfe.Variable(
      '', name='np_random_state')
  py_random_state_var = contrib_eager_python_tfe.Variable(
      '', name='py_random_state')

  saver = contrib_eager_python_tfe.Saver(
      model.variables + [replay_buffer_var] +
      [gym_random_state_var, np_random_state_var, py_random_state_var])
  tf.gfile.MakeDirs(FLAGS.save_dir)

  reward_scale = contrib_eager_python_tfe.Variable(1, name='reward_scale')
  eval_saver = contrib_eager_python_tfe.Saver(model.actor.variables +
                                              [reward_scale])
  tf.gfile.MakeDirs(FLAGS.eval_save_dir)

  last_checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
  if last_checkpoint is None:
    replay_buffer = ReplayBuffer()
    total_numsteps = 0
    prev_save_timestep = 0
    prev_eval_save_timestep = 0
  else:
    saver.restore(last_checkpoint)
    replay_buffer = pickle.loads(zlib.decompress(replay_buffer_var.numpy()))
    total_numsteps = int(last_checkpoint.split('-')[-1])
    assert len(replay_buffer) == total_numsteps
    prev_save_timestep = total_numsteps
    prev_eval_save_timestep = total_numsteps
    env.unwrapped.np_random.set_state(
        pickle.loads(gym_random_state_var.numpy()))
    np.random.set_state(pickle.loads(np_random_state_var.numpy()))
    random.setstate(pickle.loads(py_random_state_var.numpy()))

  with summary_writer.as_default():
    while total_numsteps < FLAGS.training_steps:
      rollout_reward, rollout_timesteps = do_rollout(
          env,
          model.actor,
          replay_buffer,
          noise_scale=FLAGS.exploration_noise,
          rand_actions=rand_actions)
      total_numsteps += rollout_timesteps

      logging.info('Training: total timesteps %d, episode reward %f',
                   total_numsteps, rollout_reward)

      print('Training: total timesteps {}, episode reward {}'.format(
          total_numsteps, rollout_reward))

      with contrib_summary.always_record_summaries():
        contrib_summary.scalar('reward', rollout_reward, step=total_numsteps)
        contrib_summary.scalar('length', rollout_timesteps, step=total_numsteps)

      if len(replay_buffer) >= FLAGS.min_samples_to_start:
        for _ in range(rollout_timesteps):
          time_step = replay_buffer.sample(batch_size=FLAGS.batch_size)
          batch = TimeStep(*zip(*time_step))
          model.update(batch)

        if total_numsteps - prev_save_timestep >= FLAGS.save_interval:
          replay_buffer_var.assign(zlib.compress(pickle.dumps(replay_buffer)))
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
