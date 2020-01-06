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

"""Script to create demonstration sets from our trained policies.
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
from absl import app
from absl import flags
from absl import logging
from common import Actor
import gym
import numpy as np
from replay_buffer import ReplayBuffer
import tensorflow.compat.v1 as tf
from utils import do_rollout
from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe
# pylint: enable=g-import-not-at-top,g-bad-import-order


FLAGS = flags.FLAGS
flags.DEFINE_float('exploration_noise', 0.1,
                   'Scale of noise used for exploration.')
flags.DEFINE_integer('random_actions', int(1e4),
                     'Number of random actions to sample to replay buffer '
                     'before sampling policy actions.')
flags.DEFINE_integer('training_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_string('env', 'Hopper-v1',
                    'Environment for training/evaluation.')
flags.DEFINE_string('expert_dir', '', 'Directory to load the expert model.')
flags.DEFINE_integer('num_expert_trajectories', 100,
                     'Number of trajectories taken from the expert.')
flags.DEFINE_string('save_dir', '', 'Directory to save models.')
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

  if FLAGS.expert_dir.find(FLAGS.env) == -1:
    raise ValueError('Expert directory must contain the environment name')

  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  env = gym.make(FLAGS.env)
  env.seed(FLAGS.seed)

  obs_shape = env.observation_space.shape
  act_shape = env.action_space.shape

  expert_replay_buffer_var = contrib_eager_python_tfe.Variable(
      '', name='expert_replay_buffer')

  saver = contrib_eager_python_tfe.Saver([expert_replay_buffer_var])
  tf.gfile.MakeDirs(FLAGS.save_dir)

  with tf.variable_scope('actor'):
    actor = Actor(obs_shape[0], act_shape[0])
  expert_saver = contrib_eager_python_tfe.Saver(actor.variables)

  best_checkpoint = None
  best_reward = float('-inf')

  checkpoint_state = tf.train.get_checkpoint_state(FLAGS.expert_dir)

  for checkpoint in checkpoint_state.all_model_checkpoint_paths:
    expert_saver.restore(checkpoint)
    expert_reward, _ = do_rollout(
        env, actor, replay_buffer=None, noise_scale=0.0, num_trajectories=10)

    if expert_reward > best_reward:
      best_reward = expert_reward
      best_checkpoint = checkpoint

  expert_saver.restore(best_checkpoint)

  expert_replay_buffer = ReplayBuffer()
  expert_reward, _ = do_rollout(
      env,
      actor,
      replay_buffer=expert_replay_buffer,
      noise_scale=0.0,
      num_trajectories=FLAGS.num_expert_trajectories)

  logging.info('Expert reward %f', expert_reward)
  print('Expert reward {}'.format(expert_reward))

  expert_replay_buffer_var.assign(pickle.dumps(expert_replay_buffer))
  saver.save(os.path.join(FLAGS.save_dir, 'expert_replay_buffer'))


if __name__ == '__main__':
  app.run(main)
