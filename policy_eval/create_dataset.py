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

r"""Run data collection."""

import os
import pickle
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment

import policy_eval.actor as actor_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Reacher-v2',
                    'Environment for training/evaluation.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_integer('model_seed', 0, 'Random seed used for model training.')
flags.DEFINE_integer('num_episodes', 1000, 'Num episodes to unroll.')
flags.DEFINE_integer('max_episode_len', 1000,
                     'Max episode length to estimate returns.')
flags.DEFINE_float('discount', 0.99, 'Discount for MC returns.')
flags.DEFINE_float('std', None, 'Behavior policy noise scale.')
flags.DEFINE_string('save_dir', '/tmp/policy_eval/trajectory_datasets/',
                    'Directory to save results to.')
flags.DEFINE_string('models_dir', None, 'Model to load for evaluation.')


def main(_):
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.models_dir is None:
    raise ValueError('You must set a value for models_dir.')

  env = suite_mujoco.load(FLAGS.env_name)
  env.seed(FLAGS.seed)
  env = tf_py_environment.TFPyEnvironment(env)

  sac = actor_lib.Actor(env.observation_spec().shape[0], env.action_spec())

  model_filename = os.path.join(FLAGS.models_dir, 'DM-' + FLAGS.env_name,
                                str(FLAGS.model_seed), '1000000')
  sac.load_weights(model_filename)

  if FLAGS.std is None:
    if 'Reacher' in FLAGS.env_name:
      std = 0.5
    elif 'Ant' in FLAGS.env_name:
      std = 0.4
    elif 'Walker' in FLAGS.env_name:
      std = 2.0
    else:
      std = 0.75
  else:
    std = FLAGS.std

  def get_action(state):
    _, action, log_prob = sac(state, std)
    return action, log_prob

  dataset = dict(
      model_filename=model_filename,
      behavior_std=std,
      trajectories=dict(
          states=[],
          actions=[],
          log_probs=[],
          next_states=[],
          rewards=[],
          masks=[]))

  for i in range(FLAGS.num_episodes):
    timestep = env.reset()
    trajectory = dict(
        states=[],
        actions=[],
        log_probs=[],
        next_states=[],
        rewards=[],
        masks=[])

    while not timestep.is_last():
      action, log_prob = get_action(timestep.observation)
      next_timestep = env.step(action)

      trajectory['states'].append(timestep.observation)
      trajectory['actions'].append(action)
      trajectory['log_probs'].append(log_prob)
      trajectory['next_states'].append(next_timestep.observation)
      trajectory['rewards'].append(next_timestep.reward)
      trajectory['masks'].append(next_timestep.discount)

      timestep = next_timestep

    for k, v in trajectory.items():
      dataset['trajectories'][k].append(tf.concat(v, 0).numpy())

    logging.info('%d trajectories', i + 1)

  data_save_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name,
                               str(FLAGS.model_seed))
  if not tf.io.gfile.isdir(data_save_dir):
    tf.io.gfile.makedirs(data_save_dir)

  save_filename = os.path.join(data_save_dir, f'dualdice_{FLAGS.std}.pckl')
  with tf.io.gfile.GFile(save_filename, 'wb') as f:
    pickle.dump(dataset, f)


if __name__ == '__main__':
  app.run(main)
