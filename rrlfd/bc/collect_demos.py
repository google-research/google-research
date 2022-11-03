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

"""Execute a script agent to gather demonstrations on mime."""

import os

from absl import app
from absl import flags
import gym
from mime.agent import ScriptAgent
import numpy as np

from tensorflow.io import gfile
from rrlfd.bc import pickle_dataset


flags.DEFINE_string('task', 'Pick', 'Mime task.')
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to record.')
flags.DEFINE_integer('seed', 0, 'Experiment seed.')
flags.DEFINE_boolean('increment_seed', False,
                     'If True, increment seed at every episode.')
flags.DEFINE_integer('image_size', None, 'Size of rendered images.')
flags.DEFINE_integer('num_cameras', 1, 'The number of viewpoints to record.')
flags.DEFINE_boolean('view_rand', False,
                     'If True, vary camera pose for each episode.')
flags.DEFINE_boolean('compress_images', True, 'If True, save frames as png.')

flags.DEFINE_string('script_type', 'default',
                    'Choose which script to use if mime defines multiple for '
                    'the task.')
flags.DEFINE_boolean('stop_early', False,
                     'If True, stop when task is finished according to scene. '
                     'Otherwise stop when script finishes.')
flags.DEFINE_boolean('record_failed', False,
                     'If True, save failed demonstrations.')
flags.DEFINE_boolean('add_noise', False,
                     'If True, add noise to expert actions.')
flags.DEFINE_float('lateral_friction', 0.5, 'Friction coefficient for cube.')

flags.DEFINE_string('logdir', None, 'Location to save demonstrations to.')
flags.DEFINE_string('run_id', None,
                    'If set, a custom string to append to saved demonstrations '
                    'file name.')

flags.DEFINE_boolean('render', False, 'If True, render environment.')
flags.DEFINE_boolean('use_egl', False, 'If True, use EGL for rendering mime.')

FLAGS = flags.FLAGS


def make_noised(action):
  if 'joint_velocity' in action:
    action['joint_velocity'] += np.random.normal(scale=0.01, size=6)
  if 'linear_velocity' in action:
    action['linear_velocity'] += np.random.normal(scale=0.007, size=3)
  if 'angular_velocity' in action:
    action['angular_velocity'] += np.random.normal(scale=0.04, size=3)


def env_loop(env, add_noise, num_episodes, log_path, record_failed, stop_early,
             seed, increment_seed, compress_images):
  """Loop for collecting demos with a scripted agent in a Mime environment."""
  if log_path is None:
    log_f = None
    success_f = None
    demo_writer = None
  else:
    log_f = gfile.GFile(log_path + '_log.txt', 'w')
    success_f = gfile.GFile(log_path + '_success.txt', 'w')
    demo_writer = pickle_dataset.DemoWriter(log_path + '.pkl', compress_images)
    print('Writing demos to', log_path + '.pkl')
  e = 0
  # Counter to keep track of seed offset, if not recording failed episodes.
  skipped_seeds = 0
  num_successes = 0
  num_attempts = 0
  while e < num_episodes:
    if e % 10 == 0 and e > 0:
      print(f'Episode {e} / {num_episodes}; '
            f'Success rate {num_successes} / {num_attempts}')
    if increment_seed:
      env.seed(seed + skipped_seeds + e)
    obs = env.reset()
    # To define a different script, use forked version of mime.
    # agent = ScriptAgent(env, FLAGS.script_type)
    agent = ScriptAgent(env)

    done = False
    action = agent.get_action()
    if add_noise:
      make_noised(action)
    observations = []
    actions = []

    while (not (stop_early and done)) and action is not None:
      observations.append(obs)
      actions.append(action)
      obs, unused_reward, done, info = env.step(action)
      action = agent.get_action()
      if add_noise and action is not None:
        make_noised(action)

    if info['success']:
      print(f'{num_attempts}: success')
      if log_f is not None:
        log_f.write(f'{num_attempts}: success' + '\n')
        log_f.flush()
      if success_f is not None:
        success_f.write('success\n')
        success_f.flush()
      num_successes += 1
    else:
      if action is None:
        info['failure_message'] = 'End of Script.'
      print(f'{num_attempts}: failure:', info['failure_message'])
      if log_f is not None:
        log_f.write(
            f'{num_attempts}: failure: ' + info['failure_message'] + '\n')
        log_f.flush()
      if success_f is not None:
        success_f.write('failure\n')
        success_f.flush()
    num_attempts += 1

    if info['success'] or record_failed:
      e += 1
      if demo_writer is not None:
        demo_writer.write_episode(observations, actions)
    elif not record_failed:
      skipped_seeds += 1

  print(f'Done; Success rate {num_successes} / {num_attempts}')
  if log_f is not None:
    log_f.write(f'Done; Success rate {num_successes} / {num_attempts}\n')
    log_f.close()


def main(_):
  rand_str = 'Rand' if FLAGS.view_rand else ''
  cam_num_str = str(FLAGS.num_cameras) if FLAGS.num_cameras > 1 else ''
  egl_str = '-EGL' if FLAGS.use_egl else ''
  env_id = f'UR5{egl_str}-{FLAGS.task}{cam_num_str}{rand_str}CamEnv-v0'
  print('Creating', env_id)

  env = gym.make(env_id)
  scene = env.unwrapped.scene
  scene.renders(FLAGS.render)
  env.seed(FLAGS.seed)
  if FLAGS.task == 'Push':
    scene.lateral_friction = FLAGS.lateral_friction
  im_size = FLAGS.image_size
  if im_size is not None:
    env.env._cam_resolution = (im_size, im_size)  # pylint: disable=protected-access

  if FLAGS.logdir is None:
    log_path = None
  else:
    logdir = os.path.join(FLAGS.logdir, f'{FLAGS.task}')

    increment_str = 'i' if FLAGS.increment_seed else ''
    noisy = '_noisy' if FLAGS.add_noise else ''
    stop_early = '_stopearly' if FLAGS.stop_early else ''
    view_rand = 'rand' if FLAGS.view_rand else ''
    cam_num_str = str(FLAGS.num_cameras) if FLAGS.num_cameras > 1 else ''
    if not cam_num_str and not view_rand:
      cam_str = ''
    else:
      cam_str = f'_{cam_num_str}{view_rand}cam'
    run_id = '_' + FLAGS.run_id if FLAGS.run_id else ''
    log_path = os.path.join(
        logdir,
        f's{FLAGS.seed}{increment_str}_e{FLAGS.num_episodes}{noisy}{stop_early}'
        f'{cam_str}{run_id}')
    gfile.makedirs(os.path.dirname(log_path))
    print('Writing to', log_path)

  env_loop(
      env, FLAGS.add_noise, FLAGS.num_episodes, log_path, FLAGS.record_failed,
      FLAGS.stop_early, FLAGS.seed, FLAGS.increment_seed, FLAGS.compress_images)


if __name__ == '__main__':
  app.run(main)
