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

"""Render image observations for hand manipulation suite tasks."""

import os
import pickle
import time

from absl import app
from absl import flags
import gym

from rrlfd import adroit_ext  # pylint: disable=unused-import
from rrlfd.bc import pickle_dataset
from tensorflow.io import gfile


flags.DEFINE_string('top_dir', None,
                    'Top directory under which dataset is saved.')
flags.DEFINE_string('in_dir', '',
                    'Directory under topdir from which to read data.')
flags.DEFINE_string('out_dir', 'vil_propr',
                    'Directory under topdir to which to write data.')
flags.DEFINE_enum('task', None, ['door', 'hammer', 'pen', 'relocate'],
                  'Task name.')

flags.DEFINE_boolean('compress_images', True,
                     'If True, compress image observations as png.')
flags.DEFINE_integer('max_demos_to_include', None,
                     'Number of demonstrations to output, at maximum.')

FLAGS = flags.FLAGS


def get_observations_for_demo(env, demo):
  """Add observations from env to demonstration trajectory."""
  env.set_env_state(demo['init_state_dict'])
  demo_length = len(demo['observations'])
  obs = env.get_obs()
  observations = [obs]
  actions = demo['actions']

  for t in range(demo_length - 1):
    obs, _, done, info = env.step(demo['actions'][t])
    observations.append(obs)
    if 'TimeLimit.truncated' not in info or not info['TimeLimit.truncated']:
      # Ignore time limit if demonstration is longer than limit.
      assert not done, f'Done at step {t} out of {demo_length}'
  if not isinstance(env.env, adroit_ext.VisualPenEnvV0):
    # Pen doesn't terminate on success.
    _, _, done, _ = env.step(demo['actions'][-1])
    assert done
  return observations, actions


def main(_):
  task = FLAGS.task
  env = gym.make(f'visual-{task}-v0')

  topdir = FLAGS.top_dir
  in_path = os.path.join(topdir, FLAGS.in_dir, f'{task}-v0_demos.pickle')
  with gfile.GFile(in_path, 'rb') as f:
    dataset = pickle.load(f)

  out_path = os.path.join(topdir, FLAGS.out_dir, f'{task}-v0_demos.pickle')
  writer = pickle_dataset.DemoWriter(out_path, compress=FLAGS.compress_images)

  old_time = time.time()

  num_demos = FLAGS.max_demos_to_include or len(dataset)
  for d in range(num_demos):
    env.reset()
    demo = dataset[d]
    observations, actions = get_observations_for_demo(env, demo)
    writer.write_episode(observations, actions)
    new_time = time.time()
    print(f'{d + 1} / {num_demos}', new_time - old_time, 's')
    old_time = new_time


if __name__ == '__main__':
  app.run(main)
