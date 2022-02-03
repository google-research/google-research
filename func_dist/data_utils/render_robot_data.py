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

"""Recreate robot dataset with new rendered observations, using e.g. EGL.
"""
import copy
import gzip
import io
import pickle

from absl import app
from absl import flags
import gym
import multiworld.envs.mujoco

from tensorflow.io import gfile


flags.DEFINE_string(
    'task', 'Image48HumanLikeSawyerPushForwardEnv-v0',
    'Name of the gym environment.')
flags.DEFINE_string(
    'robot_data_path', None,
    'Path to gzipped pickle file with robot data to use for actions.')

flags.DEFINE_string(
    'out_path', None,
    'Path to which to write new dataset with newly rendered observations.')

FLAGS = flags.FLAGS


def load_gzipped_pickle(path):
  with gfile.GFile(path, 'rb') as f:
    f = gzip.GzipFile(fileobj=f)
    data = io.BytesIO(f.read())
    data = pickle.load(data)
  return data


def render_episodes(env, data):
  """Create new image observations matching environment states in data."""
  new_data = copy.deepcopy(data)
  for e, episode in enumerate(data):
    # Pop in order to remove mujoco_py dependency in the pickle file.
    new_data[e].pop('env_states')
    for t, env_state in enumerate(episode['env_states']):
      # Env state "t": the environment state before step t.
      env.set_env_state(env_state)
      state = env._get_flat_img()  # pylint: disable=protected-access

      if t <= len(new_data[e]['observations']) - 1:
        new_data[e]['observations'][t] = state
      if t > 0:
        new_data[e]['next_observations'][t - 1] = state
  return new_data


def main(_):
  multiworld.envs.mujoco.register_goal_example_envs()
  env = gym.make(FLAGS.task)

  # Load raw data without preprocessing.
  data = load_gzipped_pickle(FLAGS.robot_data_path)
  new_data = render_episodes(env, data)

  with gfile.GFile(FLAGS.out_path, 'wb') as f:
    pickle.dump(new_data, f)


if __name__ == '__main__':
  app.run(main)
