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

"""Script to generate expert data in our format from Ho et al.'s DDPG policies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from absl import app
from absl import flags
import h5py
from replay_buffer import ReplayBuffer
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

FLAGS = flags.FLAGS

flags.DEFINE_string('src_data_dir', None, 'Directory containing *.h5.')
flags.DEFINE_string('dst_data_dir', None, 'Directory to store output.')

flags.mark_flag_as_required('src_data_dir')
flags.mark_flag_as_required('dst_data_dir')


def main(_):
  tf.enable_eager_execution()

  envs = ['HalfCheetah-v1', 'Hopper-v1', 'Ant-v1', 'Walker2d-v1', 'Reacher-v1']
  for ienv, env in enumerate(envs):
    print('Processing environment %d of %d: %s' % (ienv + 1, len(envs), env))
    h5_filename = os.path.join(FLAGS.src_data_dir, '%s.h5' % env)
    trajectories = h5py.File(h5_filename, 'r')

    if (set(trajectories.keys()) !=
        set(['a_B_T_Da', 'len_B', 'obs_B_T_Do', 'r_B_T'])):
      raise ValueError('Unexpected key set in file %s' % h5_filename)

    replay_buffer = ReplayBuffer()

    if env.find('Reacher') > -1:
      max_len = 50
    else:
      max_len = 1000

    for i in range(50):
      print('  Processing trajectory %d of 50 (len = %d)' % (
          i + 1, trajectories['len_B'][i]))
      for j in range(trajectories['len_B'][i]):
        mask = 1
        if j + 1 == trajectories['len_B'][i]:
          if trajectories['len_B'][i] == max_len:
            mask = 1
          else:
            mask = 0
        replay_buffer.push_back(
            trajectories['obs_B_T_Do'][i][j], trajectories['a_B_T_Da'][i][j],
            trajectories['obs_B_T_Do'][i][(j + 1) % trajectories['len_B'][i]],
            [trajectories['r_B_T'][i][j]],
            [mask], j == trajectories['len_B'][i] - 1)

    replay_buffer_var = tfe.Variable('', name='expert_replay_buffer')
    saver = tfe.Saver([replay_buffer_var])
    odir = os.path.join(FLAGS.dst_data_dir, env)
    print('Saving results to checkpoint in directory: %s' % odir)
    tf.gfile.MakeDirs(odir)
    replay_buffer_var.assign(pickle.dumps(replay_buffer))
    saver.save(os.path.join(odir, 'expert_replay_buffer'))

if __name__ == '__main__':
  app.run(main)
