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

"""Data generation script for Maze environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from .env import environment

import h5py
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('savepath', '/tmp/test.hdf5',
                    'Path to save the HDF5 dataset')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  forward_ep = 10
  steps_per_ep = 100
  datapath = FLAGS.savepath

  f = h5py.File(datapath, 'w')
  sim_data = f.create_group('sim')
  sim_data.create_dataset('ims', (forward_ep, steps_per_ep, 64, 64, 3),
                          dtype='f')
  sim_data.create_dataset('actions', (forward_ep, steps_per_ep, 2), dtype='f')

  env = environment.Environment()
  for ep in range(forward_ep):
    time_step = env.reset()
    _, im = env.get_observation()
    step = 0
    while not time_step.last():
      action = np.random.uniform(-3, 3, size=(2,))
      f['sim']['ims'][ep, step] = im
      f['sim']['actions'][ep, step] = action
      step += 1
      time_step = env.step(action)
      _, im = env.get_observation()

    print(ep)

if __name__ == '__main__':
  app.run(main)
