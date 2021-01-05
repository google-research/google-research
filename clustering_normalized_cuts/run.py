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

"""This is the code for Clustering using our CNC framework."""
from __future__ import division
import collections
import os
from absl import app
from absl import flags

from clustering_normalized_cuts import setup
from clustering_normalized_cuts.cnc_net import run_net
from clustering_normalized_cuts.data_loader import get_data

flags.adopt_module_key_flags(setup)
FLAGS = flags.FLAGS

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(_):
  params = collections.defaultdict(lambda: None)

  # SET GENERAL HYPERPARAMETERS
  general_params = {
      'dset': FLAGS.dset,  # dataset: reuters / mnist
      'val_set_fraction': 0.1,  # fraction of training set to use as validation
      'siam_batch_size': 128,  # minibatch size for siamese net
      'main_path': FLAGS.main_path,
      'result_path': FLAGS.result_path
  }
  params.update(general_params)

  # SET DATASET SPECIFIC HYPERPARAMETERS
  if FLAGS.dset == 'mnist':
    mnist_params = setup.set_mnist_params()
    params.update(mnist_params)
  # LOAD DATA
  setup.seed_init()
  data = get_data(params)

  # RUN EXPERIMENT
  run_net(data, params)


if __name__ == '__main__':
  app.run(main)
