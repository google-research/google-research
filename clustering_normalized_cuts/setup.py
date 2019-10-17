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

"""Iinitialize seeds and save config."""

from absl import flags
from numpy.random import seed
from tensorflow import set_random_seed

flags.DEFINE_string('dset', 'mnist', 'dataset')
flags.DEFINE_float('cnc_lr', 0.005, 'cnc learning rate')
flags.DEFINE_float('cnc_tau', 1.5, 'temperature')
flags.DEFINE_float('min_tem', 1.0, 'min temperature')
flags.DEFINE_float('cnc_drop', 0.5, 'cnc drop')
flags.DEFINE_integer('cnc_patience', 35, 'cnc patience')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('cnc_epochs', 1000, 'Number of epochs')
flags.DEFINE_integer('n_nbrs', 3, 'Number of neighbors')
flags.DEFINE_integer('scale_nbr', 5, 'neighbor used to determine scale')
flags.DEFINE_float('bal_reg', 0.01, 'bal_reg')
flags.DEFINE_string('main_path', '', 'main path')
flags.DEFINE_string('result_path', ' ', 'path to save the results')

FLAGS = flags.FLAGS


def seed_init():
  seed(3554)
  set_random_seed(2483)


def set_mnist_params():
  """Set hyper parameters."""
  mnist_params = {
      'n_clusters': 10,  # number of output clusters
      'use_code_space': False,  # enable / disable code space embedding
      'affinity': 'siamese',  # affinity type: siamese / knn
      'n_nbrs': FLAGS.n_nbrs,  # number of neighbors for graph Laplacian
      'scale_nbr': FLAGS.scale_nbr,  # scale of Gaussian graph Laplacian
      'siam_k': 2,  # siamese net: number of neighbors to use (the 'k' in knn)
      # to construct training pairs
      'siam_ne': 400,  # siamese net: number of training epochs
      'cnc_epochs': FLAGS.cnc_epochs,  # CNC: number of training epochs
      'siam_lr': 1e-3,  # siamese net: initial learning rate
      'cnc_lr': FLAGS.cnc_lr,  # CNC: initial learning rate
      'cnc_tau': FLAGS.cnc_tau,  # CNC: initial tempreture
      'min_tem': FLAGS.min_tem,
      'siam_patience': 10,  # siamese net: early stopping patience
      'cnc_patience': FLAGS.cnc_patience,  # CNC: early stopping patience
      'siam_drop': 0.1,  # siamese net: learning rate scheduler decay
      'cnc_drop': FLAGS.cnc_drop,  # CNC: learning rate decay
      'batch_size': FLAGS.batch_size,  # CNC: batch size
      'bal_reg': FLAGS.bal_reg,
      'siam_reg': None,  # siamese net: regularization parameter
      'cnc_reg': None,  # CNC: regularization parameter
      'siam_n': None,  # siamese net: subset of data to construct training pairs
      'siamese_tot_pairs': 600000,  # siamese net: total number of pairs
      'siam_arch': [  # siamese network architecture.
          {
              'type': 'relu',
              'size': 1024
          },
          {
              'type': 'relu',
              'size': 1024
          },
          {
              'type': 'relu',
              'size': 512
          },
          {
              'type': 'relu',
              'size': 10
          },
      ],
      'cnc_arch': [  # CNC network architecture.
          {
              'type': 'tanh',
              'size': 512
          },
          {
              'type': 'tanh',
              'size': 512
          },
          {
              'type': 'relu',
              'size': 10
          },
      ],
      'generalization_metrics':
          True,  # enable to check out of set generalization error and nmi
      'use_all_data':
          False,  # enable to use all data for training (no test set)
  }
  return mnist_params
