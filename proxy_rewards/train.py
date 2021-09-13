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

"""Experiment script for learning a proxy reward."""

from absl import app
from absl import flags
import numpy as np
import pandas as pd

from tf.io import gfile
from proxy_rewards import train_proxy


FLAGS = flags.FLAGS

flags.DEFINE_integer('nsteps', 1001, 'The number of steps to run training for.')
flags.DEFINE_float('learning_rate', 1e-1, 'Hyperparameter: learning rate.')
flags.DEFINE_float('tol', 1e-8,
                   'Hyperparameter: If grad norm less than tol, stop early.')
flags.DEFINE_float('erm_weight', 1., 'Hyperparameter: Weight on ERM loss.')
flags.DEFINE_float('bias_lamb', 0.,
                   'Hyperparameter: Weight on policy evaluation loss.')
flags.DEFINE_integer('seed', 0, 'Random Seed for model initialization.')
flags.DEFINE_integer('data_seed', 2, 'Random Seed for train/valid split.')
flags.DEFINE_enum(
    name='bias_norm',
    default='max',
    enum_values=['max', 'l2'],
    help='Calculation of policy loss via max or weighted L2 norm over policies.'
    )

DEFAULT_DATA_PATH = None
DEFAULT_DATA_FILE = None

flags.DEFINE_string('data_path', DEFAULT_DATA_PATH,
                    'Path to MovieLens Data')
flags.DEFINE_string('data_file', DEFAULT_DATA_FILE,
                    'File name (in data_path) for the simulated interactions.')
flags.DEFINE_string('simulation_dir', 'simulation_alt',
                    'Directory (in data_path) for simulation results')
flags.DEFINE_string('embed_file', 'movielens_factorization.json',
                    'File name (in data_path) for embeddings')


def load_and_train():
  """Load data from file and return checkpoints from training."""
  simulation_path = f'{FLAGS.data_path}/{FLAGS.simulation_dir}'
  with gfile.GFile(f'{simulation_path}/{FLAGS.data_file}', 'r') as f:
    df = pd.read_csv(f)

  # Split this into train and validate
  rng = np.random.default_rng(FLAGS.data_seed)
  users = np.unique(df['user'])
  users = rng.permutation(users)

  n_users = users.shape[0]
  n_train_users = int(n_users / 2)

  users_train = users[:n_train_users]
  users_val = users[n_train_users:]
  assert users_val.shape[0] + users_train.shape[0] == n_users

  df_tr = df.query('user in @users_train').copy()
  df_val = df.query('user in @users_val').copy()

  a_tr = df_tr['rec'].to_numpy()
  m_tr = df_tr[['diversity', 'rating']].to_numpy()
  y_tr = df_tr['ltr'].to_numpy()
  t_tr = np.ones_like(a_tr)

  a_val = df_val['rec'].to_numpy()
  m_val = df_val[['diversity', 'rating']].to_numpy()
  y_val = df_val['ltr'].to_numpy()
  t_val = np.ones_like(a_val)

  model = train_proxy.LogisticReg()

  data_tr = {
      'a': a_tr,
      'm': m_tr,
      'y': y_tr,
      't': t_tr,
  }

  data_val = {
      'a': a_val,
      'm': m_val,
      'y': y_val,
      't': t_val,
  }

  init_params = train_proxy.initialize_params(
      model, mdim=2, seed=FLAGS.seed)

  loss_tr = train_proxy.make_loss_func(
      model, data_tr,
      erm_weight=FLAGS.erm_weight,
      bias_lamb=FLAGS.bias_lamb,
      bias_norm=FLAGS.bias_norm)
  loss_val = train_proxy.make_loss_func(
      model, data_val,
      erm_weight=FLAGS.erm_weight,
      bias_lamb=FLAGS.bias_lamb,
      bias_norm=FLAGS.bias_norm)

  _, checkpoints = train_proxy.train(
      loss_tr, init_params,
      validation_loss=loss_val,
      lr=FLAGS.learning_rate,
      nsteps=FLAGS.nsteps,
      tol=FLAGS.tol, verbose=True, log=True)

  return checkpoints
