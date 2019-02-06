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

"""Runs worker code for parameter tuning.

Results must be aggregated by tuning_collate.py.
"""

import itertools
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as shuffle_coordinately
import tensorflow as tf
from sparse_data import utils
from sparse_data.exp_framework import dnn
from sparse_data.exp_framework import gbdt
from sparse_data.exp_framework import linear
from sparse_data.exp_framework import random_forest
from sparse_data.experiment import parse_method_flag

FILE_PATH = '/trees/out'
SEED = 123834838

LINEAR_REGR_PARAM_GRID = {
    'alpha': [2e-5, 2e-4, 2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3, 2e4, 2e5]
}
LINEAR_CLF_PARAM_GRID = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]}
RF_PARAM_GRID = {
    'n_estimators': [16, 64, 128, 256, 512],
    'max_features': ['auto', None],
    'max_depth': [1, 3, 5, 10, 50, None]
}
GBDT_PARAM_GRID = {
    'booster': ['gbtree'],
    'tree_method': ['exact'],
    'n_estimators': [16, 64, 128, 256, 512],
    'learning_rate': [0.1, 0.3, 0.5, 0.8, 1.0],
    'max_depth': [1, 3, 5, 10],
    'reg_alpha': [0.0],  # no regularization by default
    'reg_lambda': [0.0],
    'silent': [1]
}
DNN_PARAM_GRID = {
    'embedding_dim': [-1, 256, 512, 1024],
    'num_hidden_layer': [1, 2, 3, 4],
    'l1': [0.0],  # no regularization by default
    'l2': [0.0],
    'dropout': [0.0, 0.25, 0.5, 0.75],
    'hidden_layer_dim': [32, 64, 128, 256, 512, 1024],
    'activation': ['prelu', 'elu'],
    'learning_rate': [1e-4, 3e-4, 1e-3, 1e-2, 1e-1],
    'batch_size': [64, 128, 256, 512],
    'epochs': [2, 4, 8, 16, 32]
}

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_search', 100, 'Number (max) of parameter searches '
    'for DNNs and GBDTs.')
flags.DEFINE_integer('search_idx', 1, 'Index of parameter to be searched.')
flags.DEFINE_integer('k', 5, 'Number of folds in k-fold cross-validation')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite existing '
                     'output.')


def generate_param_configs(param_grid, num_iteration, seed=1):
  """Generate a list of parameter configurations from grid of parameter values.

  Uses exhaustive grid or random search depending on the size of the space
  of the search parameter values, and the number of searches specified.

  Args:
    param_grid: {str: [?]} dictionary of parameters and all possible values
    num_iteration: number of iterations (searches)
    seed: int seed value for reproducibility in random sampling

  Returns:
    out: [{str: ?}]
      a list of parameter configurations represented as dictionaries
  """
  rng = np.random.RandomState()
  rng.seed(seed)

  out = []
  num_param_config = np.prod([len(v) for v in param_grid.values()])
  if num_param_config <= num_iteration:  # exhaustive grid
    for values in itertools.product(*param_grid.values()):
      out.append({k: v for k, v in zip(param_grid.keys(), values)})
    assert len(out) <= num_iteration
  else:  # random
    for _ in range(num_iteration):
      out.append({k: v[rng.randint(len(v))] for k, v in param_grid.items()})
    assert len(out) == num_iteration
  return out


def main(_):
  experiment_string = utils.get_experiment_string(
      FLAGS.dataset,
      vec=FLAGS.vec,
      dim_red=FLAGS.dim_red,
      features=FLAGS.features,
      problem=FLAGS.problem,
      alternate=FLAGS.alternate)
  worker_dir = '{}/params/{}/worker_out'.format(FILE_PATH, experiment_string)

  logging.info('\n' * 3)
  logging.info('Parameter search search_idx=%d for %s and method=%s',
               FLAGS.search_idx, experiment_string, FLAGS.method)
  logging.info('\n' * 3)

  if FLAGS.debug:
    logging.warn('Running in debug mode')

  # setup methods
  methods = parse_method_flag(FLAGS.method)

  # setup datasets
  assert FLAGS.problem in ['classification', 'regression']
  if (FLAGS.dataset not in ['20news', 'sentiment_sentences'] and
      FLAGS.vec != 'none'):
    raise ValueError(
        'Should not be using text vectorization with {} dataset'.format(
            FLAGS.dataset))

  datasets = []
  if 'sim' in FLAGS.dataset:
    d = utils.get_sim(
        FLAGS.dataset,
        problem=FLAGS.problem,
        features=FLAGS.features,
        alternate=FLAGS.alternate)
    for _ in range(FLAGS.num_dataset):
      d.reset()
      # must use get() over generate() for consistency in random sampling calls
      x_train, y_train, _, _ = d.get()

      if FLAGS.debug:  # use smaller subset
        x_train, y_train = shuffle_coordinately(x_train, y_train)
        x_train, y_train = x_train[:500, :250], y_train[:500]
      datasets.append((x_train, y_train))
  else:
    assert FLAGS.num_dataset == 1
    x_train, y_train, _, _ = utils.load_nonsim_data(
        FLAGS.dataset, vec=FLAGS.vec, dim_red=FLAGS.dim_red)

    if FLAGS.debug:  # use smaller subset
      x_train, y_train = shuffle_coordinately(x_train, y_train)
      x_train, y_train = x_train[:500, :250], y_train[:500]

    datasets.append((x_train, y_train))

  for method in methods:
    # define methods and parameter grids here
    if method == 'l1_linear' and FLAGS.problem == 'regression':
      submodule = linear
      param_grid = LINEAR_REGR_PARAM_GRID.copy()
      param_grid['penalty'] = ['l1']
    elif method == 'l1_linear':
      submodule = linear
      param_grid = LINEAR_CLF_PARAM_GRID.copy()
      param_grid['penalty'] = ['l1']
    elif method == 'l2_linear' and FLAGS.problem == 'regression':
      submodule = linear
      param_grid = LINEAR_REGR_PARAM_GRID.copy()
      param_grid['penalty'] = ['l2']
    elif method == 'l2_linear':
      submodule = linear
      param_grid = LINEAR_CLF_PARAM_GRID.copy()
      param_grid['penalty'] = ['l2']
    elif method == 'random_forest':
      submodule = random_forest
      param_grid = RF_PARAM_GRID.copy()
    elif method == 'l1_gbdt':
      submodule = gbdt
      param_grid = GBDT_PARAM_GRID.copy()
      param_grid['reg_alpha'] = [0.0, 0.5, 1.0, 2.0, 4.0, 10.]
    elif method == 'l2_gbdt':
      submodule = gbdt
      param_grid = GBDT_PARAM_GRID.copy()
      param_grid['reg_lambda'] = [0.0, 0.5, 1.0, 2.0, 4.0, 10.]
    elif method == 'l1_dnn':
      submodule = dnn
      param_grid = DNN_PARAM_GRID.copy()
      param_grid['l1'] = [0.0, 1e-3, 1e-2, 1e-1]
    elif method == 'l2_dnn':
      submodule = dnn
      param_grid = DNN_PARAM_GRID.copy()
      param_grid['l2'] = [0.0, 1e-3, 1e-2, 1e-1]
    else:
      raise ValueError('Unknown learning method: {}'.format(method))

    params = generate_param_configs(
        param_grid, num_iteration=FLAGS.num_search, seed=SEED)

    if FLAGS.search_idx >= len(params):  # less configs than number of searches
      continue

    param_dict = params[FLAGS.search_idx]

    for dataset_idx, (x_train, y_train) in enumerate(datasets):
      # recursively make parent directory
      save_dir = '{}/dataset_idx={}_{}/{}'.format(worker_dir, dataset_idx,
                                                  FLAGS.num_dataset, method)
      tf.gfile.MakeDirs(save_dir)

      # skip search if already performed
      save_path = '{}/search_idx={}_{}.out'.format(save_dir, FLAGS.search_idx,
                                                   FLAGS.num_search)
      if tf.gfile.Exists(save_path) and not FLAGS.overwrite and not FLAGS.debug:
        logging.info('Parameter search already completed for %s, dataset %d/%d',
                     method, dataset_idx, FLAGS.num_dataset)
        continue

      # k-fold cross-validation
      start = time.time()
      tuning_scores = []
      kf = KFold(n_splits=FLAGS.k, shuffle=True, random_state=SEED)
      for cv_train_idx, cv_test_idx in kf.split(x_train):
        x_train_cv, y_train_cv = x_train[cv_train_idx], y_train[cv_train_idx]
        x_test_cv, y_test_cv = x_train[cv_test_idx], y_train[cv_test_idx]

        _, metrics = submodule.pipeline(
            x_train_cv,
            y_train_cv,
            x_test_cv,
            y_test_cv,
            param_dict,
            problem=FLAGS.problem)

        # assume that we maximize the score
        if FLAGS.problem == 'regression':
          tuning_scores.append(-metrics['test_mse'])
        else:
          tuning_scores.append(metrics['test_acc'])
        if method == 'dnn':
          dnn.clear_keras_session()

      mean_score = np.mean(tuning_scores)

      logging.info(
          'Worker result for method=%s, search %d/%d, dataset %d/%d '
          '(%.3f s)', method, FLAGS.search_idx, FLAGS.num_search, dataset_idx,
          FLAGS.num_dataset,
          time.time() - start)
      logging.info(param_dict)
      logging.info('%.4f', mean_score)
      logging.info('\n' * 2)

      if not FLAGS.debug:
        # save parameters and worker results to file
        with tf.gfile.GFile(save_path, 'w') as f:
          s = ','.join(['{}={}'.format(k, v) for k, v in param_dict.items()])
          f.write(s)
          f.write('\n')
          f.write('{:.8f}'.format(mean_score))
        logging.info('Saved results to %s', save_path)
    logging.info('\n\n')


if __name__ == '__main__':
  app.run()
