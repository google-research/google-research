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

"""Provides utilties for loading and retrieving data from CNS."""

from absl import flags
from absl import logging
import numpy as np
from scipy import io
import tensorflow as tf
from sparse_data.data import decomposition
from sparse_data.data import real
from sparse_data.data import sim

DATA_PATH = 'trees/cached_datasets'

flags.DEFINE_boolean('debug', False, 'Whether to run in debug mode.')
flags.DEFINE_enum('dataset', 'sim_sparsity', [
    '20news', 'sentiment_sentences', 'mnist', 'sim_sparsity', 'sim_cardinality',
    'sim_linear', 'sim_multiplicative', 'sim_xor'
], 'Name of dataset.')
flags.DEFINE_enum('which_features', 'all', ['all', 'inform', 'uninform'],
                  'Types of features to use during learning.')
flags.DEFINE_enum('problem', 'classification', ['classification', 'regression'],
                  'Type of learning problem.')
flags.DEFINE_boolean(
    'alternate', False, 'Whether to perform alternate '
    'experiment, if available.')
flags.DEFINE_enum(
    'vec', 'none', ['none', 'tf-idf', 'binary_bow', 'bow'],
    'Name of vectorization method; only applicable for text '
    'datasets.')
flags.DEFINE_enum(
    'dim_red', 'none', ['none', 'svd', 'lda', 'random_projection'],
    'Name of dimensionality reduction method; only applicable '
    'for non-simulation (real) datasets.')
flags.DEFINE_enum('method', 'all', [
    'all', 'all_except_dnn', 'linear', 'gbdt', 'dnn', 'l1_linear', 'l2_linear',
    'random_forest', 'l1_gbdt', 'l2_gbdt', 'l1_dnn', 'l2_dnn'
], 'Which learning method(s) to use.')

flags.DEFINE_integer(
    'num_dataset', 10, 'Number of datasets to evaluate on; '
    'only applicable for simulation experiments; will not '
    'affect running of non-simulation (e.g., real) datasets.')


def get_experiment_string(dataset,
                          vec,
                          dim_red,
                          which_features='all',
                          problem='classification',
                          alternate=False):
  """Get a string describing the experiment.

  Args:
    dataset: str dataset name
    vec: str vectorization method
    dim_red: str dimensionality reduction method
    which_features: str type of features to use; values = 'all', 'inform',
      'uninform'
    problem: str type of learning problem; values = 'classification',
      'regression'
    alternate: bool whether alternate experiment is used

  Returns:
    experiment_string: str
      string describing experiment settings
  """
  out = dataset
  out += '_{}'.format(vec) * (dim_red != 'none')
  out += '_{}'.format(dim_red) * (vec != 'none')
  out += '_{}'.format(problem) * (problem != 'classification')
  out += '_{}'.format(which_features) * (which_features != 'all')
  out += '_alternate' * alternate

  return out


def save(path, save_func, obj):
  """Save an object to path with GFile and the given save function.

  Args:
    path: string path
    save_func: function some callable with save_func(file-buffer-like, object)
      signature
    obj: some object
  """
  with tf.gfile.GFile(path, 'w') as f:
    save_func(f, obj)


def load(path, load_func):
  """Load and return an object from path with GFile and the given load function.

  Args:
    path: string path
    load_func: function some callable with load_func(file-buffer-like) signature

  Returns:
    obj: loaded object
  """
  with tf.gfile.GFile(path, 'r') as f:
    return load_func(f)


def get_sim(dataset,
            problem='classification',
            which_features='all',
            alternate=False,
            **kwargs):
  """Get simulated dataset.

  Args:
    dataset: str dataset name
    problem: str type of learning problem; values = 'classification',
      'regression'
    which_features: str type of features to use; values = 'all', 'inform',
      'uninform'
    alternate: bool whether alternate experiment is used
    **kwargs: Additional args.

  Returns:
    d: dataset object
      API defined in TOOD(jisungkim)

  Raises:
    ValueError: if dataset is unknown
  """
  if dataset == 'sim_sparsity':
    return sim.SparsitySimulation(
        problem=problem,
        which_features=which_features,
        alternate=alternate,
        **kwargs)
  if dataset == 'sim_cardinality':
    return sim.CardinalitySimulation(
        problem=problem,
        which_features=which_features,
        alternate=alternate,
        **kwargs)
  if dataset == 'sim_linear':
    return sim.LinearSimulation(problem=problem, **kwargs)
  if dataset == 'sim_multiplicative':
    return sim.MultiplicativeSimulation(problem=problem, **kwargs)
  if dataset == 'sim_xor':
    return sim.XORSimulation(problem=problem, **kwargs)
  else:
    raise ValueError('Unknown dataset: {}'.format(dataset))


def get_nonsim_data(dataset, vec='tf-idf', dim_red='none'):
  """Gets a non-simulation dataset and applies relevant transformations.

  Args:
    dataset: str dataset name
    vec: str vectorization method
    dim_red: str dimensionality reduction method

  Returns:
    x_train: array-like
      matrix of features of the training data
    y_train: list-like
      list of labels of the training data
    x_test: array-like
      matrix of features of the test data
    y_test: list-like
      list of labels of the test data

  Raises:
    ValueError: if dataset or dim_red is unknown
  """
  experiment_string = get_experiment_string(dataset, vec=vec, dim_red=dim_red)
  logging.info('experiment_string = %s', experiment_string)

  if dataset == '20news':
    d = real.TwentyNewsgroups(vectorizer=vec)
  elif dataset == 'sentiment_sentences':
    d = real.SentimentSentences(vectorizer=vec)
  elif dataset == 'mnist':
    d = real.MNIST()
  else:
    raise ValueError('Unknown dataset: {}'.format(dataset))

  x_train, y_train, x_test, y_test = d.get()

  if dim_red == 'none':
    pass
  elif dim_red == 'svd':
    x_train, x_test = decomposition.truncated_svd(x_train, x_test)
  elif dim_red == 'lda':
    x_train, x_test = decomposition.lda(x_train, y_train, x_test)
  elif dim_red == 'random_projection':
    x_train, x_test = decomposition.random_projection(x_train, x_test)
  else:
    raise ValueError('Unknown dim_red: \'{}\''.format(dim_red))

  return x_train, y_train, x_test, y_test


def load_nonsim_data(dataset, vec='none', dim_red='none'):
  """Loads a non-simulation dataset from CNS.

  Args:
    dataset: str dataset name
    vec: str vectorization method
    dim_red: str dimensionality reduction method

  Returns:
    x_train: array-like
      matrix of features of the training data
    y_train: list-like
      list of labels of the training data
    x_test: array-like
      matrix of features of the test data
    y_test: list-like
      list of labels of the test data
  """
  assert 'sim' not in dataset

  experiment_string = get_experiment_string(dataset, vec=vec, dim_red=dim_red)

  if dim_red == 'none' and vec != 'none':

    def x_load_func(f):
      return io.mmread(f).tocsr()

    x_save_func = io.mmwrite
    x_ext = 'mtx'
  else:
    x_load_func = np.load
    x_save_func = np.save
    x_ext = 'npy'

  try:
    data_path = '{}/{}'.format(DATA_PATH, experiment_string)
    x_train = load('{}/{}.{}'.format(data_path, 'x_train', x_ext), x_load_func)
    x_test = load('{}/{}.{}'.format(data_path, 'x_test', x_ext), x_load_func)
    y_train = load('{}/{}'.format(data_path, 'y_train.npy'), np.load)
    y_test = load('{}/{}'.format(data_path, 'y_test.npy'), np.load)
    logging.info('Loaded data file from: %s', data_path)
    logging.info('\n')
  except:  # pylint: disable=bare-except
    x_train, y_train, x_test, y_test = get_nonsim_data(
        dataset, vec=vec, dim_red=dim_red)

    if x_save_func is not None and x_ext is not None:  # save data
      data_path = '{}/{}'.format(DATA_PATH, experiment_string)
      try:
        tf.gfile.MakeDirs(data_path)
      except:  # pylint: disable=bare-except
        pass
      try:
        save('{}/{}.{}'.format(data_path, 'x_train', x_ext), x_save_func,
             x_train)
        save('{}/{}.{}'.format(data_path, 'x_test', x_ext), x_save_func, x_test)
        save('{}/{}'.format(data_path, 'y_train.npy'), np.save, y_train)
        save('{}/{}'.format(data_path, 'y_test.npy'), np.save, y_test)
        logging.info('Saved data files to %s', data_path)
        logging.info('\n')
      except:  # pylint: disable=bare-except
        logging.error('Unable to save data files to %s', data_path)
        logging.error('\n')

  return x_train, y_train, x_test, y_test
