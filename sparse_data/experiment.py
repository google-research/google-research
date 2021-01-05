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

"""Runs general sparse data learning experiments with different models.

Evaluates linear models, random forest ensembles, gradient boosted decision
trees and DNNs.
"""
import os
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
from sklearn.utils import shuffle as shuffle_coordinately
import tensorflow.compat.v1 as tf
from sparse_data import utils
from sparse_data.exp_framework import dnn
from sparse_data.exp_framework import gbdt
from sparse_data.exp_framework import linear
from sparse_data.exp_framework import random_forest
from sparse_data.exp_framework.evaluate import accuracy
from sparse_data.exp_framework.evaluate import mean_squared_error

FILE_PATH = 'trees/out'
SEED = 49 + 32 + 67 + 111 + 114 + 32 + 49 + 51 + 58 + 52 + 45 + 56

FLAGS = flags.FLAGS

flags.DEFINE_boolean('logtofile', False, 'Whether to log results to file.')
flags.DEFINE_boolean(
    'fast', False, 'Whether to conduct experiment even '
    'without having done parameter tuning.')


def parse_value(s):
  """Parse a given string to a boolean, integer or float."""
  if s.lower() == 'true':
    return True
  elif s.lower() == 'false':
    return False
  elif s.lower() == 'none':
    return None

  try:
    return int(s)
  except:  # pylint: disable=bare-except
    pass

  try:
    return float(s)
  except:  # pylint: disable=bare-except
    return s


def get_prop_inf_feature_importance(method, model, num_inf_feature):
  """TODO(jisungkim): Update how it is calculated."""
  num_inf_feature = int(num_inf_feature)
  if 'gbdt' in method.lower():
    weights = model.get_booster().get_score(importance_type='weight')
    weights = {int(f[1:]): w for f, w in weights.items()}
    inf_weights = [w for f, w in weights.items() if f < num_inf_feature]
    weights = [w for f, w in weights.items()]
  elif 'random_forest' in method.lower():
    weights = model.feature_importances_
    inf_weights = weights[:num_inf_feature]
  elif 'linear' in method.lower():
    weights = np.abs(np.squeeze(model.coef_))
    inf_weights = weights[:num_inf_feature]
  else:
    raise NotImplementedError('Feature importance for {}'.format(method))

  return np.true_divide(np.sum(inf_weights), np.sum(weights))


def parse_method_flag(method_flag):
  """Parse a method FLAG value and get a list of methods."""
  if method_flag == 'all':
    return [
        'l1_linear', 'l2_linear', 'random_forest', 'l1_gbdt', 'l2_gbdt',
        'l1_dnn', 'l2_dnn'
    ]
  elif method_flag == 'all_except_dnn':
    return ['l1_linear', 'l2_linear', 'random_forest', 'l1_gbdt', 'l2_gbdt']
  elif method_flag == 'linear':
    return ['l1_linear', 'l2_linear']
  elif method_flag == 'gbdt':
    return ['l1_gbdt', 'l2_gbdt']
  elif method_flag == 'dnn':
    return ['l1_dnn', 'l2_dnn']
  else:
    return [method_flag]


def main(_):
  experiment_string = utils.get_experiment_string(
      FLAGS.dataset,
      vec=FLAGS.vec,
      dim_red=FLAGS.dim_red,
      which_features=FLAGS.which_features,
      problem=FLAGS.problem,
      alternate=FLAGS.alternate)
  file_path = os.path.expanduser(FILE_PATH)
  logging.info('DIRECTORY USED %s', file_path)
  param_dir = '{}/params/{}'.format(file_path, experiment_string)
  save_dir = '{}/logs/{}'.format(file_path, experiment_string)
  tf.gfile.MakeDirs(save_dir)

  logging.info('\n' * 3)
  logging.info('Experiment %s and method=%s', experiment_string, FLAGS.method)
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
    oracle_preds = []
    d = utils.get_sim(
        FLAGS.dataset,
        problem=FLAGS.problem,
        which_features=FLAGS.which_features,
        alternate=FLAGS.alternate)
    try:
      num_inf_feature = d._num_inf_feature  # pylint: disable=protected-access
    except:  # pylint: disable=bare-except
      num_inf_feature = None

    for _ in range(FLAGS.num_dataset):
      d.reset()
      # must use get() over generate() for consistency in random sampling calls
      x_train, y_train, x_test, y_test = d.get()

      if FLAGS.debug:  # use smaller subset
        x_train, y_train = shuffle_coordinately(x_train, y_train)
        x_train, y_train = x_train[:500, :], y_train[:500]
        x_test, y_test = shuffle_coordinately(x_test, y_test)
        x_test, y_test = x_test[:500, :], y_test[:500]

      datasets.append((x_train, y_train, x_test, y_test))
      oracle_preds.append((d.oracle_predict(x_train), d.oracle_predict(x_test)))
  else:
    x_train, y_train, x_test, y_test = utils.load_nonsim_data(
        FLAGS.dataset, vec=FLAGS.vec, dim_red=FLAGS.dim_red)

    if FLAGS.debug:  # use smaller subset
      x_train, y_train = shuffle_coordinately(x_train, y_train)
      x_train, y_train = x_train[:500, :250], y_train[:500]
      x_test, y_test = shuffle_coordinately(x_test, y_test)
      x_test, y_test = x_test[:500, :250], y_test[:500]

    datasets.append((x_train, x_test, x_test, y_test))

  # evaluate oracle if experiment involves a simulation dataset
  if 'sim' in FLAGS.dataset:
    if FLAGS.problem == 'regression':
      oracle_metrics = {'train_mse': [], 'test_mse': []}
    else:
      oracle_metrics = {'train_acc': [], 'test_acc': []}

    for ((_, y_train, _, y_test), (y_train_pred, y_test_pred)) in zip(
        datasets, oracle_preds):
      if FLAGS.problem == 'regression':
        oracle_metrics['train_mse'].append(
            mean_squared_error(y_train, y_train_pred))
        oracle_metrics['test_mse'].append(
            mean_squared_error(y_test, y_test_pred))
      else:
        oracle_metrics['train_acc'].append(accuracy(y_train, y_train_pred))
        oracle_metrics['test_acc'].append(accuracy(y_test, y_test_pred))

    logging.info('\n' * 3)
    logging.info('oracle_results')
    logging.info('---')
    oracle_metrics = sorted(oracle_metrics.items(), key=lambda x: x[0])
    print_out = '\n'.join([
        '{}={:.6f}'.format(metric, np.mean(values))
        for metric, values in oracle_metrics
    ])
    print_out += '\n\n'
    print_out += '\n'.join([
        '{}_SE={:.6f}'.format(
            metric, np.true_divide(np.std(values), np.sqrt(FLAGS.num_dataset)))
        for metric, values in oracle_metrics
    ])
    logging.info(print_out)

    if not FLAGS.debug and FLAGS.logtofile:
      save_path = '{}/{}.log'.format(save_dir, 'oracle')
      with tf.gfile.GFile(save_path, 'w') as f:  # save logs to file
        f.write(print_out)
      logging.info('Saved oracle results to %s', save_path)

    logging.info('\n' * 2)

  # evaluate learning methods
  for method in methods:
    if method in ['l1_linear', 'l2_linear']:
      submodule = linear
    elif method == 'random_forest':
      submodule = random_forest
    elif method in ['l1_gbdt', 'l2_gbdt']:
      submodule = gbdt
    elif method in ['l1_dnn', 'l2_dnn']:
      submodule = dnn
    else:
      raise ValueError('Unknown learning method: {}'.format(method))

    start = time.time()
    all_metrics = {}
    other_info = {}
    for d_idx, (x_train, y_train, x_test, y_test) in enumerate(datasets):
      load_path = '{}/dataset_idx={}_{}/{}.param'.format(
          param_dir, d_idx, FLAGS.num_dataset, method)

      if tf.gfile.Exists(load_path):
        with tf.gfile.GFile(load_path,
                            'r') as f:  # load best parameters from file
          lines = f.read().splitlines()
          param_str = lines[0]
          param_dict = {
              i.split('=')[0]: parse_value(i.split('=')[1])
              for i in param_str.split(',')
          }
      else:
        if FLAGS.fast:
          logging.warn(
              'No tuned parameters found (at %s), but using default '
              'parameters since running in FAST mode.', load_path)
          param_dict = None
        else:
          raise RuntimeError('{} does not exist on Colossus'.format(load_path))

      model, metrics = submodule.pipeline(
          x_train,
          y_train,
          x_test,
          y_test,
          param_dict=param_dict,
          problem=FLAGS.problem)
      for k, v in metrics.items():
        if k not in all_metrics:
          all_metrics[k] = []
        all_metrics[k].append(v)

      if FLAGS.problem == 'classification':
        if 'class_props' not in other_info:
          other_info['class_props'] = []
        class_prop = np.true_divide(np.sum(y_test), np.size(y_test, 0))
        other_info['class_props'].append(class_prop)

      if (num_inf_feature is not None and FLAGS.which_features == 'all' and
          'dnn' not in method):
        if 'prop_inf_feat_importance' not in other_info:
          other_info['prop_inf_feat_importance'] = []
        other_info['prop_inf_feat_importance'].append(
            get_prop_inf_feature_importance(method, model, num_inf_feature))

      if 'dnn' in method:
        dnn.clear_keras_session()

    logging.info('Experiment results for method=%s, (%d datasets, %.3f s)',
                 method, FLAGS.num_dataset,
                 time.time() - start)
    logging.info('---')

    all_metrics = sorted(all_metrics.items(), key=lambda x: x[0])
    print_out = '\n'.join([
        '{}={:.6f}'.format(metric, np.mean(values))
        for metric, values in all_metrics
    ])
    print_out += '\n\n'
    print_out += '\n'.join([
        '{}_SE={:.6f}'.format(
            metric, np.true_divide(np.std(values), np.sqrt(FLAGS.num_dataset)))
        for metric, values in all_metrics
    ])

    if other_info:
      print_out += '\n\n'
      other_info = sorted(other_info.items(), key=lambda x: x[0])
      print_out += '\n'.join([
          '{}={:.6f}'.format(metric, np.mean(values))
          for metric, values in other_info
      ])

    logging.info(print_out)

    if not FLAGS.debug and FLAGS.logtofile:
      save_path = '{}/{}.log'.format(save_dir, method)
      with tf.gfile.GFile(save_path, 'w') as f:  # save logs to file
        f.write(print_out)
      logging.info('Saved %s results to %s', method, save_path)

    logging.info('\n' * 2)


if __name__ == '__main__':
  app.run(main)
