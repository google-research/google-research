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

"""Train a model for influence function evaluation.

We train a classifier on some dataset, holding out some subset of classes.
We also save a subset of the train, valid, test, and OOD splits to disk so we
can access them later in an easier way.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
from extrapolation.classifier import classifier
from extrapolation.classifier import train_cnn
from extrapolation.utils import dataset_utils
from extrapolation.utils import utils

flags.DEFINE_string('expname', 'temp', 'name of this experiment directory')
flags.DEFINE_integer('max_steps', 1000, 'number of steps of optimization')
flags.DEFINE_integer('max_steps_test', 10, 'number of steps of testing')
flags.DEFINE_integer('run_avg_len', 50,
                     'number of steps of average losses over')
flags.DEFINE_integer('print_freq', 50, 'number of steps between printing')
flags.DEFINE_float('lr', 0.001, 'Adam learning rate')
flags.DEFINE_string('conv_dims', '80,40,20',
                    'comma-separated list of integers for conv layer sizes')
flags.DEFINE_string('conv_sizes', '5,5,5',
                    'comma-separated list of integers for conv filter sizes')
flags.DEFINE_string('dense_sizes', '100',
                    'comma-separated list of integers for dense hidden sizes')
flags.DEFINE_string('mpl_format', 'pdf',
                    'format to save matplotlib  figures in, also '
                    'becomes filename extension')
flags.DEFINE_string('results_dir',
                    '/tmp',
                    'main folder for experimental results')
flags.DEFINE_integer('patience', 50, 'steps of patience for early stopping')
flags.DEFINE_integer('seed', 0, 'random seed for Tensorflow')
flags.DEFINE_integer('n_classes', 10, 'number of classes in prediction')
flags.DEFINE_string('early_stopping_metric', 'loss',
                    'which metric to track for early stopping (loss or error)')
flags.DEFINE_string('training_results_dir',
                    '/tmp',
                    'Output directory for experimental results.')
flags.DEFINE_string('ood_classes', '5', 'a comma-separated list of'
                    'which labels to consider OoD')
flags.DEFINE_float('label_noise', '0', 'what percentage of labels to flip')
flags.DEFINE_string('dataset_name', 'mnist', 'what dataset to use')
# Note: --conv_dims=50,20 --conv_sizes=5,5 is a reasonable default

FLAGS = flags.FLAGS



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  params = FLAGS.flag_values_dict()
  tf.set_random_seed(params['seed'])

  params['results_dir'] = utils.make_subdir(
      params['training_results_dir'], params['expname'])
  params['figdir'] = utils.make_subdir(params['results_dir'], 'figs')
  params['ckptdir'] = utils.make_subdir(params['results_dir'], 'ckpts')
  params['logdir'] = utils.make_subdir(params['results_dir'], 'logs')
  params['tensordir'] = utils.make_subdir(params['results_dir'], 'tensors')

  # Load the classification model.
  conv_dims = [int(x) for x in (params['conv_dims'].split(',')
                                if params['conv_dims'] else [])]
  conv_sizes = [int(x) for x in (params['conv_sizes'].split(',')
                                 if params['conv_sizes'] else [])]
  dense_sizes = [int(x) for x in (params['dense_sizes'].split(',')
                                  if params['dense_sizes'] else [])]
  params['n_layers'] = len(conv_dims)
  clf = classifier.CNN(conv_dims, conv_sizes, dense_sizes,
                       params['n_classes'], onehot=True)

  # Checkpoint the initialized model, in case we want to re-run it from there.
  utils.checkpoint_model(clf, params['ckptdir'], 'initmodel')

  # Load the "in-distribution" and "out-of-distribution" classes as
  # separate splits.
  ood_classes = [int(x) for x in params['ood_classes'].split(',')]
  # We assume we train on all non-OOD classes.
  all_classes = range(params['n_classes'])
  ind_classes = [x for x in all_classes if x not in ood_classes]
  (itr_train, itr_valid, itr_test, itr_test_ood
  ) = dataset_utils.load_dset_ood_supervised_onehot(
      ind_classes, ood_classes, label_noise=(params['label_noise']),
      dset_name=params['dataset_name'])
  # Train and test the model in-distribution, and save test outputs.
  train_cnn.train_classifier(clf, itr_train, itr_valid, params)
  train_cnn.test_classifier(clf, itr_test, params, 'test')

  # Save model outputs on the training set.
  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'train_tensors')
  train_cnn.test_classifier(clf, itr_train, params, 'train')

  # Save model outputs on the OOD set.
  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'ood_tensors')
  train_cnn.test_classifier(clf, itr_test_ood, params, 'ood')

  params['tensordir'] = utils.make_subdir(
      params['results_dir'], 'tensors')

  # Save to disk samples of size 1000 from the train, valid, test and OOD sets.
  train_data = utils.aggregate_batches(itr_train, 1000,
                                       ['train_x_infl', 'train_y_infl'])

  validation_data = utils.aggregate_batches(itr_valid, 1000,
                                            ['valid_x_infl', 'valid_y_infl'])

  test_data = utils.aggregate_batches(itr_test, 1000,
                                      ['test_x_infl', 'test_y_infl'])

  ood_data = utils.aggregate_batches(itr_test_ood, 1000,
                                     ['ood_x_infl', 'ood_y_infl'])
  utils.save_tensors(train_data.items() + validation_data.items() +
                     test_data.items() + ood_data.items(),
                     params['tensordir'])


if __name__ == '__main__':
  app.run(main)
