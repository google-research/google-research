# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Main function for running a CNN classifier on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from extrapolation.classifier.classifier import CNN as CNN
from extrapolation.classifier.classifier import train_cnn
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

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('print this')

  params = FLAGS.flag_values_dict()
  tf.set_random_seed(params['seed'])

  params['results_dir'] = utils.make_subdir(
      params['results_dir'], params['expname'])
  params['figdir'] = utils.make_subdir(params['results_dir'], 'figs')
  params['ckptdir'] = utils.make_subdir(params['results_dir'], 'ckpts')
  params['logdir'] = utils.make_subdir(params['results_dir'], 'logs')
  params['tensordir'] = utils.make_subdir(params['results_dir'], 'tensors')

  itr_train, itr_valid, itr_test = dataset_utils.load_dset_supervised()

  conv_dims = [int(x) for x in params['conv_dims'].split(',')]
  conv_sizes = [int(x) for x in params['conv_sizes'].split(',')]
  dense_sizes = [int(x) for x in params['dense_sizes'].split(',')]
  clf = CNN(conv_dims, conv_sizes, dense_sizes, params['n_classes'])
  params['n_layers'] = len(conv_dims)

  train_cnn.train_classifier(clf, itr_train, itr_valid, params)
  train_cnn.test_classifier(clf, itr_test, params, 'test')

if __name__ == '__main__':
  app.run(main)
