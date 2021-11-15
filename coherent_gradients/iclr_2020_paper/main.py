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

"""Run experiments from ICLR 2020 paper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import uuid

from absl import app
from absl import flags
from absl import logging

import coherent_gradients.iclr_2020_paper.utils as utils

################################################################################
# set up FLAGS
################################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string('idx', str(uuid.uuid4()),
                    'unique identifier for this experiment')
flags.DEFINE_string('output_dir', '/tmp/test', 'output directory')
flags.DEFINE_string('dataset', 'mnist',
                    'the dataset to use (e.g. mnist, fashion_mnist, cifar10)')
flags.DEFINE_integer('dummy', 0,
                     'dummy parameter to study run-to-run variation')
flags.DEFINE_integer('hidden', 3, 'number of hidden layers')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_integer('log_every', 600, 'how often to log training stats')
flags.DEFINE_bool('log_gradients', False, 'log gradients or not')
flags.DEFINE_integer('max_epochs', 100, 'maximum number of epochs to train')
flags.DEFINE_integer('mb_size', 100, 'how many examples in each minibatch')
flags.DEFINE_string('policy', 'sum',
                    'policy to combine gradients from each example')
flags.DEFINE_integer('randomize_labels_pct', 0,
                     'pct of labels to randomize in training set')
flags.DEFINE_bool('use_gpu', True, 'use gpu')
flags.DEFINE_integer('width', 256, 'width of each hidden layer')
flags.DEFINE_integer('winsorize_pct', 10,
                     'percentile to cutoff for winsorization')


def read_config(status):
  """Construct configuration from flags."""
  assert status in ['running', 'completed']
  config = collections.OrderedDict([
      ('idx', FLAGS.idx),
      ('status', status),
      ('output_dir', FLAGS.output_dir),
      ('dataset', FLAGS.dataset),
      ('dummy', FLAGS.dummy),
      ('hidden', FLAGS.hidden),
      ('learning_rate', FLAGS.learning_rate),
      ('log_every', FLAGS.log_every),
      ('log_gradients', FLAGS.log_gradients),
      ('max_epochs', FLAGS.max_epochs),
      ('mb_size', FLAGS.mb_size),
      ('policy', FLAGS.policy),
      ('randomize_labels_pct', FLAGS.randomize_labels_pct),
      ('use_gpu', FLAGS.use_gpu),
      ('width', FLAGS.width),
      ('winsorize_pct', FLAGS.winsorize_pct),
  ])
  return config


def dump_config(config):
  with open('{}/config-{}.json'.format(config['output_dir'], config['idx']),
            'w') as f:
    json.dump(config, f)

################################################################################
# main
################################################################################


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  os.mkdir(FLAGS.output_dir)

  logging.info('uuid = %s', FLAGS.idx)
  logging.info('start, output_dir = %s', FLAGS.output_dir)

  # from FLAGS to our format
  config = read_config('running')
  dump_config(config)

  logging.info('config = %s', config)

  dataset = utils.load_data(config['dataset'], config['randomize_labels_pct'])
  net = utils.network(dataset, config)
  net.train()

  config = read_config('completed')
  dump_config(config)

  logging.info('done, output_dir = %s', config['output_dir'])

if __name__ == '__main__':
  app.run(main)
