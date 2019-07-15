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

# Lint as: python2, python3
"""Trains a feature-column model on Criteo Kaggle data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019.criteo import data_lib
from uq_benchmark_2019.criteo import hparams_lib
from uq_benchmark_2019.criteo import models_lib

FLAGS = flags.FLAGS


def _declare_flags():
  """Declare flags; not invoked when this module is imported as a library."""
  flags.DEFINE_enum('method', None, models_lib.METHODS,
                    'Name of modeling method.')
  flags.DEFINE_string('output_dir', None, 'Output directory.')
  flags.DEFINE_integer('test_level', 0, 'Testing level.')
  flags.DEFINE_integer('train_epochs', 1, 'Number of epochs for training.')
  flags.DEFINE_integer('task', 0, 'Borg task number.')


def run(method, output_dir, num_epochs, fake_data=False, fake_training=False):
  """Trains a model and records its predictions on configured datasets.

  Args:
    method: Modeling method to experiment with.
    output_dir: Directory to record the trained model and output stats.
    num_epochs: Number of training epochs.
    fake_data: If true, use fake data.
    fake_training: If true, train for a trivial number of steps.
  Returns:
    Trained Keras model.
  """
  tf.io.gfile.makedirs(output_dir)
  data_config_train = data_lib.DataConfig(split='train', fake_data=fake_data)
  data_config_valid = data_lib.DataConfig(split='valid', fake_data=fake_data)

  hparams = hparams_lib.get_tuned_hparams(method, parameterization='C')
  model_opts = hparams_lib.model_opts_from_hparams(hparams, method,
                                                   parameterization='C',
                                                   fake_training=fake_training)

  experiment_utils.record_config(model_opts, output_dir+'/model_options.json')

  model = models_lib.build_and_train_model(
      model_opts, data_config_train, data_config_valid,
      output_dir=output_dir,
      num_epochs=num_epochs,
      fake_training=fake_training)

  logging.info('Saving model to output_dir.')
  model.save_weights(output_dir + '/model.ckpt')
  # TODO(yovadia): Looks like Keras save_model does not work with Python3.
  # (e.g. see b/129323565).
  # experiment_utils.save_model(model, output_dir)
  return model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run(FLAGS.method,
      FLAGS.output_dir,
      num_epochs=FLAGS.train_epochs,
      fake_data=FLAGS.test_level > 1,
      fake_training=FLAGS.test_level > 0)

if __name__ == '__main__':

  tf.enable_v2_behavior()
  _declare_flags()
  app.run(main)
