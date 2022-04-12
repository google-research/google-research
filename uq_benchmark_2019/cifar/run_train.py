# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Trains a ResNet model on CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019 import image_data_utils
from uq_benchmark_2019.cifar import data_lib
from uq_benchmark_2019.cifar import hparams_lib
from uq_benchmark_2019.cifar import models_lib

FLAGS = flags.FLAGS


def _declare_flags():
  """Declare flags; not invoked when this module is imported as a library."""
  flags.DEFINE_enum('method', None, models_lib.METHODS,
                    'Name of modeling method.')
  flags.DEFINE_string('output_dir', None, 'Output directory.')
  flags.DEFINE_integer('test_level', 0, 'Testing level.')
  flags.DEFINE_integer('task', 0, 'Task number.')


def run(method, output_dir, fake_data=False, fake_training=False):
  """Trains a model and records its predictions on configured datasets.

  Args:
    method: Modeling method to experiment with.
    output_dir: Directory to record the trained model and output stats.
    fake_data: If true, use fake data.
    fake_training: If true, train for a trivial number of steps.
  Returns:
    Trained Keras model.
  """
  tf.io.gfile.makedirs(output_dir)
  model_opts = hparams_lib.model_opts_from_hparams(hparams_lib.HPS_DICT[method],
                                                   method,
                                                   fake_training=fake_training)
  if fake_training:
    model_opts.batch_size = 32
    model_opts.examples_per_epoch = 256
    model_opts.train_epochs = 1

  experiment_utils.record_config(model_opts, output_dir+'/model_options.json')

  dataset_train = data_lib.build_dataset(image_data_utils.DATA_CONFIG_TRAIN,
                                         is_training=True,
                                         fake_data=fake_data)
  dataset_test = data_lib.build_dataset(image_data_utils.DATA_CONFIG_TEST,
                                        fake_data=fake_data)
  model = models_lib.build_and_train(model_opts,
                                     dataset_train, dataset_test,
                                     output_dir)

  logging.info('Saving model to output_dir.')
  model.save_weights(output_dir + '/model.ckpt')
  # TODO(yovadia): Figure out why save_model() wants to serialize ModelOptions.
  return model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run(FLAGS.method,
      FLAGS.output_dir.replace('%task%', str(FLAGS.task)),
      fake_data=FLAGS.test_level > 1,
      fake_training=FLAGS.test_level > 0)

if __name__ == '__main__':
  _declare_flags()
  app.run(main)
