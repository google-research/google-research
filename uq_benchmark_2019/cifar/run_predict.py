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

# Lint as: python2, python3
"""Generate predictions from a trained model on a range of datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
from uq_benchmark_2019 import array_utils
from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019 import image_data_utils
from uq_benchmark_2019.cifar import data_lib
from uq_benchmark_2019.cifar import models_lib

_BATCH_SIZE = 256

FLAGS = flags.FLAGS


def _declare_flags():
  """Declare flags; not invoked when this module is imported as a library."""
  flags.DEFINE_string('model_dir', None, 'Path to Keras model.')
  flags.DEFINE_string('output_dir', None, 'Output directory.')

  flags.DEFINE_integer('predictions_per_example', 1,
                       'Number of prediction samples to generate per example.')
  flags.DEFINE_integer('max_examples', None,
                       'Maximum number of examples to process per dataset.')

  flags.DEFINE_string('dataset_name', None, 'Configured dataset name.')
  flags.DEFINE_integer('task', 0, 'Task number.')


def run(dataset_name, model_dir,
        predictions_per_example, max_examples,
        output_dir,
        fake_data=False):
  """Runs predictions on the given dataset using the specified model."""
  tf.io.gfile.makedirs(output_dir)
  data_config = image_data_utils.get_data_config(dataset_name)
  dataset = data_lib.build_dataset(data_config, fake_data=fake_data)
  if max_examples:
    dataset = dataset.take(max_examples)

  model = models_lib.load_model(model_dir)
  logging.info('Starting predictions.')
  predictions = experiment_utils.make_predictions(
      model, dataset.batch(_BATCH_SIZE), predictions_per_example)

  logging.info('Done computing predictions; recording results to disk.')
  array_utils.write_npz(output_dir, 'predictions_%s.npz' % dataset_name,
                        predictions)
  del predictions['logits_samples']
  array_utils.write_npz(output_dir, 'predictions_small_%s.npz' % dataset_name,
                        predictions)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run(FLAGS.dataset_name,
      FLAGS.model_dir,
      FLAGS.predictions_per_example,
      FLAGS.max_examples,
      FLAGS.output_dir.replace('%task%', str(FLAGS.task)))


if __name__ == '__main__':
  _declare_flags()
  app.run(main)
