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

# Lint as: python2, python3
"""Configures and runs distributional-skew UQ experiments on MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v2 as tf

from uq_benchmark_2019 import array_utils
from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019.mnist import data_lib
from uq_benchmark_2019.mnist import hparams_lib
from uq_benchmark_2019.mnist import models_lib
gfile = tf.io.gfile


def get_experiment_config(method, architecture,
                          test_level, output_dir=None):
  """Returns model and data configs."""
  data_opts_list = data_lib.DATA_OPTIONS_LIST
  if test_level:
    data_opts_list = data_opts_list[:4]

  model_opts = hparams_lib.get_tuned_model_options(architecture, method,
                                                   fake_data=test_level > 1,
                                                   fake_training=test_level > 0)
  if output_dir:
    experiment_utils.record_config(model_opts, output_dir+'/model_options.json')
  return model_opts, data_opts_list


def run(method, architecture, output_dir, test_level):
  """Trains a model and records its predictions on configured datasets.

  Args:
    method: Name of modeling method (vanilla, dropout, svi, ll_svi).
    architecture: Name of DNN architecture (mlp or dropout).
    output_dir: Directory to record the trained model and output stats.
    test_level: Zero indicates no testing. One indicates testing with real data.
      Two is for testing with fake data.
  """
  fake_data = test_level > 1
  gfile.makedirs(output_dir)
  model_opts, data_opts_list = get_experiment_config(method, architecture,
                                                     test_level=test_level,
                                                     output_dir=output_dir)

  # Separately build dataset[0] with shuffle=True for training.
  dataset_train = data_lib.build_dataset(data_opts_list[0], fake_data=fake_data)
  dataset_eval = data_lib.build_dataset(data_opts_list[1], fake_data=fake_data)
  model = models_lib.build_and_train(model_opts,
                                     dataset_train, dataset_eval, output_dir)
  logging.info('Saving model to output_dir.')
  model.save_weights(output_dir + '/model.ckpt')

  for idx, data_opts in enumerate(data_opts_list):
    dataset = data_lib.build_dataset(data_opts, fake_data=fake_data)
    logging.info('Running predictions for dataset #%d', idx)
    stats = models_lib.make_predictions(model_opts, model, dataset)
    array_utils.write_npz(output_dir, 'stats_%d.npz' % idx, stats)
    del stats['logits_samples']
    array_utils.write_npz(output_dir, 'stats_small_%d.npz' % idx, stats)
