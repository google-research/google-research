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

# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

r"""ResNet-50 implemented with Keras running on TPU or multi-GPU.

This file shows how you can run ResNet-50 on a TPU or multi-GPU using TensorFlow
Keras support. This is configured for ImageNet (e.g. 1000 classes), but you can
easily adapt to your own datasets by changing the code appropriately.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019 import image_data_utils
from uq_benchmark_2019.imagenet import data_lib
from uq_benchmark_2019.imagenet import hparams_lib
from uq_benchmark_2019.imagenet import learning_rate_lib
from uq_benchmark_2019.imagenet import models_lib

gfile = tf.io.gfile
kb = tf.keras.backend
WEIGHTS_TXT = 'resnet50_weights.h5'

FLAGS = flags.FLAGS


def _declare_flags():
  """Declare flags; not invoked when this module is imported as a library."""
  # Common flags for TPU models.
  flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
  flags.DEFINE_boolean('use_tpu', True, 'Use TPU, otherwise use CPU.')
  flags.DEFINE_string('master', None, 'Master')
  flags.DEFINE_integer('task', 0, 'Task number')
  flags.DEFINE_string('output_dir', None, 'The directory where the model '
                      'weights and training/evaluation summaries are stored. If'
                      'not specified, save to /tmp/resnet50.')
  flags.DEFINE_enum('method', None, models_lib.METHODS, 'Name of modeling '
                    'method.')
  flags.DEFINE_integer('test_level', 0, 'Testing level.')

  # Special flags for Resnet50.
  flags.DEFINE_bool(
      'eval_top_5_accuracy', True,
      'Eval both top 1 and top 5 accuracy. Otherwise, eval top 1 accuracy.')
  flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores.')
  flags.DEFINE_integer('num_gpus', 1, 'For multi-gpu training.')
  flags.DEFINE_integer('num_replicas', 1, 'For distributed GPU training.')


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  """TPU version of sparse_top_k_categorical_accuracy."""
  y_pred_rank = tf.convert_to_tensor(y_pred).get_shape().ndims
  y_true_rank = tf.convert_to_tensor(y_true).get_shape().ndims
  # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      kb.int_shape(y_true)) == len(kb.int_shape(y_pred))):
    y_true = tf.squeeze(y_true, [-1])

  y_true = tf.cast(y_true, 'int32')

  in_top_k_on_device = tf.nn.in_top_k(y_true, y_pred, k)
  return kb.mean(in_top_k_on_device, axis=-1)


def save_model(model, output_dir, method, use_tpu, task_number):
  """Save the trained model."""
  if use_tpu or task_number == 0:
    # On GPU this seems to cause the different gpus to overwrite each other
    model.save_weights(os.path.join(output_dir, 'model.ckpt'))

    # saving model to hdf5 doesn't work for svi methods
    if 'svi' not in method:
      weights_file_path = os.path.join(output_dir, WEIGHTS_TXT)

      # Google internal case.
      if weights_file_path.startswith('/cns'):
        logging.info('Saving weights and optimizer states into %s',
                     weights_file_path)
        logging.info('This might take a while...')
        save_model_to_cns(model, weights_file_path,
                          overwrite=True, include_optimizer=True)

      logging.info('Saving weights and optimizer states into %s',
                   weights_file_path)
      logging.info('This might take a while...')
      model.save(weights_file_path, overwrite=True, include_optimizer=True)


def run(method, output_dir, task_number, use_tpu, tpu, metrics, fake_data=False,
        fake_training=False):
  """Train a ResNet model on ImageNet."""
  gfile.makedirs(output_dir)
  model_opts = hparams_lib.model_opts_from_hparams(hparams_lib.HPS_DICT[method],
                                                   method, use_tpu, tpu,
                                                   fake_training=fake_training)
  if fake_training:
    model_opts.batch_size = 32
    model_opts.examples_per_epoch = 256
    model_opts.train_epochs = 1

  if use_tpu:
    logging.info('Use TPU at %s',
                 model_opts.tpu if model_opts.tpu is not None else 'local')
    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=model_opts.tpu)
    tf.contrib.distribute.initialize_tpu_system(resolver)
    strategy = tf.contrib.distribute.TPUStrategy(resolver)
  else:
    strategy = experiment_utils.get_distribution_strategy(
        distribution_strategy='default',
        num_gpus=model_opts.num_gpus,
        num_workers=model_opts.num_replicas,
        all_reduce_alg=None)

  logging.info('Use global batch size: %s.', model_opts.batch_size)
  logging.info('Use bfloat16: %s.', model_opts.use_bfloat16)
  experiment_utils.record_config(model_opts, output_dir+'/model_options.json')

  imagenet_train = data_lib.build_dataset(
      image_data_utils.DATA_CONFIG_TRAIN,
      batch_size=model_opts.batch_size,
      is_training=True,
      fake_data=fake_data,
      use_bfloat16=model_opts.use_bfloat16)

  imagenet_eval = data_lib.build_dataset(
      image_data_utils.DATA_CONFIG_TEST,
      batch_size=model_opts.batch_size,
      fake_data=fake_data,
      use_bfloat16=model_opts.use_bfloat16)

  if fake_training:
    model = models_lib.build_and_train(
        model_opts, imagenet_train, imagenet_eval, output_dir, metrics)
  else:
    with strategy.scope():
      model = models_lib.build_and_train(
          model_opts, imagenet_train, imagenet_eval, output_dir, metrics)

  save_model(model, output_dir, method, use_tpu, task_number)


def main(unused_argv):
  logging.info('Base LR: %s.', learning_rate_lib.BASE_LEARNING_RATE)
  logging.info('Enable top 5 accuracy: %s.', FLAGS.eval_top_5_accuracy)

  metrics = ['sparse_categorical_crossentropy']
  if FLAGS.eval_top_5_accuracy:
    metrics.append(sparse_top_k_categorical_accuracy)

  run(FLAGS.method,
      FLAGS.output_dir.replace('%task%', str(FLAGS.task)),
      task_number=FLAGS.task,
      use_tpu=FLAGS.use_tpu,
      tpu=FLAGS.tpu,
      metrics=metrics,
      fake_data=FLAGS.test_level > 1,
      fake_training=FLAGS.test_level > 0)


if __name__ == '__main__':

  tf.enable_v2_behavior()  # Required due to b/128610213.
  tf.logging.set_verbosity(tf.logging.INFO)
  _declare_flags()
  app.run(main)
