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

"""Trains and evaluates unprocessing neural network.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from unprocessing import dataset
from unprocessing import estimator
from unprocessing import network

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir',
    None,
    'Location at which to save model logs and checkpoints.')

flags.DEFINE_string(
    'train_pattern',
    None,
    'Pattern for directory containing source JPG images for training.')

flags.DEFINE_string(
    'test_pattern',
    None,
    'Pattern for directory containing source JPG images for testing.')

flags.DEFINE_integer(
    'image_size',
    256,
    'Width and height to crop training and testing frames. '
    'Must be a multiple of 16',
    lower_bound=16)

flags.DEFINE_integer(
    'batch_size',
    16,
    'Training batch size.',
    lower_bound=1)

flags.DEFINE_float(
    'learning_rate',
    2e-5,
    'Learning rate for Adam optimization.',
    lower_bound=0.0)

flags.register_validator(
    'image_size',
    lambda image_size: image_size % 16 == 0,
    message='\'image_size\' must multiple of 16.')

flags.mark_flag_as_required('model_dir')
flags.mark_flag_as_required('train_pattern')
flags.mark_flag_as_required('test_pattern')


def main(_):
  inference_fn = network.inference
  hparams = tf.contrib.training.HParams(learning_rate=FLAGS.learning_rate)
  model_fn = estimator.create_model_fn(inference_fn, hparams)
  config = tf.estimator.RunConfig(FLAGS.model_dir)
  tf_estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

  train_dataset_fn = dataset.create_dataset_fn(
      FLAGS.train_pattern,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      batch_size=FLAGS.batch_size)

  eval_dataset_fn = dataset.create_dataset_fn(
      FLAGS.test_pattern,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      batch_size=FLAGS.batch_size)

  train_spec, eval_spec = estimator.create_train_and_eval_specs(
      train_dataset_fn, eval_dataset_fn)

  tf.estimator.train_and_evaluate(tf_estimator, train_spec, eval_spec)


if __name__ == '__main__':
  tf.app.run(main)
