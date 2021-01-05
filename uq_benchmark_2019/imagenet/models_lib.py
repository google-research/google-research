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
"""Build and train image models for UQ experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import attr
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from uq_benchmark_2019.imagenet import learning_rate_lib
from uq_benchmark_2019.imagenet import resnet50_model
from tensorflow.python.keras.optimizer_v2 import gradient_descent  # pylint: disable=g-direct-tensorflow-import

keras = tf.keras
tfd = tfp.distributions

METHODS = ['vanilla', 'll_dropout', 'll_svi', 'dropout', 'svi',
           'dropout_nofirst']


@attr.s
class ModelOptions(object):
  """Parameters for model construction and fitting."""
  # Modeling options
  method = attr.ib()
  # Data options
  image_shape = attr.ib()
  num_classes = attr.ib()
  examples_per_epoch = attr.ib()
  validation_size = attr.ib()
  use_bfloat16 = attr.ib()
  # SGD Options.
  train_epochs = attr.ib()
  batch_size = attr.ib()
  dropout_rate = attr.ib()  # Only used for dropout-based methods.
  init_learning_rate = attr.ib()
  # VI options.
  std_prior_scale = attr.ib()
  init_prior_scale_mean = attr.ib()
  init_prior_scale_std = attr.ib()
  num_updates = attr.ib()
  # TPU/GPU
  use_tpu = attr.ib()
  num_cores = attr.ib()
  num_gpus = attr.ib()
  num_replicas = attr.ib()
  tpu = attr.ib()


def build_model(opts):
  """Builds a ResNet keras.models.Model."""
  return resnet50_model.ResNet50(
      opts.method, opts.num_classes, opts.num_updates, opts.dropout_rate)


def build_and_train(opts, dataset_train, dataset_eval, output_dir, metrics):
  """Returns a trained image model and saves it to output_dir.

  Args:
    opts: ModelOptions
    dataset_train: tf.data.Dataset for training.
    dataset_eval: tf.data.Dataset for continuous eval during training.
    output_dir: Directory for the saved model.
    metrics: Train/eval metrics to track.
  Returns:
    Trained Keras model.
  """
  model = build_model(opts)
  logging.info('Compiling model.')
  model.compile(
      optimizer=gradient_descent.SGD(
          learning_rate=learning_rate_lib.BASE_LEARNING_RATE,
          momentum=0.9, nesterov=True),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=metrics)

  tensorboard_cb = keras.callbacks.TensorBoard(
      log_dir=output_dir, write_graph=False)

  bs = opts.batch_size
  training_steps_per_epoch = opts.examples_per_epoch // bs

  lr_schedule_cb = learning_rate_lib.LearningRateBatchScheduler(
      schedule=learning_rate_lib.learning_rate_schedule_wrapper(
          training_steps_per_epoch))
  # checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
  #     os.path.join(model_dir, 'cp_{epoch:04d}.ckpt'), save_weights_only=True,
  #     verbose=1)

  model.fit(
      dataset_train,
      steps_per_epoch=training_steps_per_epoch,
      epochs=opts.train_epochs,
      validation_data=dataset_eval,
      validation_steps=int(opts.validation_size // opts.batch_size),
      callbacks=[tensorboard_cb, lr_schedule_cb],
      validation_freq=[10, 20, 30, 40, 50, 60, 70, 80, 90])
  return model
