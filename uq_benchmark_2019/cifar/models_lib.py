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
"""Build and train image models for UQ experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import attr
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019 import uq_utils
from uq_benchmark_2019.cifar import resnet

keras = tf.keras
tfd = tfp.distributions

METHODS = ['vanilla', 'll_dropout', 'll_svi', 'dropout', 'svi',
           'dropout_nofirst', 'wide_dropout']
_VALIDATION_STEPS = 100


@attr.s
class ModelOptions(object):
  """Parameters for model construction and fitting."""
  # Modeling options
  method = attr.ib()
  resnet_depth = attr.ib()
  num_resnet_filters = attr.ib()
  # Data options
  image_shape = attr.ib()
  num_classes = attr.ib()
  examples_per_epoch = attr.ib()
  # SGD Options.
  train_epochs = attr.ib()
  batch_size = attr.ib()
  dropout_rate = attr.ib()  # Only used for dropout-based methods.
  init_learning_rate = attr.ib()
  # VI options.
  # TODO(yovadia): Maybe remove defaults.
  std_prior_scale = attr.ib(1.5)
  init_prior_scale_mean = attr.ib(-1)
  init_prior_scale_std = attr.ib(.1)


def load_model(model_dir):
  model_opts = experiment_utils.load_config(model_dir + '/model_options.json')
  model_opts = ModelOptions(**model_opts)
  logging.info('Loaded model options: %s', model_opts)

  model = build_model(model_opts)
  logging.info('Loading model weights...')
  model.load_weights(model_dir + '/model.ckpt')
  logging.info('done loading model weights.')
  return model


def build_model(opts):
  """Builds a ResNet keras.models.Model."""
  is_dropout_last = opts.method in (
      'll_dropout', 'dropout', 'dropout_nofirst', 'wide_dropout')
  is_dropout_all = opts.method in ('dropout', 'dropout_nofirst', 'wide_dropout')
  all_dropout_rate = opts.dropout_rate if is_dropout_all else None
  last_dropout_rate = opts.dropout_rate if is_dropout_last else None

  eb_prior_fn = uq_utils.make_prior_fn_for_empirical_bayes(
      opts.init_prior_scale_mean, opts.init_prior_scale_std)

  keras_in = keras.layers.Input(shape=opts.image_shape)
  net = resnet.build_resnet_v1(
      keras_in, depth=opts.resnet_depth,
      variational=opts.method == 'svi',
      std_prior_scale=opts.std_prior_scale,
      eb_prior_fn=eb_prior_fn,
      always_on_dropout_rate=all_dropout_rate,
      no_first_layer_dropout=opts.method == 'dropout_nofirst',
      examples_per_epoch=opts.examples_per_epoch,
      num_filters=opts.num_resnet_filters)
  if opts.method == 'vanilla':
    keras_out = keras.layers.Dense(
        opts.num_classes, kernel_initializer='he_normal')(net)
  elif is_dropout_last:
    net = keras.layers.Dropout(last_dropout_rate)(net, training=True)
    keras_out = keras.layers.Dense(
        opts.num_classes, kernel_initializer='he_normal')(net)
  elif opts.method in ('svi', 'll_svi'):
    divergence_fn = uq_utils.make_divergence_fn_for_empirical_bayes(
        opts.std_prior_scale, opts.examples_per_epoch)

    keras_out = tfp.layers.DenseReparameterization(
        opts.num_classes,
        kernel_prior_fn=eb_prior_fn,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            loc_initializer=keras.initializers.he_normal()),
        kernel_divergence_fn=divergence_fn)(net)

  return keras.models.Model(inputs=keras_in, outputs=keras_out)


def _make_lr_scheduler(init_lr):
  """Builds a keras LearningRateScheduler."""

  def schedule_fn(epoch):
    """Learning rate schedule function."""
    rate = init_lr
    if epoch > 180:
      rate *= 0.5e-3
    elif epoch > 160:
      rate *= 1e-3
    elif epoch > 120:
      rate *= 1e-2
    elif epoch > 80:
      rate *= 1e-1
    logging.info('Learning rate=%f for epoch=%d ', rate, epoch)
    return rate
  return keras.callbacks.LearningRateScheduler(schedule_fn)


def build_and_train(opts, dataset_train, dataset_eval, output_dir):
  """Returns a trained image model and saves it to output_dir.

  Args:
    opts: ModelOptions
    dataset_train: tf.data.Dataset for training.
    dataset_eval: tf.data.Dataset for continuous eval during training.
    output_dir: Directory for the saved model.
  Returns:
    Trained Keras model.
  """
  model = build_model(opts)
  model.compile(
      keras.optimizers.Adam(opts.init_learning_rate),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  tensorboard_cb = keras.callbacks.TensorBoard(
      log_dir=output_dir, write_graph=False)
  lr_scheduler = _make_lr_scheduler(opts.init_learning_rate)
  bs = opts.batch_size
  dataset_eval = dataset_eval.take(bs * _VALIDATION_STEPS).repeat().batch(bs)
  model.fit(
      dataset_train.repeat().shuffle(10*bs).batch(bs),
      steps_per_epoch=opts.examples_per_epoch // bs,
      epochs=opts.train_epochs,
      validation_data=dataset_eval,
      validation_steps=_VALIDATION_STEPS,
      callbacks=[tensorboard_cb, lr_scheduler],
  )
  return model
