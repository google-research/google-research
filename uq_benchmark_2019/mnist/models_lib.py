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
"""Build and train MNIST models for UQ experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import attr
import numpy as np
import scipy.special
import six
from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uq_benchmark_2019 import uq_utils
keras = tf.keras

_NUM_CLASSES = 10
_MNIST_SHAPE = (28, 28, 1)
_NUM_IMAGE_EXAMPLES_TO_RECORD = 32
_BATCH_SIZE_FOR_PREDICT = 1024

ARCHITECTURES = ['mlp', 'lenet']
METHODS = ['vanilla', 'dropout', 'svi', 'll_dropout', 'll_svi']


@attr.s
class ModelOptions(object):
  """Parameters for model construction and fitting."""
  train_epochs = attr.ib()
  num_train_examples = attr.ib()
  batch_size = attr.ib()
  learning_rate = attr.ib()
  method = attr.ib()
  architecture = attr.ib()
  mlp_layer_sizes = attr.ib()
  dropout_rate = attr.ib()
  num_examples_for_predict = attr.ib()
  predictions_per_example = attr.ib()


def _build_mlp(opts):
  """Builds a multi-layer perceptron Keras model."""
  layer_builders = uq_utils.get_layer_builders(opts.method, opts.dropout_rate,
                                               opts.num_train_examples)
  _, dense_layer, dense_last, dropout_fn, dropout_fn_last = layer_builders

  inputs = keras.layers.Input(_MNIST_SHAPE)
  net = keras.layers.Flatten(input_shape=_MNIST_SHAPE)(inputs)
  for size in opts.mlp_layer_sizes:
    net = dropout_fn(net)
    net = dense_layer(size, activation='relu')(net)
  net = dropout_fn_last(net)
  logits = dense_last(_NUM_CLASSES)(net)
  return keras.Model(inputs=inputs, outputs=logits)


def _build_lenet(opts):
  """Builds a LeNet Keras model."""
  layer_builders = uq_utils.get_layer_builders(opts.method, opts.dropout_rate,
                                               opts.num_train_examples)
  conv2d, dense_layer, dense_last, dropout_fn, dropout_fn_last = layer_builders

  inputs = keras.layers.Input(_MNIST_SHAPE)
  net = inputs
  net = conv2d(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=_MNIST_SHAPE)(net)
  net = conv2d(64, (3, 3), activation='relu')(net)
  net = keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
  net = dropout_fn(net)
  net = keras.layers.Flatten()(net)
  net = dense_layer(128, activation='relu')(net)
  net = dropout_fn_last(net)
  logits = dense_last(_NUM_CLASSES)(net)
  return keras.Model(inputs=inputs, outputs=logits)


def build_model(opts):
  """Builds (uncompiled) Keras model from ModelOptions instance."""
  return {'mlp': _build_mlp, 'lenet': _build_lenet}[opts.architecture](opts)


def build_and_train(opts, dataset_train, dataset_eval, output_dir):
  """Returns a trained MNIST model and saves it to output_dir.

  Args:
    opts: ModelOptions
    dataset_train: Pair of images, labels np.ndarrays for training.
    dataset_eval: Pair of images, labels np.ndarrays for continuous eval.
    output_dir: Directory for the saved model and tensorboard events.
  Returns:
    Trained Keras model.
  """
  model = build_model(opts)
  model.compile(
      keras.optimizers.Adam(opts.learning_rate),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )

  tensorboard_cb = keras.callbacks.TensorBoard(
      log_dir=output_dir, write_graph=False)

  train_images, train_labels = dataset_train
  assert len(train_images) == opts.num_train_examples, (
      '%d != %d' % (len(train_images), opts.num_train_examples))
  model.fit(
      train_images, train_labels,
      epochs=opts.train_epochs,
      # NOTE: steps_per_epoch will cause OOM for some reason.
      validation_data=dataset_eval,
      batch_size=opts.batch_size,
      callbacks=[tensorboard_cb],
  )
  return model


def make_predictions(opts, model, dataset):
  """Build a dictionary of model predictions on a given dataset.

  Args:
    opts: ModelOptions.
    model: Trained Keras model.
    dataset: tf.data.Dataset of <image, label> pairs.
  Returns:
    Dictionary containing labels and model logits.
  """
  if opts.num_examples_for_predict:
    dataset = tuple(x[:opts.num_examples_for_predict] for x in dataset)

  batched_dataset = (tf.data.Dataset.from_tensor_slices(dataset)
                     .batch(_BATCH_SIZE_FOR_PREDICT))
  out = collections.defaultdict(list)
  for images, labels in tfds.as_numpy(batched_dataset):
    logits_samples = np.stack(
        [model.predict(images) for _ in range(opts.predictions_per_example)],
        axis=1)  # shape: [batch_size, num_samples, num_classes]
    probs = scipy.special.softmax(logits_samples, axis=-1).mean(-2)
    out['labels'].extend(labels)
    out['logits_samples'].extend(logits_samples)
    out['probs'].extend(probs)
    if len(out['image_examples']) < _NUM_IMAGE_EXAMPLES_TO_RECORD:
      out['image_examples'].extend(images)

  return {k: np.stack(a) for k, a in six.iteritems(out)}
