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
"""Library of models for experiemnts on Criteo data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import attr
import numpy as np
from six.moves import range
import tensorflow.compat.v2 as tf
from uq_benchmark_2019 import experiment_utils
from uq_benchmark_2019 import uq_utils
from uq_benchmark_2019.criteo import data_lib
keras = tf.keras

METHODS = ['vanilla', 'll_dropout', 'll_svi', 'dropout', 'svi']


@attr.s
class ModelOptions(object):
  """Options for Criteo model."""
  # Model parameters.
  method = attr.ib()
  layer_sizes = attr.ib()
  num_hash_buckets = attr.ib()
  num_embed_dims = attr.ib()
  dropout_rate = attr.ib()

  # Training parameters.
  learning_rate = attr.ib()
  batch_size = attr.ib()
  # TODO(yovadia): Maybe add l2_regularization.


def make_input_layers():
  """Defines an input layer for Keras model with int32 and string dtypes."""
  out = {}
  for idx in range(1, data_lib.NUM_TOTAL_FEATURES+1):
    dtype = tf.int32 if idx <= data_lib.NUM_INT_FEATURES else tf.string
    name = data_lib.feature_name(idx)
    out[name] = keras.layers.Input([], dtype=dtype, name=name)
  return out


def make_feature_columns(opts):
  """Build feature_columns for converting features to a dense vector."""
  tffc = tf.feature_column
  out_cat = []
  for idx in data_lib.CAT_FEATURE_INDICES:
    name = data_lib.feature_name(idx)
    cat_idx = idx - data_lib.NUM_INT_FEATURES - 1
    num_buckets = opts.num_hash_buckets[cat_idx]
    num_embed_dims = opts.num_embed_dims[cat_idx]

    hash_col = tffc.categorical_column_with_hash_bucket(name, num_buckets)
    cat_col = (tffc.embedding_column(hash_col, num_embed_dims)
               if num_embed_dims else tffc.indicator_column(hash_col))
    out_cat.append(cat_col)

  out_int = []
  for idx in data_lib.INT_FEATURE_INDICES:
    name = data_lib.feature_name(idx)
    out_int.append(tffc.numeric_column(name))
  return out_int, out_cat


def load_trained_model(model_dir, load_weights=True, as_components=False):
  """Load a trained model using recorded options and weights."""
  model_opts = experiment_utils.load_config(model_dir + '/model_options.json')
  model_opts = ModelOptions(**model_opts)
  logging.info('Loaded model options: %s', model_opts)

  model = build_model(model_opts, as_components=as_components)
  if load_weights:
    logging.info('Loading model weights...')
    if as_components:
      _ = [m.load_weights(model_dir + '/model.ckpt') for m in model]
    else:
      model.load_weights(model_dir + '/model.ckpt')
    logging.info('done loading model weights.')
  return model


def build_model(opts, as_components=False):
  """Builds a Keras model for Criteo data."""
  layers_tup = uq_utils.get_layer_builders(opts.method, opts.dropout_rate,
                                           data_lib.NUM_TRAIN_EXAMPLES)
  _, dense_layer, dense_last, dropout_fn, dropout_fn_last = layers_tup

  fcs_int, fcs_cat = make_feature_columns(opts)
  input_layer = make_input_layers()
  features = input_layer
  dense_int = keras.layers.DenseFeatures(fcs_int)(features)
  dense_cat = keras.layers.DenseFeatures(fcs_cat)(features)
  net = tf.concat([dense_int, dense_cat], axis=-1)
  logging.info('Dense layer shape: %s', net.shape)
  # TODO(yovadia): Consider explicit normalization according to data stats.
  net = keras.layers.BatchNormalization()(net)
  for size in opts.layer_sizes:
    net = dropout_fn(net)
    net = dense_layer(size, activation='relu')(net)
  prelogits = dropout_fn_last(net)
  # Sigmoid output necessary to get useful AUC metric outputs.
  lastlayer = dense_last(1, activation='sigmoid')
  probs = lastlayer(prelogits)
  if as_components:
    trunc = keras.Model(inputs=input_layer, outputs=prelogits)
    embedings_in = keras.layers.Input(shape=prelogits.shape[1:])
    head = keras.Model(inputs=embedings_in, outputs=lastlayer(embedings_in))
    return trunc, head
  return keras.Model(inputs=input_layer, outputs=probs)


def build_and_train_model(opts,
                          data_config_train, data_config_eval,
                          output_dir, num_epochs, fake_training=False):
  """Compile and fit a Keras Criteo model."""
  model = build_model(opts)
  logging.info('Compiling model...')
  model.compile(keras.optimizers.Adam(opts.learning_rate),
                loss=keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()])

  logging.info('Building datasets...')
  bs = opts.batch_size
  steps_per_epoch = 2 if fake_training else data_lib.NUM_TRAIN_EXAMPLES // bs
  dataset_eval = data_lib.build_dataset(data_config_eval, batch_size=bs)
  dataset_train = data_lib.build_dataset(data_config_train, batch_size=bs,
                                         is_training=True)
  tensorboard_cb = keras.callbacks.TensorBoard(
      update_freq=int(1e5), log_dir=output_dir, write_graph=False)

  logging.info('Starting training...')
  model.fit(dataset_train,
            validation_data=dataset_eval,
            epochs=1 if fake_training else num_epochs,
            validation_steps=100,
            callbacks=[tensorboard_cb],
            steps_per_epoch=steps_per_epoch)
  return model


def make_predictions(model, batched_dataset, predictions_per_example):
  """Generate labels and predictions for a model on a dataset."""
  labels, probs = [], []
  for i, (inputs_i, labels_i) in enumerate(batched_dataset):
    logging.info('predict iteration=%d, len(labels)=%d', i, len(labels))
    probs_i = np.stack([
        model.predict(inputs_i).squeeze(-1)
        for _ in range(predictions_per_example)
    ],
                       axis=1)
    labels.extend(labels_i)
    probs.append(probs_i)

  labels = np.stack(labels, axis=0).astype(np.int32)
  probs = np.concatenate(probs, axis=0).astype(np.float32)
  return {'labels': labels, 'probs_samples': probs, 'probs': probs.mean(-1)}
