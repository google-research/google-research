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
"""Utilities to help set up and run experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path

from absl import logging

import numpy as np
import scipy.special
from six.moves import range
from six.moves import zip
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

gfile = tf.io.gfile


class _SimpleJsonEncoder(json.JSONEncoder):

  def default(self, o):
    return o.__dict__


def json_dumps(x):
  return json.dumps(x, indent=2, cls=_SimpleJsonEncoder)


def record_config(config, path):
  out = json_dumps(config)
  logging.info('Recording config to %s\n %s', path, out)
  gfile.makedirs(os.path.dirname(path))
  with gfile.GFile(path, 'w') as fh:
    fh.write(out)


def load_config(path):
  logging.info('Loading config from %s', path)
  with gfile.GFile(path) as fh:
    return json.loads(fh.read())


def save_model(model, output_dir):
  """Save Keras model weights and architecture as HDF5 file."""
  save_path = '%s/model.hdf5' % output_dir
  logging.info('Saving model to %s', save_path)
  model.save(save_path, include_optimizer=False)
  return save_path


def load_model(path):
  logging.info('Loading model from %s', path)
  return tf.keras.models.load_model(path)


def metrics_from_stats(stats):
  """Compute metrics to report to hyperparameter tuner."""
  labels, probs = stats['labels'], stats['probs']
  # Reshape binary predictions to 2-class.
  if len(probs.shape) == 1:
    probs = np.stack([1-probs, probs], axis=-1)
  assert len(probs.shape) == 2

  predictions = np.argmax(probs, axis=-1)
  accuracy = np.equal(labels, predictions)

  label_probs = probs[np.arange(len(labels)), labels]
  log_probs = np.maximum(-1e10, np.log(label_probs))
  brier_scores = np.square(probs).sum(-1) - 2 * label_probs

  return {'accuracy': accuracy.mean(0),
          'brier_score': brier_scores.mean(0),
          'log_prob': log_probs.mean(0)}


def make_predictions(
    model, batched_dataset, predictions_per_example=1, writers=None,
    predictions_are_logits=True, record_image_samples=True, max_batches=1e6):
  """Build a dictionary of predictions for examples from a dataset.

  Args:
    model: Trained Keras model.
    batched_dataset: tf.data.Dataset that yields batches of image, label pairs.
    predictions_per_example: Number of predictions to generate per example.
    writers: `dict` with keys 'small' and 'full', containing
      array_utils.StatsWriter instances for full prediction results and small
      prediction results (omitting logits).
    predictions_are_logits: Indicates whether model outputs are logits or
      probabilities.
    record_image_samples: `bool` Record one batch of input examples.
    max_batches: `int`, maximum number of batches.
  Returns:
    Dictionary containing:
      labels: Labels copied from the dataset (shape=[N]).
      logits_samples: Samples of model predict outputs for each example
          (shape=[N, M, K]).
      probs: Probabilities after averaging over samples (shape=[N, K]).
      image_samples: One batch of input images (for sanity checking).
  """
  if predictions_are_logits:
    samples_key = 'logits_samples'
    avg_probs_fn = lambda x: scipy.special.softmax(x, axis=-1).mean(-2)
  else:
    samples_key = 'probs_samples'
    avg_probs_fn = lambda x: x.mean(-2)

  labels, outputs = [], []
  predict_fn = model.predict if hasattr(model, 'predict') else model
  for i, (inputs_i, labels_i) in enumerate(tfds.as_numpy(batched_dataset)):
    logging.info('iteration: %d', i)
    outputs_i = np.stack(
        [predict_fn(inputs_i) for _ in range(predictions_per_example)], axis=1)

    if writers is None:
      labels.extend(labels_i)
      outputs.append(outputs_i)
    else:
      avg_probs_i = avg_probs_fn(outputs_i)
      prediction_batch = dict(labels=labels_i, probs=avg_probs_i)
      if i == 0 and record_image_samples:
        prediction_batch['image_samples'] = inputs_i

      writers['small'].write_batch(prediction_batch)
      prediction_batch[samples_key] = outputs_i
      writers['full'].write_batch(prediction_batch)

    # Don't predict whole ImageNet training set
    if i > max_batches:
      break

  if writers is None:
    image_samples = inputs_i  # pylint: disable=undefined-loop-variable
    labels = np.stack(labels, axis=0)
    outputs = np.concatenate(outputs, axis=0)

    stats = {'labels': labels, 'image_samples': image_samples,
             samples_key: outputs, 'probs': avg_probs_fn(outputs)}
    if record_image_samples:
      stats['image_samples'] = image_samples
    return stats


def download_dataset(dataset, batch_size_for_dl=1024):
  logging.info('Starting dataset download...')
  tup = list(zip(*tfds.as_numpy(dataset.batch(batch_size_for_dl))))
  logging.info('dataset download complete.')
  return tuple(np.concatenate(x, axis=0) for x in tup)
