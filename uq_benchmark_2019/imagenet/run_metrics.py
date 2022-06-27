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

"""Calculate accuracy and calibration metrics from model predictions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from uq_benchmark_2019 import array_utils
from uq_benchmark_2019 import metrics_lib

gfile = tf.io.gfile

FLAGS = flags.FLAGS

_ENSEMBLE_SIZE = 12
_ECE_BINS = 30
_MAX_PREDICTIONS = 50000

tfd = tfp.distributions

tf.enable_v2_behavior()


def _declare_flags():
  """Declare flags; not invoked when this module is imported as a library."""
  flags.DEFINE_string('prediction_path', None, 'Filepath to predictions.')
  flags.DEFINE_string('model_dir_ensemble', None, 'Pattern for ensemble dirs.')
  flags.DEFINE_boolean('use_temp_scaling', False, 'If True, apply temperature '
                       'scaling to logits.')
  flags.DEFINE_string('imagenet_validation_path', None, 'Filepath to '
                      'predictions on held-out validation set.')


def inverse_softmax(x):
  return np.log(x / x[Ellipsis, :1])


def split_prediction_path(path):
  path = os.path.normpath(path)
  return path.split(os.sep)


def get_ensemble_stats(model_dir_ensemble, prediction_path):
  """Average results in all parallel ensemble directories."""

  split_path = split_prediction_path(prediction_path)
  filname = split_path[-1]

  ensemble_dirs = gfile.glob(model_dir_ensemble)[:_ENSEMBLE_SIZE]

  probs = 0.
  count = 0
  for d in ensemble_dirs:
    # TODO(emilyaf) support replicas
    glob_path = os.path.join(d, 'r0/predictions/*/*', filname)
    paths = gfile.glob(glob_path)
    if not paths:
      continue
    count += 1

    # Use most recent prediction in ensemble
    filestats = gfile.stat(glob_path)
    idx = np.argmax([s.mtime_nsecs for s in filestats])
    path = paths[idx]

    stats = array_utils.load_stats_from_tfrecords(
        path, max_records=_MAX_PREDICTIONS)
    probs += stats['probs']
  probs /= count

  return probs.astype(np.float32), stats['labels'].astype(np.int32)


def run(prediction_path, validation_path, model_dir_ensemble,
        use_temp_scaling=False):
  """Runs predictions on the given dataset using the specified model."""

  logging.info('Loading predictions...')
  out_file_prefix = 'metrics_'
  if model_dir_ensemble:
    probs, labels = get_ensemble_stats(model_dir_ensemble, prediction_path)
    out_file_prefix = 'metrics_ensemble_'
  else:
    stats = array_utils.load_stats_from_tfrecords(
        prediction_path, max_records=_MAX_PREDICTIONS)
    probs = stats['probs'].astype(np.float32)
    labels = stats['labels'].astype(np.int32)

  if len(labels.shape) > 1:
    labels = np.squeeze(labels, -1)

  probs = metrics_lib.soften_probabilities(probs=probs)
  logits = inverse_softmax(probs)

  if use_temp_scaling:
    predictions_base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(prediction_path)))
    json_dir = os.path.join(predictions_base_dir, '*/*/temperature_hparam.json')

    paths = gfile.glob(json_dir)
    filestats = gfile.stat(json_dir)
    idx = np.argmax([s.mtime_nsecs for s in filestats])
    temperature_hparam_path = paths[idx]
    with gfile.GFile(temperature_hparam_path) as fh:
      temp = json.loads(fh.read())['temperature']

    logits /= temp
    probs = tf.nn.softmax(logits).numpy()
    probs = metrics_lib.soften_probabilities(probs=probs)

    out_file_prefix = 'metrics_temp_scaled_'

  # confidence vs accuracy
  thresholds = np.linspace(0, 1, 10, endpoint=False)
  accuracies, counts = metrics_lib.compute_accuracies_at_confidences(
      labels, probs, thresholds)

  overall_accuracy = (probs.argmax(-1) == labels).mean()
  accuracy_top5 = metrics_lib.accuracy_top_k(probs, labels, 5)
  accuracy_top10 = metrics_lib.accuracy_top_k(probs, labels, 10)

  probs_correct_class = probs[np.arange(probs.shape[0]), labels]
  negative_log_likelihood = -np.log(probs_correct_class).mean()
  entropy_per_example = tfd.Categorical(probs=probs).entropy().numpy()

  uncertainty, resolution, reliability = metrics_lib.brier_decomposition(
      labels=labels, logits=logits)
  brier = tf.reduce_mean(metrics_lib.brier_scores(labels=labels, logits=logits))

  ece = metrics_lib.expected_calibration_error_multiclass(
      probs, labels, _ECE_BINS)
  ece_top5 = metrics_lib.expected_calibration_error_multiclass(
      probs, labels, _ECE_BINS, top_k=5)
  ece_top10 = metrics_lib.expected_calibration_error_multiclass(
      probs, labels, _ECE_BINS, top_k=10)

  validation_stats = array_utils.load_stats_from_tfrecords(
      validation_path, max_records=20000)
  validation_probs = validation_stats['probs'].astype(np.float32)

  bins = metrics_lib.get_quantile_bins(_ECE_BINS, validation_probs)
  q_ece = metrics_lib.expected_calibration_error_multiclass(
      probs, labels, bins)

  bins = metrics_lib.get_quantile_bins(_ECE_BINS, validation_probs, top_k=5)
  q_ece_top5 = metrics_lib.expected_calibration_error_multiclass(
      probs, labels, bins, top_k=5)

  bins = metrics_lib.get_quantile_bins(_ECE_BINS, validation_probs, top_k=10)
  q_ece_top10 = metrics_lib.expected_calibration_error_multiclass(
      probs, labels, bins, top_k=10)

  metrics = {
      'accuracy': overall_accuracy,
      'accuracy_top5': accuracy_top5,
      'accuracy_top10': accuracy_top10,
      'accuracy_at_confidence': accuracies,
      'confidence_thresholds': thresholds,
      'confidence_counts': counts,
      'ece': ece,
      'ece_top5': ece_top5,
      'ece_top10': ece_top10,
      'q_ece': q_ece,
      'q_ece_top5': q_ece_top5,
      'q_ece_top10': q_ece_top10,
      'ece_nbins': _ECE_BINS,
      'entropy_per_example': entropy_per_example,
      'brier_uncertainty': uncertainty.numpy(),
      'brier_resolution': resolution.numpy(),
      'brier_reliability': reliability.numpy(),
      'brier_score': brier.numpy(),
      'true_labels': labels,
      'pred_labels': probs.argmax(-1),
      'prob_true_label': probs[np.arange(len(labels)), labels],
      'prob_pred_label': probs.max(-1),
      'negative_log_likelihood': negative_log_likelihood,
  }

  save_dir = os.path.dirname(prediction_path)
  split_path = split_prediction_path(prediction_path)
  prediction_file = split_path[-1]
  dataset_name = '-'.join(prediction_file.split('_')[2:])

  out_file = out_file_prefix + dataset_name + '.npz'
  array_utils.write_npz(save_dir, out_file, metrics)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run(FLAGS.prediction_path, FLAGS.validation_path, FLAGS.model_dir_ensemble,
      FLAGS.use_temp_scaling)

if __name__ == '__main__':
  _declare_flags()
  app.run(main)
