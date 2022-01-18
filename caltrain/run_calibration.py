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

"""Compute sample estimate of bias."""
import copy
import json
import os

import numpy as np
import tensorflow as tf

import caltrain as caltrain
import caltrain.calibration_metrics as calibration_metrics
from caltrain.glm_modeling import get_datasets
import caltrain.simulation.logistic as logistic
import caltrain.simulation.polynomial as polynomial
import caltrain.utils as utils


def get_true_dataset(config):
  """Return true dataset given config."""
  dataset = config['dataset']
  assert dataset in caltrain.TRUE_DATASETS

  a = config['a']
  b = config['b']
  alpha = config['alpha']
  beta = config['beta']
  d = config['d']
  num_samples = config['num_samples']

  if dataset == 'logistic':
    true_dataset = logistic.TrueLogisticUniform(a=a, b=b, n_samples=num_samples)
  elif dataset == 'logistic_beta':
    true_dataset = logistic.TrueLogisticBeta(
        a=a, b=b, alpha=alpha, beta=beta, n_samples=num_samples)
  elif dataset == 'logistic_log_odds':
    true_dataset = logistic.TrueLogisticLogOdds(
        a=a, b=b, alpha=alpha, beta=beta, n_samples=num_samples)
  elif dataset == 'polynomial':
    true_dataset = polynomial.TruePolynomial(
        alpha=a, beta=b, d=d, n_samples=num_samples)
  elif dataset == 'flip_polynomial':
    true_dataset = polynomial.TrueFlipPolynomial(
        alpha=a, beta=b, d=d, n_samples=num_samples)
  elif dataset == 'two_param_polynomial':
    true_dataset = polynomial.TrueTwoParamPolynomial(
        alpha=alpha, beta=beta, a=a, b=b, n_samples=num_samples)
  elif dataset == 'two_param_flip_polynomial':
    true_dataset = polynomial.TrueTwoParamFlipPolynomial(
        alpha=alpha, beta=beta, a=a, b=b, n_samples=num_samples)
  elif dataset == 'logistic_two_param_flip_polynomial':
    true_dataset = logistic.TrueLogisticTwoParamFlipPolynomial(
        alpha=alpha, beta=beta, a=a, b=b, n_samples=num_samples)
  else:
    raise NotImplementedError

  return true_dataset


def estimate_ece(config, data_dir):
  """Compute sample estimate of bias and variance."""

  config = copy.deepcopy(config)
  true_dataset = get_true_dataset(config)
  true_calibration_error = 100 * true_dataset.true_calib_error()

  calibration_results_cache_file = os.path.join(data_dir,
                                                'calibration_results.json')
  with tf.io.gfile.GFile(calibration_results_cache_file, 'r') as f:
    result = json.load(f)

  hash_key = utils.get_hash_key(config)
  if hash_key in result:
    print(f'{hash_key} already computed, loading cached result.')
    return result[hash_key]['bias'], result[hash_key]['var'], result[hash_key][
        'mse']

  saved_ece = []
  for _ in range(config['num_reps']):
    ce = calibrate(config, true_dataset)
    saved_ece.append(ce)
  config['bias'] = np.mean(saved_ece - true_calibration_error)
  config['mse'] = np.mean(np.square(saved_ece - true_calibration_error))
  config['var'] = np.sqrt(np.var(saved_ece))
  config['true_ce'] = true_calibration_error

  print(hash_key)
  result[hash_key] = config
  with tf.io.gfile.GFile(calibration_results_cache_file, 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

  return config['bias'], config['var'], config['mse']


def calibrate(config, true_dataset=None):
  """Compute estimated calibration error."""
  dataset = config['dataset']
  ce_type = config['ce_type']
  num_bins = config['num_bins']
  bin_method = config['bin_method']
  norm = config['norm']
  num_samples = config['num_samples']
  datasets = get_datasets(None)

  if dataset in caltrain.TRUE_DATASETS:
    scores, _, raw_labels = true_dataset.dataset()
    scores = scores.reshape((num_samples, 1))
    raw_labels = raw_labels.reshape((num_samples, 1))
  elif dataset in {key for key, _ in datasets.items()}:
    scores, _, raw_labels = true_dataset.dataset()
    scores = scores.reshape((num_samples, 1))
    raw_labels = raw_labels.reshape((num_samples, 1))
  else:
    raise NotImplementedError

  cm = calibration_metrics.CalibrationMetric(ce_type, num_bins, bin_method,
                                             norm)
  ce = cm.compute_error(scores, raw_labels)

  return 100 * ce
