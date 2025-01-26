# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""numpy implementations of prediction quality metrics.

Partly adapted from
https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/uncertainty/uncertainty.py
"""

import numpy as onp


def _flatten_batch_axes(arr):
  pred_dim = arr.shape[-1]
  batch_dim = arr.size // pred_dim
  return arr.reshape((batch_dim, pred_dim))


def _flatten_outputs_labels(outputs, labels):
  return _flatten_batch_axes(outputs), labels.reshape(-1)


def accuracy(outputs, labels):
  """Negative log-likelihood."""
  outputs, labels = _flatten_outputs_labels(outputs, labels)
  labels = labels.astype(int)
  preds = onp.argmax(outputs, axis=1)
  return (preds == labels).mean()


def nll(outputs, labels, normalized=True):
  """Negative log-likelihood."""
  outputs, labels = _flatten_outputs_labels(outputs, labels)
  labels = labels.astype(int)
  idx = (onp.arange(labels.size), labels)
  log_ps = onp.log(outputs[idx])
  if normalized:
    return -log_ps.mean()
  else:
    return -log_ps.sum()


def calibration_curve(outputs, labels, num_bins=20, equispaced=False):
  """Compute calibration curve and ECE."""
  outputs, labels = _flatten_outputs_labels(outputs, labels)
  predictions = onp.argmax(outputs, 1)
  confidences = onp.max(outputs, 1)
  num_inputs = confidences.shape[0]
  if equispaced:
    # Estimate ECE with confidence-equispaced bins as Guo et al., 2017
    bins = onp.linspace(0., 1., num=1+num_bins)
  else:
    step = (num_inputs + num_bins - 1) // num_bins
    bins = onp.sort(confidences)[::step]
    if num_inputs % step != 1:
      bins = onp.concatenate((bins, [onp.max(confidences)]))
  bin_lowers = bins[:-1]
  bin_uppers = bins[1:]

  accuracies = (predictions == labels)

  bin_confidences = []
  bin_accuracies = []
  bin_proportions = []
  bin_samples = []

  ece = 0.0
  for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    in_bin = (confidences > bin_lower) * (confidences < bin_upper)
    prop_in_bin = in_bin.mean()
    samples_in_bin = in_bin.sum()
    if prop_in_bin > 0:
      accuracy_in_bin = accuracies[in_bin].mean()
      avg_confidence_in_bin = confidences[in_bin].mean()
      ece += onp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
      bin_confidences.append(avg_confidence_in_bin)
      bin_accuracies.append(accuracy_in_bin)
      bin_proportions.append(prop_in_bin)
      bin_samples.append(samples_in_bin)

  bin_confidences, bin_accuracies, bin_proportions, bin_samples = map(
      lambda lst: onp.array(lst),
      (bin_confidences, bin_accuracies, bin_proportions, bin_samples))

  bins[0] = 0.
  bins[-1] = 1.
  return {
      # "num_inputs": num_inputs,
      # "bins": bins,
      "confidence": bin_confidences,
      "accuracy": bin_accuracies,
      "proportions": bin_proportions,
      # "samples": bin_samples,  # Num samples in bins
      "ece": ece,
  }


def mse(predictions, targets, y_scale=1.):
  mus, sigmas = onp.split(predictions, [1], axis=-1)
  assert mus.shape == targets.shape, (
      "Predictions and targets should have the same shape, "
      "got {} and {}".format(mus.shape, targets.shape))
  return ((mus - targets)**2).mean() * y_scale**2


def rmse(predictions, targets, y_scale=1.):
  return onp.sqrt(mse(predictions, targets, y_scale))


def regression_nll(predictions, targets, y_scale=1.):
  #ToDo: check
  mus, sigmas = onp.split(predictions, [1], axis=-1)
  se = (mus - targets)**2
  nll = 0.5 * (se / sigmas**2 + onp.log(2 * onp.pi * sigmas**2)).mean()
  nll += onp.log(y_scale)
  return nll

def summarize_1d_regr(predictions, targets, y_scale=1.):
  pred_means, pred_stds = onp.split(predictions, [1], axis=-1)
  return ",".join([f"{mu:.1f}+-{std:.1f}" for mu, std in zip(pred_means[0][:, 0], pred_stds[0][:, 0])])
