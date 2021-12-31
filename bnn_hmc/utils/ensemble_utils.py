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

import numpy as onp


def running_average(old_avg_val, new_val, n_avg):
  new_avg_val = old_avg_val + (new_val - old_avg_val) / (n_avg + 1)
  return new_avg_val


def compute_updated_ensemble_predictions_classification(
    ensemble_predicted_probs, num_ensembled, new_predicted_probs):
  """Update ensemble predictive categorical distribution."""
  #ToDo: test
  if num_ensembled:
    new_ensemble_predicted_probs = running_average(ensemble_predicted_probs,
                                                   new_predicted_probs,
                                                   num_ensembled)
  else:
    new_ensemble_predicted_probs = new_predicted_probs
  return new_ensemble_predicted_probs


def compute_updated_ensemble_predictions_regression(ensemble_predictions,
                                                    num_ensembled,
                                                    new_predictions):
  """Update ensemble predictive distribution assuming Gaussian likelihood."""
  mus, sigmas = onp.split(new_predictions, [1], axis=-1)

  if num_ensembled:
    old_mus, old_sigmas = onp.split(ensemble_predictions, [1], axis=-1)
    new_mus = running_average(old_mus, mus, num_ensembled)
    old_sigmas_corrected = old_sigmas**2 + old_mus**2 - new_mus**2
    new_sigmas = onp.sqrt(
        running_average(old_sigmas_corrected, sigmas**2 + mus**2 - new_mus**2,
                        num_ensembled))
    new_ensemble_predictions = onp.concatenate([new_mus, new_sigmas], axis=-1)
  else:
    new_ensemble_predictions = new_predictions
  return new_ensemble_predictions
