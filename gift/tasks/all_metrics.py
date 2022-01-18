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

"""Mapping of all defined metrics.

 metric name --> metric function.
"""

from flax.training import common_utils
from gift.tasks import metrics

# TODO(samiraabnar): Refactor this.
CLASSIFICATION_METRICS = {
    'accuracy': metrics.weighted_accuracy,
    'top1_accuracy': metrics.weighted_top_one_accuracy,
    'loss': metrics.weighted_cross_entropy_loss,
    'entropy': metrics.mean_logits_entropy,
    'confidence': metrics.mean_confidence,
}

BINARY_CLASSIFICATION_METRICS = {
    'accuracy': metrics.weighted_binary_accuracy,
    'loss': metrics.weighted_sigmoid_cross_entropy_loss,
}

MULTI_ENV_CLASSIFICATION_METRICS = {'irm_penalty': metrics.irm_env_penalty}
MULTI_ENV_CLASSIFICATION_METRICS.update(CLASSIFICATION_METRICS)

MULTI_ENV_BINARY_CLASSIFICATION_METRICS = {
    'irm_penalty': metrics.binary_irm_env_penalty
}
MULTI_ENV_BINARY_CLASSIFICATION_METRICS.update(BINARY_CLASSIFICATION_METRICS)

ALL_LOSSES = {
    'categorical_cross_entropy': metrics.categorical_cross_entropy_loss,
    'sigmoid_cross_entropy': metrics.sigmoid_cross_entropy_loss,
    'softmax_hinge': metrics.softmax_hinge_loss,
    'sigmoid_hinge': metrics.sigmoid_hinge_loss,
}


def classification_metrics_function(logits, batch, target_is_onehot=False):
  """Calculates metrics for the classification task.

  Args:
   logits: float array; Output of model in shape [batch, length, num_classes].
   batch: dict; Batch of data that has 'label' and optionally 'weights'.
   target_is_onehot: bool; If the target is a one-hot vector.

  Returns:
    a dict of metrics.
  """
  if target_is_onehot:
    one_hot_targets = batch['label']
  else:
    one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])
  weights = batch.get('weights')  # weights might not be defined
  metrics_dic = {
      key: CLASSIFICATION_METRICS[key](logits, one_hot_targets, weights)
      for key in CLASSIFICATION_METRICS
  }
  return metrics_dic
