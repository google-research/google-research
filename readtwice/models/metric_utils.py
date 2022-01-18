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

"""Utilities for training metrics."""

import tensorflow.compat.v1 as tf


def get_multi_doc_mask(block_ids):
  """Computes a mask on whether there are other blocks from the same doc."""
  # [batch_size]
  num_blocks_from_the_same_doc = tf.reduce_sum(
      tf.cast(
          tf.logical_and(
              tf.equal(
                  tf.expand_dims(block_ids, 1), tf.expand_dims(block_ids, 0)),
              tf.not_equal(tf.expand_dims(block_ids, 1), 0)),
          dtype=tf.int32),
      axis=1)

  block_has_other_blocks_from_the_same_doc = tf.cast(
      tf.greater(num_blocks_from_the_same_doc, 1), tf.float32)

  return block_has_other_blocks_from_the_same_doc


def masked_lm_metrics(
    mlm_loss_per_sample, mlm_accuracy_per_sample, mlm_weight_per_sample,
    block_ids, mlm_loss_per_entity_sample, mlm_accuracy_per_entity_sample,
    mlm_weight_per_entity_sample, mlm_loss_per_non_entity_sample,
    mlm_accuracy_per_non_entity_sample, mlm_weight_per_non_entity_sample,
    is_train, metrics_name):
  """Computes the loss and accuracy of the model."""

  assert "/" not in metrics_name

  def _add_metric(values, weights=None):
    """Adds a given metric to the metric dict."""
    assert values is not None
    if is_train:
      # Compute average metric over the current batch.
      if weights is not None:
        return tf.reduce_sum(values * weights) / (tf.reduce_sum(weights) + 1e-5)
      else:
        return tf.reduce_mean(values)
    else:
      # Convert to a streaming metric for eval.
      # tf.metrics would allow to compute a proper mean over the whole dataset.
      return tf.metrics.mean(values, weights)

  multi_blocks_mask = get_multi_doc_mask(block_ids)
  block_weights = {
      "": 1.0,
      "_multi_blocks": multi_blocks_mask,
      "_single_blocks": 1 - get_multi_doc_mask(block_ids),
  }
  metrics_dict = {}

  for suffix, weight in block_weights.items():
    if weight is not None:
      metrics_dict["pct" + suffix] = _add_metric(weight)

    metrics_dict["mlm_loss" + suffix] = _add_metric(
        mlm_loss_per_sample, weight * mlm_weight_per_sample)
    metrics_dict["mlm_accuracy" + suffix] = _add_metric(
        mlm_accuracy_per_sample, weight * mlm_weight_per_sample)

    if mlm_weight_per_entity_sample is not None:
      metrics_dict["mlm_loss_entity" + suffix] = _add_metric(
          mlm_loss_per_entity_sample, weight * mlm_weight_per_entity_sample)
      metrics_dict["mlm_accuracy_entity" + suffix] = _add_metric(
          mlm_accuracy_per_entity_sample, weight * mlm_weight_per_entity_sample)

    if mlm_weight_per_non_entity_sample is not None:
      metrics_dict["mlm_loss_non_entity" + suffix] = _add_metric(
          mlm_loss_per_non_entity_sample,
          weight * mlm_weight_per_non_entity_sample)
      metrics_dict["mlm_accuracy_non_entity" + suffix] = _add_metric(
          mlm_accuracy_per_non_entity_sample,
          weight * mlm_weight_per_non_entity_sample)

  return {
      metrics_name + "/" + name: value
      for (name, value) in metrics_dict.items()
  }
