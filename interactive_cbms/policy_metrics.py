# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Metric computation utilities for use in intervention policies."""

import functools
from typing import Any, List, Tuple

import numpy as np
import sklearn.metrics
import tensorflow as tf
import tensorflow_probability as tfp

from interactive_cbms import enum_utils

tfk = tf.keras
tfd = tfp.distributions


def row_mean_rec_rank(y_true, y_pred):
  """The Mean Reciprocal Rank metric.

  Args:
    y_true: Ground truth class labels with shape=(None,) and dtype=tf.int64.
    y_pred: Predicted class logits or probabilities with shape=(None, n_classes)
      and dtype=tf.float32.

  Returns:
    An array containing the reciprocal rank of the true class for each test
    instance.
  """
  assert tf.rank(y_true) == 1 and tf.rank(y_pred) == 2
  ranked_classes = tf.cast(
      tf.argsort(y_pred, axis=1, direction='DESCENDING'), dtype=tf.int64)
  mean_rec_rank = 1 / (tf.where(ranked_classes == y_true[:, None])[:, 1] + 1)
  return mean_rec_rank


def cat_auc(y_true, y_pred):
  """The categorical AUC metric.

  Args:
    y_true: Ground truth class labels with shape=(None,) and dtype=tf.int64.
    y_pred: Predicted class logits with shape=(None, n_classes)
      and dtype=tf.float32.

  Returns:
    The one vs. rest categorical AUC score.
  """
  y_pred_exp = np.exp(y_pred)
  y_probs = y_pred_exp/y_pred_exp.sum(axis=1, keepdims=True)
  return sklearn.metrics.roc_auc_score(y_true, y_probs, multi_class='ovr')


def label_metric_utils(
    # Using Any for type annotation instead of
    # intervened_dataset.IntervenedDataset to avoid circular imports.
    intervened_ds, concept_name_to_intervene,
    curr_intervention_mask
):
  """Common utilities for computing concept importance metrics.



  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.
    curr_intervention_mask: Intervention mask being used for the current
      intervention step.

  Returns:
    concept_probs: The mean predicted probability of the concept group, for each
      test example.
    dist: The predicted class label distribution using <curr_intervention_mask>.
    dist_0: The predicted class label distribution using an intervention mask
      that also reveals <concept_name_to_intervene> concept group assuiming 0 to
      be the true target for all binary concepts in the group.
    dist_1: The predicted class label distribution using an intervention mask
      that also reveals <concept_name_to_intervene> concept group assuiming 1 to
      be the true target for all binary concepts in the group.
  """
  b_concepts_to_reveal = intervened_ds.concept_groups[concept_name_to_intervene]
  concept_probs = tf.reduce_mean(
      tf.sigmoid(
          tf.gather(intervened_ds.pred_concepts, indices=b_concepts_to_reveal,
                    axis=1)),
      axis=1)
  b_concepts_to_reveal_multihot = tf.cast(
      tf.reduce_sum(
          tf.one_hot(b_concepts_to_reveal, intervened_ds.n_concepts), axis=0),
      dtype=tf.bool)
  mask_temp = tf.where(
      b_concepts_to_reveal_multihot, True, curr_intervention_mask)
  true_concepts_0 = tf.where(
      b_concepts_to_reveal_multihot, 0, intervened_ds.true_concepts)
  true_concepts_1 = tf.where(
      b_concepts_to_reveal_multihot, 1, intervened_ds.true_concepts)
  intervened_concepts = intervened_ds.intervene_on_batch(
      intervened_ds.true_concepts, intervened_ds.concept_uncertainty,
      intervened_ds.pred_concepts, curr_intervention_mask)
  intervened_concepts_0 = intervened_ds.intervene_on_batch(
      true_concepts_0, intervened_ds.concept_uncertainty,
      intervened_ds.pred_concepts, mask_temp)
  intervened_concepts_1 = intervened_ds.intervene_on_batch(
      true_concepts_1, intervened_ds.concept_uncertainty,
      intervened_ds.pred_concepts, mask_temp)
  if intervened_ds.n_classes == 1:
    dist_class = tfd.Bernoulli
  else:
    dist_class = tfd.Categorical
  dist = dist_class(
      logits=tf.squeeze(
          intervened_ds.ctoy_model(intervened_concepts, training=False)[0]))
  dist_0 = dist_class(
      logits=tf.squeeze(
          intervened_ds.ctoy_model(intervened_concepts_0, training=False)[0]))
  dist_1 = dist_class(
      logits=tf.squeeze(
          intervened_ds.ctoy_model(intervened_concepts_1, training=False)[0]))
  return concept_probs, dist, dist_0, dist_1


def label_metric_utilsv2(
    intervened_ds, concept_name_to_intervene,
    curr_intervention_mask
):
  """Common utilities for computing concept importance metrics.

  This v2 method deals with categorical concepts differently by instead
  returning two lists of predicted label distributions - dists_0 and dists_1,
  each of length equal to the no. of binary concepts in the
  <concept_name_to_intervene> concept group. Each label distribution in these
  lists corresponds to predictions of the ctoy_model after revealing *only* the
  respective binary concept assuming 0 (for dists_0) or 1 (for dists_1) to be
  its true value. In comparison, the original method reveals *all* binary
  concepts in the <concept_name_to_intervene> concept group at once.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.
    curr_intervention_mask: Intervention mask being used for the current
      intervention step.

  Returns:
    concept_probs: The predicted probability of all binary concepts in the
      concept group, for each test example.
    dist: The predicted class label distribution using <curr_intervention_mask>.
    dists_0: A list containing the predicted class label distribution for each
      binary concept in the <concept_name_to_intervene> concept group. The
      distributions are computed using intervention masks that also reveal the
      respective binary concept in the concept group assuming 0 to be the true
      target for the binary concept.
    dists_1: A list containing the predicted class label distribution for each
      binary concept in the <concept_name_to_intervene> concept group. The
      distributions are computed using intervention masks that also reveal the
      respective binary concept in the concept group assuming 1 to be the true
      target for the binary concept.
  """
  if intervened_ds.n_classes == 1:
    dist_class = tfd.Bernoulli
  else:
    dist_class = tfd.Categorical
  b_concepts_to_reveal = intervened_ds.concept_groups[concept_name_to_intervene]
  concept_probs = tf.sigmoid(tf.gather(
      intervened_ds.pred_concepts, indices=b_concepts_to_reveal, axis=1))
  b_concepts_to_reveal_multihot = tf.cast(
      tf.reduce_sum(
          tf.one_hot(b_concepts_to_reveal, intervened_ds.n_concepts), axis=0),
      dtype=tf.bool)
  true_concepts_0 = tf.where(
      b_concepts_to_reveal_multihot, 0, intervened_ds.true_concepts)
  true_concepts_1 = tf.where(
      b_concepts_to_reveal_multihot, 1, intervened_ds.true_concepts)
  intervened_concepts = intervened_ds.intervene_on_batch(
      intervened_ds.true_concepts, intervened_ds.concept_uncertainty,
      intervened_ds.pred_concepts, curr_intervention_mask)
  dist = dist_class(
      logits=tf.squeeze(
          intervened_ds.ctoy_model(intervened_concepts, training=False)[0]))
  dists_0 = []
  dists_1 = []
  for concept_idx in b_concepts_to_reveal:
    b_concepts_to_reveal_onehot = tf.cast(
        tf.one_hot(concept_idx, intervened_ds.n_concepts), dtype=tf.bool)
    mask_temp = tf.where(b_concepts_to_reveal_onehot, True,
                         curr_intervention_mask)
    intervened_concepts_0 = intervened_ds.intervene_on_batch(
        true_concepts_0, intervened_ds.concept_uncertainty,
        intervened_ds.pred_concepts, mask_temp)
    intervened_concepts_1 = intervened_ds.intervene_on_batch(
        true_concepts_1, intervened_ds.concept_uncertainty,
        intervened_ds.pred_concepts, mask_temp)
    dist_0 = dist_class(
        logits=tf.squeeze(
            intervened_ds.ctoy_model(intervened_concepts_0, training=False)[0]))
    dist_1 = dist_class(
        logits=tf.squeeze(
            intervened_ds.ctoy_model(intervened_concepts_1, training=False)[0]))
    dists_0.append(dist_0)
    dists_1.append(dist_1)

  return concept_probs, dist, dists_0, dists_1


def label_entropy_change(
    intervened_ds, concept_name_to_intervene,
    curr_intervention_mask, signed):
  """Metric that measures the importance of a concept using the expected change in label distribution entropy.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.
    curr_intervention_mask: Intervention mask being used for the current
      intervention step.
    signed: Whether to use the signed difference for entropy to measure change.

  Returns:
    The expected change in label distribution entropy.
  """
  concept_probs, dist, dist_0, dist_1 = label_metric_utils(
      intervened_ds, concept_name_to_intervene, curr_intervention_mask)

  entropy = dist.entropy()
  entropy_0 = dist_0.entropy()
  entropy_1 = dist_1.entropy()
  change_0 = entropy - entropy_0
  change_1 = entropy - entropy_1

  if signed:
    metric = concept_probs * change_1 + (1 - concept_probs) * change_0
  else:
    metric = concept_probs * tf.abs(change_1) + (
        1 - concept_probs) * tf.abs(change_0)

  return metric


def label_entropy_changev2(
    intervened_ds, concept_name_to_intervene,
    curr_intervention_mask, signed):
  """Metric that measures the importance of a concept using the expected change in label distribution entropy.

  This v2 method deals with categorical concepts differently by taking
  expectation under the distribution for each predicted binary concept in the
  <concept_name_to_intervene> concept group separately and then summing the
  expectations.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.
    curr_intervention_mask: Intervention mask being used for the current
      intervention step.
    signed: Whether to use the signed difference for entropy to measure change.

  Returns:
    The expected change in label distribution entropy.
  """
  concept_probs, dist, dists_0, dists_1 = label_metric_utilsv2(
      intervened_ds, concept_name_to_intervene, curr_intervention_mask)

  entropy = dist.entropy()[:, None]
  entropies_0 = tf.stack([dist_0.entropy().numpy() for dist_0 in dists_0],
                         axis=1)
  entropies_1 = tf.stack([dist_1.entropy().numpy() for dist_1 in dists_1],
                         axis=1)
  changes_0 = entropies_0 - entropy
  changes_1 = entropies_1 - entropy

  if signed:
    metric = tf.reduce_sum(
        concept_probs * changes_1 + (1 - concept_probs) * changes_0, axis=1)
  else:
    metric = tf.reduce_sum(
        (concept_probs * tf.abs(changes_1) +
         (1 - concept_probs) * tf.abs(changes_0)),
        axis=1)

  return metric


def label_confidence_change(
    intervened_ds, concept_name_to_intervene,
    curr_intervention_mask, signed):
  """Metric that measures the importance of a concept using the expected change in the predicted label confidence.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.
    curr_intervention_mask: Intervention mask being used for the current
      intervention step.
    signed: Whether to use the signed difference for entropy to measure change.

  Returns:
    The expected change in prediced label confidence.
  """
  concept_probs, dist, dist_0, dist_1 = label_metric_utils(
      intervened_ds, concept_name_to_intervene, curr_intervention_mask)

  if intervened_ds.n_classes == 1:
    pred_class = tf.cast(dist.logits > 0, tf.float32)
  else:
    pred_class = tf.argmax(dist.logits, axis=1)
  pred_class_prob = dist.prob(pred_class)
  pred_class_prob_0 = dist_0.prob(pred_class)
  pred_class_prob_1 = dist_1.prob(pred_class)
  change_0 = pred_class_prob_0 - pred_class_prob
  change_1 = pred_class_prob_1 - pred_class_prob

  if signed:
    metric = concept_probs * change_1 + (1 - concept_probs) * change_0
  else:
    metric = (concept_probs * tf.abs(change_1)
              + (1 - concept_probs) * tf.abs(change_0))

  return metric


def label_confidence_changev2(
    intervened_ds, concept_name_to_intervene,
    curr_intervention_mask, signed):
  """Metric that measures the importance of a concept using the expected change in the predicted label confidence.

  This v2 method deals with categorical concepts differently by taking
  expectation under the distribution each predicted binary concept in the
  <concept_name_to_intervene> concept group separately and then summing the
  expectations.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.
    curr_intervention_mask: Intervention mask being used for the current
      intervention step.
    signed: Whether to use the signed difference for entropy to measure change.

  Returns:
    The expected change in prediced label confidence.
  """
  concept_probs, dist, dists_0, dists_1 = label_metric_utilsv2(
      intervened_ds, concept_name_to_intervene, curr_intervention_mask)

  if intervened_ds.n_classes == 1:
    pred_class = tf.cast(dist.logits > 0, tf.float32)
  else:
    pred_class = tf.argmax(dist.logits, axis=1)
  pred_class_prob = dist.prob(pred_class)[:, None]
  pred_class_prob_0 = tf.stack([dist_0.prob(pred_class) for dist_0 in dists_0],
                               axis=1)
  pred_class_prob_1 = tf.stack([dist_1.prob(pred_class) for dist_1 in dists_1],
                               axis=1)
  changes_0 = pred_class_prob_0 - pred_class_prob
  changes_1 = pred_class_prob_1 - pred_class_prob

  if signed:
    metric = tf.reduce_sum(
        concept_probs * changes_1 + (1 - concept_probs) * changes_0, axis=1)
  else:
    metric = tf.reduce_sum(
        (concept_probs * tf.abs(changes_1) +
         (1 - concept_probs) * tf.abs(changes_0)),
        axis=1)
  return metric


def label_kld(
    intervened_ds, concept_name_to_intervene,
    curr_intervention_mask):
  """Metric that measures the importance of a concept using the expected change in the predicted label distibution.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.
    curr_intervention_mask: Intervention mask being used for the current
      intervention step.

  Returns:
    The expected change in the predicted label distibution.
  """
  concept_probs, dist, dist_0, dist_1 = label_metric_utils(
      intervened_ds, concept_name_to_intervene, curr_intervention_mask)
  kld_0 = tfd.kl_divergence(dist_0, dist) + tfd.kl_divergence(dist, dist_0)
  kld_1 = tfd.kl_divergence(dist_1, dist) + tfd.kl_divergence(dist, dist_1)

  return concept_probs * kld_1 + (1 - concept_probs) * kld_0


def concept_confidence(
    intervened_ds, concept_name_to_intervene):
  """Metric that measures the prediction uncertainty of a concept using the negative concept prediction confidence.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.

  Returns:
    Sum of the negative prediction confidence of each binary concept in the
    <concept_name_to_intervene> concept group.
  """
  b_concepts_to_reveal = intervened_ds.concept_groups[concept_name_to_intervene]
  concept_probs = tf.sigmoid(tf.gather(
      intervened_ds.pred_concepts, indices=b_concepts_to_reveal, axis=1))
  confidence = tf.where(concept_probs > 0.5, concept_probs, 1 - concept_probs)
  return -tf.reduce_sum(confidence, axis=1)


def concept_entropy(
    intervened_ds, concept_name_to_intervene):
  """Metric that measures the prediction uncertainty of a concept using the entropy of the predicted concept distribution.

  Args:
    intervened_ds: The IntervenedDataset object to use for metric computation.
    concept_name_to_intervene: The name of the concept group to intervene on in
      the next intervention step.

  Returns:
    Sum of the entropy of the predicted distributions of each binary concept in
    the <concept_name_to_intervene> concept group.
  """
  b_concepts_to_reveal = intervened_ds.concept_groups[concept_name_to_intervene]
  concept_logits = tf.gather(
      intervened_ds.pred_concepts, indices=b_concepts_to_reveal, axis=1)
  return tf.reduce_sum(
      tfd.Bernoulli(logits=concept_logits).entropy(), axis=1)


def get_metric_fn(
    metric, reduce_mean = False
):
  """Utility function to get metric functions.

  Args:
    metric: Name of the metric to get.
    reduce_mean: Whether to apply mean reduction to the metric.

  Returns:
    A function that computes the desired metric using y_true and y_pred.

  Raises:
    ValueError when the metric name is not recognized.
  """

  metric = enum_utils.Metric(metric)
  if metric is enum_utils.Metric.MEAN_REC_RANK:
    metric_fn = row_mean_rec_rank
  elif metric is enum_utils.Metric.AUC:
    if not reduce_mean:
      raise ValueError('reduce_mean cannot be False if metric=="auc"')
    metric_fn = sklearn.metrics.roc_auc_score
  elif metric is enum_utils.Metric.CAT_AUC:
    if not reduce_mean:
      raise ValueError('reduce_mean cannot be False if metric=="cat_auc"')
    metric_fn = cat_auc
  elif metric is enum_utils.Metric.BINARY_XENT:
    metric_fn = tfk.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  elif metric is enum_utils.Metric.CAT_XENT:
    metric_fn = tfk.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  elif metric is enum_utils.Metric.CONCEPT_ENTROPY:
    metric_fn = concept_entropy
  elif metric is enum_utils.Metric.CONCEPT_CONFIDENCE:
    metric_fn = concept_confidence
  elif metric is enum_utils.Metric.LABEL_ENTROPY_CHANGE:
    metric_fn = functools.partial(label_entropy_change, signed=False)
  elif metric is enum_utils.Metric.LABEL_ENTROPY_DECREASE:
    metric_fn = functools.partial(label_entropy_change, signed=True)
  elif metric is enum_utils.Metric.LABEL_CONFIDENCE_CHANGE:
    metric_fn = functools.partial(label_confidence_change, signed=False)
  elif metric is enum_utils.Metric.LABEL_CONFIDENCE_INCREASE:
    metric_fn = functools.partial(label_confidence_change, signed=True)
  elif metric is enum_utils.Metric.LABEL_KLD:
    metric_fn = label_kld
  elif metric is enum_utils.Metric.LABEL_ENTROPY_CHANGEV2:
    metric_fn = functools.partial(label_entropy_changev2, signed=False)
  elif metric is enum_utils.Metric.LABEL_ENTROPY_DECREASEV2:
    metric_fn = functools.partial(label_entropy_changev2, signed=True)
  elif metric is enum_utils.Metric.LABEL_CONFIDENCE_CHANGEV2:
    metric_fn = functools.partial(label_confidence_changev2, signed=False)
  elif metric is enum_utils.Metric.LABEL_CONFIDENCE_INCREASEV2:
    metric_fn = functools.partial(label_confidence_changev2, signed=True)
  else:
    raise ValueError(f'Metric: {metric} not recognized.')

  if reduce_mean:
    return lambda *args, **kwargs: tf.reduce_mean(metric_fn(*args, **kwargs))
  else:
    return metric_fn


def new_best_metric(curr_metric, best_metric,
                    mode):
  """Utility function to indicate whether the current value for a metric is the new best.

  Args:
    curr_metric: Value of the metric at the current interveniton step.
    best_metric: Best value of the metric obtained in the past intervention
      steps.
    mode: Whether to maximize or minimize the metric. Allowed values are "max"
      or "min".

  Returns:
    True if the current metric is the new best else False.

  Raises:
    ValueError when "mode" is not recognized.
  """
  if mode == 'min':
    return curr_metric < best_metric
  elif mode == 'max':
    return curr_metric > best_metric
  else:
    raise ValueError('"mode" not recognized.')
