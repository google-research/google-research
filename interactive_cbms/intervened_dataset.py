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

"""The IntervenedDataset class."""

import random
from typing import Any, Dict, Generator, List, Optional, Tuple
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from interactive_cbms import enum_utils
from interactive_cbms import policy_metrics

tfk = tf.keras
tfd = tfp.distributions

_MAX_TEST_SIZE = 1000000


class IntervenedDataset:
  """Implements intervention policies and enables intervention on an existing tf.data.Dataset object."""

  def __init__(self, policy,
               intervention_format,
               include_uncertain, steps, xtoc_model,
               ctoy_model, policy_dataset,
               concept_groups,
               **policy_kwargs):
    """Initialises an IntervenedDataset object.

    Args:
      policy: Name of the policy to use for intervention.
      intervention_format: Concept representation format to use for
        intervention.
      include_uncertain: Whether to intervene on concepts that are labelled as
        uncertain/not visible in the image. If include_uncertain=False, the true
        values of uncertain concepts are not revealed in the intervened
        batch/dataset even when the intervention policy requests them.
      steps: Nunber of intervention steps.
      xtoc_model: X->C tfk.Model instance.
      ctoy_model: C->Y tfk.Model instance.
      policy_dataset: A validation or test dataset that an intervention policy
        can use to determine the next best concept to request true labels for.
      concept_groups: A dictionary that maps a concept group name to the list of
        concept indices that belong to that group. This is used from group-based
        intervention.
      **policy_kwargs: Keyword arguments to pass on to the intervention policy.
    """
    self.include_uncertain = include_uncertain
    self.steps = min(steps, len(concept_groups))
    self.xtoc_model = xtoc_model
    self.ctoy_model = ctoy_model
    if isinstance(ctoy_model.layers[-1], tf.keras.Sequential):
      self.n_classes = ctoy_model.layers[-1].layers[-1].units
    else:
      self.n_classes = ctoy_model.layers[-1].units
    self.policy_dataset = policy_dataset
    self.true_concepts, self.true_labels, self.concept_uncertainty = (
        policy_dataset.unbatch().batch(_MAX_TEST_SIZE).get_single_element()[1:])
    self.n_test = self.true_concepts.shape[0]
    self.concept_groups = concept_groups
    self.n_concepts = sum(map(len, concept_groups.values()))
    self.concept_group_names = list(self.concept_groups.keys())
    self.pred_concepts = xtoc_model.predict(policy_dataset, verbose=0)[0]
    self.logits_min = np.percentile(self.pred_concepts, 5)
    self.logits_max = np.percentile(self.pred_concepts, 95)
    self.probs_min = np.percentile(tf.sigmoid(self.pred_concepts), 5)
    self.probs_max = np.percentile(tf.sigmoid(self.pred_concepts), 95)
    self.policy_type = policy.split('_')[0]

    if self.n_concepts != self.true_concepts.shape[1]:
      raise ValueError(
          (f'No. of concepts in concept_groups (={self.n_concepts}) does not '
           'match the no. of concepts in the provided policy_dataset'
           f' (={self.true_concepts.shape[1]})'))
    if self.n_concepts != self.xtoc_model.layers[-1].units:
      raise ValueError(
          (f'No. of concepts in concept_groups (={self.n_concepts}) does not '
           'match the no. of predicted concepts in the provided xtoc_model'
           f' (={self.xtoc_model.layers[-1].units})'))
    if self.n_concepts != self.ctoy_model.layers[0].weights[0].shape[0]:
      raise ValueError(
          (f'No. of concepts in concept_groups (={self.n_concepts}) does not '
           'match the no. of expected concepts in the provided ctoy_model'
           f' (={self.ctoy_model.layers[0].weights[0].shape[0]})'))

    if policy is enum_utils.InterventionPolicy.GLOBAL_RANDOM:
      self.policy = self.global_random_policy(**policy_kwargs)
    elif policy is enum_utils.InterventionPolicy.INSTANCE_RANDOM:
      self.policy = self.instance_random_policy(**policy_kwargs)
    elif policy is enum_utils.InterventionPolicy.GLOBAL_GREEDY:
      self.policy = self.global_greedy_policy(**policy_kwargs)
    elif policy is enum_utils.InterventionPolicy.INSTANCE_GREEDY:
      self.policy = self.instance_greedy_policy(**policy_kwargs)
    elif policy is enum_utils.InterventionPolicy.COOP:
      self.policy = self.coop_policy(**policy_kwargs)
    else:
      raise ValueError(f'Policy: {policy} not recognized')

    self.format = intervention_format

  def intervene_on_batch(self, true_concepts,
                         concept_uncertainty,
                         pred_concepts,
                         intervention_mask):
    """Performs intervention on concepts for a single batch of data.

    Args:
      true_concepts: Batch of true concepts.
      concept_uncertainty: Batch of concept annotation uncertainty scores (for
        the CUB dataset).
      pred_concepts: Batch of concept predictions (logits) obtained from the
        XtoC model.
      intervention_mask: Boolean mask to use for intervention.

    Returns:
      intervened_concepts: The intervened concept batch
    """

    true_concepts = tf.cast(true_concepts, tf.float32)
    intervention_mask = tf.cast(intervention_mask, tf.float32)
    if not self.include_uncertain:
      # To prevent leaking information about invisible concepts during
      # intervention, CBM authors change the true labels for invisible concepts
      # to 0 and perform intervention as usual. This will force set the
      # invisible concept predictions to 0 upon intervention:
      # true_concepts = tf.where(concept_uncertainty != 1, true_concepts, 0.)

      # Instead, we can choose not to intervene on invivsible concepts even when
      # explicitly requested by the intervention policy. This method will still
      # retain the predictions of the XtoC model for the invisible concepts upon
      # intervention, which seems to be more reasonable behaviour:
      intervention_mask = tf.where(concept_uncertainty != 1, intervention_mask,
                                   0.)

    if self.format is enum_utils.InterventionFormat.LOGITS:
      true_concepts = true_concepts * self.logits_max + (
          1 - true_concepts) * self.logits_min
    elif self.format is enum_utils.InterventionFormat.PROBS:
      pred_concepts = tf.sigmoid(pred_concepts)
      true_concepts = true_concepts * self.probs_max + (
          1 - true_concepts) * self.probs_min
    elif self.format is enum_utils.InterventionFormat.BINARY:
      pred_concepts = tf.sigmoid(pred_concepts)

    return true_concepts * intervention_mask + pred_concepts * (
        1 - intervention_mask)

  def global_greedy_policy(
      self, primary_metric, primary_mode,
      secondary_metric = None,
      secondary_mode = None,
      concepts_revealed = None,
      **kwargs):
    """Global Greedy intervention policy.

    For a given validation dataset, this policy selects a global ordering of
    concepts to query using a greedy ranking scheme that obtains the maximum
    incremental improvement in performance (as measured by the primary and
    secondary metrics) at each intervention step.

    Args:
      primary_metric: Name of the primary metric to optimize.
      primary_mode: Whether to maximize or minimize the primary metric. Allowed
        values are "min" and "max".
      secondary_metric: Name of the secondary metric to optimize. This is
        optional, and is only used when the primary metric results in tie.
      secondary_mode:  Whether to maximize or minimize the secondary metric.
        Allowed values are "min" and "max".
      concepts_revealed: Provided when an existing global greedy policy is to be
        loaded, determining the order in which to reveal the concepts according
        to the loaded policy.
      **kwargs: Unused kwargs kept for compatibility.

    Yields:
      intervention_mask: Boolean mask to use for intervention
      next_best_concept: Concept to request for the current intervention
        step.
    """
    if kwargs:
      warnings.warn(
          f'Unnecessary kwargs passed to intervention policy: {kwargs}')
    intervention_mask = np.zeros((self.n_concepts,), dtype=bool)
    yield intervention_mask, ''

    if concepts_revealed is not None:
      for next_best_concept in concepts_revealed[1:]:
        intervention_mask[self.concept_groups[next_best_concept]] = True
        yield intervention_mask, next_best_concept
      return

    primary_metric_fn = policy_metrics.get_metric_fn(primary_metric,
                                                     reduce_mean=True)
    if secondary_metric is not None:
      if secondary_mode is None:
        raise ValueError(
            '"secondary_mode" cannot be None if "secondary_metric" is not None')
      secondary_metric_fn = policy_metrics.get_metric_fn(secondary_metric,
                                                         reduce_mean=True)

    for _ in range(self.steps):
      best_primary_metric = {'min': np.inf, 'max': -np.inf}[primary_mode]
      if secondary_mode is not None:
        best_secondary_metric = {'min': np.inf, 'max': -np.inf}[secondary_mode]
      for concept_group_name in self.concept_group_names:
        b_concepts_to_reveal = self.concept_groups[concept_group_name]
        if intervention_mask[b_concepts_to_reveal].all():
          continue
        mask_temp = intervention_mask.copy()
        mask_temp[b_concepts_to_reveal] = True

        intervened_concepts = self.intervene_on_batch(self.true_concepts,
                                                      self.concept_uncertainty,
                                                      self.pred_concepts,
                                                      mask_temp)
        pred_labels = self.ctoy_model(intervened_concepts, training=False)[0]

        curr_primary_metric = primary_metric_fn(self.true_labels, pred_labels)
        if secondary_mode is not None:
          curr_secondary_metric = secondary_metric_fn(self.true_labels,
                                                      pred_labels)

        if policy_metrics.new_best_metric(
            curr_primary_metric, best_primary_metric, mode=primary_mode):
          best_primary_metric = curr_primary_metric
          if secondary_mode is not None:
            best_secondary_metric = curr_secondary_metric
          next_best_concept = concept_group_name
        elif (secondary_mode is not None and
              curr_primary_metric == best_primary_metric and
              policy_metrics.new_best_metric(
                  curr_secondary_metric,
                  best_secondary_metric,
                  mode=secondary_mode)):
          best_secondary_metric = curr_secondary_metric
          next_best_concept = concept_group_name

      intervention_mask[self.concept_groups[next_best_concept]] = True
      yield intervention_mask, next_best_concept

  def instance_greedy_policy(
      self, primary_metric, primary_mode,
      secondary_metric = None,
      secondary_mode = None,
      **kwargs):
    """Instance Greedy intervention policy.

    For a given test instance, this policy selects an ordering of concepts to
    query using a greedy ranking scheme that obtains the maximum incremental
    improvement in performance (as measured by the primary metric, and, in the
    case of a tie, the secondary metric) at each intervention step.

    Args:
      primary_metric: Name of the primary metric to optimize.
      primary_mode: Whether to maximize or minimize the primary metric. Allowed
        values are "min" and "max".
      secondary_metric: Name of the secondary metric to optimize. This is
        optional, and is only used when the primary metric results in tie.
      secondary_mode:  Whether to maximize or minimize the secondary metric.
        Allowed values are "min" and "max".
      **kwargs: Unused kwargs kept for compatibility.

    Yields:
      intervention_mask: Boolean mask to use for intervention
      next_best_concept: Concepts to request for the current intervention
        step.
    """
    if kwargs:
      warnings.warn(
          f'Unnecessary kwargs passed to intervention policy: {kwargs}')
    primary_metric_fn = policy_metrics.get_metric_fn(primary_metric,
                                                     reduce_mean=False)
    if secondary_metric is not None:
      if secondary_mode is None:
        raise ValueError(
            '"secondary_mode" cannot be None if "secondary_metric" is not None')
      secondary_metric_fn = policy_metrics.get_metric_fn(secondary_metric,
                                                         reduce_mean=False)

    intervention_mask = np.zeros((self.n_test, self.n_concepts), dtype=bool)
    next_best_concept = np.zeros(shape=(self.n_test,), dtype=object)
    yield intervention_mask, next_best_concept.copy()

    for _ in range(self.steps):
      best_primary_metric = {'min': np.inf, 'max': -np.inf}[primary_mode]
      best_primary_metric = np.ones(shape=(self.n_test,)) * best_primary_metric
      if secondary_mode is not None:
        best_secondary_metric = {'min': np.inf, 'max': -np.inf}[secondary_mode]
        best_secondary_metric = np.ones(
            shape=(self.n_test,)) * best_secondary_metric

      for concept_group_name in self.concept_group_names:
        b_concepts_to_reveal = self.concept_groups[concept_group_name]
        already_revealed = intervention_mask[:,
                                             b_concepts_to_reveal].all(axis=1)
        mask_temp = intervention_mask.copy()
        mask_temp[:, b_concepts_to_reveal] = True
        intervened_concepts = self.intervene_on_batch(self.true_concepts,
                                                      self.concept_uncertainty,
                                                      self.pred_concepts,
                                                      mask_temp)

        pred_labels = self.ctoy_model(intervened_concepts, training=False)[0]
        curr_primary_metric = primary_metric_fn(self.true_labels, pred_labels)

        update_mask = policy_metrics.new_best_metric(
            curr_primary_metric, best_primary_metric, mode=primary_mode)

        if secondary_mode is not None:
          curr_secondary_metric = secondary_metric_fn(self.true_labels,
                                                      pred_labels)
          tie_mask = curr_primary_metric == best_primary_metric
          update_mask = update_mask | (
              tie_mask & policy_metrics.new_best_metric(
                  curr_secondary_metric,
                  best_secondary_metric,
                  mode=secondary_mode))
        update_mask = update_mask & (~already_revealed)

        best_primary_metric[update_mask] = curr_primary_metric[update_mask]
        next_best_concept[update_mask] = concept_group_name
        if secondary_mode is not None:
          best_secondary_metric[update_mask] = curr_secondary_metric[
              update_mask]

      for i in range(self.n_test):
        intervention_mask[i, self.concept_groups[next_best_concept[i]]] = True
      yield intervention_mask, next_best_concept.copy()

  def global_random_policy(
      self, seed,
      **kwargs):
    """Global Random intervention policy.

    For a given test dataset, this policy selects a random ordering of
    concepts to query.

    Args:
      seed: Random seed
      **kwargs: Unused kwargs kept for compatibility.

    Yields:
      intervention_mask: Boolean mask to use for intervention
      next_best_concept: Concept to request for the current intervention
        step.
    """
    if kwargs:
      warnings.warn(
          f'Unnecessary kwargs passed to intervention policy: {kwargs}')
    random.seed(seed)
    concept_group_names = self.concept_group_names.copy()
    random.shuffle(concept_group_names)
    intervention_mask = np.zeros((self.n_concepts,), dtype=bool)
    yield intervention_mask, ''

    for i in range(self.steps):
      b_concepts_to_reveal = self.concept_groups[concept_group_names[i]]
      intervention_mask[b_concepts_to_reveal] = True
      yield intervention_mask, concept_group_names[i]

  def instance_random_policy(
      self, seed,
      **kwargs):
    """Instance Random intervention policy.

    For a given test instance, this policy selects a random ordering of
    concepts to query.

    Args:
      seed: Random seed
      **kwargs: Unused kwargs kept for compatibility.

    Yields:
      intervention_mask: Boolean mask to use for intervention
      next_best_concept: Concept chosen for the current intervention
        step.
    """
    if kwargs:
      warnings.warn(
          f'Unnecessary kwargs passed to intervention policy: {kwargs}')
    random.seed(seed)
    intervention_mask = np.zeros((self.n_test, self.n_concepts), dtype=bool)
    next_best_concept = np.zeros(shape=(self.n_test,), dtype=object)
    yield intervention_mask, next_best_concept.copy()

    instance_concept_names = []
    for _ in range(self.n_test):
      concept_group_names = self.concept_group_names.copy()
      random.shuffle(concept_group_names)
      instance_concept_names.append(concept_group_names)

    for step in range(self.steps):
      for i in range(self.n_test):
        b_concepts_to_reveal = self.concept_groups[instance_concept_names[i]
                                                   [step]]
        intervention_mask[i, b_concepts_to_reveal] = True
        next_best_concept[i] = instance_concept_names[i][step]
      yield intervention_mask, next_best_concept.copy()

  def coop_policy(
      self, concept_metric, label_metric,
      concept_costs, label_metric_weight,
      cost_weight,
      **kwargs):
    """The CooP policy.

    For a given test instance, this policy computes the score using a weighted
    linear combination of three quantities -
      1. concept prediction uncertainty given by the X->C model,
      2. concept importance given by the C-Y model, and,
      3. concept label acquisition cost,
    and selects an ordering that contains the highest scoring concept in each
    iteration of intervention.

    Args:
      concept_metric: Name of the metric to use for measuring concept prediction
        uncertainty.
      label_metric: Name of the metric to use for measuring concept importance.
      concept_costs: A dictionary mapping concept group names to concept
        acquisition costs.
      label_metric_weight: Weight for the concept importance term in the score.
      cost_weight: Weight for the concept acquisition cost term in the score.
      **kwargs: Unused kwargs kept for compatibility.

    Yields:
      intervention_mask: Boolean mask to use for intervention
      next_best_concept: Concept chosen for the current intervention
        step.
    """
    if kwargs:
      warnings.warn(
          f'Unnecessary kwargs passed to intervention policy: {kwargs}')
    concept_metric_fn = policy_metrics.get_metric_fn(concept_metric,
                                                     reduce_mean=False)
    label_metric_fn = policy_metrics.get_metric_fn(label_metric,
                                                   reduce_mean=False)

    intervention_mask = np.zeros((self.n_test, self.n_concepts), dtype=bool)
    next_best_concept = np.zeros(shape=(self.n_test,), dtype=object)
    yield intervention_mask, next_best_concept.copy()

    for _ in range(self.steps):
      best_score = np.ones(shape=(self.n_test,)) * -np.inf
      for concept_group_name in self.concept_group_names:
        b_concepts_to_reveal = self.concept_groups[concept_group_name]
        already_revealed = intervention_mask[:,
                                             b_concepts_to_reveal].all(axis=1)
        concept_metric_value = concept_metric_fn(self, concept_group_name)
        label_metric_value = label_metric_fn(
            self, concept_group_name, intervention_mask)
        score = (concept_metric_value
                 + label_metric_weight * label_metric_value
                 - cost_weight * concept_costs[concept_group_name])
        update_mask = (~already_revealed) & policy_metrics.new_best_metric(
            score, best_score, mode='max')
        best_score[update_mask] = score[update_mask]
        next_best_concept[update_mask] = concept_group_name

      for i in range(self.n_test):
        intervention_mask[i, self.concept_groups[next_best_concept[i]]] = True
      yield intervention_mask, next_best_concept.copy()

  def load_dataset(
      self, intervention_mask, batch_size,
      output_signature,
  ):
    """Returns the intervened dataset for a given intervention mask.

    Args:
      intervention_mask: Boolean mask to use for intervention
      batch_size: Batch size for the intervened dataset
      output_signature: Output signature of the intervened dataset

    Returns:
      intervened_data: The interevened dataset
    """

    policy_dataset = self.policy_dataset.unbatch().batch(batch_size)

    pred_concepts_ds = tf.data.Dataset.from_tensor_slices(
        self.pred_concepts).batch(batch_size)

    if self.policy_type == 'global':
      assert intervention_mask.ndim == 1
      intervention_mask = np.repeat(
          intervention_mask[None, :], repeats=self.n_test, axis=0)
    else:
      assert (intervention_mask.ndim == 2) and (intervention_mask.shape[0]
                                                == self.n_test)

    mask_ds = tf.data.Dataset.from_tensor_slices(intervention_mask).batch(
        batch_size)

    def _generator():
      for true_batch, pred_concept_batch, mask_batch in zip(
          policy_dataset, pred_concepts_ds, mask_ds):
        intervened_concepts = self.intervene_on_batch(true_batch[1],
                                                      true_batch[3],
                                                      pred_concept_batch,
                                                      mask_batch)
        yield true_batch[0], intervened_concepts, true_batch[2]

    intervened_data = tf.data.Dataset.from_generator(
        _generator, output_signature=output_signature)
    return intervened_data
