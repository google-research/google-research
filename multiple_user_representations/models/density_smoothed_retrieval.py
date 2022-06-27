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

"""Defines a retrieval model that is trained on next-item prediction task using dynamic item density smoothing.

The retrieval model trains the user and item tower to generate low-dimensional
user and item embeddings, which are used to retrieve candidate items for the
next item engagement. The sample_weights are computed based on iterative updates
using dynamic item density smoothing as proposed in the following project doc -
http://shortn/_uPej1Fh7Jq#heading=h.sd8nixbhxgso.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
import numpy as np
from scipy import stats
import tensorflow as tf

from multiple_user_representations.models import retrieval
from multiple_user_representations.models import util


class DensityWeightedRetrievalModel(retrieval.RetrievalModel):
  """Retrieval model for next-item task with iterative density item weights."""

  def __init__(self,
               user_model,
               candidate_model,
               task,
               num_items,
               use_disagreement_loss = False,
               l2_normalize = False):
    """See the base class."""

    self._use_sample_weight = True
    self._critic = stats.gaussian_kde
    super().__init__(user_model, candidate_model, task, num_items,
                     use_disagreement_loss, l2_normalize,
                     self._use_sample_weight)

  def _update_item_weights_using_density(
      self,
      item_dataset,
      item_count_weights,
      num_samples_for_density_model = 10000,
      momentum = 0.9,
      save_state_path = None):
    """Updates item weights based on the item density in the embedding space.

    Args:
      item_dataset: The tf dataset containing ids of all items.
      item_count_weights: A dictionary mapping item_id to (item_count,
        item_weight), where item_count refers to the frequency in the training
        dataset.
      num_samples_for_density_model: Num samples to use when learning density
        model.
      momentum: A scalar momentum used to update item weights iteratively.
      save_state_path: If not None, saves item_embeddings and sample_weight for
        visualization at save_state_path.

    Returns:
      updated_item_count_weight: A dictionary mapping item_id to (item_count,
        updated_item_weight).
    """

    def item_map(batched_items):
      return tf.squeeze(
          self.candidate_model(tf.expand_dims(batched_items, axis=1)))

    item_embeddings = list(
        item_dataset.batch(100).map(item_map).unbatch().as_numpy_iterator())
    item_embeddings = np.array(item_embeddings)

    train_next_items = np.array(list(item_count_weights.keys()))
    train_item_counts, train_item_weights = zip(
        *list(item_count_weights.values()))
    train_item_weights = np.array(train_item_weights)

    sampled_items = np.random.choice(
        train_next_items,
        p=train_item_weights,
        size=num_samples_for_density_model)

    # Get embeddings and smooth the data with normal noise.
    train_data = item_embeddings[sampled_items]
    train_data += np.random.normal(loc=0.0, scale=0.1, size=train_data.shape)

    # Learning the density function.
    kernel = self._critic(train_data.T)

    # Update weights
    train_next_item_embeddings = item_embeddings[train_next_items]
    item_densities = kernel(train_next_item_embeddings.T) + 1e-4
    item_density_weights = (1.0 / item_densities)
    item_density_weights /= np.sum(item_density_weights)

    updated_item_weights = momentum * train_item_weights + (
        1 - momentum) * item_density_weights
    updated_item_count_weight = dict(
        zip(train_next_items, zip(train_item_counts, updated_item_weights)))

    if save_state_path is not None:
      embeddings_file = os.path.join(save_state_path, "embeddings.npy")
      current_weights_file = os.path.join(save_state_path,
                                          "current_weights.npy")
      new_weights_file = os.path.join(save_state_path, "new_weights.npy")
      user_query_file = os.path.join(save_state_path, "user_queries.npy")

      util.save_np(embeddings_file, item_embeddings)
      util.save_np(current_weights_file, item_count_weights)
      util.save_np(new_weights_file, updated_item_count_weight)
      util.save_np(user_query_file, self.user_model.query_head.numpy())

    return updated_item_count_weight

  def iterative_training(self,
                         fit_retrieval_model_fn,
                         train_dataset,
                         item_dataset,
                         item_count_weights,
                         results_dir = None,
                         delta = 0.005,
                         momentum = 0.9,
                         max_iterations = 20):
    """Performs iterative training of the retrieval model.

    For each iteration the sample_weights are updated in the training dataset.
    The weights are computed using the self._update_item_weights_using_density
    method.

    Args:
      fit_retrieval_model_fn: A function that trains the retrieval model on the
        given dataset. The function should take tf.data.Dataset as argument.
      train_dataset: The train dataset for the next_item prediction task.
      item_dataset: The item dataset containing the item ids.
      item_count_weights: A dictionary mapping item_ids to (item_count,
        item_sampling_probability).
      results_dir: Path to save intermediate results.
      delta: Stop iterative_training when ||w_{t+1} - w_t|| < delta.
      momentum: Momentum for weight updates.
      max_iterations: Max number of iterations for weight updates.

    Returns:
      The object returned from the fit_retrieval_model_fn in the last iteration.
    """

    converged = False
    iteration = 0
    history = None

    while not converged:

      path = os.path.join(
          results_dir,
          f"iteration_{iteration+1}") if results_dir is not None else None
      updated_item_count_weights = self._update_item_weights_using_density(
          item_dataset,
          item_count_weights,
          momentum=momentum,
          save_state_path=path)
      train_dataset = util.update_train_dataset_with_sample_weights(
          train_dataset, updated_item_count_weights)

      prior_weights = []
      updated_weights = []
      for item_id in item_count_weights.keys():
        prior_weights.append(item_count_weights[item_id][1])
        updated_weights.append(updated_item_count_weights[item_id][1])

      prior_weights = np.array(prior_weights)
      updated_weights = np.array(updated_weights)

      delta_w = np.linalg.norm(prior_weights - updated_weights, ord=np.inf)
      item_count_weights = updated_item_count_weights
      converged = delta_w < delta

      if not converged:
        history = fit_retrieval_model_fn(train_dataset=train_dataset)

      iteration += 1
      logging.info("Iteration %d Delta_W: %.4f", iteration, delta_w)

      if iteration >= max_iterations:
        logging.info("Max iteration ({%d}) reached! Delta_W: %.4f", iteration,
                     delta_w)
        break

    return history
