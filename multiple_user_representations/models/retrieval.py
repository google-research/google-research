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

"""Defines a retrieval model that is trained on next-item prediction task.

The retrieval model trains the user and item tower to generate low-dimensional
user and item embeddings, which are used to retrieve candidate items for the
next item engagement.
"""

from typing import Dict, Optional, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs

from multiple_user_representations.models import util


class RetrievalModel(tfrs.Model):
  """Retrieval model class for next-item engagement."""

  def __init__(self,
               user_model,
               candidate_model,
               task,
               num_items,
               use_disagreement_loss = False,
               l2_normalize = False,
               use_sample_weight = False):
    """Initializes the retrieval model.

    Args:
      user_model: User tower model. Outputs user embedding of shape [B, H, D]
        given sequence of shape [B, T], where B is the batch_size, H is the
        number of user representations and D is the output dimension.
      candidate_model: Item tower model. Outputs item embedding of size D.
      task: The next-item task that computes the loss function and metrics.
      num_items: Number of items in the dataset.
      use_disagreement_loss: Whether to use the disagreement loss or not.
      l2_normalize: True if the output embeddings are to be normalized.
      use_sample_weight: True if item imbalance weights are to be used.
    """

    super().__init__()
    self.candidate_model = candidate_model
    self.user_model = user_model
    self._task = task
    self._use_disagreement_loss = use_disagreement_loss
    self._l2_normalize = l2_normalize
    self._num_items = num_items
    self._use_sample_weight = use_sample_weight

  def compute_loss(self,
                   features,
                   training = False):
    """Computes the loss function to train the user and item tower.

    Args:
      features: The input batch data.
      training: Whether the model is training mode or not.

    Returns:
      loss: A scalar tensor for loss value.
    """

    # Consider adding a feature extraction layer for user
    # and item tower to make the retrieval model more generic.
    user_embeddings = self.user_model(features["user_item_sequence"])

    positive_item_embeddings = self.candidate_model(
        tf.reshape(features["next_item"], (-1, 1)))
    positive_item_embeddings = tf.squeeze(positive_item_embeddings)

    candidate_sampling_probability = features.get(
        "candidate_sampling_probability", None)
    eval_negative_item_embeddings = None
    if "user_negative_items" in features:
      eval_negative_item_embeddings = self.candidate_model(
          features["user_negative_items"])

    if self._l2_normalize:
      user_embeddings = tf.linalg.normalize(user_embeddings, axis=-1)
      positive_item_embeddings = tf.linalg.normalize(positive_item_embeddings,
                                                     axis=-1)
      if eval_negative_item_embeddings is not None:
        eval_negative_item_embeddings = tf.linalg.normalize(
            eval_negative_item_embeddings, axis=-1)

    sample_weight = None
    if self._use_sample_weight:
      sample_weight_in_features = features.get(util.SAMPLE_WEIGHT, None)
      if sample_weight_in_features is not None:
        # sample_weight provided in the input.
        sample_weight = sample_weight_in_features
      elif candidate_sampling_probability is not None:
        # Use class imbalance weights: w_j = num_samples/(samples_j * num_items)
        sample_weight = 1.0 / (candidate_sampling_probability * self._num_items)

    # The task computes the loss and the metrics.
    loss = self._task(
        user_embeddings,
        positive_item_embeddings,
        sample_weight=sample_weight,
        candidate_sampling_probability=candidate_sampling_probability,
        eval_candidate_embeddings=eval_negative_item_embeddings,
        is_head_item=features["is_head_item"])

    # Check if the task supports disagreement loss.
    compute_disagr_op = getattr(self._task, "compute_cosine_disagreement_loss",
                                None)
    if self._use_disagreement_loss and callable(compute_disagr_op):
      disagreement_loss = compute_disagr_op(self.user_model.query_head)
      loss += disagreement_loss

    return loss

  def set_sample_weighting(self, sample_weighting = True):
    self._use_sample_weight = sample_weighting

  @property
  def get_candidate_embeddings(self):
    return self.candidate_model.layers[0].weights[0]
