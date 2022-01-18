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

"""Implementation of the triplet loss for contrastive learning."""

import random

import tensorflow as tf


def triplet_semihard_loss_fn(anchor_embedding,
                             positive_embedding,
                             negative_embedding,
                             margin=1.0):
  """Implements the triplet loss function.

  Args:
    anchor_embedding: The embedding vector of the anchor example.
    positive_embedding: The embedding vector of the positive example.
    negative_embedding: The embedding vector of the negative example.
    margin: The loss margin.

  Returns:
    The value of the loss.
  """

  return tf.nn.relu(
      tf.norm(anchor_embedding - positive_embedding) -
      tf.norm(anchor_embedding - negative_embedding) + margin)


class TripletLoss(tf.keras.losses.Loss):
  """Computes the triplet loss over a batch of data."""

  def __init__(self,
               embedding_size,
               triplet_loss_margin=0.1,
               train_ratio=0.1,
               num_partitions=1000,
               min_partition_size=10,
               chunk_count=5,
               num_clusters_per_block=16,
               num_neighbors=100,
               num_partitions_to_search=140,
               anchor_reuse_count_max=20,
               **kwargs):
    # TODO(sahandm): Add docstring
    super(TripletLoss, self).__init__(**kwargs)
    self.embedding_size = embedding_size
    self.train_ratio = train_ratio
    self.num_partitions = num_partitions
    self.min_partition_size = min_partition_size
    self.num_clusters_per_block = num_clusters_per_block
    self.num_neighbors = num_neighbors
    self.num_partitions_to_search = num_partitions_to_search
    self.anchor_reuse_count_max = anchor_reuse_count_max
    self.chunk_sizes = [
        (self.embedding_size - 1) // chunk_count
    ] * (chunk_count - 1) + [((self.embedding_size - 1) % chunk_count) + 1]

  def internal_create_ah_tree_searcher(self, embedding_matrix):
    # TODO(sahandm): Add docstring

    # TODO(sahandm): Comment in this code once Scann sarcher is added.
    # dataset_size = embedding_matrix.shape[0]
    # train_size = tf.cast(
    #     tf.cast(dataset_size, tf.float32) * tf.constant(self.train_ratio),
    #     tf.uint32)
    #
    # training_data = tf.slice(
    #     tf.random.shuffle(embedding_matrix), [0, 0],
    #     tf.cast(
    #         tf.convert_to_tensor([train_size, self.embedding_size]),tf.int32),
    #     name="GetTrainingData")
    #
    # approx_num_neighbors = self.num_neighbors * 10

    # TODO(sahandm) Create AH tree Scann searcher here and return it.
    pass

  def internal_get_nearest_neighbors(self, embedding_matrix):
    # TODO(sahandm): Explain args and return values in docstring
    """Returns the nearest neighbors for each data point."""
    # TODO(sahandm): Comment in this code once Scann sarcher call is added.
    # searcher = self.internal_create_ah_tree_searcher(embedding_matrix)
    # query_data = embedding_matrix

    # TODO(sahandm): Call Scann to get nearest neighbors and return them.
    return (None, None)

  def internal_get_easy_positives_and_hard_negatives(self, embedding_matrix,
                                                     labels):
    # TODO(sahandm): Explain args and return values in docstring
    """Gets easy positives and hard negative for each data point."""
    dataset_size = embedding_matrix.shape[0]

    # Search
    approx_dis, approx_idx = self.internal_get_nearest_neighbors(
        embedding_matrix)
    inf_dist = tf.math.reduce_max(approx_dis) + 1

    # Extract positives
    is_positive = labels[approx_idx] == tf.reshape(
        tf.repeat(labels, self.num_neighbors),
        [dataset_size, self.num_neighbors])
    positive_order = tf.argsort(
        tf.where(is_positive, approx_dis, inf_dist), axis=1)
    easy_positives = tf.gather(
        tf.where(is_positive, approx_idx, -1), positive_order)

    # Extract negatives
    is_negative = tf.math.logical_not(is_positive)
    negative_order = tf.argsort(
        tf.where(is_negative, approx_dis, inf_dist), axis=1)
    hard_negatives = tf.gather(
        tf.where(is_negative, approx_idx, -1), negative_order)

    return easy_positives, hard_negatives

  def generate_triplets(self, embedding_matrix, labels, sample_weight):
    # TODO(sahandm): Explain args and return values in docstring
    """Generates triplets of easy positives and hard negatives."""

    def cmp_key(a, p, n):
      return tf.norm(embedding_matrix[a, :] -
                     embedding_matrix[p, :]) + tf.norm(embedding_matrix[a, :] -
                                                       embedding_matrix[n, :])

    easy_positives, hard_negatives = self.internal_get_easy_positives_and_hard_negatives(
        embedding_matrix, labels)

    for anchor_id in range(embedding_matrix.shape[0]):
      anchor_easy_positives = easy_positives[anchor_id, :].eval(
          session=tf.compat.v1.Session())
      anchor_hard_negatives = hard_negatives[anchor_id, :].eval()
      positive_neighbors = []
      negative_neighbors = []

      for positive_index in range(self.num_neighbors):
        if anchor_easy_positives[positive_index] == -1:
          break
        positive_neighbors.append(anchor_easy_positives[positive_index])
      positive_neighbors.reverse()

      for negative_index in range(self.num_neighbors):
        if anchor_hard_negatives[positive_index] == -1:
          break
        negative_neighbors.append(anchor_hard_negatives[negative_index])

      triplets = []
      for positive_id in positive_neighbors:
        for negative_id in negative_neighbors:
          if sample_weight is None or random.random(
          ) < sample_weight[anchor_id] * sample_weight[
              positive_id] * sample_weight[negative_id]:
            triplets.append((anchor_id, positive_id, negative_id))
      triplets.sort(key=cmp_key)
      for i in range(min(len(triplets), self.anchor_reuse_count_max)):
        yield triplets[i]

  def call(self, y_true, y_pred, sample_weight=None):
    # TODO(sahandm): Add docstring
    self.y_true = y_true
    self.y_pred = y_pred
    self.triplets = []

    loss = 0
    for (anchor_id, positive_id,
         negative_id) in self.generate_triplets(y_pred, y_true, sample_weight):
      self.triplets.append((anchor_id, positive_id, negative_id))

      anchor_embedding = y_pred[anchor_id, :]
      positive_embedding = y_pred[positive_id, :]
      negative_embedding = y_pred[negative_id, :]

      loss += triplet_semihard_loss_fn(anchor_embedding, positive_embedding,
                                       negative_embedding,
                                       self.triplet_loss_margin)

    return loss
