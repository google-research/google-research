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

"""Functions to compute the cycle consistency loss."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow as tf


@tf.function
def pairwise_l2_distance(pts1, pts2):
  """Computes squared L2 distances between each element of each set of points.

  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.

  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  norm1 = tf.reduce_sum(tf.square(pts1), axis=1, keepdims=True)
  norm2 = tf.reduce_sum(tf.square(pts2), axis=1)
  norm2 = tf.reshape(norm2, [1, -1])
  distance_matrix = tf.maximum(
      norm1 + norm2 - 2.0 * tf.matmul(pts1, pts2, transpose_b=True), 0.0)
  return distance_matrix


@tf.function
def pairwise_l1_distance(pts1, pts2):
  """Computes L1 distances between each element of each set of points.

  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.

  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  stack_size2 = pts2.shape[0]
  pts1_tiled = tf.tile(tf.expand_dims(pts1, 1), [1, stack_size2, 1])
  distance_matrix = tf.reduce_sum(tf.abs(pts1_tiled-pts2), -1)
  return distance_matrix


@tf.function
def pairwise_linf_distance(pts1, pts2):
  """Computes Chebyshev distances between each element of each set of points.

  The Chebyshev/chessboard distance is the L_infinity distance between two
  points, the maximum difference between any of their dimensions.

  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.

  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  stack_size2 = pts2.shape[0]
  pts1_tiled = tf.tile(tf.expand_dims(pts1, 1), [1, stack_size2, 1])
  distance_matrix = tf.reduce_max(tf.abs(pts1_tiled-pts2), -1)
  return distance_matrix


def get_scaled_similarity(embeddings1,
                          embeddings2,
                          similarity_type,
                          temperature):
  """Returns matrix of similarities between two sets of embeddings.

  Similarity is a scalar relating two embeddings, such that a more similar pair
  of embeddings has a higher value of similarity than a less similar pair.  This
  is intentionally vague to emphasize the freedom in defining measures of
  similarity. For the similarities defined, the distance-related ones range from
  -inf to 0 and cosine similarity ranges from -1 to 1.

  Args:
    embeddings1: [N, d] float tensor of embeddings.
    embeddings2: [M, d] float tensor of embeddings.
    similarity_type: String with the method of computing similarity between
      embeddings. Implemented:
        l2sq -- Squared L2 (Euclidean) distance
        l2 -- L2 (Euclidean) distance
        l1 -- L1 (Manhattan) distance
        linf -- L_inf (Chebyshev) distance
        cosine -- Cosine similarity, the inner product of the normalized vectors
    temperature: Float value which divides all similarity values, setting a
      scale for the similarity values.  Should be positive.

  Raises:
    ValueError: If the similarity type is not recognized.
  """
  eps = 1e-9
  if similarity_type == 'l2sq':
    similarity = -1.0 * pairwise_l2_distance(embeddings1, embeddings2)
  elif similarity_type == 'l2':
    # Add a small value eps in the square root so that the gradient is always
    # with respect to a nonzero value.
    similarity = -1.0 * tf.sqrt(
        pairwise_l2_distance(embeddings1, embeddings2) + eps)
  elif similarity_type == 'l1':
    similarity = -1.0 * pairwise_l1_distance(embeddings1, embeddings2)
  elif similarity_type == 'linf':
    similarity = -1.0 * pairwise_linf_distance(embeddings1, embeddings2)
  elif similarity_type == 'cosine':
    embeddings1, _ = tf.linalg.normalize(embeddings1, ord=2, axis=-1)
    embeddings2, _ = tf.linalg.normalize(embeddings2, ord=2, axis=-1)
    similarity = tf.matmul(embeddings1, embeddings2, transpose_b=True)
  else:
    raise ValueError('Similarity type not implemented: ', similarity_type)

  similarity /= temperature
  return similarity


@tf.function
def quantify_unambiguous_cycles(embeddings1,
                                embeddings2,
                                similarity_type,
                                temperature):
  """Map from the first set of embeddings to the second and come back.

  Finds the soft nearest neighbor for each point in embeddings1, in embeddings2,
  according to the method of similarity and the temperature.  Then this soft
  nearest neighbor is compared by the same means to all elements of embeddings1,
  with the label being the identity (unambiguous cyclic consistency).

  Args:
    embeddings1: [N, d] float tensor of embeddings.
    embeddings2: [M, d] float tensor of embeddings.
    similarity_type: String with the method of computing similarity between
      embeddings.
    temperature: Float value which rescales the similarity matrix, setting a
      scale for the similarity values.  Should be positive.

  Returns:
    logits: [N, N] tensor of cycle consistency values, used as logits in a
      cross entropy loss, where the [i, j] component is the scaled similarity
      between the soft nearest neighbor of the ith vector in embeddings1 and the
      jth vector in embeddings1.
    labels: [N, N] tensor of one-hot labels for the cycle consistency.
  """
  stack_size1 = tf.shape(embeddings1)[0]
  similarity12 = get_scaled_similarity(embeddings1, embeddings2,
                                       similarity_type, temperature)
  softmaxed_similarity12 = tf.nn.softmax(similarity12, axis=1)
  nn_embeddings1 = tf.matmul(softmaxed_similarity12, embeddings2)
  similarity121 = get_scaled_similarity(nn_embeddings1, embeddings1,
                                        similarity_type, temperature)
  logits = similarity121
  labels = tf.one_hot(tf.range(stack_size1), stack_size1)
  return logits, labels


def classification_loss(logits, labels, top_k_num=0):
  """Computes the cross-entropy loss, possibly only including the best k losses.

  Args:
    logits: [N, N] tensor of logits.
    labels: [N, N] tensor of one-hot classification labels.
    top_k_num: An optional integer which averages over only the top-k values of
      the loss (top meaning best, so the smallest k values).

  Returns:
    loss: Float tensor of the losses averaged over the stack.
  """

  labels = tf.stop_gradient(labels)
  losses = tf.keras.losses.categorical_crossentropy(y_true=labels,
                                                    y_pred=logits,
                                                    from_logits=True)
  if top_k_num:
    loss = tf.reduce_mean(-tf.math.top_k(-losses, k=top_k_num)[0])
  else:
    loss = tf.reduce_mean(losses)
  return loss
