# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Useful functions for handling embedding vectors.
"""
import sys
import tensorflow as tf
from tf3d.utils import instance_segmentation_utils


def embedding_to_cluster_centers_random(embedding, num_samples_per_center=400):
  """Randomly picks vectors as centers from the embedding vectors.

  Args:
    embedding: A tf.float32 tensor of size [N, embedding_size].
    num_samples_per_center: Number of input embedding samples per center.

  Returns:
    centers: A tf.float32 tensor of size [k, embedding_size].
    indices: The indices of the centers in embedding such that
             centers = embedding[indices, :].
  """
  n = tf.shape(embedding)[0]
  num_centers = n / num_samples_per_center
  indices = tf.slice(tf.random.shuffle(tf.range(n)),
                     begin=[0],
                     size=[num_centers])
  return tf.gather(embedding, indices), indices


def embedding_to_cluster_centers_kpp(embedding, num_samples_per_center=400):
  """Initializes centers using kmeans plus plus initialization.

  Args:
    embedding: A tf.float32 tensor of size [height, width, embedding_size].
    num_samples_per_center: Number of input embedding samples per center.

  Returns:
    embedding_seeds: A tf.float32 tensor of size [k, embedding_size].
    indices: Indices of embedding_seeds
  """
  n = tf.shape(embedding)[0]
  num_centers = n // num_samples_per_center
  return instance_segmentation_utils.kmeans_initialize_centers_plus_plus(
      embedding, num_centers)


def embedding_to_cluster_centers_given_scores(embedding,
                                              centerness_scores,
                                              num_samples_per_center=400):
  """Initializes centers given centerness scores.

  Args:
    embedding: A tf.float32 tensor of size [N, embedding_size].
    centerness_scores: A tf.float32 tensor of size [N]. it is
                       assumed that centerness scores are between 0 and 1.
    num_samples_per_center: Number of input embedding samples per center.

  Returns:
    embedding_seeds: A tf.float32 tensor of size [k, embedding_size].
    indices: Indices of embedding_seeds
  """
  n = tf.shape(embedding)[0]
  num_centers = n // num_samples_per_center
  center_indices = tf.squeeze(
      tf.random.categorical(
          tf.math.log(
              tf.expand_dims(centerness_scores, axis=0) + sys.float_info.min),
          num_centers),
      axis=0)
  return tf.gather(embedding, center_indices), center_indices


def embedding_centers_to_soft_masks_by_dot_product(embedding, centers):
  """Masks from embedding and centers.

  Args:
    embedding: A tf.float32 tensor of size [N, embedding_size].
    centers: A tf.float32 tensor of size [k, embedding_size].

  Returns:
    A tf.float32 tensor of [k, N] that contains k soft masks.
  """
  centers_embedding_dotprod = tf.matmul(centers, embedding, transpose_b=True)
  return tf.nn.sigmoid(centers_embedding_dotprod)


def embedding_centers_to_soft_masks_by_relative_dot_product(embedding, centers):
  """Returns a soft mask for each embedding center.

  For each center vector, returns a soft mask which has values between [0, 1].
  The mask value for each pixel with embedding vector e and a center with
  embedding vector c is computed as min(max(exp(e.c) / exp(c.c), 0.0), 1.0)

  Args:
    embedding: A tf.float32 embedding tensor of size [N, dims] where N is the
               number of embedding vectors and dims is the embedding dimension.
    centers: A tf.float32 embedding centers tensor of size [k, dims] where k is
             the number of centers.

  Returns:
    A tf.float32 soft masks tensor of [k, N].
  """
  n = tf.shape(embedding)[0]
  centers_norm = tf.reduce_sum(centers * centers, axis=1)
  centers_norm = tf.tile(tf.expand_dims(centers_norm, axis=1), [1, n])
  centers_embedding_dotprod = tf.matmul(centers, embedding, transpose_b=True)
  return tf.exp(tf.minimum(centers_embedding_dotprod - centers_norm, 0.0))


def embedding_centers_to_soft_masks_by_distance(embedding, centers):
  """Masks from embedding and centers based on distance to centers.

  Args:
    embedding: A tf.float32 tensor of size [N, embedding_size].
    centers: A tf.float32 tensor of size [k, embedding_size].

  Returns:
    A tf.float32 tensor of [k, N] that contains k soft masks.
  """
  dists_to_centers = instance_segmentation_utils.inputs_distances_to_centers(
      centers, embedding)
  return tf.nn.sigmoid(-dists_to_centers) * 2.0


def embedding_centers_to_soft_masks_by_distance2(embedding, centers):
  """Masks from embedding and centers based on distance to centers.

  Args:
    embedding: A tf.float32 tensor of size [N, embedding_size].
    centers: A tf.float32 tensor of size [k, embedding_size].

  Returns:
    A tf.float32 tensor of [k, N] that contains k soft masks.
  """
  max_dist = 100.0
  dists_to_centers = instance_segmentation_utils.inputs_distances_to_centers(
      embedding, centers)
  return tf.transpose(tf.minimum(
      tf.maximum((max_dist - dists_to_centers) / max_dist, 0.0), 1.0))


def get_similarity_between_corresponding_embedding_vectors(embedding_vectors1,
                                                           embedding_vectors2,
                                                           similarity_strategy):
  """Similarity between corresponding embedding vectors given strategy.

  Args:
    embedding_vectors1: A tf.float32 tensor of size [N, embedding_size].
    embedding_vectors2: A tf.float32 tensor of size [N, embedding_size].
    similarity_strategy: Strategy for computing the similarity between
                         embedding vectors.

  Returns:
    A tf.float32 tensor of [k, 1] containing the similarities.

  Raises:
    ValueError: If similarity_strategy is unknown.
  """
  if similarity_strategy == 'dotproduct':
    return tf.expand_dims(
        tf.reduce_sum(embedding_vectors1 * embedding_vectors2, axis=1), axis=1)
  elif similarity_strategy == 'distance':
    return tf.expand_dims(
        -tf.square(tf.norm(embedding_vectors1 - embedding_vectors2, axis=1)),
        axis=1)
  else:
    raise ValueError('Similarity strategy is unknown')


def embedding_centers_to_logits(
    embedding, centers, similarity_strategy, max_value=10.0):
  """logits from embedding and centers based on distance to centers.

  Args:
    embedding: A tf.float32 tensor of size [N, embedding_size].
    centers: A tf.float32 tensor of size [k, embedding_size].
    similarity_strategy: Strategy for computing the similarity between
                         embedding vectors.
    max_value: Maximum absolute value of any dimension of the logits.

  Returns:
    A tf.float32 tensor of [k, N] that contains k logit maps.

  Raises:
    ValueError: If similarity_strategy is unknown.
  """
  if similarity_strategy == 'dotproduct':
    logits = tf.matmul(centers, embedding, transpose_b=True)
  elif similarity_strategy == 'distance':
    logits = -instance_segmentation_utils.inputs_distances_to_centers(
        centers, embedding)
  else:
    raise ValueError('Similarity strategy is unknown')
  return tf.minimum(tf.maximum(logits, -max_value), max_value)


def embedding_centers_to_soft_masks(
    embedding,
    centers,
    similarity_strategy):
  """Masks from embedding and centers based on distance to centers.

  Args:
    embedding: A tf.float32 tensor of size [N, embedding_size].
    centers: A tf.float32 tensor of size [k, embedding_size].
    similarity_strategy: Strategy for computing the similarity between
                         embedding vectors.

  Returns:
    A tf.float32 tensor of [k, N] that contains k soft masks.

  Raises:
    ValueError: If similarity_strategy is unknown.
  """
  if similarity_strategy == 'dotproduct':
    return embedding_centers_to_soft_masks_by_dot_product(
        embedding, centers)
  elif similarity_strategy == 'distance':
    return embedding_centers_to_soft_masks_by_distance(
        embedding, centers)
  else:
    raise ValueError('Similarity strategy is unknown')
