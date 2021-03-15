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

"""Seed point sampling algorithms."""
import sys
import tensorflow as tf

from tf3d.utils import instance_segmentation_utils


def sample_based_on_scores_and_distances(inputs,
                                         scores,
                                         num_samples,
                                         scores_coef):
  """Iteratively samples points based on distances and scores.

  Each point i's current total score is computed as
      total_score[i] = scores_coef * log(scores[i]) +
                       log(dist(i, closest previously picked point))

  Args:
    inputs: A tf.float32 tensor of size [N, dims] containing the input vectors.
    scores: A tf.float32 tensor of size [N] containing the scores. Scores
            should be all positive.
    num_samples: A tf.int32 scalar defining the number of samples.
    scores_coef: A tf.float32 coeffcieint that determines that weight assigned
                 to scores in comparison to distances.

  Returns:
    input_samples: The input vectors that are sampled. A tf.float32 tensor of
                   size [num_samples, dims].
    indices: A tf.int32 tensor of size [num_samples] containing the indices of
             the sampled points.
  """
  log_scores = tf.math.log(scores + sys.float_info.min)
  dims = tf.shape(inputs)[1]
  init_samples = tf.zeros(tf.stack([num_samples, dims]), dtype=tf.float32)
  indices = tf.zeros([num_samples], dtype=tf.int32)
  index_0 = tf.expand_dims(
      tf.cast(tf.math.argmax(scores, axis=0), dtype=tf.int32), axis=0)
  init_sample_0 = tf.gather(inputs, index_0)
  min_distances = tf.squeeze(
      instance_segmentation_utils.inputs_distances_to_centers(
          inputs, init_sample_0),
      axis=1)
  init_samples += tf.pad(
      init_sample_0,
      paddings=tf.stack([tf.stack([0, num_samples-1]), tf.stack([0, 0])]))
  indices += tf.pad(index_0, paddings=[[0, num_samples-1]])
  i = tf.constant(1, dtype=tf.int32)

  def body(i, init_samples, indices, min_distances):
    """while loop body. Pick the next center given previously picked centers.

    Args:
      i: For loop index.
      init_samples: Initial samples that is modified inside the body.
      indices: Indices of picked samples.
      min_distances: Minimum distance of the embedding to centers so far.

    Returns:
      i and init_centers.
    """
    best_new_sample_ind = tf.cast(
        tf.math.argmax(
            scores_coef * log_scores +
            tf.math.log(min_distances + sys.float_info.min),
            axis=0),
        dtype=tf.int32)
    indices += tf.pad(tf.stack([best_new_sample_ind]),
                      paddings=[[i, num_samples-i-1]])
    init_samples_i = tf.expand_dims(tf.gather(inputs, best_new_sample_ind), 0)
    init_samples += tf.pad(init_samples_i,
                           paddings=tf.stack([tf.stack([i, num_samples-i-1]),
                                              tf.stack([0, 0])]))
    min_distances = tf.minimum(
        tf.squeeze(
            instance_segmentation_utils.inputs_distances_to_centers(
                inputs, init_samples_i),
            axis=1), min_distances)
    i += 1
    return i, init_samples, indices, min_distances

  (i, init_samples, indices, min_distances) = tf.while_loop(
      lambda i, init_samples, indices, min_distances: i < num_samples,
      body,
      [i, init_samples, indices, min_distances])
  return init_samples, indices
