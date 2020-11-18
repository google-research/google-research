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

"""Instance segmentation post-processing function."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.utils import instance_sampling_utils
from tf3d.utils import instance_segmentation_utils
from tf3d.utils import metric_learning_utils as embedding_utils


def _get_top_k_indices(scores, k):
  max_score = tf.reduce_max(scores, axis=1)
  num_boxes = tf.shape(max_score)[0]
  [_, indices] = tf.nn.top_k(max_score, k=tf.minimum(num_boxes, k))
  return indices


def _sample_furthest_voxels(scores, instance_embeddings,
                            num_furthest_voxel_samples,
                            sampler_score_vs_distance_coef):
  """Samples voxels based on distance in embedding space and scores."""
  num_furthest_voxel_samples = tf.minimum(num_furthest_voxel_samples,
                                          tf.shape(scores)[0])
  _, seed_indices = instance_sampling_utils.sample_based_on_scores_and_distances(
      inputs=instance_embeddings,
      scores=tf.reduce_max(scores, axis=1),
      num_samples=num_furthest_voxel_samples,
      scores_coef=sampler_score_vs_distance_coef)
  scores = tf.gather(scores, seed_indices)
  instance_embeddings = tf.gather(instance_embeddings, seed_indices)
  return scores, instance_embeddings


def _keep_top_k_instances(scores, instance_embeddings, k):
  """Keeps the top k highest score boxes.

  Args:
    scores: A tf.float32 tensor of size [num_voxels, num_classes].
    instance_embeddings: A tf.float32 tensor of
      size [num_voxels, embedding_dims].
    k: Number of instances.

  Returns:
    top k scores and instance embeddings.
  """
  indices = _get_top_k_indices(scores, k)
  scores = tf.gather(scores, indices)
  instance_embeddings = tf.gather(instance_embeddings, indices)
  return scores, instance_embeddings


def _remove_low_score_instances(scores,
                                instance_embeddings,
                                score_threshold,
                                min_num_instances=100):
  """Removes the instances that have an score lower than threshold.

  Args:
    scores: A tf.float32 tensor of size [num_voxels, num_classes].
    instance_embeddings: A tf.float32 tensor of
      size [num_voxels, embedding_dims].
    score_threshold: A float corresponding to score threshold.
    min_num_instances: Minimum number of instances.

  Returns:
    High scores and their corresponding instance embeddings.
  """
  max_score = tf.reduce_max(scores, axis=1)
  score_mask = tf.greater_equal(max_score, score_threshold)
  high_score_indices = tf.cast(
      tf.reshape(tf.where(score_mask), [-1]), dtype=tf.int32)
  top_k_indices = _get_top_k_indices(scores=scores, k=min_num_instances)
  high_score_indices = tf.cond(
      tf.shape(high_score_indices)[0] > min_num_instances,
      lambda: high_score_indices, lambda: top_k_indices)
  scores = tf.gather(scores, high_score_indices)
  instance_embeddings = tf.gather(instance_embeddings, high_score_indices)
  return scores, instance_embeddings


def postprocess(outputs,
                num_furthest_voxel_samples=200,
                sampler_score_vs_distance_coef=0.5,
                embedding_similarity_strategy='distance',
                embedding_similarity_threshold=0.5,
                apply_nms=False,
                nms_score_threshold=0.1,
                nms_iou_threshold=0.5):
  """Postprocess the outputs of our network, including untiling.

  Args:
    outputs: A dict of `Tensor` objects with network outputs.
    num_furthest_voxel_samples: Number of voxels to be sampled using furthest
      voxel sampling.
    sampler_score_vs_distance_coef: The coefficient that balances the weight
      between furthest voxel sampling and highest score sampling.
    embedding_similarity_strategy: Defines the method for computing similarity
      between embedding vectors. Possible values are 'dotproduct'
      and 'distance'.
    embedding_similarity_threshold: Similarity threshold used to decide if two
      point embedding vectors belong to the same instance.
    apply_nms: If True, performs non-maximum suppression after proposing
      instances.
    nms_score_threshold: Score threshold used for non-maximum suppression.
    nms_iou_threshold: IOU threshold used for non-maximum suppression.

  Returns:
    outputs: Our dict of `Tensor` objects with outputs post processed.
  """
  # Softmax semantic scores, removing backgorund scores
  voxel_scores = tf.nn.softmax(
      outputs[standard_fields.DetectionResultFields.object_semantic_voxels])
  voxel_scores = voxel_scores[:, 1:]
  voxel_instance_embeddings = outputs[
      standard_fields.DetectionResultFields.instance_embedding_voxels]

  # Remove low score instances
  top_scores, top_instance_embeddings = _remove_low_score_instances(
      scores=voxel_scores,
      instance_embeddings=voxel_instance_embeddings,
      score_threshold=nms_score_threshold)

  # Sample furthest high score voxels
  top_scores, top_instance_embeddings = _sample_furthest_voxels(
      scores=top_scores,
      instance_embeddings=top_instance_embeddings,
      num_furthest_voxel_samples=num_furthest_voxel_samples,
      sampler_score_vs_distance_coef=sampler_score_vs_distance_coef)

  # Setting instance segment masks, scores and class
  predicted_soft_masks = embedding_utils.embedding_centers_to_soft_masks(
      embedding=voxel_instance_embeddings,
      centers=top_instance_embeddings,
      similarity_strategy=embedding_similarity_strategy)
  outputs[standard_fields.DetectionResultFields
          .instance_segments_voxel_mask] = tf.cast(
              tf.greater(predicted_soft_masks, embedding_similarity_threshold),
              dtype=tf.float32)
  if apply_nms:
    num_classes = top_scores.get_shape().as_list()[1]
    (outputs[
        standard_fields.DetectionResultFields.instance_segments_voxel_mask],
     outputs[standard_fields.DetectionResultFields.objects_score],
     outputs[standard_fields.DetectionResultFields.objects_class]) = (
         instance_segmentation_utils.instance_non_maximum_suppression_2d_scores(
             masks=tf.expand_dims(
                 outputs[standard_fields.DetectionResultFields
                         .instance_segments_voxel_mask],
                 axis=2),
             scores=top_scores,
             num_classes=num_classes,
             min_score_thresh=nms_score_threshold,
             min_iou_thresh=nms_iou_threshold))
    outputs[standard_fields.DetectionResultFields
            .instance_segments_voxel_mask] = tf.squeeze(
                outputs[standard_fields.DetectionResultFields
                        .instance_segments_voxel_mask],
                axis=2)
  else:
    outputs[
        standard_fields.DetectionResultFields.objects_class] = tf.math.argmax(
            top_scores, axis=1)
    outputs[standard_fields.DetectionResultFields
            .objects_score] = tf.math.reduce_max(
                top_scores, axis=1)
  outputs[standard_fields.DetectionResultFields.objects_class] += 1
  outputs[standard_fields.DetectionResultFields.objects_score] = tf.expand_dims(
      outputs[standard_fields.DetectionResultFields.objects_score], axis=1)
  outputs[
      standard_fields.DetectionResultFields.objects_class] = tf.expand_dims(
          outputs[standard_fields.DetectionResultFields.objects_class],
          axis=1)
  outputs[standard_fields.DetectionResultFields.objects_class] = tf.cast(
      outputs[standard_fields.DetectionResultFields.objects_class], tf.int32)
