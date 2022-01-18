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

"""Contains voxel classification losses."""
import functools
import gin
import gin.tf
import tensorflow as tf
from tf3d import standard_fields
from tf3d.losses import utils as loss_utils
from tf3d.utils import instance_segmentation_utils
from tf3d.utils import mask_utils
from tf3d.utils import metric_learning_utils
from tf3d.utils import sampling_utils


def _get_voxels_valid_mask(inputs_1):
  """Returns the mask that removes voxels that are not within image."""
  return mask_utils.num_voxels_mask(inputs=inputs_1)


def _get_voxels_valid_inputs_outputs(inputs_1, outputs_1):
  """Applies the valid mask to input and output voxel tensors."""
  valid_mask = _get_voxels_valid_mask(inputs_1=inputs_1)
  inputs_1 = mask_utils.apply_mask_to_input_voxel_tensors(
      inputs=inputs_1, valid_mask=valid_mask)
  mask_utils.apply_mask_to_output_voxel_tensors(
      outputs=outputs_1, valid_mask=valid_mask)
  return inputs_1, outputs_1


def classification_loss_fn(logits,
                           labels,
                           num_valid_voxels=None,
                           weights=1.0):
  """Semantic segmentation cross entropy loss."""
  logits_rank = len(logits.get_shape().as_list())
  labels_rank = len(labels.get_shape().as_list())
  if logits_rank != labels_rank:
    raise ValueError(
        'Logits and labels should have the same rank.')
  if logits_rank != 2 and logits_rank != 3:
    raise ValueError('Logits and labels should have either 2 or 3 dimensions.')
  if logits_rank == 2:
    if num_valid_voxels is not None:
      raise ValueError(
          '`num_valid_voxels` should be None if not using batched logits.')
  elif logits_rank == 3:
    if num_valid_voxels is None:
      raise ValueError(
          '`num_valid_voxels` cannot be None if using batched logits.')
  if logits_rank == 3:
    if (isinstance(weights, tf.Tensor) and
        len(weights.get_shape().as_list()) == 3):
      use_weights = True
    else:
      use_weights = False
    batch_size = logits.get_shape().as_list()[0]
    logits_list = []
    labels_list = []
    weights_list = []
    for i in range(batch_size):
      num_valid_voxels_i = num_valid_voxels[i]
      logits_list.append(logits[i, 0:num_valid_voxels_i, :])
      labels_list.append(labels[i, 0:num_valid_voxels_i, :])
      if use_weights:
        weights_list.append(weights[i, 0:num_valid_voxels_i, :])
    logits = tf.concat(logits_list, axis=0)
    labels = tf.concat(labels_list, axis=0)
    if use_weights:
      weights = tf.concat(weights_list, axis=0)
  weights = tf.convert_to_tensor(weights, dtype=tf.float32)
  if labels.get_shape().as_list()[-1] == 1:
    num_classes = logits.get_shape().as_list()[-1]
    labels = tf.one_hot(tf.reshape(labels, shape=[-1]), num_classes)
  losses = tf.nn.softmax_cross_entropy_with_logits(
      labels=tf.stop_gradient(labels), logits=logits)
  return tf.reduce_mean(losses * tf.reshape(weights, [-1]))


@gin.configurable('classification_loss', denylist=['inputs', 'outputs'])
def classification_loss(inputs, outputs):
  """Applies categorical crossentropy loss to voxel predictions.

  Note that `labels` and `weights` are resized to match `logits`.

  Args:
    inputs: A dictionary of `Tensors` that contains ground-truth.
    outputs: A dictionary of `Tensors` that contains predictions.

  Returns:
    The loss `Tensor`.

  Raises:
    ValueError: If the loss method is unknown.
    ValueError: If voxel logits and labels have different dimensions.
    ValueError: If num_valid_voxels is None in batch mode.
  """
  if standard_fields.InputDataFields.object_class_voxels not in inputs:
    raise ValueError('`object_class_voxels` is missing in inputs.')
  if (standard_fields.DetectionResultFields.object_semantic_voxels not in
      outputs):
    raise ValueError('`object_semantic_voxels` is missing in outputs.')
  logits = outputs[standard_fields.DetectionResultFields.object_semantic_voxels]
  labels = inputs[standard_fields.InputDataFields.object_class_voxels]
  if standard_fields.InputDataFields.num_valid_voxels in inputs:
    num_valid_voxels = inputs[standard_fields.InputDataFields.num_valid_voxels]
  else:
    num_valid_voxels = None
  if standard_fields.InputDataFields.voxel_loss_weights in inputs:
    weights = inputs[standard_fields.InputDataFields.voxel_loss_weights]
  else:
    weights = 1.0
  return classification_loss_fn(
      logits=logits,
      labels=labels,
      num_valid_voxels=num_valid_voxels,
      weights=weights)


def _box_classification_loss_unbatched(inputs_1, outputs_1, is_intermediate,
                                       is_balanced, mine_hard_negatives,
                                       hard_negative_score_threshold):
  """Loss function for input and outputs of batch size 1."""
  valid_mask = _get_voxels_valid_mask(inputs_1=inputs_1)
  if is_intermediate:
    logits = outputs_1[standard_fields.DetectionResultFields
                       .intermediate_object_semantic_voxels]
  else:
    logits = outputs_1[
        standard_fields.DetectionResultFields.object_semantic_voxels]
  num_classes = logits.get_shape().as_list()[-1]
  if num_classes is None:
    raise ValueError('Number of classes is unknown.')
  logits = tf.boolean_mask(tf.reshape(logits, [-1, num_classes]), valid_mask)
  labels = tf.boolean_mask(
      tf.reshape(
          inputs_1[standard_fields.InputDataFields.object_class_voxels],
          [-1, 1]), valid_mask)
  if mine_hard_negatives or is_balanced:
    instances = tf.boolean_mask(
        tf.reshape(
            inputs_1[standard_fields.InputDataFields.object_instance_id_voxels],
            [-1]), valid_mask)
  params = {}
  if mine_hard_negatives:
    negative_scores = tf.reshape(tf.nn.softmax(logits)[:, 0], [-1])
    hard_negative_mask = tf.logical_and(
        tf.less(negative_scores, hard_negative_score_threshold),
        tf.equal(tf.reshape(labels, [-1]), 0))
    hard_negative_labels = tf.boolean_mask(labels, hard_negative_mask)
    hard_negative_logits = tf.boolean_mask(logits, hard_negative_mask)
    hard_negative_instances = tf.boolean_mask(
        tf.ones_like(instances) * (tf.reduce_max(instances) + 1),
        hard_negative_mask)
    logits = tf.concat([logits, hard_negative_logits], axis=0)
    instances = tf.concat([instances, hard_negative_instances], axis=0)
    labels = tf.concat([labels, hard_negative_labels], axis=0)
  if is_balanced:
    weights = loss_utils.get_balanced_loss_weights_multiclass(
        labels=tf.expand_dims(instances, axis=1))
    params['weights'] = weights
  return classification_loss_fn(
      logits=logits,
      labels=labels,
      **params)


@gin.configurable
def box_classification_loss(inputs,
                            outputs,
                            is_intermediate=False,
                            is_balanced=False,
                            mine_hard_negatives=False,
                            hard_negative_score_threshold=0.5):
  """Calculates the voxel level classification loss.

  Args:
    inputs: A dictionary of tf.Tensors with our input label data.
    outputs: A dictionary of tf.Tensors with the network output.
    is_intermediate: If True, loss will be computed on intermediate tensors.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for foreground vs. background voxels.
    mine_hard_negatives: If True, mines hard negatives and applies loss on them
      too.
    hard_negative_score_threshold: A prediction is a hard negative if its label
      is 0 and the score for the 0 class is less than this threshold.

  Returns:
    loss: A tf.float32 scalar corresponding to softmax classification loss.

  Raises:
    ValueError: If the size of the third dimension of the predicted logits is
      unknown at graph construction.
  """
  return loss_utils.apply_unbatched_loss_on_voxel_tensors(
      inputs=inputs,
      outputs=outputs,
      unbatched_loss_fn=functools.partial(
          _box_classification_loss_unbatched,
          is_intermediate=is_intermediate,
          is_balanced=is_balanced,
          mine_hard_negatives=mine_hard_negatives,
          hard_negative_score_threshold=hard_negative_score_threshold))


def _box_classification_using_center_distance_loss_unbatched(
    inputs_1,
    outputs_1,
    is_intermediate,
    is_balanced,
    max_positive_normalized_distance):
  """Loss function for input and outputs of batch size 1."""
  inputs_1, outputs_1 = _get_voxels_valid_inputs_outputs(
      inputs_1=inputs_1, outputs_1=outputs_1)
  if is_intermediate:
    output_object_centers = outputs_1[
        standard_fields.DetectionResultFields.intermediate_object_center_voxels]
    output_object_length = outputs_1[
        standard_fields.DetectionResultFields.intermediate_object_length_voxels]
    output_object_height = outputs_1[
        standard_fields.DetectionResultFields.intermediate_object_height_voxels]
    output_object_width = outputs_1[
        standard_fields.DetectionResultFields.intermediate_object_width_voxels]
    output_object_rotation_matrix = outputs_1[
        standard_fields.DetectionResultFields
        .intermediate_object_rotation_matrix_voxels]
    logits = outputs_1[standard_fields.DetectionResultFields
                       .intermediate_object_semantic_voxels]
  else:
    output_object_centers = outputs_1[
        standard_fields.DetectionResultFields.object_center_voxels]
    output_object_length = outputs_1[
        standard_fields.DetectionResultFields.object_length_voxels]
    output_object_height = outputs_1[
        standard_fields.DetectionResultFields.object_height_voxels]
    output_object_width = outputs_1[
        standard_fields.DetectionResultFields.object_width_voxels]
    output_object_rotation_matrix = outputs_1[
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels]
    logits = outputs_1[
        standard_fields.DetectionResultFields.object_semantic_voxels]
  normalized_center_distance = loss_utils.get_normalized_corner_distances(
      predicted_boxes_center=output_object_centers,
      predicted_boxes_length=output_object_length,
      predicted_boxes_height=output_object_height,
      predicted_boxes_width=output_object_width,
      predicted_boxes_rotation_matrix=output_object_rotation_matrix,
      gt_boxes_center=inputs_1[
          standard_fields.InputDataFields.object_center_voxels],
      gt_boxes_length=inputs_1[
          standard_fields.InputDataFields.object_length_voxels],
      gt_boxes_height=inputs_1[
          standard_fields.InputDataFields.object_height_voxels],
      gt_boxes_width=inputs_1[
          standard_fields.InputDataFields.object_width_voxels],
      gt_boxes_rotation_matrix=inputs_1[
          standard_fields.InputDataFields.object_rotation_matrix_voxels])
  labels = tf.reshape(
      inputs_1[standard_fields.InputDataFields.object_class_voxels], [-1])
  instances = tf.reshape(
      inputs_1[standard_fields.InputDataFields.object_instance_id_voxels], [-1])
  params = {}
  if is_balanced:
    weights = loss_utils.get_balanced_loss_weights_multiclass(
        labels=tf.expand_dims(instances, axis=1))
    params['weights'] = weights

  def loss_fn():
    """Loss function."""
    num_classes = logits.get_shape().as_list()[-1]
    if num_classes is None:
      raise ValueError('Number of classes is unknown.')
    labels_one_hot = tf.one_hot(indices=(labels - 1), depth=(num_classes - 1))
    inverse_distance_coef = tf.maximum(
        tf.minimum(
            1.0 - normalized_center_distance / max_positive_normalized_distance,
            1.0), 0.0)
    labels_one_hot = tf.reshape(inverse_distance_coef, [-1, 1]) * labels_one_hot
    background_label = 1.0 - tf.math.reduce_sum(
        labels_one_hot, axis=1, keepdims=True)
    labels_one_hot = tf.concat([background_label, labels_one_hot], axis=1)
    loss = classification_loss_fn(
        logits=logits,
        labels=labels_one_hot,
        **params)
    return loss

  return tf.cond(
      tf.greater(tf.shape(labels)[0], 0), loss_fn,
      lambda: tf.constant(0.0, dtype=tf.float32))


@gin.configurable
def box_classification_using_center_distance_loss(
    inputs,
    outputs,
    is_intermediate=False,
    is_balanced=False,
    max_positive_normalized_distance=0.3):
  """Calculates the loss based on predicted center distance from gt center.

  Computes the loss using the object properties of the voxel tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input label data.
    outputs: A dictionary of tf.Tensors with the network output.
    is_intermediate: If True, loss will be computed on intermediate tensors.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for foreground vs. background voxels.
    max_positive_normalized_distance: Maximum distance of a predicted box from
      the ground-truth box that we use to classify the predicted box as
      positive.

  Returns:
    loss: A tf.float32 scalar corresponding to distance confidence loss.
  """
  return loss_utils.apply_unbatched_loss_on_voxel_tensors(
      inputs=inputs,
      outputs=outputs,
      unbatched_loss_fn=functools.partial(
          _box_classification_using_center_distance_loss_unbatched,
          is_intermediate=is_intermediate,
          is_balanced=is_balanced,
          max_positive_normalized_distance=max_positive_normalized_distance))


def classification_loss_using_mask_iou_func_unbatched(
    embeddings, instance_ids, sampled_embeddings,
    sampled_instance_ids, sampled_class_labels, sampled_logits,
    similarity_strategy, is_balanced):
  """Classification loss using mask iou.

  Args:
    embeddings: A tf.float32 tensor of size [n, f].
    instance_ids: A tf.int32 tensor of size [n].
    sampled_embeddings: A tf.float32 tensor of size [num_samples, f].
    sampled_instance_ids: A tf.int32 tensor of size [num_samples].
    sampled_class_labels: A tf.int32 tensor of size [num_samples, 1].
    sampled_logits: A tf.float32 tensor of size [num_samples, num_classes].
    similarity_strategy: Defines the method for computing similarity between
                         embedding vectors. Possible values are 'dotproduct' and
                         'distance'.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for foreground vs. background voxels.

  Returns:
    A tf.float32 loss scalar tensor.
  """
  predicted_soft_masks = metric_learning_utils.embedding_centers_to_soft_masks(
      embedding=embeddings,
      centers=sampled_embeddings,
      similarity_strategy=similarity_strategy)
  predicted_masks = tf.cast(
      tf.greater(predicted_soft_masks, 0.5), dtype=tf.float32)
  gt_masks = tf.cast(
      tf.equal(
          tf.expand_dims(sampled_instance_ids, axis=1),
          tf.expand_dims(instance_ids, axis=0)),
      dtype=tf.float32)
  pairwise_iou = instance_segmentation_utils.points_mask_pairwise_iou(
      masks1=predicted_masks, masks2=gt_masks)
  num_classes = sampled_logits.get_shape().as_list()[1]
  sampled_class_labels_one_hot = tf.one_hot(
      indices=tf.reshape(sampled_class_labels, [-1]), depth=num_classes)
  sampled_class_labels_one_hot_fg = sampled_class_labels_one_hot[:, 1:]
  iou_coefs = tf.tile(tf.reshape(pairwise_iou, [-1, 1]), [1, num_classes - 1])
  sampled_class_labels_one_hot_fg *= iou_coefs
  sampled_class_labels_one_hot_bg = tf.maximum(1.0 - tf.math.reduce_sum(
      sampled_class_labels_one_hot_fg, axis=1, keepdims=True), 0.0)
  sampled_class_labels_one_hot = tf.concat(
      [sampled_class_labels_one_hot_bg, sampled_class_labels_one_hot_fg],
      axis=1)
  params = {}
  if is_balanced:
    weights = loss_utils.get_balanced_loss_weights_multiclass(
        labels=tf.expand_dims(sampled_instance_ids, axis=1))
    params['weights'] = weights
  return classification_loss_fn(
      logits=sampled_logits, labels=sampled_class_labels_one_hot, **params)


def classification_loss_using_mask_iou_func(embeddings,
                                            logits,
                                            instance_ids,
                                            class_labels,
                                            num_samples,
                                            valid_mask=None,
                                            max_instance_id=None,
                                            similarity_strategy='dotproduct',
                                            is_balanced=True):
  """Classification loss using mask iou.

  Args:
    embeddings: A tf.float32 tensor of size [batch_size, n, f].
    logits: A tf.float32 tensor of size [batch_size, n, num_classes]. It is
      assumed that background is class 0.
    instance_ids: A tf.int32 tensor of size [batch_size, n].
    class_labels: A tf.int32 tensor of size [batch_size, n]. It is assumed
      that the background voxels are assigned to class 0.
    num_samples: An int determining the number of samples.
    valid_mask: A tf.bool tensor of size [batch_size, n] that is True when an
      element is valid and False if it needs to be ignored. By default the value
      is None which means it is not applied.
    max_instance_id: If set, instance ids larger than that value will be
      ignored. If not set, it will be computed from instance_ids tensor.
    similarity_strategy: Defines the method for computing similarity between
                         embedding vectors. Possible values are 'dotproduct' and
                         'distance'.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for foreground vs. background voxels.

  Returns:
    A tf.float32 scalar loss tensor.
  """
  batch_size = embeddings.get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError('Unknown batch size at graph construction time.')
  if max_instance_id is None:
    max_instance_id = tf.reduce_max(instance_ids)
  class_labels = tf.reshape(class_labels, [batch_size, -1, 1])
  sampled_embeddings, sampled_instance_ids, sampled_indices = (
      sampling_utils.balanced_sample(
          features=embeddings,
          instance_ids=instance_ids,
          num_samples=num_samples,
          valid_mask=valid_mask,
          max_instance_id=max_instance_id))
  losses = []
  for i in range(batch_size):
    embeddings_i = embeddings[i, :, :]
    instance_ids_i = instance_ids[i, :]
    class_labels_i = class_labels[i, :, :]
    logits_i = logits[i, :]
    sampled_embeddings_i = sampled_embeddings[i, :, :]
    sampled_instance_ids_i = sampled_instance_ids[i, :]
    sampled_indices_i = sampled_indices[i, :]
    sampled_class_labels_i = tf.gather(class_labels_i, sampled_indices_i)
    sampled_logits_i = tf.gather(logits_i, sampled_indices_i)
    if valid_mask is not None:
      valid_mask_i = valid_mask[i]
      embeddings_i = tf.boolean_mask(embeddings_i, valid_mask_i)
      instance_ids_i = tf.boolean_mask(instance_ids_i, valid_mask_i)
    loss_i = classification_loss_using_mask_iou_func_unbatched(
        embeddings=embeddings_i,
        instance_ids=instance_ids_i,
        sampled_embeddings=sampled_embeddings_i,
        sampled_instance_ids=sampled_instance_ids_i,
        sampled_class_labels=sampled_class_labels_i,
        sampled_logits=sampled_logits_i,
        similarity_strategy=similarity_strategy,
        is_balanced=is_balanced)
    losses.append(loss_i)
  return tf.math.reduce_mean(tf.stack(losses))


@gin.configurable(
    'classification_loss_using_mask_iou', denylist=['inputs', 'outputs'])
def classification_loss_using_mask_iou(inputs,
                                       outputs,
                                       num_samples,
                                       max_instance_id=None,
                                       similarity_strategy='distance',
                                       is_balanced=True,
                                       is_intermediate=False):
  """Classification loss with an iou threshold.

  Args:
    inputs: A dictionary that contains
      num_valid_voxels - A tf.int32 tensor of size [batch_size].
      instance_ids - A tf.int32 tensor of size [batch_size, n].
      class_labels - A tf.int32 tensor of size [batch_size, n]. It is assumed
        that the background voxels are assigned to class 0.
    outputs: A dictionart that contains
      embeddings - A tf.float32 tensor of size [batch_size, n, f].
      logits - A tf.float32 tensor of size [batch_size, n, num_classes]. It is
        assumed that background is class 0.
    num_samples: An int determining the number of samples.
    max_instance_id: If set, instance ids larger than that value will be
      ignored. If not set, it will be computed from instance_ids tensor.
    similarity_strategy: Defines the method for computing similarity between
                         embedding vectors. Possible values are 'dotproduct' and
                         'distance'.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for foreground vs. background voxels.
    is_intermediate: True if applied to intermediate predictions;
      otherwise, False.

  Returns:
    A tf.float32 scalar loss tensor.
  """
  instance_ids_key = standard_fields.InputDataFields.object_instance_id_voxels
  class_labels_key = standard_fields.InputDataFields.object_class_voxels
  num_voxels_key = standard_fields.InputDataFields.num_valid_voxels
  if is_intermediate:
    embedding_key = (
        standard_fields.DetectionResultFields
        .intermediate_instance_embedding_voxels)
    logits_key = (
        standard_fields.DetectionResultFields
        .intermediate_object_semantic_voxels)
  else:
    embedding_key = (
        standard_fields.DetectionResultFields.instance_embedding_voxels)
    logits_key = standard_fields.DetectionResultFields.object_semantic_voxels
  if instance_ids_key not in inputs:
    raise ValueError('instance_ids is missing in inputs.')
  if class_labels_key not in inputs:
    raise ValueError('class_labels is missing in inputs.')
  if num_voxels_key not in inputs:
    raise ValueError('num_voxels is missing in inputs.')
  if embedding_key not in outputs:
    raise ValueError('embedding is missing in outputs.')
  if logits_key not in outputs:
    raise ValueError('logits is missing in outputs.')
  batch_size = inputs[num_voxels_key].get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError('batch_size is not defined at graph construction time.')
  num_valid_voxels = inputs[num_voxels_key]
  num_voxels = tf.shape(inputs[instance_ids_key])[1]
  valid_mask = tf.less(
      tf.tile(tf.expand_dims(tf.range(num_voxels), axis=0), [batch_size, 1]),
      tf.expand_dims(num_valid_voxels, axis=1))
  return classification_loss_using_mask_iou_func(
      embeddings=outputs[embedding_key],
      logits=outputs[logits_key],
      instance_ids=tf.reshape(inputs[instance_ids_key], [batch_size, -1]),
      class_labels=inputs[class_labels_key],
      num_samples=num_samples,
      valid_mask=valid_mask,
      max_instance_id=max_instance_id,
      similarity_strategy=similarity_strategy,
      is_balanced=is_balanced)


def _voxel_hard_negative_classification_loss_unbatched(
    inputs_1, outputs_1, is_intermediate, gamma):
  """Loss function for input and outputs of batch size 1."""
  inputs_1, outputs_1 = _get_voxels_valid_inputs_outputs(
      inputs_1=inputs_1, outputs_1=outputs_1)
  if is_intermediate:
    logits = outputs_1[standard_fields.DetectionResultFields
                       .intermediate_object_semantic_voxels]
  else:
    logits = outputs_1[
        standard_fields.DetectionResultFields.object_semantic_voxels]
  labels = tf.reshape(
      inputs_1[standard_fields.InputDataFields.object_class_voxels], [-1])
  background_mask = tf.equal(labels, 0)
  num_background_points = tf.reduce_sum(
      tf.cast(background_mask, dtype=tf.int32))

  def loss_fn():
    """Loss function."""
    num_classes = logits.get_shape().as_list()[-1]
    if num_classes is None:
      raise ValueError('Number of classes is unknown.')
    masked_logits = tf.boolean_mask(logits, background_mask)
    masked_weights = tf.pow(
        1.0 - tf.reshape(tf.nn.softmax(masked_logits)[:, 0], [-1, 1]), gamma)
    num_points = tf.shape(masked_logits)[0]
    masked_weights = masked_weights * tf.cast(
        num_points, dtype=tf.float32) / tf.reduce_sum(masked_weights)
    masked_labels_one_hot = tf.one_hot(
        indices=tf.boolean_mask(labels, background_mask), depth=num_classes)
    loss = classification_loss_fn(
        logits=masked_logits,
        labels=masked_labels_one_hot,
        weights=masked_weights)
    return loss

  cond = tf.logical_and(
      tf.greater(num_background_points, 0), tf.greater(tf.shape(labels)[0], 0))
  return tf.cond(cond, loss_fn, lambda: tf.constant(0.0, dtype=tf.float32))


@gin.configurable
def hard_negative_classification_loss(
    inputs,
    outputs,
    is_intermediate=False,
    gamma=1.0):
  """Calculates the loss based on predicted center distance from gt center.

  Computes the loss using the object properties of the voxel tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input label data.
    outputs: A dictionary of tf.Tensors with the network output.
    is_intermediate: If True, loss will be computed on intermediate tensors.
    gamma: Gamma similar to how it is used in focal loss.

  Returns:
    loss: A tf.float32 scalar corresponding to distance confidence loss.
  """
  return loss_utils.apply_unbatched_loss_on_voxel_tensors(
      inputs=inputs,
      outputs=outputs,
      unbatched_loss_fn=functools.partial(
          _voxel_hard_negative_classification_loss_unbatched,
          is_intermediate=is_intermediate,
          gamma=gamma))
