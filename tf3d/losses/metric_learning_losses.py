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

"""Losses that are useful for instance segmentation."""

import gin
import gin.tf
import tensorflow.compat.v1 as tf
from tf3d import standard_fields
from tf3d.utils import metric_learning_losses
from tf3d.utils import sampling_utils


def npair_loss_func(embeddings,
                    instance_ids,
                    num_samples,
                    valid_mask=None,
                    max_instance_id=None,
                    similarity_strategy='dotproduct',
                    loss_strategy='softmax'):
  """N-pair metric learning loss for learning feature embeddings.

  Args:
    embeddings: A tf.float32 tensor of size [batch_size, n, f].
    instance_ids: A tf.int32 tensor of size [batch_size, n].
    num_samples: An int determinig the number of samples.
    valid_mask: A tf.bool tensor of size [batch_size, n] that is True when an
      element is valid and False if it needs to be ignored. By default the value
      is None which means it is not applied.
    max_instance_id: If set, instance ids larger than that value will be
      ignored. If not set, it will be computed from instance_ids tensor.
    similarity_strategy: Defines the method for computing similarity between
                         embedding vectors. Possible values are 'dotproduct' and
                         'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.

  Returns:
    A tf.float32 scalar loss tensor.
  """
  batch_size = embeddings.get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError('Unknown batch size at graph construction time.')
  if max_instance_id is None:
    max_instance_id = tf.reduce_max(instance_ids)
  sampled_embeddings, sampled_instance_ids, _ = sampling_utils.balanced_sample(
      features=embeddings,
      instance_ids=instance_ids,
      num_samples=num_samples,
      valid_mask=valid_mask,
      max_instance_id=max_instance_id)
  losses = []
  for i in range(batch_size):
    sampled_instance_ids_i = sampled_instance_ids[i, :]
    sampled_embeddings_i = sampled_embeddings[i, :, :]
    min_ids_i = tf.math.reduce_min(sampled_instance_ids_i)
    max_ids_i = tf.math.reduce_max(sampled_instance_ids_i)
    target_i = tf.one_hot(
        sampled_instance_ids_i,
        depth=(max_instance_id + 1),
        dtype=tf.float32)

    # pylint: disable=cell-var-from-loop
    def npair_loss_i():
      return metric_learning_losses.npair_loss(
          embedding=sampled_embeddings_i,
          target=target_i,
          similarity_strategy=similarity_strategy,
          loss_strategy=loss_strategy)
# pylint: enable=cell-var-from-loop

    loss_i = tf.cond(
        max_ids_i > min_ids_i, npair_loss_i,
        lambda: tf.constant(0.0, dtype=tf.float32))
    losses.append(loss_i)
  return tf.math.reduce_mean(losses)


@gin.configurable('npair_loss', blacklist=['inputs', 'outputs'])
def npair_loss(inputs,
               outputs,
               num_samples,
               max_instance_id=None,
               similarity_strategy='distance',
               loss_strategy='softmax',
               is_intermediate=False):
  """N-pair metric learning loss for learning feature embeddings.

  Args:
    inputs: A dictionary that contains
      instance_ids - A tf.int32 tensor of size [batch_size, n].
      valid_mask - A tf.bool tensor of size [batch_size, n] that is True when an
        element is valid and False if it needs to be ignored. By default the
        value is None which means it is not applied.
    outputs: A dictionary that contains
      embeddings - A tf.float32 tensor of size [batch_size, n, f].
    num_samples: An int determinig the number of samples.
    max_instance_id: If set, instance ids larger than that value will be
      ignored. If not set, it will be computed from instance_ids tensor.
    similarity_strategy: Defines the method for computing similarity between
                         embedding vectors. Possible values are 'dotproduct' and
                         'distance'.
    loss_strategy: Defines the type of loss including 'softmax' or 'sigmoid'.
    is_intermediate: True if applied to intermediate predictions;
      otherwise, False.

  Returns:
    A tf.float32 scalar loss tensor.
  """
  instance_ids_key = standard_fields.InputDataFields.object_instance_id_voxels
  num_voxels_key = standard_fields.InputDataFields.num_valid_voxels
  if is_intermediate:
    embedding_key = (
        standard_fields.DetectionResultFields
        .intermediate_instance_embedding_voxels)
  else:
    embedding_key = (
        standard_fields.DetectionResultFields.instance_embedding_voxels)
  if instance_ids_key not in inputs:
    raise ValueError('object_instance_id_voxels is missing in inputs.')
  if num_voxels_key not in inputs:
    raise ValueError('num_voxels is missing in inputs.')
  if embedding_key not in outputs:
    raise ValueError('embedding key is missing in outputs.')
  batch_size = inputs[num_voxels_key].get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError('batch_size is not defined at graph construction time.')
  num_valid_voxels = inputs[num_voxels_key]
  num_voxels = tf.shape(inputs[instance_ids_key])[1]
  valid_mask = tf.less(
      tf.tile(tf.expand_dims(tf.range(num_voxels), axis=0), [batch_size, 1]),
      tf.expand_dims(num_valid_voxels, axis=1))
  return npair_loss_func(
      embeddings=outputs[embedding_key],
      instance_ids=tf.reshape(inputs[instance_ids_key], [batch_size, -1]),
      num_samples=num_samples,
      valid_mask=valid_mask,
      max_instance_id=max_instance_id,
      similarity_strategy=similarity_strategy,
      loss_strategy=loss_strategy)


@gin.configurable(
    'embedding_regularization_loss', blacklist=['inputs', 'outputs'])
def embedding_regularization_loss(inputs,
                                  outputs,
                                  lambda_coef=0.0001,
                                  regularization_type='unit_length',
                                  is_intermediate=False):
  """Classification loss with an iou threshold.

  Args:
    inputs: A dictionary that contains
      num_valid_voxels - A tf.int32 tensor of size [batch_size].
      instance_ids - A tf.int32 tensor of size [batch_size, n].
    outputs: A dictionart that contains
      embeddings - A tf.float32 tensor of size [batch_size, n, f].
    lambda_coef: Regularization loss coefficient.
    regularization_type: Regularization loss type. Supported values are 'msq'
      and 'unit_length'. 'msq' stands for 'mean square' which penalizes the
      embedding vectors if they have a length far from zero. 'unit_length'
      penalizes the embedding vectors if they have a length far from one.
    is_intermediate: True if applied to intermediate predictions;
      otherwise, False.

  Returns:
    A tf.float32 scalar loss tensor.
  """
  instance_ids_key = standard_fields.InputDataFields.object_instance_id_voxels
  num_voxels_key = standard_fields.InputDataFields.num_valid_voxels
  if is_intermediate:
    embedding_key = (
        standard_fields.DetectionResultFields
        .intermediate_instance_embedding_voxels)
  else:
    embedding_key = (
        standard_fields.DetectionResultFields.instance_embedding_voxels)
  if instance_ids_key not in inputs:
    raise ValueError('instance_ids is missing in inputs.')
  if embedding_key not in outputs:
    raise ValueError('embedding is missing in outputs.')
  if num_voxels_key not in inputs:
    raise ValueError('num_voxels is missing in inputs.')
  batch_size = inputs[num_voxels_key].get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError('batch_size is not defined at graph construction time.')
  num_valid_voxels = inputs[num_voxels_key]
  num_voxels = tf.shape(inputs[instance_ids_key])[1]
  valid_mask = tf.less(
      tf.tile(tf.expand_dims(tf.range(num_voxels), axis=0), [batch_size, 1]),
      tf.expand_dims(num_valid_voxels, axis=1))
  valid_mask = tf.reshape(valid_mask, [-1])
  embedding_dims = outputs[embedding_key].get_shape().as_list()[-1]
  if embedding_dims is None:
    raise ValueError(
        'Embedding dimension is unknown at graph construction time.')
  embedding = tf.reshape(outputs[embedding_key], [-1, embedding_dims])
  embedding = tf.boolean_mask(embedding, valid_mask)
  return metric_learning_losses.regularization_loss(
      embedding=embedding,
      lambda_coef=lambda_coef,
      regularization_type=regularization_type)
