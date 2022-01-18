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

"""Supervised pixel contrastive loss implementation."""

from typing import Any, Tuple, List

import tensorflow as tf
import tf_slim as slim


def generate_same_image_mask(num_pixels):
  """Generates a mask indicating if two pixels belong to the same image or not.

  Args:
    num_pixels: List with `num_image` elements representing the number of
      pixels in each image.

  Returns:
    A tensor of shape [1, num_total_pixels, num_total_pixels] indicating
      which pixel pairs belong to the same image.
  """
  image_ids = []
  num_total_pixels = 0
  for img_id, pixel_count in enumerate(num_pixels):
    image_ids += ([img_id] * pixel_count)
    num_total_pixels += pixel_count

  image_ids = tf.reshape(tf.constant(image_ids), shape=(num_total_pixels, 1))
  same_image_mask = tf.cast(
      tf.equal(image_ids, tf.transpose(image_ids)), dtype=tf.float32)
  return tf.expand_dims(same_image_mask, axis=0)


def generate_ignore_mask(labels,
                         ignore_labels):
  """Generates ignore mask used by contrastive loss.

  Args:
    labels: A tensor of shape [batch_size, height, width, 1]
    ignore_labels: A list of labels to ignore. Pixels with these labels will
      be ignored in the loss computation

  Returns:
    A tensor of shape [batch, num_pixels, num_pixels], indicating which pixel
      pairs are invalid.
  """
  ignore_mask = tf.math.reduce_any(
      tf.equal(labels, tf.constant(ignore_labels, dtype=labels.dtype)),
      axis=2, keepdims=True)
  ignore_mask = tf.cast(
      tf.logical_or(ignore_mask, tf.transpose(ignore_mask, [0, 2, 1])),
      tf.float32)
  return ignore_mask


def generate_positive_and_negative_masks(
    labels):
  """Generates positive and negative masks used by contrastive loss.

  Args:
    labels: A tensor of shape [batch_size, height, width, 1]

  Returns:
    positive_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
      which pixel pairs are positive pairs.
    negative_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
      which pixel pairs are negative pairs.
  """
  positive_mask = tf.cast(
      tf.equal(labels, tf.transpose(labels, [0, 2, 1])), tf.float32)
  negative_mask = 1 - positive_mask
  return positive_mask, negative_mask


def collapse_spatial_dimensions(inputs):
  """Collapses height and width dimensions into one dimension.

  Args:
    inputs: A tensor of shape [batch_size, height, width, num_channels]

  Returns:
    A tensor of shape [batch_size, height * width, num_channels]
  """
  batch_size, _, _, num_channels = inputs.get_shape().as_list()
  return tf.reshape(inputs, [batch_size, -1, num_channels])


def projection_head(features, num_projection_layers,
                    num_projection_channels):
  """Implements the projection head used before contrastive loss.

  This projection head uses 1x1 convolutions.

  Args:
    features: A tensor of shape [batch_size, height, width, num_input_channels]
    num_projection_layers: Number of layers in the projection head
    num_projection_channels: Number of channels used for the projection layers

  Returns:
    A tensor of shape [batch_size, num_pixels, num_proj_channels]
  """
  for ind in range(num_projection_layers - 1):
    features = slim.conv2d(features, num_projection_channels, [1, 1],
                           activation_fn=tf.nn.relu,
                           reuse=tf.compat.v1.AUTO_REUSE,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params={'is_training': True},
                           scope=f'proj/conv_proj{ind + 1}')

  features = slim.conv2d(features, num_projection_channels, [1, 1],
                         reuse=tf.compat.v1.AUTO_REUSE,
                         activation_fn=None,
                         scope='proj/conv_final')

  return tf.math.l2_normalize(features, -1)


def resize_and_project(features,
                       resize_size,
                       num_projection_layers,
                       num_projection_channels):
  """Resizes input features and passes them through a projection head.

  Args:
    features: A [batch_size, height, width, num_channels] tensor
    resize_size: A tuple of (height, width) to resize the features/labels
      before computing the loss
    num_projection_layers: Number of layers in the projection head
    num_projection_channels: Number of channels used for the projection layers

  Returns:
    A [batch_size, resize_height, resize_width, num_projection_channels] tensor
  """
  resized_features = tf.compat.v1.image.resize_bilinear(
      features, resize_size, align_corners=True)

  return projection_head(
      features=resized_features, num_projection_layers=num_projection_layers,
      num_projection_channels=num_projection_channels)


def compute_contrastive_loss(logits,
                             positive_mask,
                             negative_mask,
                             ignore_mask):
  """Contrastive loss function.

  Args:
    logits: A tensor of shape [batch, num_pixels, num_pixels] with each value
      corresponding to a pixel pair.
    positive_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
      which pixel pairs are positive pairs.
    negative_mask: A tensor of shape [batch, num_pixels, num_pixels] indicating
      which pixel pairs are negative pairs.
    ignore_mask: A tensor of shape [batch, num_pixels, num_pixels], indicating
      which pixel pairs are invalid.

  Returns:
    A scalar tensor with contrastive loss
  """
  validity_mask = 1 - ignore_mask
  positive_mask *= validity_mask
  negative_mask *= validity_mask

  exp_logits = tf.exp(logits) * validity_mask

  normalized_exp_logits = tf.math.divide_no_nan(
      exp_logits,
      exp_logits + tf.reduce_sum(exp_logits * negative_mask, 2, keepdims=True))
  neg_log_likelihood = -tf.math.log(
      normalized_exp_logits * validity_mask + ignore_mask)

  # normalize weight and sum in dimension 2
  normalized_weight = positive_mask / tf.maximum(
      1e-6, tf.reduce_sum(positive_mask, 2, keepdims=True))
  neg_log_likelihood = tf.reduce_sum(
      neg_log_likelihood * normalized_weight, axis=2)

  # normalize weight and sum in dimension 1
  positive_mask_sum = tf.reduce_sum(positive_mask, 2)
  valid_index = 1 - tf.cast(tf.equal(positive_mask_sum, 0), tf.float32)
  normalized_weight = valid_index / tf.maximum(
      1e-6, tf.reduce_sum(valid_index, 1, keepdims=True))
  neg_log_likelihood = tf.reduce_sum(
      neg_log_likelihood * normalized_weight, axis=1)
  loss = tf.reduce_mean(neg_log_likelihood)

  return loss


def supervised_pixel_contrastive_loss(features_orig,
                                      features_aug,
                                      labels_orig,
                                      labels_aug,
                                      ignore_labels,
                                      resize_size,
                                      num_projection_layers = 2,
                                      num_projection_channels = 256,
                                      temperature = 0.07,
                                      within_image_loss = False):
  """Computes pixel-level supervised contrastive loss.

  Args:
    features_orig: A [batch_size, height, width, num_channels] tensor
      representing the features extracted from the original images
    features_aug: A [batch_size, height, width, num_channels] tensor
      representing the features extracted from the augmented images
    labels_orig: A tensor of shape [batch_size, height, width, 1] representing
      the labels of original images
    labels_aug: A tensor of shape [batch_size, height, width, 1] representing
      the labels of augmented images
    ignore_labels: A list of labels to ignore. Pixels with these labels will
      be ignored in the loss computation
    resize_size: A tuple of (height, width) to resize the features/labels
      before computing the loss
    num_projection_layers: Number of layers in the projection head
    num_projection_channels: Number of channels used for the projection layers
    temperature: Temperature to use in contrastive loss
    within_image_loss: Computes within image contrastive loss is set to true,
      and cross-image contrastive loss otherwise.

  Returns:
   Contrastive loss tensor
  """
  features_orig = resize_and_project(
      features=features_orig, resize_size=resize_size,
      num_projection_layers=num_projection_layers,
      num_projection_channels=num_projection_channels)

  features_aug = resize_and_project(
      features=features_aug, resize_size=resize_size,
      num_projection_layers=num_projection_layers,
      num_projection_channels=num_projection_channels)

  labels_orig = tf.compat.v1.image.resize_nearest_neighbor(
      labels_orig, resize_size, align_corners=True)

  labels_aug = tf.compat.v1.image.resize_nearest_neighbor(
      labels_aug, resize_size, align_corners=True)

  features_orig = collapse_spatial_dimensions(features_orig)
  features_aug = collapse_spatial_dimensions(features_aug)
  labels_orig = collapse_spatial_dimensions(labels_orig)
  labels_aug = collapse_spatial_dimensions(labels_aug)

  if within_image_loss:
    within_image_loss_orig = within_image_supervised_pixel_contrastive_loss(
        features=features_orig, labels=labels_orig,
        ignore_labels=ignore_labels, temperature=temperature)

    within_image_loss_aug = within_image_supervised_pixel_contrastive_loss(
        features=features_aug, labels=labels_aug,
        ignore_labels=ignore_labels, temperature=temperature)

    return within_image_loss_orig + within_image_loss_aug

  batch_size = labels_orig.get_shape().as_list()[0]
  indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
  shuffled_indices = tf.random.shuffle(indices)
  shuffled_features_aug = tf.gather(features_aug, shuffled_indices, axis=0)
  shuffled_labels_aug = tf.gather(labels_aug, shuffled_indices, axis=0)

  return cross_image_supervised_pixel_contrastive_loss(
      features1=features_orig,
      features2=shuffled_features_aug,
      labels1=labels_orig,
      labels2=shuffled_labels_aug,
      ignore_labels=ignore_labels,
      temperature=temperature)


def within_image_supervised_pixel_contrastive_loss(
    features, labels, ignore_labels,
    temperature):
  """Computes within-image supervised pixel contrastive loss.

  Args:
    features: A tensor of shape [batch_size, num_pixels, num_channels]
    labels: A tensor of shape [batch_size, num_pixels, 1]
    ignore_labels: A list of labels to ignore. Pixels with these labels will
      be ignored in the loss computation
    temperature: Temperature to use in contrastive loss

  Returns:
   Contrastive loss tensor
  """
  logits = tf.matmul(features, features, transpose_b=True) / temperature
  positive_mask, negative_mask = generate_positive_and_negative_masks(labels)
  ignore_mask = generate_ignore_mask(labels, ignore_labels)

  return compute_contrastive_loss(
      logits, positive_mask, negative_mask, ignore_mask)


def cross_image_supervised_pixel_contrastive_loss(
    features1, features2, labels1,
    labels2, ignore_labels,
    temperature):
  """Computes cross-image suprvised pixel contrastive loss.

  Args:
    features1: A tensor of shape [batch_size, num_pixels, num_channels]
    features2: A tensor of shape [batch_size, num_pixels, num_channels]
    labels1: A tensor of shape [batch_size, num_pixels, 1]
    labels2: A tensor of shape [batch_size, num_pixels, 1]
    ignore_labels: A list of labels too ignore. Pixels with these labels will
      be ignored in the loss computation
    temperature: Temperature to use in contrastive loss

  Returns:
   Contrastive loss tensor
  """
  num_pixels1 = features1.get_shape().as_list()[1]
  num_pixels2 = features2.get_shape().as_list()[1]

  features = tf.concat([features1, features2], axis=1)
  labels = tf.concat([labels1, labels2], axis=1)

  same_image_mask = generate_same_image_mask([num_pixels1, num_pixels2])

  logits = tf.matmul(features, features, transpose_b=True) / temperature
  positive_mask, negative_mask = generate_positive_and_negative_masks(labels)
  negative_mask *= same_image_mask
  ignore_mask = generate_ignore_mask(labels, ignore_labels)

  return compute_contrastive_loss(
      logits, positive_mask, negative_mask, ignore_mask)
