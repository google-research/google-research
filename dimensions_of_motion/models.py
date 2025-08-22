# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# -*- coding: utf-8 -*-
"""Build model and compute losses."""
import loss
import nets
import tensorflow as tf


# We use resolution-independent texture coordinates that range from (0,0) in the
# top-left of an image to (1,1) in the bottom right. Thus, a principal point in
# center of the image would be at (0.5, 0.5). Similarly, a flow of (0.1, -0.5)
# would represent a pixel that moves 10% of the width of the image to the right,
# and 50% of the height of the image upwards.


def run_model(image, flow, embedding_dimension, principal_point):
  """Run model on a batch of images and and compute loss with ground truth flow.

  Args:
    image: [B, H, W, 3] RGB input images.
    flow: [B, H, W, 2] Ground truth flow.
    embedding_dimension: Number of embedding dimensions to predict, or 0.
    principal_point: [B, 2] Principal point, or None.
  Returns:
    dictionary of losses. Key 'total' maps to overall total loss.
  """
  batch_size = image.shape.as_list()[0]

  if principal_point is None:
    # Default to center of image
    principal_point = tf.tile([[0.5, 0.5]], [batch_size, 1])

  (disparity, embeddings, motion_basis_losses) = (
      nets.predict_scene_representation(image, embedding_dimension))

  (losses_and_weights, summary_images) = loss.compute_motion_loss(
      image, flow, disparity, embeddings, principal_point)

  def add_loss(key, value, weight):
    losses_and_weights[key] = (value, weight)

  # Losses that are not per-item in the batch, but global:
  disparity_activation_weight = 1e-6
  embedding_activation_weight = 1e-6
  regularization_weight = 1e-6

  add_loss('disparity_activation',
           motion_basis_losses['disparity_activation'],
           disparity_activation_weight)

  if 'embedding_activation' in motion_basis_losses:
    add_loss('embedding_activation',
             motion_basis_losses['embedding_activation'],
             embedding_activation_weight)

  add_loss('regularization',
           motion_basis_losses['regularization'],
           regularization_weight)

  # Convert all batched losses to scalars by taking the mean, and compute a
  # total loss using the weights for each loss.
  losses = {}
  weighted_losses = []
  for name, (value, weight) in losses_and_weights.items():
    if value.shape.rank > 0:
      value = tf.reduce_mean(value)
    losses[name] = value
    weighted_losses.append(value * weight)

  losses['total'] = tf.add_n(weighted_losses)

  return losses, summary_images
