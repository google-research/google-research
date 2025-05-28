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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Near Duplicate Constrastive Learning Retrieval Model."""

import numpy as np
import tensorflow as tf
from tensorflow_addons import image as tfa_image

layers = tf.keras.layers
Model = tf.keras.Model
gaussian_filter2d = tfa_image.gaussian_filter2d
eps = 1e-6

contrastive_augmenter = {
    "brightness": 0.8,
    "name": "contrastive_augmenter",
    "scale": (0.1, 1.0),
}


class L2Normalization(tf.keras.layers.Layer):
  """Normalization layer using L2 norm."""

  def __init__(self):
    """Initialization of the L2Normalization layer."""
    super().__init__()
    # A lower bound value for the norm.
    self.eps = eps

  def call(self, x, axis=1):
    """Invokes the L2Normalization instance.

    Args:
      x: A Tensor.
      axis: Dimension along which to normalize. A scalar or a vector of
        integers.

    Returns:
      norm: A Tensor with the same shape as `x`.
    """
    return tf.nn.l2_normalize(x, axis, epsilon=self.eps)


def ssl_local_loss(embs, t_embs, t_indices, temperature=0.5):
  """Calculate the local loss.

  Args:
    embs:
    t_embs:
    t_indices:
    temperature:

  Returns:

  """
  # Reshape input.
  # pylint: disable=invalid-name
  B, _, _, D = embs.shape
  # pylint: enable=invalid-name
  embs = tf.reshape(embs, [B, -1, D])
  t_embs = tf.reshape(t_embs, [B, -1, D])
  t_indices = tf.reshape(t_indices, [B, -1])

  # Normalize embs.
  # pylint: disable=invalid-name
  L2Norm = L2Normalization()
  # pylint: enable=invalid-name
  embs = L2Norm(embs, axis=2)
  t_embs = L2Norm(t_embs, axis=2)
  t_indices = tf.cast(t_indices, tf.int32)

  valid_inds = t_indices >= 0

  # Fix indices.
  # Invalid index were set to -1.
  t_indices = tf.where(valid_inds, t_indices, 0)
  maped_embs = tf.gather(embs, t_indices, batch_dims=1)

  # Calculate Similarity matrix.
  sim_matrix = maped_embs @ tf.transpose(
      t_embs, perm=[0, 2, 1])  # B, nBlock*nBlocks

  logits = tf.exp(sim_matrix / temperature)

  matches = tf.eye(
      num_rows=sim_matrix.shape[1],
      num_columns=sim_matrix.shape[2],
      batch_shape=[sim_matrix.shape[0]],
      dtype=tf.bool)

  not_matches = tf.cast(~matches, tf.float32)
  # Only consider valid tranformed pixels.
  # i.e. pixels that actually were transformted from imgA to imgB.
  # Valid_inds=(B, M*N).
  valid_inds = tf.expand_dims(valid_inds, axis=2)

  logits = tf.where(valid_inds, logits, 0)

  # Calculate partitions.
  partitions = tf.expand_dims(
      tf.reduce_sum(not_matches * logits, axis=1) + eps, axis=1)
  partitions = tf.where(valid_inds, partitions, eps)

  # Validate Blocks that should be matched with the augmented masks.
  matches = tf.where(valid_inds, not_matches == 0, False)
  matches = tf.cast(matches, tf.float32)

  # Trying to Smooth likehood of each block with gaussian filter.
  # matches = gaussian_filter2d(
  #        tf.reshape(matches, (B*M*N, M, N, 1)),
  #        filter_shape=3,
  #        sigma=0.425)
  # matches = tf.reshape(matches, (B, M*N, M*N))

  # Calculate probs.
  probabilities = logits / (partitions + eps)
  matches_probs = -tf.math.log(probabilities) * matches
  # Considerate only valid probs
  matches_probs = tf.where(valid_inds, matches_probs, 0)

  valid_inds = tf.cast(valid_inds, tf.float32)

  loss = tf.reduce_sum(matches_probs) / (tf.reduce_sum(valid_inds) + eps)
  return loss


def ssl_loss(embeddings,
             labels,
             infonce_temperature=0.05,
             entropy_weight=30,
             mixup=False):
  """Calculate a contrastive loss added by entropy."""

  # Calculate cosine similairty,
  similarity_matrix = embeddings @ tf.transpose(embeddings, perm=[1, 0])

  # Get logits.
  logits = tf.exp(similarity_matrix / infonce_temperature)

  # Get Match matrix based on the provided labels.
  match_matrix = tf.cast(
      tf.identity(labels),
      dtype=tf.bool)  # Copied items are set to True in the same line (Batch).
  non_matches_matrix = ~match_matrix
  non_matches_matrix = tf.cast(non_matches_matrix, tf.float32)
  identity = tf.eye(labels.shape[0], dtype=tf.bool)
  nontrivial_matches = tf.cast(
      tf.math.logical_and(labels, (~identity)), tf.float32)

  # Calculate partitions.
  partitions = logits + (tf.reduce_sum(non_matches_matrix * logits, axis=1) +
                         eps)  # Sum of all logits per line
  probabilities = logits / partitions  # This is not summing one???

  # Calculate InfoNCE loss.
  batch_size = similarity_matrix.shape[0]
  if mixup:

    nontrivial_sum = tf.reduce_sum(tf.cast(nontrivial_matches, tf.float32))
    infonce_loss = tf.reduce_sum(
        -tf.math.log(probabilities) * nontrivial_matches) / nontrivial_sum
  else:
    infonce_loss = tf.reduce_sum(
        -tf.math.log(probabilities) * nontrivial_matches) / batch_size

  # Calculate entropy_loss.
  small_value = tf.constant(
      -100.0, dtype=tf.float32)  # any value > max L2 normalized distance.
  max_non_match_sim = tf.reduce_max(
      tf.where(
          tf.cast(non_matches_matrix, tf.bool), similarity_matrix, small_value),
      axis=1  #, keepdims=True
  )
  closest_distance = tf.sqrt(
      tf.clip_by_value(
          2 - (2 * max_non_match_sim), clip_value_min=1e-6,
          clip_value_max=1)  # Max = 1 since it is a similairity score.
  )
  entropy_loss = -tf.reduce_mean(
      tf.math.log(closest_distance), axis=0) * entropy_weight

  loss = infonce_loss + entropy_loss
  return loss, infonce_loss, entropy_loss


class Backbone(Model):

  def __init__(self, backbone=None):
    super().__init__()
    self._backbone_name = backbone

  def _build(self):
    if (self._backbone_name is None) or (self._backbone_name == "ResNet50"):
      self._backbone_name = "ResNet50"
      return tf.keras.applications.ResNet50V2(
          # weights="imagenet",
          weights=None,
          include_top=False,
          pooling=None,
      )


class NDCLR(Model):
  """Near-duplicate retrieval constrastive model."""

  def __init__(self,
               block_local_dims=8,
               local_emb_dims=2048,
               backbone=None,
               batch_size=16,
               conv_output_layer="conv4_block5_out"):
    super().__init__()
    self._backbone_name = backbone
    self.backbone = Backbone(backbone)
    self.backbone = self.backbone._build()
    self.backbone.trainable = True

    self.local_feat_layer = Model(
        inputs=self.backbone.input,
        outputs=self.backbone.get_layer(conv_output_layer).output)

    # Adds a hidden_layer to the model, besides the projection one.
    # self.hidden_layer = layers.Dense(
    #     units=2048, activation='relu', name="hidden_layer")
    self.projector = layers.Dense(
        units=512, activation="relu", name="projector")
    # pylint: disable=invalid-name
    self.L2Norm = L2Normalization()
    # pylint: enable=invalid-name
    self.flatten = layers.Flatten()
    self.clip_val = tf.constant(10.0)

    # Classification layers.
    # Adds a classification layer, when trained with labeled data.
    # self.classification = layers.Dense(1000, name="outputs")
    # self.softmax = layers.Softmax()

    # Layers from the idea of combining local features.
    # Trying to combine the local features from different convolutional layers.
    # self.local_feat_conv2 = Model(
    #        inputs=self.backbone.input,
    #       outputs=self.backbone.get_layer('conv2_block3_out').output)
    # self.local_feat_conv3 = Model(
    #        inputs=self.backbone.input,
    #        outputs=self.backbone.get_layer('conv3_block3_out').output)
    # self.local_feat_linear = layers.Dense(256, name="local_feat_linear")

    # Online augmentation.
    # self.ssl_augmenter = augmenter(**contrastive_augmenter)
    # self.augmenter = augmenter
    # self.batch_size = batch_size
    # self.class_loss = tf.keras.losses.CategoricalCrossentropy(
    #     reduction=tf.keras.losses.Reduction.NONE)

    # Create model masks.
    self.local_emb_dims = local_emb_dims
    self.block_local_dims = block_local_dims
    self.masks = np.arange(0, self.block_local_dims**2).reshape(
        (1, self.block_local_dims, self.block_local_dims))
    self.masks = np.repeat(self.masks, batch_size, axis=0)

    # Consider a input embedding (B, 16,16,1024).
    # Deconvolution Layer.
    # self.local_decode = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(16, 16, 1024)),
    #         tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    #         tf.keras.layers.Reshape(target_shape=(16, 16, 256)),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=64, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=32, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         # No activation
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=128, kernel_size=3, strides=1, padding='same'),
    #     ]
    # )

  def gem(self, x, axis=None, power=3.):
    """Performs generalized mean pooling (GeM).

    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
      power: Float, power > 0 is an inverse exponent parameter (GeM power).

    Returns:
      output: [B, D] A float32 Tensor.
    """
    # the original gem of SSCD is a GlobalGeMPool2d.
    # https://github.com/facebookresearch/sscd-copy-detection/blob/main/sscd/models/gem_pooling.py
    if axis is None:
      axis = [1, 2]
    tmp = tf.pow(tf.maximum(x, eps), power)
    out = tf.pow(tf.reduce_mean(tmp, axis=axis, keepdims=False), 1. / power)
    return out

  # Combining the local features during the local locass calculation.
  # def call_local(self, inputs, training=True):
  #  emb_conv2 = self.local_feat_conv2(inputs, training=training)
  #  emb_conv3 = self.local_feat_conv3(inputs, training=training)

  #  #emb = tf.concat([emb_conv2, emb_conv3], axis=-1)
  #  #emb = self.local_feat_linear(emb, training=training)
  #  #return emb
  #  #emb = tf.expand_dims(
  #  #     tf.norm(emb_conv2 , axis=-1) + tf.norm(emb_conv3, axis=-1),
  #  #     axis=-1)
  #  #return emb
  #
  #  #emb = tf.concat([emb_conv2, emb_conv3], axis=-1)
  #  #return emb

  #  emb_conv3 = self.local_feat_linear(emb_conv3, training=training)
  #  return emb_conv2 + emb_conv3

  def __call__(self, inputs, training=True):

    # Backbone
    x = self.backbone(inputs, training=training)  # Shape (B, H, W, C)
    # Shape (B, 16, 16, 2048)
    # Exploring the model with and without hidden_layer
    # x = self.hidden_layer(x, training=training)
    x = self.gem(x)
    x = self.projector(x, training=training)
    embeddings = self.L2Norm(x)

    # When training with classisfication head:
    # logits = self.classification(embeddings, training=training)
    # return self.softmax(logits, training=training), embeddings

    # Without a classification head:
    return embeddings

  def prepare_gt(self, batch_size, mixup=False):
    if mixup:
      gt = np.concatenate([np.eye(batch_size), np.eye(batch_size)], axis=0)
      gt = np.concatenate([gt, gt], axis=1)
      gt = np.flipud(gt) / 2 + gt
      gt = tf.convert_to_tensor(gt, dtype=tf.bool)
      return gt

    gt = tf.concat([tf.eye(batch_size), tf.eye(batch_size)], axis=0)
    gt = tf.concat([gt, gt], axis=1)
    return tf.cast(gt, dtype=tf.bool)
