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

# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cropped Positional Embedding.

Adapted from Praxis ViT layer:
https://github.com/google/praxis/blob/main/praxis/layers/vits.py
"""

import jax
import jax.numpy as jnp

from findit import spatial_transform_ops


def _interpolate_embedding_2d(emb, source_emb_shape, target_emb_shape):
  """Interpolates a 2D positional embedding to a new shape.

  Args:
    emb: JTensor, (1, HxW, D), flattened 2D positional embedding.
    source_emb_shape: Tuple, (H, W), height and width of the source embedding.
    target_emb_shape: Tuple, (H', W'), height and width of the target embedding.

  Returns:
    Interpolated embedding of shape (1, H', W', D)
  """

  if len(emb.shape) > 3 or emb.shape[0] != 1:
    raise ValueError('The shape of the embedding should be (1, H * W, D)')

  if emb.shape[1] != source_emb_shape[0] * source_emb_shape[1]:
    raise ValueError('The shape of the embedding does NOT match input specs.')

  emb_dims = emb.shape[2]
  emb = jnp.reshape(emb, (source_emb_shape[0], source_emb_shape[1], emb_dims))

  target_emb = jax.image.resize(
      emb, (target_emb_shape[0], target_emb_shape[1], emb_dims),
      method='bilinear')
  target_emb = jnp.reshape(
      target_emb, (1, target_emb_shape[0], target_emb_shape[1], emb_dims))

  return target_emb


def cropped_positional_embedding(
    pos_emb,
    pos_emb_size,
    pos_emb_crop_region,
    up_pos_emb_size,
    batch_size,
):
  """Cropped Positional Embedding.

  First, we up-sample the positional embeddings from the size typical for
  pretraining, e.g., 14x14 to that typical for detection tasks, e.g., 64x64
  (i.e., 'up_pos_emb_size'). Then we randomly crop and resize a region
  from the up-sampled positional embeddings and use that as the image-level
  positional embeddings during pretraining. The regions (i.e.,
  'pos_emb_crop_region') are uniformly sampled from the coordinates using
  Random Resized Crop (RRC) augmentation, while keeping the crop scale ratio
  in [0.1, 1.0] and aspect ratio in [0.5, 2.0].
  Intuitively, this causes the model to view an image not as a full image in
  itself, but as a region crop from some larger unknown image. This better
  matches the downstream use case of detection where recognition occurs at
  region- rather than image-level.

  Example usage (in Vision Transformer):
    features = PatchProjectionLayer(image_patches)
    batch, num_patches, dims = features.shape
    pos_emb = Param([num_patches])  # shape: [1, num_patches, dims]
    pos_emb = cropped_positional_embedding(
        pos_emb, pos_emb_size, pos_emb_crop_region, up_pos_emb_size, batch
    )  # shape: [batch, num_patches, dims]
    features = features + pos_emb
    features = TransformerBlocks(features)
    ...

  Args:
    pos_emb: Tensor [1, HxW, D], flattened 2D positional embedding.
    pos_emb_size: height/width of 2D positional embedding, e.g., 14.
    pos_emb_crop_region: Crop region tensor [B, 1, 4].
    up_pos_emb_size: Desired height/width of the upsampled 2D positional
      embedding, e.g., 64.
    batch_size: batch size.

  Returns:
    Output tensor [B, HxW, D], cropped positional embedding.
  """
  output_dims = pos_emb.shape[-1]
  pos_emb = _interpolate_embedding_2d(
      pos_emb, (pos_emb_size, pos_emb_size), (up_pos_emb_size, up_pos_emb_size))
  feature_map = {0: jnp.tile(pos_emb, [batch_size, 1, 1, 1])}
  cropped_pos_emb = spatial_transform_ops.multilevel_crop_and_resize(
      feature_map, pos_emb_crop_region, output_size=pos_emb_size)
  pos_emb = jnp.reshape(
      cropped_pos_emb, [batch_size, pos_emb_size * pos_emb_size, output_dims])
  return pos_emb
