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

"""Preprocessing ops for Multiscale Transformer."""

from typing import List, Optional, Tuple, Union

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2


def _ceil_divide_int(x,
                     y):
  """Returns ceil(x / y) as tf.int32."""
  z = tf.truediv(x, y)
  tf.debugging.check_numerics(
      z, message='_ceil_divide_int output is NaN or Inf.')
  z = tf.math.ceil(z)
  z = tf.cast(z, dtype=tf.int32)
  return z


def resize_preserve_aspect_ratio(
    image, h, w,
    longer_side_length
):
  """Aspect-ratio-preserving resizing with tf.image.ResizeMethod.GAUSSIAN.

  Args:
    image: The image tensor (h, w, c).
    h: Height of the input image.
    w: Width of the input image.
    longer_side_length: The length of the longer side after resizing.

  Returns:
    A tuple of [Image after resizing, Resized height, Resized width].

  """
  # Computes the height and width after aspect-ratio-preserving resizing.
  ratio = (
      tf.cast(longer_side_length, tf.float32) /
      tf.cast(tf.maximum(h, w), tf.float32))
  tf.debugging.check_numerics(ratio, message='Resize ratio is NaN or Inf.')
  rh = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
  rw = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

  resized = tf2.image.resize(
      image, (rh, rw), method=tf2.image.ResizeMethod.GAUSSIAN)
  resized = tf.image.convert_image_dtype(resized, dtype=image.dtype)
  return resized, rh, rw


def _pad_or_cut_to_max_seq_len(x,
                               max_seq_len):
  """Pads (or cuts) patch tensor `max_seq_len`.

  Args:
      x: input tensor of shape (n_crops, num_patches, c).
      max_seq_len: max sequence length.

  Returns:
      The padded or cropped tensor of shape (n_crops, max_seq_len, c).
  """
  # Shape of x (n_crops, num_patches, c)
  assert len(tf.shape(x)) == 3
  # Padding makes sure that # patches > max_seq_length. Note that it also
  # makes the input mask zero for shorter input.
  paddings = tf.zeros([tf.shape(x)[0], max_seq_len,
                       tf.shape(x)[-1]],
                      dtype=x.dtype)
  x = tf.concat([x, paddings], axis=1)
  # Cuts to max_seq_len number of patches.
  x = x[:, :max_seq_len, :]
  return x


def get_hashed_spatial_pos_emb_index(grid_size, count_h,
                                     count_w):
  """Get hased spatial pos embedding index for each patch.

  The size H x W is hashed to grid_size x grid_size.

  Args:
    grid_size: grid size G for the hashed-based spatial positional embedding.
    count_h: number of patches in each row for the image.
    count_w: number of patches in each column for the image.

  Returns:
    hashed position of shape (HxW, 1). Each value corresponded to the hashed
    position index in [0, grid_size x grid_size).

  """
  pos_emb_grid = tf.range(grid_size, dtype=tf.int32)
  pos_emb_grid = tf.reshape(pos_emb_grid, [grid_size, 1, 1])
  pos_emb_hash_w = tf.image.resize(
      pos_emb_grid, [count_w, 1],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      name='pos_emb_hash_w')[:, 0, 0]
  pos_emb_hash_w = tf.cast(pos_emb_hash_w, dtype=tf.int32)
  pos_emb_hash_w = tf.expand_dims(pos_emb_hash_w, axis=0)
  pos_emb_hash_w = tf.tile(pos_emb_hash_w, (count_h, 1))

  pos_emb_hash_h = tf.image.resize(
      pos_emb_grid, [count_h, 1],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      name='pos_emb_hash_h')[:, 0, 0]
  pos_emb_hash_h = tf.cast(pos_emb_hash_h, dtype=tf.int32)
  pos_emb_hash_h = tf.cast(pos_emb_hash_h, dtype=tf.int32)
  pos_emb_hash_h = tf.expand_dims(pos_emb_hash_h, axis=1)
  pos_emb_hash_h = tf.tile(pos_emb_hash_h, (1, count_w))

  pos_emb_hash = pos_emb_hash_h * grid_size + pos_emb_hash_w

  # Shape (num_patches, 1)
  pos_emb_hash = tf.reshape(pos_emb_hash, (-1, 1))
  pos_emb_hash = tf.cast(pos_emb_hash, tf.float32)
  return pos_emb_hash


def _extract_patches_and_positions_from_image(
    image, patch_size, patch_stride, hse_grid_size,
    n_crops, h, w,
    c, scale_id, max_seq_len):
  """Extracts patches and positional embedding lookup indexes for a given image.

  Args:
    image: the input image of shape [n_crops, h, w, c]
    patch_size: the extracted patch size.
    patch_stride: stride for extracting patches.
    hse_grid_size: grid size for hash-based spatial positional embedding.
    n_crops: number of crops from the input image.
    h: height of the image.
    w: width of the image.
    c: number of channels for the image.
    scale_id: the scale id for the image in the multi-scale representation.
    max_seq_len: maximum sequence length for the number of patches. If
      max_seq_len = 0, no patch is returned. If max_seq_len < 0 then we return
      all the patches.

  Returns:
    A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
    is (n_crops, num_patches, patch_size * patch_size * c + 3).
  """
  p = tf.image.extract_patches(
      image, [1, patch_size, patch_size, 1], [1, patch_stride, patch_stride, 1],
      [1, 1, 1, 1],
      padding='SAME')

  p = tf.reshape(p, [n_crops, -1, patch_size * patch_size * c])

  count_h = _ceil_divide_int(h, patch_stride)
  count_w = _ceil_divide_int(w, patch_stride)

  # Shape (num_patches, 1)
  spatial_p = get_hashed_spatial_pos_emb_index(hse_grid_size, count_h, count_w)
  # Shape (1, num_patches, 1)
  spatial_p = tf.expand_dims(spatial_p, axis=0)
  # Shape (n_crops, num_patches, 1)
  spatial_p = tf.tile(spatial_p, (n_crops, 1, 1))
  spatial_p = tf.cast(spatial_p, dtype=p.dtype)
  # Shape (n_crops, num_patches, 1)
  scale_p = tf.ones_like(spatial_p, dtype=p.dtype) * scale_id
  # Shape (n_crops, num_patches, 1)
  mask_p = tf.ones_like(spatial_p, dtype=p.dtype)

  # Concatenating is a hacky way to pass both patches, positions and input
  # mask to the model.
  # Shape (n_crops, num_patches, patch_size * patch_size * c + 3)
  out = tf.concat([p, spatial_p, scale_p, mask_p], axis=2)
  if max_seq_len >= 0:
    out = _pad_or_cut_to_max_seq_len(out, max_seq_len)
    out = tf.reshape(out,
                     [n_crops, max_seq_len, c * patch_size * patch_size + 3])
  else:
    out = tf.reshape(out, [n_crops, -1, c * patch_size * patch_size + 3])
  return out


def get_multiscale_patches(
    image,
    patch_size,
    patch_stride,
    hse_grid_size,
    longer_side_lengths,
    max_seq_len_from_original_res = None):
  """Extracts image patches from multi-scale representation.

  Args:
    image: the input image either [n_crops, h, w, 3] or [h, w, 3].
      We only handle 3-channel images for now.
    patch_size: patch size.
    patch_stride: patch stride.
    hse_grid_size: Hash-based positional embedding grid size.
    longer_side_lengths: List of longer-side lengths for each scale in the
      multi-scale representation.
    max_seq_len_from_original_res: Maximum number of patches extracted from
      original resolution. <0 means use all the patches from the original
      resolution. None means we don't use original resolution input.

  Returns:
    A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
    is (n_crops, num_patches, patch_size * patch_size * c + 3).
  """
  # Sorting the list to ensure a deterministic encoding of the scale position.
  longer_side_lengths = sorted(longer_side_lengths)

  # Input channels.
  c = 3
  if len(image.get_shape().as_list()) == 3:
    n_crops = 1
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = tf.expand_dims(image, axis=0)
  else:
    n_crops, h, w = (tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2])

  outputs = []
  for scale_id, longer_size in enumerate(longer_side_lengths):
    resized_image, rh, rw = resize_preserve_aspect_ratio(
        image, h, w, longer_size)

    max_seq_len = int(np.ceil(longer_size / patch_stride)**2)
    out = _extract_patches_and_positions_from_image(resized_image, patch_size,
                                                    patch_stride, hse_grid_size,
                                                    n_crops, rh, rw, c,
                                                    scale_id, max_seq_len)
    outputs.append(out)

  if max_seq_len_from_original_res is not None:
    out = _extract_patches_and_positions_from_image(
        image, patch_size, patch_stride, hse_grid_size, n_crops, h, w, c,
        len(longer_side_lengths), max_seq_len_from_original_res)
    outputs.append(out)

  # Shape: (n_crops, num_total_patches, patch_size * patch_size * c + 3)
  outputs = tf.concat(outputs, axis=1)
  if n_crops == 1:
    # Shape: (num_total_patches, patch_size * patch_size * c + 3).
    # Training mode. 4 dim wasn't handled by loss.
    outputs = outputs[0]
  return outputs


def normalize_value_range(image,
                          vmin = -1,
                          vmax = 1,
                          in_min = 0,
                          in_max = 255.0,
                          clip_values = False):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    image: the input image tensor.
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.

  Returns:
    Rescaled image.
  """
  if in_min >= in_max or vmin >= vmax:
    raise ValueError('min must be strictly less than max')

  in_min_t = tf.constant(in_min, tf.float32)
  in_max_t = tf.constant(in_max, tf.float32)
  image = tf.cast(image, tf.float32)
  image = (image - in_min_t) / (in_max_t - in_min_t)
  image = vmin + image * (vmax - vmin)
  if clip_values:
    image = tf.clip_by_value(image, vmin, vmax)
  return image


def decode_image(encoded_image_bytes):
  """Decodes an image.

  Args:
    encoded_image_bytes: encoded image string.

  Returns:
    The decoded image tensor [H, W, 3].
  """
  return tf.image.decode_jpeg(encoded_image_bytes)


def get_preprocess_fn(**preprocessing_kwargs):
  """Gets the preprocessing function for Multiscale Image Quality Transformer.

  The output preprocessing function expects a dictinary as input. This
  dictionary should have a key "image" that corresponds to a 3D tensor
  (height x width x channel).

  Args:
    **preprocessing_kwargs: kwargs passed to get_multiscale_patches.

  Returns:
    preprocessing function.

  Raises:
    ValueError: if input data
  """

  def _preprocess_fn(data):
    """The preprocessing function that is returned."""

    # Validate input
    if not isinstance(data, dict) or 'image' not in data:
      raise ValueError('Argument `data` must be a dictionary, '
                       'not %s' % str(type(data)))

    # Apply all the individual steps in sequence.
    logging.info('Data before pre-processing:\n%s', data)
    image = data['image']
    image = decode_image(image)
    image = normalize_value_range(image)
    image = get_multiscale_patches(image, **preprocessing_kwargs)

    data['image'] = image
    logging.info('Data after pre-processing:\n%s', data)
    return data

  return _preprocess_fn
