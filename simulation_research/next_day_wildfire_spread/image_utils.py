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

"""Library of image transformation utils."""

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf


def _tensor_to_list_of_channel_tensors(img):
  """Converts a tensor with dimensions of HWC or BHWC to a list of channels."""
  if len(img.shape) == 3:  # HWC.
    img_list = [img[:, :, c] for c in range(img.shape[-1])]
  elif len(img.shape) == 4:  # BHWC.
    img_list = [img[:, :, :, c] for c in range(img.shape[-1])]
  else:
    raise ValueError(f'Input must be HWC or BHWC and not of shape {img.shape}.')
  return img_list


def _update_azimuth_flip_left_right(img,
                                    azimuth_channel):
  """For HWB input, recalculate azimuth channel after a left-right flip.

  The input img has already undergone a flip_left_right transformation and the
  azimuth channel needs to be updated. The azimuth is the direction in number of
  degrees clockwise from North. After a left-right flip of the tensor (i.e. a
  reflection along the North-South axis), the new azimuth is `(360 -
  original_azimuth)`.

  Args:
    img: Tensor with dimensions HWC or BHWC.
    azimuth_channel: Channel index for the azimuth.

  Returns:
    Tensor with recalculated azimuth channel.
  """
  img = _tensor_to_list_of_channel_tensors(img)
  img[azimuth_channel] = 360 - img[azimuth_channel]
  img = tf.stack(img, axis=-1)
  return img


def _update_azimuth_flip_up_down(img,
                                 azimuth_channel):
  """For HWB input, recalculates azimuth channel after an up down flip.

  The azimuth is the number of degrees clockwise from North.

  Args:
    img: Tensor with dimensiosn HWC or BHWC.
    azimuth_channel: Channel index for the azimuth.

  Returns:
    Tensor with recalculated azimuth channel.
  """
  img = _tensor_to_list_of_channel_tensors(img)
  new_azimuth = (180. - img[azimuth_channel]) % 360.
  img[azimuth_channel] = new_azimuth
  img = tf.stack(img, axis=-1)
  return img


def random_flip_input_and_output_images(
    input_img,
    output_img,
    azimuth_in_channel = None,
    azimuth_out_channel = None,
):
  """Randomly flip up-down and left-right input and output images.

  There are four possible flips, each with equal probability:
    1. Left-right flip
    2. Up-down flip
    3. Left-right and up-down flip
    4. No flip

  If the tensors have an azimuth channel, defined as a direction in degrees
  clockwise from North, then the azimuth channel is also updated to reflect the
  transformation. I.e. if the image is flipped, then the direction of the
  azimuth is also flipped to match the new image.

  Args:
    input_img: Tensor with dimensions HWC or BHWC.
    output_img: Tensor with dimensions HWC or BHWC.
    azimuth_in_channel: Channel index of azimuth channel in input, else `None`
      if there is no azimuth channel.
    azimuth_out_channel: Channel index of azimuth channel in output, else `None`
      if there is no azimuth channel.

  Returns:
    input_img: Tensor with dimensions HWC or BHWC.
    output_img: Tensor with dimensions HWC or BHWC.
  """
  if not ((len(input_img.shape) == 3) or (len(input_img.shape) == 4)):
    raise ValueError(
        f'Input must be HWC or BHWC and not of shape {input_img.shape}.')
  random_num = tf.random.uniform(())
  if random_num < 0.25:
    input_img = tf.image.flip_left_right(input_img)
    output_img = tf.image.flip_left_right(output_img)
    if azimuth_in_channel:
      input_img = _update_azimuth_flip_left_right(input_img, azimuth_in_channel)
    if azimuth_out_channel:
      output_img = _update_azimuth_flip_left_right(output_img,
                                                   azimuth_out_channel)
  elif random_num < 0.5:
    input_img = tf.image.flip_up_down(input_img)
    output_img = tf.image.flip_up_down(output_img)
    if azimuth_in_channel:
      input_img = _update_azimuth_flip_up_down(input_img, azimuth_in_channel)
    if azimuth_out_channel:
      output_img = _update_azimuth_flip_up_down(output_img, azimuth_out_channel)
  elif random_num < 0.75:
    input_img = tf.image.flip_left_right(input_img)
    output_img = tf.image.flip_left_right(output_img)
    input_img = tf.image.flip_up_down(input_img)
    output_img = tf.image.flip_up_down(output_img)
    if azimuth_in_channel:
      input_img = _update_azimuth_flip_left_right(input_img, azimuth_in_channel)
      input_img = _update_azimuth_flip_up_down(input_img, azimuth_in_channel)
    if azimuth_out_channel:
      output_img = _update_azimuth_flip_left_right(output_img,
                                                   azimuth_out_channel)
      output_img = _update_azimuth_flip_up_down(output_img, azimuth_out_channel)
  return input_img, output_img


def _update_azimuth_rotate90(img, azimuth_channel,
                             num_rotations):
  """Calculates new azimuth after 90 degree counterclockwise rotations.

  The input img has already undergone a `rotate90` transformation and the
  azimuth channel needs to be updated. The azimuth is the direction in number of
  degrees clockwise from North.

  Args:
    img: Tensor with dimensiosn HWC or BHWC.
    azimuth_channel: Channel index for the azimuth.
    num_rotations: Number of counterclockwise rotations of 90 degrees.

  Returns:
    Tensor with recalculated azimuth channel.
  """
  img_list = _tensor_to_list_of_channel_tensors(img)
  new_azimuth = (img_list[azimuth_channel] - 90. * float(num_rotations)) % 360.
  img_list[azimuth_channel] = new_azimuth
  img = tf.stack(img_list, axis=-1)
  return img


def random_rotate90_input_and_output_images(
    input_img,
    output_img,
    azimuth_in_channel = None,
    azimuth_out_channel = None,
):
  """Randomly rotate input and output images in increments of 90.

  Args:
    input_img: Tensor with dimensions HWC or BHWC.
    output_img: Tensor with dimensions HWC or BHWC.
    azimuth_in_channel: Channel index of azimuth channel in input. Default value
      `None` means there is no azimuth channel.
    azimuth_out_channel: Channel index of azimuth channel in output. Default
      value `None` means there is no azimuth channel.

  Returns:
    input_img: Tensor with dimensions HWC or BHWC.
    output_img: Tensor with dimensions HWC or BHWC.
  """
  num_rotations = tf.random.categorical(
      tf.math.log([[0.25, 0.25, 0.25, 0.25]]), 1, dtype='int32')[0][0]
  input_img = tf.image.rot90(input_img, k=num_rotations)
  output_img = tf.image.rot90(output_img, k=num_rotations)

  if azimuth_in_channel:
    input_img = _update_azimuth_rotate90(input_img, azimuth_in_channel,
                                         num_rotations)
  if azimuth_out_channel:
    output_img = _update_azimuth_rotate90(output_img, azimuth_out_channel,
                                          num_rotations)

  return input_img, output_img


def random_crop_input_and_output_images(
    input_img,
    output_img,
    sample_size,
    num_in_channels,
    num_out_channels,
):
  """Randomly axis-align crop input and output image tensors.

  Args:
    input_img: Tensor with dimensions HWC.
    output_img: Tensor with dimensions HWC.
    sample_size: Side length (square) to crop to.
    num_in_channels: Number of channels in input_img.
    num_out_channels: Number of channels in output_img.

  Returns:
    input_img: Tensor with dimensions HWC.
    output_img: Tensor with dimensions HWC.
  """
  combined = tf.concat([input_img, output_img], axis=2)
  combined = tf.image.random_crop(
      combined, [sample_size, sample_size, num_in_channels + num_out_channels])
  input_img = combined[:, :, :num_in_channels]
  output_img = combined[:, :, -num_out_channels:]
  return input_img, output_img


def center_crop_input_and_output_images(
    input_img,
    output_img,
    sample_size,
):
  """Center crops input and output image tensors.

  Args:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
    sample_size: side length (square) to crop to.

  Returns:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
  """
  central_fraction = sample_size / input_img.shape[0]
  input_img = tf.image.central_crop(input_img, central_fraction)
  output_img = tf.image.central_crop(output_img, central_fraction)
  return input_img, output_img


def _get_coarse_value(input_img):
  """Maps input with values from `{1, 0, -1}` to a coarse value in `{[0, 1], -1}`.

  Args:
    input_img: Tensor with values from `{1, 0, -1}`, where `1` is positive, `0`
      is negative, and `-1` is uncertain.

  Returns:
    A scalar tensor with value in the range `[0, 1]` indicating the ratio of
    positives to the total `positives / (positives + negatives)`, or `-1` if
    `(positives + negatives)` is `0`.
  """
  positives = tf.math.count_nonzero(tf.equal(input_img, 1))
  negatives = tf.math.count_nonzero(tf.equal(input_img, 0))
  total_pos_neg = tf.math.add(positives, negatives)
  return tf.cond(
      tf.equal(total_pos_neg, 0), lambda: tf.constant(-1.0, dtype=tf.float32),
      lambda: tf.cast(tf.math.divide(positives, total_pos_neg)))


def _get_coarse_label(input_img,
                      downsample_threshold = 0.0,
                      binarize_output = True):
  """Maps input with values from `{1, 0, -1}` to a coarse label value.

  Args:
    input_img: Tensor with values from `{1, 0, -1}`, where `1` is positive, `0`
      is negative, and `-1` is uncertain.
    downsample_threshold: Threshold to determine the coarse label from a tensor
      of `1` (positive), `0` (negative), `-1` (uncertain) labels. Ignoring
      uncertain labels, if the ratio of positive/all labels is higher than this
      threshold, then downsampled label is `1`, otherwise `0`. Value is ignored
      if `binarize_output = False`.
    binarize_output: Whether to binarize the output values. If `True`, then
      output values are in `{1, 0, -1}`, else in `{[0, 1], -1}`.

  Returns:
    Tensor with float values from `{[0, 1], -1}` or from `{1, 0, -1}` depending
    on binarize_output:
      If `binarize_output = False`:
        `-1`, if `positives + negatives = 0`;
        else `positives / (positives + negatives)`.
      If `binarize_output = True`:
        `1`, if `positives / (positives + negatives) > downsample_threshold`;
        `0`, if `positives / (positives + negatives) <= downsample_threshold`.
  """
  coarse_val = _get_coarse_value(input_img)
  if binarize_output:
    coarse_val = tf.cond(
        tf.equal(coarse_val, -1.0),
        lambda: tf.constant(-1.0, dtype=tf.float32),
        lambda: tf.cond(  # pylint: disable=g-long-lambda
            tf.math.greater(coarse_val, downsample_threshold),
            lambda: tf.constant(1.0, dtype=tf.float32),  # pylint: disable=g-long-lambda
            lambda: tf.constant(0.0, dtype=tf.float32)))
  return coarse_val


def downsample_output_image(img,
                            output_sample_size,
                            downsample_threshold = 0.0,
                            binarize_output = True):
  """Downsamples the given img to output_sample_size.

  Args:
    img: Tensor with dimensions `[batch_size, sample_size, sample_size,
      num_out_channels]`, where sample_size is height and width of input. Values
      are from `{1, 0, -1}`.
    output_sample_size: Size of the output tiles (square), usually same or
      smaller than the sample_size such that sample_size is a perfect multiple.
    downsample_threshold: Threshold to determine the downsampled coarse label
      from a tensor of `1` (positive), `0` (negative), `-1` (uncertain) labels.
      Ignoring uncertain labels, if the ratio of positive/all labels is higher
      than this threshold, then downsampled label is `1`, otherwise `0`. Value
      is ignored if `output_sample_size = sample_size` or if `binarize_output =
      False`.
    binarize_output: Whether to binarize the output values. If `True`, then
      output values are in `{1, 0, -1}`, else in `{[0, 1], -1}`.

  Returns:
    Output tensor of labels with dimensions
    `[batch_size, output_sample_size, output_sample_size, num_out_channels]`
    and the downsampled values are determined as follows: if,
      `output_sample_size = sample_size`, img is returned unchanged;
      `output_sample_size > sample_size`, raises an error;
      `output_sample_size < sample_size`, returns downsampled labels that are
        in `{[0, 1], -1}` if `binarize_output = False`, and `{1, 0, -1}`
        otherwise. See `_get_coarse_label` docstring for details.

  Raises:
    ValueError: if img shape or `output_sample_size` is invalid.
  """
  img_shape = img.shape.as_list()
  if not ((len(img.shape) == 3) or (len(img.shape) == 4)):
    raise ValueError(f'Input must be HWC or BHWC and not of shape {img.shape}.')
  if img_shape[-2] != img_shape[-3]:
    raise ValueError(
        f'Image shape is {img_shape}. Height and Width must be equal.')
  batch_size = (img.shape[0] if len(img.shape) == 4 else 1)
  sample_size = img.shape[-2]
  num_out_channels = img.shape[-1]

  if sample_size < output_sample_size:
    raise ValueError('output_sample_size must be <= sample_size.')
  elif output_sample_size == 0:
    raise ValueError('output_sample_size must be > 0.')
  elif sample_size == output_sample_size:
    return img
  size, rem = divmod(sample_size, output_sample_size)
  if rem > 0:
    raise ValueError('sample_size must be a multiple of output_sample_size')

  new_shape = img_shape
  new_shape[-2] = output_sample_size
  new_shape[-3] = output_sample_size

  output = tf.zeros(shape=new_shape)
  num_updates = (
      batch_size * output_sample_size * output_sample_size * num_out_channels)
  indices_array = np.zeros([num_updates, len(img.shape)])
  updates = []
  update_count = 0

  for b in range(0, batch_size):
    for h in range(0, output_sample_size):
      for w in range(0, output_sample_size):
        for c in range(0, num_out_channels):
          indices_array[update_count] = ([b, h, w, c]
                                         if len(img.shape) == 4 else [h, w, c])
          # Get slice at index [b,h,w,c] of h x w of size size x size.
          if len(img.shape) == 3:
            img_slice = img[h * size:(h + 1) * size, w * size:(w + 1) * size, c]
          else:  # len(img.shape) == 4
            img_slice = img[b, h * size:(h + 1) * size, w * size:(w + 1) * size,
                            c]
          updates = tf.concat([
              updates,
              [
                  _get_coarse_label(img_slice, downsample_threshold,
                                    binarize_output)
              ]
          ],
                              axis=0)
          update_count += 1

  # Update output tensor. Equivalent to `tensor[indices] = updates`.
  output = tf.tensor_scatter_nd_update(
      tensor=output,
      indices=tf.convert_to_tensor(indices_array, dtype=tf.int32),
      updates=updates)
  return tf.cast(output, dtype=tf.float32)
