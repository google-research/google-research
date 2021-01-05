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

"""Preprocess seqeuences consistently and with respect to sequence modality.

These functions generalize tf.image functions to handle sequences of images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


def adjust_sequence_brightness(tensor, brightness_change):
  """Adjust brightness consistently across time with dim[0] as time."""
  return tf.image.adjust_brightness(tensor, brightness_change)


def adjust_sequence_contrast(tensor, contrast_change):
  """Adjust contrast consistently across time with dim[0] as time."""
  return tf.image.adjust_contrast(tensor, contrast_change)


def clip_sequence_value(tensor, lower_limit, upper_limit):
  """Clip values consistently across time with dim[0] as time."""
  return tf.clip_by_value(tensor, lower_limit, upper_limit)


def add_additive_noise_to_sequence(tensor, noise_stddev, seed=None):
  """Add Gaussian noise consistently across time (but random per frame)."""
  return tensor + noise_stddev * tf.random_normal(
      tf.shape(tensor), dtype=tf.float32, seed=seed)


def convert_image_sequence_dtype(tensor, dtype=tf.float32):
  return tf.map_fn(
      lambda x: tf.image.convert_image_dtype(x, dtype),
      tensor,
      dtype=dtype,
      back_prop=False)


def adjust_sequence_hue(tensor, hue_change):
  """Adjust hue consistently across time with dim[0] as time."""
  return tf.map_fn(
      lambda x: tf.image.adjust_hue(*x),
      [tensor,
       tf.tile([hue_change], tf.reshape(tf.shape(tensor)[0], [1]))],
      dtype=tf.float32,
      back_prop=False)


def adjust_sequence_saturation(tensor, saturation_change):
  """Adjust saturation consistently across time with dim[0] as time."""
  return tf.map_fn(
      lambda x: tf.image.adjust_saturation(*x), [
          tensor,
          tf.tile([saturation_change], tf.reshape(tf.shape(tensor)[0], [1]))
      ],
      dtype=tf.float32,
      back_prop=False)


def largest_square_crop(image_size):
  """For given image size, returns the maximum square, central crop.

  Args:
    image_size: a [height, width] tensor.

  Returns:
    output_begin, output_size and image_size.
    output_begin and output_size are three element tensors specifying the shape
    to crop using crop_sequence below. image_size is a two element
    [height, width] tensor from the input.
  """
  min_dim = tf.reduce_min(image_size[0:2])
  output_size = tf.stack([min_dim, min_dim, -1])
  height_offset = tf.to_int32((image_size[0] - min_dim) / 2)
  width_offset = tf.to_int32((image_size[1] - min_dim) / 2)
  output_begin = tf.stack([height_offset, width_offset, 0])
  return output_begin, output_size, image_size


def random_square_crop(image_size, min_scale):
  """Generates a random square crop within an image.

  Args:
    image_size: a [height, width] tensor.
    min_scale: how much the minimum dimension can be scaled down when taking a
      crop. (e.g. if the image is 480 x 640, a min_scale of 0.8 means the output
      crop can have a height and width between 480 and 384, which is 480 * 0.8.)

  Returns:
    output_begin, output_size and image_size.
    output_begin and output_size are three element tensors specifying the shape
    to crop using crop_sequence below. image_size is a two element
    [height, width] tensor from the input.
  """
  min_dim = tf.reduce_min(image_size[0:2])
  sampled_size = tf.to_int32(
      tf.to_float(min_dim) * tf.random_uniform([], min_scale, 1.0))
  output_size = tf.stack([sampled_size, sampled_size, -1])
  height_offset = tf.random_uniform([],
                                    0,
                                    image_size[0] - sampled_size + 1,
                                    dtype=tf.int32)
  width_offset = tf.random_uniform([],
                                   0,
                                   image_size[1] - sampled_size + 1,
                                   dtype=tf.int32)
  output_begin = tf.stack([height_offset, width_offset, 0])
  return output_begin, output_size, image_size


def crop_sequence(tensor,
                  single_slice_begin,
                  single_slice_size,
                  unused_image_size=None):
  """Take a crop at the same coordinates for an entire sequence.

  Args:
    tensor: the [time, ...] tensor to crop. Typically [time, height, width,
      channels].
    single_slice_begin: the starting location of the crop. Must include all
      dimensions after the time dimension (e.g. typically includes channels).
    single_slice_size: the number of from slice begin to include in the crop.
      See tf.slice for a more detailed explanation of begin and size.
    unused_image_size: included to match APIs. Unused here.

  Returns:
    The cropped tensor.
  """
  sequence_begin = tf.concat([tf.constant([0]), single_slice_begin], 0)
  sequence_size = tf.concat([tf.constant([-1]), single_slice_size], 0)
  return tf.slice(tensor, sequence_begin, sequence_size)


def resize_sequence(tensor,
                    new_size,
                    method=tf.image.ResizeMethod.BILINEAR):
  """Resize images appropriately.

  For images, apply tf.image.resize.

  Args:
    tensor: the 4D tensor to resize.
    new_size: the [height, width] to resize to.
    method: one of tf.image.ResizeMethod.{AREA,BICUBIC,BILINIEAR,
      NEAREST_NEIGHBOR}
  Returns:
  """
  return tf.image.resize(tensor, new_size, method=method)


def optionally_flip_sequence(tensor, dim, do_flip):
  """Either flip all of a sequence or flip none of it with dim[0] as time.

  Args:
   tensor: the tensor to flip a dimension of.
   dim: the dimension of the input tensor to flip. (e.g. if data is [time,
     height, width, channels], dim=1 is vertical.)
   do_flip: whether to actually flip the data in the function.

  Returns:
   The flipped tensor or the original.
  """

  def flip():
    if dim < 0 or dim >= (len(tensor.get_shape().as_list())):
      raise ValueError('dim must represent a valid dimension.')
    return tf.reverse(tensor, [dim])

  output = tf.cond(do_flip, flip, lambda: tensor)
  return output
