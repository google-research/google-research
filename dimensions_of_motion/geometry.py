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
"""Functions for sampling and warping images.

We use texture coordinates to represent points and offsets in images. They go
from (0,0) in the top-left corner of an image to (1,1) in the bottom right. It
is convenient to work with these coordinates rather than counts of pixels,
because they are resolution-independent.
"""

import tensorflow as tf
import tensorflow_addons as tfa

import utils


def check_input_shape(name, tensor, axis, value):
  """Utility function for checking tensor shapes."""
  shape = tensor.shape.as_list()
  if shape[axis] != value:
    raise ValueError('Input "%s": dimension %d should be %s. Shape = %s' %
                     (name, axis, value, shape))


def pixel_center_grid(height, width):
  """Produce a grid of (x,y) texture-coordinate pairs of pixel centers.

  Args:
    height: (integer) height, not a tensor
    width: (integer) width, not a tensor

  Returns:
    A tensor of shape [height, width, 2] where each entry gives the (x,y)
    texture coordinates of the corresponding pixel center. For example, for
    pixel_center_grid(2, 3) the result is:
       [[[1/6, 1/4], [3/6, 1/4], [5/6, 1/4]],
        [[1/6, 3/4], [3/6, 3/4], [5/6, 3/4]]]
  """
  height_float = tf.cast(height, dtype=tf.float32)
  width_float = tf.cast(width, dtype=tf.float32)
  ys = tf.linspace(0.5 / height_float, 1.0 - 0.5 / height_float, height)
  xs = tf.linspace(0.5 / width_float, 1.0 - 0.5 / width_float, width)
  xs, ys = tf.meshgrid(xs, ys)
  grid = tf.stack([xs, ys], axis=-1)
  assert grid.shape.as_list() == [height, width, 2]
  return grid


def sample_image(image, coords):
  """Sample points from an image, using bilinear filtering.

  Args:
    image: [B0, ..., Bn-1, height, width, channels] image data
    coords: [B0, ..., Bn-1, ..., 2] (x,y) texture coordinates

  Returns:
    [B0, ..., Bn-1, ..., channels] image data, in which each value is sampled
    with bilinear interpolation from the image at position indicated by the
    (x,y) texture coordinates. The image and coords parameters must have
    matching batch dimensions B0, ..., Bn-1.

  Raises:
    ValueError: if shapes are incompatible.
  """
  check_input_shape('coords', coords, -1, 2)
  tfshape = tf.shape(image)[-3:-1]
  height = tf.cast(tfshape[0], dtype=tf.float32)
  width = tf.cast(tfshape[1], dtype=tf.float32)

  # Resampler expects coordinates where (0,0) is the center of the top-left
  # pixel and (width-1, height-1) is the center of the bottom-right pixel.
  pixel_coords = coords * [width, height] - 0.5

  # tfa.image.resampler only works with exactly one batch dimension, i.e. it
  # expects image to be [batch, height, width, channels] and pixel_coords to be
  # [batch, ..., 2]. So we need to reshape, perform the resampling, and then
  # reshape back to what we had.
  batch_dims = len(image.shape.as_list()) - 3
  assert (image.shape.as_list()[:batch_dims] == pixel_coords.shape.as_list()
          [:batch_dims])

  batched_image, _ = utils.flatten_batch(image, batch_dims)
  batched_coords, unflatten_coords = utils.flatten_batch(
      pixel_coords, batch_dims)
  resampled = tfa.image.resampler(batched_image, batched_coords)

  # Convert back to the right shape to return
  resampled = unflatten_coords(resampled)
  return resampled


def bilinear_forward_warp(image, coords, weights=None):
  """Forward warp each point in an image using bilinear filtering.

  This is a sort of reverse of sample_image, in the sense that scatter is the
  reverse of gather. A new image is generated of the same size as the input, in
  which each pixel has been splatted onto the 2x2 block containing the
  corresponding coordinates, using bilinear weights (multiplied with the input
  per-pixel weights, if supplied). Thus if two or more pixels warp to the same
  point, the result will be a blend of the their values. If no pixels warp to a
  location, the result at that location will be zero.

  Args:
    image: [B0, ..., Bn-1, height, width, channels] image data
    coords: [B0, ..., Bn-1, height, width, 2] (x,y) texture coordinates
    weights: [B0, ... ,Bn-1, height, width] weights for each point. If omitted,
      all points are weighed equally. Use this to implement, for example, soft
      z-buffering.

  Returns:
    [B0, ..., Bn-1, ..., channels] image data, in which each point in the
    input image has been moved to the position indicated by the corresponding
    (x,y) texture coordinates. The image and coords parameters must have
    matching batch dimensions B0, ..., Bn-1.
  """
  # Forward-warp computed using the gradient of reverse-warp. We use a dummy
  # image of the right size for reverse-warping. An extra channel is used to
  # accumulate the total weight for each pixel which we'll then divide by.
  image_and_ones = tf.concat([image, tf.ones_like(image[Ellipsis, -1:])], axis=-1)
  dummy = tf.zeros_like(image_and_ones)
  if weights is None:
    weighted_image = image_and_ones
  else:
    weighted_image = image_and_ones * weights[Ellipsis, tf.newaxis]

  with tf.GradientTape(watch_accessed_variables=False) as g:
    g.watch(dummy)
    reverse = tf.reduce_sum(
        sample_image(dummy, coords) * weighted_image, [-3, -2])
    grads = g.gradient(reverse, dummy)
  rgb = grads[Ellipsis, :-1]
  total = grads[Ellipsis, -1:]
  result = tf.math.divide_no_nan(rgb, total)
  return result


def flow_warp(image, flow):
  """Warp images by resampling according to flow vectors.

  Args:
    image: [..., H, W, C] images
    flow: [..., H, W, 2] (x, y) texture offsets

  Returns:
    [..., H, W, C] resampled images. Each pixel in each output image has been
    bilinearly sampled from the corresponding pixel in its input image plus
    the (x, y) flow vector. The flow vectors are texture coordinate offsets,
    e.g. (1, 1) is an offset of the whole width and height of the image.
    Sampling outside the image yields zero values.
  """
  width = image.shape.as_list()[-2]
  height = image.shape.as_list()[-3]
  grid = pixel_center_grid(height, width)
  coords = grid + flow
  return sample_image(image, coords)


def flow_forward_warp(image, flow):
  """Forward-warp images according to flow vectors.

  Args:
    image: [..., H, W, C] images
    flow: [..., H, W, 2] (x, y) texture offsets

  Returns:
    [..., H, W, C] warped images. Each pixel in each image is offset according
    to the corresponding value in the flow, and splatted onto a 2x2 pixel block.
    (See bilinear_forward_warp for details.) If no points warp to a location,
    the result will be zero. The flow vectors are texture coordinate offsets,
    e.g. (1, 1) is an offset of the whole width and height of the image.
  """
  width = image.shape.as_list()[-2]
  height = image.shape.as_list()[-3]
  grid = pixel_center_grid(height, width)
  coords = grid + flow
  return bilinear_forward_warp(image, coords)
