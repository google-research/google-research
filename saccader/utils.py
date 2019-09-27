# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utility methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import numpy as np
import tensorflow as tf


def draw_bounding_boxes(images, locations_list, normalized_box_size):
  """Draws bounding boxes on images.

  Args:
    images: 4D tensor of images of size [batch, height, width, channels].
    locations_list: list of locations, each element is a tensor of size [batch,
      2]. Locations are normalized in the range of -1, 1 where:
          (-1, -1): Upper left corner.
          (-1, 1): Upper right corner.
          (1, 1): Lower right corner.
          (1, -1): Lower left corner.
    normalized_box_size: Portion of box size with respect to image size (ie
      box_size / image_size).

  Returns:
    images_with_boxes: Images with boxes drawn at locations specified.
  """
  boxes = tf.stack(locations_list, axis=1)  # Size now is times x batch x 2.
  boxes = (boxes + 1.) / 2.  # Normalize to (0,1) instead of (-1, 1) range.
  # Get bounding boxes corners
  boxes = tf.concat(
      [boxes - normalized_box_size / 2., boxes + normalized_box_size / 2.],
      axis=2)
  images_with_boxes = tf.image.draw_bounding_boxes(images, boxes)
  return images_with_boxes


def location_diversity(locations_list):
  """Computes pairwise distance from a list of locations.

  Args:
    locations_list: List of length number of glimpses (ie length time), where
      each element is a tensor of size batch by 2 indicating x, y coordinates
      for glimpse locations.

  Returns:
    d: Pairwise distances averaged across the list of locations.
  """
  num_times = len(locations_list)
  location_mtx = tf.stack(locations_list, axis=0)
  a0 = tf.expand_dims(location_mtx, 0)
  a1 = tf.expand_dims(location_mtx, 1)
  d = tf.sqrt(tf.reduce_sum(tf.squared_difference(a0, a1), axis=3))
  d = tf.reduce_sum(d, axis=[0, 1]) / (num_times * (num_times - 1))
  d = tf.reduce_mean(d)
  return d


def vectors_alignment(vectors_list):
  """Computes the degree of alignment of location vectors.

  Args:
    vectors_list: List of length number of glimpses (ie length time), where each
      element is a tensor of shape [batch, 2] indicating x, y coordinates for
      glimpse locations.

  Returns:
    alignment_metric: Number indicating how consistent glimpse locations across
      the batch (0: glimpses are different. 1: glimpse locations are the same).

  """
  # computes alignment of glimpse locations across the batch
  mtx = tf.concat(vectors_list, axis=1)  # size batch x batch
  dims = mtx.shape.as_list()[0]
  mtx /= (tf.sqrt(tf.reduce_sum(mtx**2, axis=1, keepdims=True)) + 1e-8)
  # note here we do not differentiate between 0 and 180 degrees difference.
  dot_product_mtx = tf.abs(tf.matmul(mtx, tf.transpose(mtx)))
  alignment_metric = (
      tf.reduce_sum(dot_product_mtx) - tf.trace(dot_product_mtx)) / (
          dims * (dims - 1))
  return alignment_metric


def prefix_dictionary(dictionary, prefix=None):
  """Returns a dictionary with a prefix to all input dictionary keys."""
  if prefix:
    return {os.path.join(prefix, k): v for (k, v) in dictionary.iteritems()}


def add_histogram_summary(tensor):
  """Adds a summary operation to visualize any tensor."""
  tf.summary.histogram(tensor.op.name, tensor)


def position_channels(images):
  """Constructs two channels with position information."""
  batch_size, h, w = images.shape.as_list()[0:3]
  pos_h = tf.tile(tf.linspace(-1., 1., h)[:, tf.newaxis],
                  [1, w])[tf.newaxis, :, :, tf.newaxis]

  pos_w = tf.tile(tf.linspace(-1., 1., w)[tf.newaxis, :],
                  [h, 1])[tf.newaxis, :, :, tf.newaxis]

  channels = tf.tile(
      tf.concat([pos_h, pos_w], axis=3), [batch_size, 1, 1, 1])
  channels = tf.cast(channels, dtype=images.dtype)

  return channels


def normalize_range(tensor, min_value, max_value):
  """Normalizes the range of a tensor to (min_val, max_val)."""
  tensor -= tf.reduce_min(tensor)
  tensor /= tf.reduce_max(tensor)
  tensor *= (max_value - min_value)
  tensor += min_value
  return tensor


def distorted_inputs(images):
  """Constructs distorted input for training.

  Args:
    images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.

  Returns:
    distorted_images: Distorted Images. 4D tensor of [batch_size, IMAGE_SIZE,
    IMAGE_SIZE, 3] size.
  """

  # Put image in [0, 255] range (assumes images are in -1, 1 range)
  distorted_images = ((images + 1.) / 2.) * 255.

  # Randomly flip the image horizontally.
  distorted_images = tf.image.random_flip_left_right(distorted_images)

  distorted_images = tf.image.random_brightness(distorted_images, max_delta=63)
  distorted_images = tf.image.random_contrast(
      distorted_images, lower=0.2, upper=1.8)

  # Change the range back to -1, 1 range.
  distorted_images /= 255.
  distorted_images *= 2.
  distorted_images -= 1.
  return distorted_images


def write_config(save_dir, config):
  """Writes all params in configs to text file.

  This could be read back with ast.literal_eval().
  Args:
    save_dir: Directory.
    config: Configuration object.
  """
  if not tf.gfile.Exists(save_dir):
    tf.gfile.MakeDirs(save_dir)
  with tf.gfile.Open(os.path.join(save_dir, "config.txt"), "w") as f:
    f.write(pprint.pformat(config.to_dict()))


def write_flags(save_dir, flags):
  """Writes all params in flag to text file.

  This could be read back with ast.literal_eval().
  Args:
    save_dir: Directory.
    flags: flags is flags.FLAGS.
  """
  flags_dict = flags.flag_values_dict().copy()
  if not tf.gfile.Exists(save_dir):
    tf.gfile.MakeDirs(save_dir)

  with tf.gfile.Open(os.path.join(save_dir, "flags.txt"), "w") as f:
    f.write(pprint.pformat(flags_dict))


def index_to_normalized_location(ix, image_size):
  """Converts center index of a glimpse to a normalized location in -1, 1 range.

  The image locations mapping to normalized locations are as follows:
    (-1, -1): upper left corner.
    (-1, 1): upper right corner.
    (1, 1): lower right corner.
    (1, -1): lower left corner.

  Args:
    ix: Tensor of size 2 representing the horizontal and vertical indices of the
      center of a glimpse.
    image_size: (Integer) image size.

  Returns:
    Normalized location of a glimpse.
  """
  ix = tf.cast(ix, dtype=tf.float32)
  # Location range (-1, 1). So full range is 2.
  normalized_location = (2. * ix) / image_size
  # Make center (0, 0)
  normalized_location = normalized_location - 1.
  return normalized_location


def location_guide(image,
                   image_size=32,
                   open_fraction=0.2,
                   uniform_noise=False,
                   block_probability=0.5):
  """Provides supervised signal to guide glimpsing controller.

  All image is blocked (using zeros or uniform noise) except for a square
  window. The center location of this window is sent as a guide for glimpse
  controller.

  Args:
    image: Tensor of shape [height, width, channels].
    image_size: (Integer) image size.
    open_fraction: (Float) fraction of image_size to leave intact the rest of
      the image will be blocked.
    uniform_noise: (Boolean) whether to use uniform noise to block the image or
      block with zeros.
    block_probability: [0 - 1] probability of blocking the image.

  Returns:
    image: The resulting image is a tensor of same shape as input image.
    location: Normalized location of the center of the window where the image is
      intact. If the image was not blocked this will be [0, 0]. Also, if the
      open window is exactly at the center this will be [0, 0].
    blocked_indicator: Indicator if the image was blocked or not (1: blocked,
      0: output image is the same as input image).
  """

  def location_guide_helper(x):
    """Helper function."""
    window_size = int(open_fraction * image_size)
    mask = tf.ones([window_size, window_size, 3])

    mask = tf.image.resize_image_with_crop_or_pad(
        mask, 2 * image_size - window_size, 2 * image_size - window_size)

    # range of bounding boxes is from [0, image_size-window_size]
    offset_height = tf.random_uniform(
        shape=(), minval=0, maxval=image_size - window_size, dtype=tf.int32)
    offset_width = tf.random_uniform(
        shape=(), minval=0, maxval=image_size - window_size, dtype=tf.int32)

    mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width,
                                         image_size, image_size)

    x *= mask
    if uniform_noise:
      x += tf.random_uniform((image_size, image_size, 3), 0, 1.0) * (1. - mask)

    center_ix = tf.convert_to_tensor([
        image_size - window_size - offset_height + window_size // 2,
        image_size - window_size - offset_width + window_size // 2
    ],
                                     dtype=tf.int32)

    location = index_to_normalized_location(center_ix, image_size)
    return x, location

  image, location, blocked_indicator = tf.cond(
      tf.math.less(
          tf.random_uniform([], 0, 1.0),
          block_probability), lambda: (location_guide_helper(image) + (1.,)),
      lambda: (image, tf.zeros(shape=(2,), dtype=tf.float32), 0.))
  return image, location, blocked_indicator


def extract_glimpse(images, size, offsets):
  """Extracts a glimpse from the input tensor.

  Args:
    images: A Tensor of type float32. A 4-D float tensor of shape [batch_size,
      height, width, channels].
    size: A list or tuple of 2 integers specifying the size of the glimpses to
      extract. The glimpse height must be specified first, followed by the
      glimpse width.
    offsets: A Tensor of type float32. A 2-D integer tensor of shape
      [batch_size, 2] containing the y, x locations of the center of each
      window. If parts of the glimpses to be extracted fall outside the image,
      they will be replaced with the nearest pixel inside the image.

  Returns:
    A Tensor of type float32.
  """
  batch_size, height, width, num_channels = images.shape.as_list()
  offsets = tf.cast(offsets, dtype=tf.float32)

  # Compute the coordinates of the top left of each glimpse.
  coord_height = tf.cast(
      (height * (offsets[:, 0] + 3) - tf.cast(size[0], dtype=tf.float32)) / 2,
      tf.int32) - height
  coord_width = tf.cast(
      (width * (offsets[:, 1] + 3) - tf.cast(size[1], dtype=tf.float32)) / 2,
      tf.int32) - width

  # Compute linear indices into flattened images. If the indices along the
  # height or width dimension fall outside the image, we clip them to be the
  # nearest pixel inside the image.
  indices_batch = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
  indices_height = (tf.reshape(coord_height, [batch_size, 1, 1]) +
                    tf.reshape(tf.range(size[0]), [1, size[0], 1]))
  indices_height = tf.clip_by_value(indices_height, 0, height - 1)
  indices_width = (tf.reshape(coord_width, [batch_size, 1, 1]) +
                   tf.reshape(tf.range(size[1]), [1, 1, size[1]]))
  indices_width = tf.clip_by_value(indices_width, 0, width - 1)
  indices = (height * width * indices_batch + width * indices_height
             + indices_width)

  # Gather into flattened images.
  return tf.reshape(
      tf.gather(tf.reshape(images, [-1, num_channels]),
                tf.reshape(indices, [-1])),
      [batch_size, size[0], size[1], num_channels])


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             start_at_step=0):
  """Cosine decay schedule with warm up period.

  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
    start_at_step: if > 0 start learning at global step = start_at_step.

  Returns:
    a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if start_at_step > 0:
    global_step = global_step - start_at_step
    total_steps = total_steps - start_at_step
  if total_steps < warmup_steps:
    raise ValueError("total_steps must be larger or equal to " "warmup_steps.")
  learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
      np.pi *
      (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps) /
      float(total_steps - warmup_steps - hold_base_rate_steps)))
  if hold_base_rate_steps > 0:
    learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                             learning_rate, learning_rate_base)
  if warmup_steps > 0:
    if learning_rate_base < warmup_learning_rate:
      raise ValueError("learning_rate_base must be larger or equal to "
                       "warmup_learning_rate.")
    slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * tf.cast(global_step,
                                  tf.float32) + warmup_learning_rate
    learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                             learning_rate)

  if start_at_step > 0:
    learning_rate = tf.where(global_step < 0, 0.0, learning_rate)

  return tf.where(
      global_step > total_steps, 0.0, learning_rate, name="learning_rate")


def batch_gather_nd(x, indices, axis):
  """Gathers elements using an index per example in batch.

  Args:
    x: Tensor with leading batch dimension.
    indices: Tensor of type int32 of size batch.
    axis: (Integer) axis to gather across.

  Returns:
    Tensor with leading batch dimension with 1-less rank than x with
      the specied axis reduced.

  Raises:
    ValueError if axis is not within the input dimensions.
  """
  n_dims = len(x.shape.as_list())
  if axis >= n_dims or axis < -n_dims:
    raise ValueError("axis %d is out of bounds" %axis)
  if axis < 0:
    axis += n_dims

  dims = [0, axis] + list(set(range(1, n_dims)) - set((axis,)))
  indices = tf.cast(indices, dtype=tf.int32)
  indices_nd = tf.stack(
      (tf.range(int(x.shape[0]), dtype=tf.int32), indices), 1)
  return tf.gather_nd(tf.transpose(x, dims), indices_nd)


def sort2d(tensor2d,
           ref_indices2d,
           first_k=float("inf"),
           direction="DESCENDING"):
  """Perform sort in 2D based on tensor2d values with indices in ref_indices2d.

  Args:
    tensor2d: 3D Tensor of size [batch_size, height, width]
    ref_indices2d: 4D Tensor of size [batch_size, height, width, 2]
      with reference coordinates in 2D.
    first_k: (Integer) return indices of first_k elements.
    direction: "ASCENDING" or "DESCENDING".

  Returns:
    sorted_tensor: 2D Tensor of size (first_k, batch_size)
    sorted_ref_indices: 3D Tensor of size (first_k, batch_size, 2) with
      ref_indices sorted order based on the order of the input (tensor2d).
  """
  batch_size, height, width = tensor2d.shape.as_list()
  first_k = min(first_k, height*width)
  tensor2d_reshaped = tf.reshape(
      tensor2d, [batch_size, height*width])
  ref_indices2d_reshaped = tf.reshape(
      ref_indices2d, [batch_size, height*width, 2])
  sort_indices = tf.argsort(
      tensor2d_reshaped, axis=1, direction=direction)
  sort_indices = tf.gather(sort_indices,
                           tf.cast(np.array(range(first_k)), tf.int32), axis=1)
  sorted_ref_indices = tf.reduce_sum(
      tf.eye(batch_size, batch_size)[:, :, tf.newaxis, tf.newaxis] * tf.gather(
          ref_indices2d_reshaped, sort_indices, axis=1),
      axis=1)

  sorted_tensor = tf.reduce_sum(
      tf.eye(batch_size, batch_size)[:, :, tf.newaxis] * tf.gather(
          tensor2d_reshaped, sort_indices, axis=1),
      axis=1)
  sorted_tensor = tf.transpose(sorted_tensor, [1, 0])
  sorted_ref_indices = tf.transpose(sorted_ref_indices, [1, 0, 2])
  return sorted_tensor, sorted_ref_indices


def normalized_locations_to_indices(offsets, height, width):
  """Converts normalized locations to indices.

  Args:
    offsets: Tensor of size [batch, 2] with normalized (i.e., range -1, 1)
      x and y locations.
    height: (Integer) Image height.
    width: (Integer) Image width.

  Returns:
    indices_height: (Integer) Image height index.
    indices_width: (Integer) Image width index.
  """
  offsets = tf.cast(offsets, dtype=tf.float32)

  # Compute the coordinates of the top left of each glimpse.
  indices_height = tf.cast(
      tf.round((height-1.) * (offsets[:, 0] + 1.) / 2.), tf.int32)
  indices_width = tf.cast(
      tf.round((width-1.) * (offsets[:, 1] + 1.) / 2.), tf.int32)

  # Clip to the correct size.
  indices_height = tf.clip_by_value(indices_height, 0, height-1)
  indices_width = tf.clip_by_value(indices_width, 0, width-1)
  return indices_height, indices_width


def onehot2d(input_tensor, offsets, dtype=tf.float32):
  """Computes onehot matrices.

  Values at locations offsets are 1 otherwise are 0.

  Args:
    input_tensor: A Tensor of type float32. A 4-D float tensor of shape [
      batch_size, height, width, channels].
    offsets: A Tensor of type float32. A 2-D integer tensor of shape
      [batch_size, 2] containing the x, y coordinates in normalized range
      (-1, 1).
    dtype: Data type of returned tensor.

  Returns:
    A Tensor of type float32.
  """
  batch_size, height, width, _ = input_tensor.shape.as_list()
  offsets = tf.cast(offsets, dtype=tf.float32)
  indices_height, indices_width = normalized_locations_to_indices(
      offsets, height, width)
  indices = (width * indices_height + indices_width)
  onehot = tf.reshape(
      tf.one_hot(indices, depth=height*width), (batch_size, height, width, 1)
      )
  return tf.cast(onehot, dtype=dtype)


def softmax2d(input_tensor):
  """Applies softmax to input_trensor across 2D plain (i.e., axis=[1,2])."""
  batch_size, height, width, channels = input_tensor.shape.as_list()
  reshaped_tensor = tf.reshape(
      input_tensor, (batch_size, height*width, channels))
  return tf.reshape(
      tf.nn.softmax(reshaped_tensor, axis=1),
      [batch_size, height, width, channels])


def patches_masks(locations_t, image_size, patch_size):
  """Returns masks for patches specified by locations_t.

  Args:
    locations_t: List of location samples at each time point (in range [-1, 1]).
    image_size: (Integer) Size of image.
    patch_size: (Integer) Size of patch.

  Returns:
    masks: 4-D image tensor with patch masks.
  """
  num_times = len(locations_t)
  batch_size = locations_t[0].shape.as_list()[0]
  def _helper(offsets):
    """Helper function."""
    mask = tf.ones([patch_size, patch_size, 1])

    mask = tf.image.resize_image_with_crop_or_pad(
        mask, 2 * image_size - patch_size, 2 * image_size - patch_size)

    # Compute the coordinates of the top left of each glimpse.
    size = tf.cast(patch_size, dtype=tf.float32)
    offset_height = tf.cast(
        (image_size * (offsets[0] + 3) - size) / 2, tf.int32) - image_size
    offset_width = tf.cast(
        (image_size * (offsets[1] + 3) - size) / 2, tf.int32) - image_size

    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_width,
                                         offset_height,
                                         image_size,
                                         image_size)
    return tf.transpose(
        tf.image.flip_up_down(tf.image.flip_left_right(1. - mask)), [1, 0, 2])

  masks = tf.map_fn(_helper, tf.concat(locations_t, axis=0))
  masks = tf.reduce_prod(
      tf.reshape(masks, (num_times, batch_size, image_size, image_size, 1)),
      axis=0)

  return masks


def metric_fn(logits, labels, mask, **kws):
  """Metrics to evaluate."""
  metrics = {k: tf.metrics.mean(v, weights=mask) for k, v in kws.items()}
  metrics["accuracy/top1"] = tf.metrics.accuracy(
      labels, tf.argmax(logits, axis=1), weights=mask)

  metrics["accuracy/top5"] = tf.metrics.recall_at_k(
      tf.cast(labels, tf.int64), logits, k=5, weights=mask)
  return metrics


