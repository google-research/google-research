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

"""Creates a Dataset of unprocessed images for denoising.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow as tf

from unprocessing import unprocess


def read_jpg(filename):
  """Reads an 8-bit JPG file from disk and normalizes to [0, 1]."""
  image_file = tf.read_file(filename)
  image = tf.image.decode_jpeg(image_file, channels=3)
  white_level = 255.0
  return tf.cast(image, tf.float32) / white_level


def is_large_enough(image, height, width):
  """Checks if `image` is at least as large as `height` by `width`."""
  image.shape.assert_has_rank(3)
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  return tf.logical_and(
      tf.greater_equal(image_height, height),
      tf.greater_equal(image_width, width))


def augment(image, height, width):
  """Randomly flips and crops `images` to `height` by `width`."""
  size = [height, width, tf.shape(image)[-1]]
  image = tf.random_crop(image, size)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  return image


def create_example(image):
  """Creates training example of inputs and labels from `image`."""
  image.shape.assert_is_compatible_with([None, None, 3])
  image, metadata = unprocess.unprocess(image)
  shot_noise, read_noise = unprocess.random_noise_levels()
  noisy_img = unprocess.add_noise(image, shot_noise, read_noise)
  # Approximation of variance is calculated using noisy image (rather than clean
  # image), since that is what will be avaiable during evaluation.
  variance = shot_noise * noisy_img + read_noise

  inputs = {
      'noisy_img': noisy_img,
      'variance': variance,
  }
  inputs.update(metadata)
  labels = image
  return inputs, labels


def create_dataset_fn(dir_pattern, height, width, batch_size):
  """Wrapper for creating a dataset function for unprocessing.

  Args:
    dir_pattern: A string representing source data directory glob.
    height: Height to crop images.
    width: Width to crop images.
    batch_size: Number of training examples per batch.

  Returns:
    Nullary function that returns a Dataset.
  """
  if height % 16 != 0 or width % 16 != 0:
    raise ValueError('`height` and `width` must be multiples of 16.')

  def dataset_fn_():
    """Creates a Dataset for unprocessing training."""
    autotune = tf.data.experimental.AUTOTUNE

    filenames = tf.data.Dataset.list_files(dir_pattern, shuffle=True).repeat()
    images = filenames.map(read_jpg, num_parallel_calls=autotune)
    images = images.filter(lambda x: is_large_enough(x, height, width))
    images = images.map(
        lambda x: augment(x, height, width), num_parallel_calls=autotune)
    examples = images.map(create_example, num_parallel_calls=autotune)
    return examples.batch(batch_size).prefetch(autotune)

  return dataset_fn_
