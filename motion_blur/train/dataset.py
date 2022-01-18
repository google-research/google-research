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

"""Creates a motion blur dataset.

Learning to Synthesize Motion Blur
http://timothybrooks.com/tech/motion-blur
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def read_jpg(filename):
  """Reads an 8-bit JPG file from disk and normalizes to [0, 1]."""
  image_file = tf.read_file(filename)
  image = tf.image.decode_jpeg(image_file, channels=3)
  white_level = 255.0
  return tf.cast(image, tf.float32) / white_level


def read_images(dir_name):
  """Reads motion blur training inputs and label images."""
  images = [
      read_jpg(dir_name + '/frame_0.jpg'),
      read_jpg(dir_name + '/frame_1.jpg'),
      read_jpg(dir_name + '/blurred.jpg')
  ]
  return tf.concat(images, axis=-1)




def is_large_enough(images, height, width):
  """Checks if `image` is at least as large as `height` by `width`."""
  images.shape.assert_has_rank(3)
  shape = tf.shape(images)
  image_height = shape[0]
  image_width = shape[1]
  return tf.logical_and(
      tf.greater_equal(image_height, height),
      tf.greater_equal(image_width, width))


def augment(images, height, width):
  """Randomly flips and crops `images` to `height` by `width`."""
  size = [height, width, tf.shape(images)[-1]]
  images = tf.random_crop(images, size)
  images = tf.image.random_flip_left_right(images)
  images = tf.image.random_flip_up_down(images)
  return images


def create_example(images):
  """Creates training example of inputs and label from `images`."""
  images.shape.assert_is_compatible_with([None, None, 9])

  inputs = {
      'frame_0': images[Ellipsis, 0:3],
      'frame_1': images[Ellipsis, 3:6],
  }
  label = images[Ellipsis, 6:9]
  return inputs, label


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
    images = filenames.map(read_images, num_parallel_calls=autotune)
    images = images.filter(lambda x: is_large_enough(x, height, width))
    images = images.map(
        lambda x: augment(x, height, width), num_parallel_calls=autotune)
    examples = images.map(create_example, num_parallel_calls=autotune)
    return examples.batch(batch_size).prefetch(autotune)

  return dataset_fn_
