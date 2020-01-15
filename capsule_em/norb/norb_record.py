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

"""Input utility functions for norb."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path
import numpy as np
import tensorflow.compat.v1 as tf


def _read_and_decode(filename_queue, image_pixel=96, distort=0):
  """Read a norb tf record file."""
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'meta': tf.FixedLenFeature([4], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length image_pixels) to a uint8 tensor with shape
  # [image_pixels].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  height = tf.cast(features['height'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  image = tf.reshape(image, tf.stack([depth, height, height]))
  image = tf.transpose(image, [1, 2, 0])
  image = tf.cast(image, tf.float32)
  print(image.get_shape()[0].value)
  if image_pixel < 96:
    print('image resizing to {}'.format(image_pixel))
    image = tf.image.resize_images(image, [image_pixel, image_pixel])
    orig_images = image

  if image_pixel == 48:
    new_dim = 32
  elif image_pixel == 32:
    new_dim = 22
  if distort == 1:
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.random_crop(image, tf.stack([new_dim, new_dim, depth]))
    # 0.26179938779 is 15 degress in radians
    image = tf.image.per_image_standardization(image)
    image_pixel = new_dim
  elif distort == 2:
    image = tf.image.resize_image_with_crop_or_pad(image, new_dim, new_dim)
    image = tf.image.per_image_standardization(image)
    image_pixel = new_dim
  else:
    image = image * (1.0 / 255.0)
    image = tf.div(
        tf.subtract(image, tf.reduce_min(image)),
        tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label, image_pixel, orig_images


bxs_m2 = [[1, 1], [1, -1], [-1, 1], [-1, -1]]


def inputs(train_dir,
           batch_size,
           split,
           multi,
           image_pixel=96,
           distort=False,
           patching=False):
  """Reads input data num_epochs times."""
  if multi:
    filename = os.path.join(train_dir, '{}duo-az.tfrecords'.format(split))
  else:
    filename = os.path.join(train_dir, '{}.tfrecords'.format(split))

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])

    if distort:
      d = 1 + (split == 'test')
    else:
      d = 0

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label, dim, orig_image = _read_and_decode(
        filename_queue, image_pixel=image_pixel, distort=d)
    orig_image.set_shape([48, 48, 1 + multi])
    image.set_shape([dim, dim, 1 + multi])
    image = tf.transpose(image, [2, 0, 1])

    if split == 'train':
      images, sparse_labels = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=2,
          capacity=2000 + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=2000)
    else:
      images, sparse_labels, orig_images = tf.train.batch(
          [image, label, orig_image],
          batch_size=batch_size,
          num_threads=1,
          capacity=1000 + 3 * batch_size)
      if patching:
        t_images = tf.tile(orig_images, [4, 1, 1, 1])
        c_images = tf.image.extract_glimpse(
            t_images, [32, 32], bxs_m2, centered=True, normalized=False)
        c2images = tf.image.extract_glimpse(
            t_images, [32, 32],
            2 * np.array(bxs_m2),
            centered=True,
            normalized=False)
        c3images = tf.image.extract_glimpse(
            t_images, [32, 32],
            3 * np.array(bxs_m2),
            centered=True,
            normalized=False)
        c_images = tf.map_fn(tf.image.per_image_standardization, c_images)
        c2images = tf.map_fn(tf.image.per_image_standardization, c2images)
        c3images = tf.map_fn(tf.image.per_image_standardization, c3images)
        c_images = tf.transpose(c_images, [0, 3, 1, 2])
        c2images = tf.transpose(c2images, [0, 3, 1, 2])
        c3images = tf.transpose(c3images, [0, 3, 1, 2])
        # cc_images = tf.concat([images, m_images, c_images], axis=0)
        # cc_labels = tf.tile(sparse_labels, [9])
        cc_images = tf.concat([images, c_images, c2images, c3images], axis=0)
        cc_labels = tf.tile(sparse_labels, [13])
    features = {
        'images': images,
        'labels': tf.one_hot(sparse_labels, 5),
        'recons_image': images,
        'recons_label': sparse_labels,
        'height': dim,
        'depth': 1 + multi,
        'num_classes': 5,
        'cc_images': cc_images,
        'cc_recons_label': cc_labels,
        'cc_labels': tf.one_hot(cc_labels, 5),
    }

    return features
