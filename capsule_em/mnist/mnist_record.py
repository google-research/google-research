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

"""input utility functions for mnist and mnist multi. NOT MAINTAINED.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path
import tensorflow.compat.v1 as tf


def _read_and_decode(filename_queue, image_pixel=28, distort=0):
  """Read tf records of MNIST images and labels."""
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
          'depth': tf.FixedLenFeature([], tf.int64)
      })

  # Convert from a scalar string tensor (whose single string has
  # length image_pixels) to a uint8 tensor with shape
  # [image_pixels].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.reshape(image, [image_pixel, image_pixel, 1])
  print(image.get_shape()[0].value)
  image.set_shape([image_pixel, image_pixel, 1])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255)
  if distort == 1:
    image = tf.reshape(image, [28, 28])
    image = tf.random_crop(image, [24, 24])
    # 0.26179938779 is 15 degress in radians
    # image = contrib_image.rotate(image,
    #                             random.uniform(-0.26179938779, 0.26179938779))
    image = tf.reshape(image, [24, 24, 1])
  elif distort == 2:
    image = tf.reshape(image, [28, 28])
    image = tf.expand_dims(image, 2)
    image = tf.image.central_crop(image, central_fraction=24 / 28)
    image = tf.squeeze(image, 2)
    image = tf.reshape(image, [24, 24, 1])

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def inputs(train_dir,
           batch_size,
           split,
           multi,
           shift,
           train_file='shifted_mnist.tfrecords',
           height=28,
           distort=False):
  """Reads input data num_epochs times."""
  if multi:
    print('not supported.')
  filename = os.path.join(train_dir, '{}_{}{}'.format(split, shift, train_file))

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])

    d = 0
    if distort:
      d = 1 + (split == 'test')

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = _read_and_decode(
        filename_queue, image_pixel=height, distort=d)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    if split == 'train':
      images, sparse_labels = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=2,
          capacity=20000 + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=20000)
    else:
      images, sparse_labels = tf.train.batch([image, label],
                                             batch_size=batch_size,
                                             num_threads=1,
                                             capacity=1000 + 3 * batch_size)
    features = {
        'images': images,
        'labels': tf.one_hot(sparse_labels, 10),
        'recons_image': images,
        'recons_label': sparse_labels,
        'height': 24 if distort else height,
        'depth': 1,
        'num_classes': 10,
    }

    return features
