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

"""Data provider for MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
NUM_TRAIN_EXAMPLES = 60000
NUM_TEST_EXAMPLES = 10000
NUM_TRAIN_FILES = 5


LABEL_BYTES = 1
IMAGE_BYTES = IMAGE_SIZE**2 * NUM_CHANNELS


def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder(">")
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
  """Validate that filename corresponds to images for the MNIST dataset."""
  with tf.gfile.Open(filename, "rb") as f:
    magic = read32(f)
    read32(f)  # num_images, unused
    rows = read32(f)
    cols = read32(f)
    if magic != 2051:
      raise ValueError("Invalid magic number %d in MNIST file %s" % (magic,
                                                                     f.name))
    if rows != 28 or cols != 28:
      raise ValueError(
          "Invalid MNIST file %s: Expected 28x28 images, found %dx%d" %
          (f.name, rows, cols))


def check_labels_file_header(filename):
  """Validate that filename corresponds to labels for the MNIST dataset."""
  with tf.gfile.Open(filename, "rb") as f:
    magic = read32(f)
    read32(f)  # num_items, unused
    if magic != 2049:
      raise ValueError("Invalid magic number %d in MNIST file %s" % (magic,
                                                                     f.name))


def get_image_label_from_record(image_record, label_record):
  """Decodes the image and label information from one data record."""
  # Convert from tf.string to tf.uint8.
  image = tf.decode_raw(image_record, tf.uint8)
  # Convert from tf.uint8 to tf.float32.
  image = tf.cast(image, tf.float32)

  # Reshape image to correct shape.
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
  # Normalize from [0, 255] to [0.0, 1.0]
  image /= 255.0

  # Convert from tf.string to tf.uint8.
  label = tf.decode_raw(label_record, tf.uint8)
  # Convert from tf.uint8 to tf.int32.
  label = tf.to_int32(label)
  # Reshape label to correct shape.
  label = tf.reshape(label, [])  # label is a scalar
  return image, label


class MNIST(object):
  r"""Create Dataset from MNIST files.

  First download MNIST data.

    Data extraction flow.

    Extract data records.
          |
    Shuffle dataset (if is_training)
          |
    Repeat dataset after finishing all examples (if is_training)
          |
    Map parser over dataset
          |
    Batch by batch size
          |
    Prefetch dataset
          |
    Create one shot iterator
  Attributes:
    images: 4D Tensors with images of a batch.
    labels: 1D Tensor with labels of a batch.
    num_examples: (integer) Number of examples in the data.
    subset: Data subset 'train_valid', 'train', 'valid', or, 'test'.
    num_classes: (integer) Number of classes in the data.
  """

  def __init__(self, data_dir, subset, batch_size, is_training=False):

    if subset == "train":
      images_file = os.path.join(data_dir, "train-images.idx3-ubyte")
      labels_file = os.path.join(data_dir, "train-labels.idx1-ubyte")
      num_examples = NUM_TRAIN_EXAMPLES
    elif subset == "test":
      images_file = os.path.join(data_dir, "t10k-images.idx3-ubyte")
      labels_file = os.path.join(data_dir, "t10k-labels.idx1-ubyte")
      num_examples = NUM_TEST_EXAMPLES
    else:
      raise ValueError("Invalid subset: %s" % subset)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)
    # Construct fixed length record dataset.
    dataset_images = tf.data.FixedLengthRecordDataset(
        images_file, IMAGE_BYTES, header_bytes=16)

    dataset_labels = tf.data.FixedLengthRecordDataset(
        labels_file, LABEL_BYTES, header_bytes=8)

    dataset = tf.data.Dataset.zip((dataset_images, dataset_labels))
    if is_training:
      dataset = dataset.shuffle(buffer_size=300)
    dataset = dataset.repeat(-1 if is_training else 1)
    dataset = dataset.map(get_image_label_from_record, num_parallel_calls=32)
    dataset = dataset.batch(batch_size, drop_remainder=not is_training)
    dataset = dataset.prefetch(buffer_size=max(1, int(128 / batch_size)))
    iterator = dataset.make_one_shot_iterator()
    self.images, labels = iterator.get_next()
    self.labels = tf.squeeze(labels)
    self.subset = subset
    self.num_classes = 10
    self.num_examples = num_examples

