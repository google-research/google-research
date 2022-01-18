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

"""tf.data.Dataset interface to the CIFAR10 dataset."""

from absl import logging
import tensorflow as tf

IMG_DIM = 32
NUM_CHANNELS = 3
NUM_LABELS = 10


def dataset_randomized(pattern):
  """tf.data.Dataset object for CIFAR-10 training data."""
  filenames = tf.io.gfile.glob(pattern)
  logging.info('*** Input Files ***')
  for input_file in filenames:
    logging.info('  %s', input_file)

  ds_filenames = tf.data.Dataset.from_tensor_slices(tf.constant(filenames))
  ds_filenames = ds_filenames.shuffle(buffer_size=len(filenames))

  dataset = tf.data.TFRecordDataset(ds_filenames)

  # Create a description of the features.
  feature_description = {
      'image/class/label': tf.io.FixedLenFeature([], tf.int64),
      'image/class/shuffled_label': tf.io.FixedLenFeature([], tf.int64),
      'image/encoded': tf.io.FixedLenFeature([], tf.string),
  }

  def decode_image(image):
    image = tf.io.decode_png(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [IMG_DIM * IMG_DIM * NUM_CHANNELS])
    return image

  def parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, feature_description)
    features['image/encoded'] = decode_image(features['image/encoded'])
    return features

  dataset = dataset.map(parse_function)
  return dataset
