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

# Lint as: python2, python3
"""Dataset builder for ImageNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from six.moves import range
import tensorflow.compat.v2 as tf
from uq_benchmark_2019 import image_data_utils
from uq_benchmark_2019.imagenet import resnet_preprocessing

# total training set from ImageNet competition
IMAGENET_TRAIN_AND_VALID_SHARDS = 1024

# Number of shards to split off from this set for validation
IMAGENET_VALID_SHARDS = 100

IMAGENET_SHAPE = (224, 224, 3)


class ImageNetInput(object):
  """Generates ImageNet input_fn for training, validation, and testing.

  The complete ImageNet training data is assumed to be in TFRecord format with
  keys as specified in the dataset_parser below, sharded across 1024 files,
  named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  ImageNet's validation data (which is treated as test data here) is in the same
  format but sharded in 128 files. 'Validation' data here refers to a subset of
  the ImageNet training set split off for validation.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self, is_training, data_dir, dataset_split, batch_size,
               use_bfloat16=False, fake_data=False):
    """Initialize ImageNetInput object.

    Args:
      is_training: `bool` for whether the input is for training.
      data_dir: `str` for the directory of the training and validation data.
      dataset_split: `str`, either 'train', 'valid', or 'test'.
      batch_size: `int`, dataset batch size.
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      fake_data: If True, use synthetic random data.
    """
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.data_dir = data_dir
    self.dataset_split = dataset_split
    self.fake_data = fake_data
    self.batch_size = batch_size

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.io.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.io.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        use_bfloat16=self.use_bfloat16)

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    label = tf.cast(tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[1]), dtype=tf.int32) - 1,
                    dtype=tf.float32)

    return image, label

  def input_fn(self):
    """Input function which provides a single batch for train or eval.

    Returns:
      A `tf.data.Dataset` object.
    """
    if self.fake_data:
      return image_data_utils.make_fake_data(IMAGENET_SHAPE)

    train_path_tmpl = os.path.join(self.data_dir, 'train-{0:05d}*')
    if self.dataset_split == 'train':
      file_pattern = [train_path_tmpl.format(i)
                      for i in range(IMAGENET_VALID_SHARDS,
                                     IMAGENET_TRAIN_AND_VALID_SHARDS)]
    elif self.dataset_split == 'valid':
      file_pattern = [train_path_tmpl.format(i)
                      for i in range(IMAGENET_VALID_SHARDS)]
    elif self.dataset_split == 'test':
      file_pattern = os.path.join(self.data_dir, 'validation-*')
    else:
      raise ValueError(
          "Dataset_split must be 'train', 'valid', or 'test', was %s"
          % self.dataset_split)

    # Shuffle the filenames to ensure better randomization.
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.interleave(fetch_dataset, cycle_length=16)

    if self.is_training:
      dataset = dataset.shuffle(1024)

    # Parse, pre-process, and batch the data in parallel (for speed, it's
    # necessary to apply batching here rather than using dataset.batch later)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser,
            batch_size=self.batch_size,
            num_parallel_batches=2,
            drop_remainder=True))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self.is_training:
      # Use a private thread pool and limit intra-op parallelism. Enable
      # non-determinism only for training.
      options = tf.data.Options()
      options.experimental_threading.max_intra_op_parallelism = 1
      options.experimental_threading.private_threadpool_size = 16
      options.experimental_deterministic = False
      dataset = dataset.with_options(options)

    return dataset
