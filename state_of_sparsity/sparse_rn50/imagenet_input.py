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

"""Efficient ImageNet input pipeline using tf.data.Dataset.

Branched from:
github.com/tensorflow/tpu/blob/master/models/official/resnet/imagenet_input.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple
import functools
import os
import tensorflow as tf
from state_of_sparsity.sparse_rn50 import resnet_preprocessing
from tensorflow.contrib import cloud as contrib_cloud
from tensorflow.contrib import data as contrib_data


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = resnet_preprocessing.preprocess_image(
        image_bytes=image_bytes, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


class ImageNetTFExampleInput(object):
  """Base class for ImageNet input_fn generator.

  Args:
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
    num_parallel_calls: `int` for the number of parallel threads.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               image_size=224,
               transpose_input=False,
               num_parallel_calls=8):
    self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.num_parallel_calls = num_parallel_calls

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return images, labels

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 1

    return image, label

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1

    dataset = self.make_source_dataset(current_host, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser, batch_size=batch_size,
            num_parallel_batches=self.num_parallel_calls, drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=self.num_parallel_calls)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class ImageNetInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=224,
               num_parallel_calls=8,
               cache=False):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data;
          if 'null' (the literal string 'null') or implicitly False
          then construct a null pipeline, consisting of empty images
          and blank labels.
      image_size: `int` image height and width.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
    """
    super(ImageNetInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.data_dir = data_dir
    # TODO(b/112427086):  simplify the choice of input source
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3], tf.bfloat16
                    if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        contrib_data.parallel_interleave(
            fetch_dataset, cycle_length=64, sloppy=True))

    if self.cache:
      dataset = dataset.cache().apply(
          contrib_data.shuffle_and_repeat(1024 * 16))
    else:
      dataset = dataset.shuffle(1024)
    return dataset


# Defines a selection of data from a Cloud Bigtable.
BigtableSelection = namedtuple('BigtableSelection',
                               ['project',
                                'instance',
                                'table',
                                'prefix',
                                'column_family',
                                'column_qualifier'])


class ImageNetBigtableInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a Bigtable for training or evaluation.
  """

  def __init__(self, is_training, use_bfloat16, transpose_input, selection):
    """Constructs an ImageNet input from a BigtableSelection.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      selection: a BigtableSelection specifying a part of a Bigtable.
    """
    super(ImageNetBigtableInput, self).__init__(
        is_training=is_training,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.selection = selection

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    data = self.selection
    client = contrib_cloud.BigtableClient(data.project, data.instance)
    table = client.table(data.table)
    ds = table.parallel_scan_prefix(data.prefix,
                                    columns=[(data.column_family,
                                              data.column_qualifier)])
    # The Bigtable datasets will have the shape (row_key, data)
    ds_data = ds.map(lambda index, data: data)

    if self.is_training:
      ds_data = ds_data.repeat()

    return ds_data
