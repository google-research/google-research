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

"""Used to unify various data providers under a single namespace."""

import logging
import gin
import gin.tf
import tensorflow as tf


from tf3d.datasets import rio  # pylint: disable=g-bad-import-order
from tf3d.datasets import scannet_scene  # pylint: disable=g-bad-import-order
from tf3d.datasets import waymo_object_per_frame  # pylint: disable=g-bad-import-order


_DATASET_MAP = {
    'rio': rio,
    'scannet_scene': scannet_scene,
    'waymo_object_per_frame': waymo_object_per_frame,
}


def get_file_pattern(dataset_name,
                     split_name,
                     dataset_format=None,
                     dataset_dir=None):
  """Returns the file pattern given the dataset name and split.

  Args:
    dataset_name: Dataset name.
    split_name: A train/test split name.
    dataset_format: A str of the dataset format.
    dataset_dir: The base directory of the dataset sources.

  Returns:
    A string containing the file pattern.
  """
  if dataset_dir is not None:
    return _DATASET_MAP[dataset_name].get_file_pattern(
        split_name=split_name,
        dataset_dir=dataset_dir,
        dataset_format=dataset_format)
  else:
    return _DATASET_MAP[dataset_name].get_file_pattern(split_name=split_name)


def get_decode_fn(dataset_name,
                  include_saved_predictions=False):
  decoder_params = {}
  if include_saved_predictions:
    decoder_params['include_saved_predictions'] = include_saved_predictions
  return _DATASET_MAP[dataset_name].get_decode_fn(**decoder_params)


def get_items_to_descriptions(dataset_name):
  return _DATASET_MAP[dataset_name].ITEMS_TO_DESCRIPTIONS


def get_num_samples(dataset_name, split_name):
  return _DATASET_MAP[dataset_name].SPLITS_TO_SIZES[split_name]


def _get_params(dataset_name):
  params = {}
  if _DATASET_MAP[dataset_name].IGNORE_LABEL is not None:
    params['ignore_label'] = _DATASET_MAP[dataset_name].IGNORE_LABEL
  if _DATASET_MAP[dataset_name].NUM_CLASSES is not None:
    params['num_classes'] = _DATASET_MAP[dataset_name].NUM_CLASSES
  return params




def _read_data(file_read_func, file_pattern, shuffle, num_readers,
               filenames_shuffle_buffer_size, num_epochs, read_block_length,
               shuffle_buffer_size):
  """Gets a dataset tuple.

  Args:
    file_read_func: Function to use in tf.contrib.data.parallel_interleave, to
      read every individual file into a tf.data.Dataset.
    file_pattern: A string containing a file pattern that corresponds to a set
      of files containing the data.
    shuffle: Whether data should be processed in the order they are read in, or
      shuffled randomly.
    num_readers: Number of file shards to read in parallel.
    filenames_shuffle_buffer_size: Buffer size to be used when shuffling file
      names.
    num_epochs: The number of times a data source is read. If set to zero, the
      data source will be reused indefinitely.
    read_block_length: Number of records to read from each reader at once.
    shuffle_buffer_size: Buffer size to be used when shuffling.

  Returns:
    A tf.data.Dataset.
  """
  # Shard, shuffle, and read files.
  dataset = tf.data.Dataset.list_files(
      file_pattern=file_pattern, shuffle=shuffle)
  if shuffle:
    dataset = dataset.shuffle(filenames_shuffle_buffer_size)
  elif num_readers > 1:
    logging.warning('`shuffle` is false, but the input data stream is '
                    'still slightly shuffled since `num_readers` > 1.')
  dataset = dataset.repeat(num_epochs or None)

  records_dataset = dataset.interleave(
      map_func=file_read_func,
      cycle_length=num_readers,
      block_length=read_block_length,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=shuffle)

  if shuffle:
    records_dataset = records_dataset.shuffle(shuffle_buffer_size)
  return records_dataset




def tfrecord_read_fn(filename):
  return tf.data.TFRecordDataset(filename).prefetch(1)


@gin.configurable(
    'get_tf_data_decoder', denylist=['batch_size', 'is_training'])
def get_tf_data_decoder(dataset_format,
                        decode_fn,
                        file_pattern,
                        batch_size,
                        is_training,
                        preprocess_fn=None,
                        feature_keys=None,
                        label_keys=None,
                        num_readers=64,
                        filenames_shuffle_buffer_size=100,
                        num_epochs=0,
                        read_block_length=32,
                        shuffle_buffer_size=256,
                        num_parallel_batches=8,
                        num_prefetch_batches=2,
                        ):
  """Reads a tf.data.Dataset given a decoder and outputs tensor dictionaries.

  Args:
    dataset_format: Currently 'tfexample' and 'recordio' are supported.
    decode_fn: Decoder function.
    file_pattern: A string containing the file pattern that represents the
      sstable.
    batch_size: Batch size.
    is_training: Whether reading data in training mode or not. If in training,
      data will be shuffled and if not it won't be shuffled. Also in training
      preprocessing can act different than in eval time.
    preprocess_fn: A function that preprocesses data.
    feature_keys: Either None or a list[str] with keys in features.
    label_keys: Either None or a list[str] with keys in labels.
    num_readers: Number of file shards to read in parallel.
    filenames_shuffle_buffer_size: Buffer size to be used when shuffling file
      names.
    num_epochs: The number of times a data source is read. If set to zero, the
      data source will be reused indefinitely.
    read_block_length: Number of records to read from each reader at once.
    shuffle_buffer_size: Buffer size to be used when shuffling.
    num_parallel_batches: Number of batches to produce in parallel. If this is
      run on a 2x2 TPU set this to 8.
    num_prefetch_batches: Number of batches to prefetch. Prefetch decouples
      input pipeline and model so they can be pipelined resulting in higher
      throughput. Set this to a small constant and increment linearly until the
      improvements become marginal or you exceed your cpu memory budget. Setting
      this to -1, automatically tunes this value for you.

  Returns:
    Return a tf.data.dataset where each element is a ditctionary with features
    and labels; if not executing eagerly i.e. under tf1 environment, returns a
    dictionary with features and labels instead.
  """

  def _process_fn(key, value):
    """Sets up tf graph that decodes and preprocesses input."""
    tensors_dict = decode_fn(value)
    if preprocess_fn is None:
      return tensors_dict
    else:
      output_keys = feature_keys + label_keys
      return preprocess_fn(
          inputs=tensors_dict, output_keys=output_keys, is_training=is_training)

  if dataset_format == 'tfrecord':
    read_fn = tfrecord_read_fn
  else:
    raise ValueError('Unknown dataset type')

  # Read data
  dataset = _read_data(
      file_read_func=read_fn,
      file_pattern=file_pattern,
      num_readers=num_readers,
      shuffle=is_training,
      filenames_shuffle_buffer_size=filenames_shuffle_buffer_size,
      num_epochs=num_epochs,
      read_block_length=read_block_length,
      shuffle_buffer_size=shuffle_buffer_size)

  if dataset_format == 'tfrecord':
    # insert dummy key to form (key, value pair)
    dataset = dataset.map(lambda x: (None, x))

  # Preprocess data
  dataset_dict = tf.data.Dataset.batch(
      dataset.map(
          _process_fn, num_parallel_calls=num_parallel_batches),
      batch_size=batch_size,
      drop_remainder=True)
  dataset_dict = dataset_dict.prefetch(num_prefetch_batches)

  return dataset_dict




@gin.configurable(denylist=['batch_size', 'is_training'])
def get_tf_data_dataset(dataset_name,
                        split_name,
                        batch_size,
                        is_training,
                        preprocess_fn=None,
                        feature_keys=None,
                        label_keys=None,
                        num_readers=64,
                        filenames_shuffle_buffer_size=100,
                        num_epochs=0,
                        read_block_length=32,
                        shuffle_buffer_size=256,
                        num_parallel_batches=8,
                        num_prefetch_batches=2,
                        dataset_dir=None,
                        dataset_format=None):
  """Reads a tf.data.Dataset given a dataset name and split and outputs tensors.

  Args:
    dataset_name: Dataset name.
    split_name: A train/test split name.
    batch_size: Batch size.
    is_training: Whether reading data in training mode or not. If in training,
      data will be shuffled and if not it won't be shuffled. Also in training
      preprocessing can act different than in eval time.
    preprocess_fn: A function that preprocesses data.
    feature_keys: Either None or a list[str] with keys in features.
    label_keys: Either None or a list[str] with keys in labels.
    num_readers: Number of file shards to read in parallel.
    filenames_shuffle_buffer_size: Buffer size to be used when shuffling file
      names.
    num_epochs: The number of times a data source is read. If set to zero, the
      data source will be reused indefinitely.
    read_block_length: Number of records to read from each reader at once.
    shuffle_buffer_size: Buffer size to be used when shuffling.
    num_parallel_batches: Number of batches to produce in parallel. If this is
      run on a 2x2 TPU set this to 8.
    num_prefetch_batches: Number of batches to prefetch. Prefetch decouples
      input pipeline and model so they can be pipelined resulting in higher
      throughput. Set this to a small constant and increment linearly until the
      improvements become marginal or you exceed your cpu memory budget. Setting
      this to -1, automatically tunes this value for you.
    dataset_dir: The base directory of the dataset sources.
    dataset_format: If not None, a str of dataset format, can be 'tfrecord',
    'sstable' or 'recordio'.

  Returns:
    Return a tf.data.dataset where each element is a ditctionary with features
    and labels; if not executing eagerly i.e. under tf1 environment, returns a
    dictionary with features and labels instead.
  """
  if dataset_format is None:
    dataset_format = _DATASET_MAP[dataset_name].DATASET_FORMAT
  file_pattern = get_file_pattern(
      dataset_name=dataset_name,
      split_name=split_name,
      dataset_dir=dataset_dir,
      dataset_format=dataset_format)

  decode_fn = get_decode_fn(dataset_name=dataset_name)
  if feature_keys is None:
    feature_keys = list(_DATASET_MAP[dataset_name].get_feature_keys())
  if label_keys is None:
    label_keys = list(_DATASET_MAP[dataset_name].get_label_keys())
  return get_tf_data_decoder(
      dataset_format=dataset_format,
      decode_fn=decode_fn,
      file_pattern=file_pattern,
      batch_size=batch_size,
      is_training=is_training,
      preprocess_fn=preprocess_fn,
      feature_keys=feature_keys,
      label_keys=label_keys,
      num_readers=num_readers,
      filenames_shuffle_buffer_size=filenames_shuffle_buffer_size,
      num_epochs=num_epochs,
      read_block_length=read_block_length,
      shuffle_buffer_size=shuffle_buffer_size,
      num_parallel_batches=num_parallel_batches,
      num_prefetch_batches=num_prefetch_batches)


