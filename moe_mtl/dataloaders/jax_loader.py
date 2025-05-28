# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utility to load data and feed models."""
import functools
import inspect
from typing import Any, Callable, Optional, Dict

from absl import logging

import gin
import jax
import ml_collections
import seqio
import tensorflow.compat.v2 as tf
import tensorflow as tf  # tf_google
import tensorflow_datasets as tfds
from vmoe.data.input_pipeline import get_datasets

DEFAULT_SHUFFLE_BUFFER = 50_000
Data = Dict[str, Any]

TFDS_MANUAL_DIR = None
TFDS_DATA_DIR = None


@gin.configurable
def tfrecords_sequential_loader(
    file_pattern,
    use_sstable = False,
    parse_fn = None  # pylint: disable=g-bare-generic
):
  """Prepares and returns a `tf.data.Dataset` from tfrecords files."""
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  # fetch file paths to python to split files per host.
  files = list(dataset)
  files = [
      f for i, f in enumerate(files)
  ]
  dataset = tf.data.Dataset.from_tensor_slices(files)

  if not use_sstable:
    dataset = dataset.interleave(
        lambda f: tf.data.TFRecordDataset(f).prefetch(1),
        num_parallel_calls=tf.data.AUTOTUNE)
  else:
    dataset = dataset.interleave(
        tf_google.data.SSTableDataset, num_parallel_calls=tf.data.AUTOTUNE)
  if parse_fn is not None:
    dataset = dataset.map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


@gin.configurable
def tfrecords_subset_loader(
    file_pattern,
    use_sstable = False,
    leave_out = 0,
    parse_fn = None  # pylint: disable=g-bare-generic
):
  """Prepares and returns a `tf.data.Dataset` from tfrecords files."""
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  # fetch file paths to python to split files per host.
  files = list(dataset)
  files = [
      f for i, f in enumerate(files)
  ]
  if leave_out != 0:
    if leave_out > 0:
      num_files = int(len(files) * leave_out)
      assert num_files > 1
      files = files[:-num_files]
    else:
      num_files = int(len(files) * leave_out)
      assert num_files < 1
      files = files[num_files:]

  dataset = tf.data.Dataset.from_tensor_slices(files)

  if not use_sstable:
    dataset = dataset.interleave(
        lambda f: tf.data.TFRecordDataset(f).prefetch(1),
        num_parallel_calls=tf.data.AUTOTUNE)
  else:
    dataset = dataset.interleave(
        tf_google.data.SSTableDataset, num_parallel_calls=tf.data.AUTOTUNE)
  if parse_fn is not None:
    dataset = dataset.map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


@gin.configurable
def get_pjit_input(
    loader_fn=gin.REQUIRED,
    batch_size = gin.REQUIRED,
    preprocess_fn = None,
    postprocess_fn = None,
    filter_fn = None,
    map_fn = None,
    batch_map_fn = None,
    flatten = False,
    repeat = False,
    shuffle = True,
    shuffle_multiplier = 16,
    cache = False,
    ignore_errors = False,
    unbatch = False,
    vocab = None,
    return_dataset = False):
  """Input functions."""
  if isinstance(loader_fn, str):
    # When passed in through xm.hyper flags, these do not get parsed
    # correctly. They come in a strings, e.g., '@prefix/function_name', and
    # are converted to the configured function here, and removing the '@'.
    if '()' in loader_fn:
      # Call the function.
      loader_fn = gin.get_configurable(loader_fn.replace('@', '')[:-2])()
    else:
      loader_fn = gin.get_configurable(loader_fn.replace('@', ''))

  dataset = loader_fn()
  # Calculate the batch size for each device.
  per_process_batch_size = batch_size // jax.process_count()

  logging.info('BATCH SIZE: %d', per_process_batch_size)
  num_parallel_calls = tf.data.experimental.AUTOTUNE

  if preprocess_fn:
    dataset = preprocess_fn(dataset)
  if cache:
    dataset = dataset.cache()
  if repeat:
    dataset = dataset.repeat()
  if shuffle:
    dataset = dataset.shuffle(per_process_batch_size * shuffle_multiplier)
  if map_fn:
    fn_args = set(inspect.signature(map_fn).parameters.keys())
    if 'vocab' in fn_args:
      map_fn = functools.partial(map_fn, vocab=vocab)
    dataset = dataset.map(
        map_fn, num_parallel_calls=num_parallel_calls, deterministic=False)
  if postprocess_fn:
    fn_args = set(inspect.signature(postprocess_fn).parameters.keys())
    if 'vocab' in fn_args:
      postprocess_fn = functools.partial(postprocess_fn, vocab=vocab)
    dataset = postprocess_fn(dataset)
  if flatten:
    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
  if filter_fn:
    dataset = dataset.filter(filter_fn)
  if ignore_errors:
    dataset = dataset.apply(
        tf.data.experimental.ignore_errors(log_warning=True))

  if unbatch:
    dataset = dataset.unbatch()

  if return_dataset:
    return dataset

  dataset = dataset.batch(per_process_batch_size, drop_remainder=True)
  if batch_map_fn:
    dataset = dataset.map(
        batch_map_fn,
        num_parallel_calls=num_parallel_calls,
        deterministic=False)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset, tfds.as_numpy(dataset)
  # Number of images processed in each step.


def get_data_params(
    name,
    split,
    process,
    shuffle_buffer,
    cache,
    batch_size=1024,
    data_dir=None,
    manual_dir=None):
  """Returns dataset parameters."""
  config = ml_collections.ConfigDict()
  config.name = name
  config.split = split
  config.process = process
  config.batch_size = batch_size
  config.prefetch = 'autotune'
  config.prefetch_device = 2
  config.data_dir = data_dir
  config.manual_dir = manual_dir
  if shuffle_buffer:
    config.shuffle_buffer = shuffle_buffer
  if cache:
    config.cache = cache
  return config


@gin.configurable
def get_imagenet(
    batch_size=1024,
    image_size=224,
    mode='train',
    rand_aug=True,
):
  """Get ImageNet dataloaders."""
  config = ml_collections.ConfigDict()
  config.re_init_step = 0
  config.dataset = ml_collections.ConfigDict()
  pp_common = ('value_range(-1,1)|copy("image","images")|'
               'onehot(1_000, inkey="label", outkey="labels")|'
               'keep("images", "labels")')
  if rand_aug:
    train_process = f'decode_jpeg_and_inception_crop({image_size})|flip_lr|randaug(2,9)|{pp_common}'
  else:
    train_process = f'decode_jpeg_and_inception_crop({image_size})|flip_lr|{pp_common}'
  # Dataset variation used for training.
  config.dataset.train = get_data_params(
      name='imagenet2012',
      split='train[:99%]',
      process=train_process,
      shuffle_buffer=250_000,
      batch_size=batch_size,
      cache=None)
  # Dataset variation used for validation.
  config.dataset.valid = get_data_params(
      name='imagenet2012',
      split='train[99%:]',
      process=f'decode|resize({image_size})|{pp_common}',
      shuffle_buffer=None,
      batch_size=batch_size,
      cache='batched')
  # Dataset variation used for test.
  config.dataset.test = get_data_params(
      name='imagenet2012',
      split='validation',
      process=f'decode|resize({image_size})|{pp_common}',
      shuffle_buffer=None,
      batch_size=batch_size,
      cache='batched')

  datasets = get_datasets(config.dataset)
  return datasets[mode], datasets[mode]
