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

"""Builds a dataset with uniref sequences."""

import os
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import gin
import numpy as np
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
import typing_extensions

from dedal.data import serialization


# Type aliases.
FeatureConverter = seqio.feature_converters.FeatureConverter


@gin.configurable
class TFRecordsLoader:
  """Creates tf.data.Dataset instances for TFRecords."""

  FILE_EXT = 'tfrecords'

  def __init__(self,
               folder,
               split_folder = True,
               coder=serialization.Coder,
               input_sequence_key = 'seq',
               output_sequence_key = 'sequence'):
    self._folder = folder
    self._coder = coder
    self._input_sequence_key = input_sequence_key
    self._split_folder = split_folder
    if isinstance(self._input_sequence_key, str):
      self._input_sequence_key = (self._input_sequence_key,)

    self._output_sequence_key = output_sequence_key
    if isinstance(self._output_sequence_key, str):
      self._output_sequence_key = (self._output_sequence_key,)

    in_len = len(self._input_sequence_key)
    out_len = len(self._output_sequence_key)
    if in_len != out_len:
      raise ValueError(
          f'input_sequence_key and output_sequence_key must agree in length. '
          f'Got {in_len} and {out_len}, respectively.')

  def rename(self, example):
    for k_in, k_out in zip(self._input_sequence_key, self._output_sequence_key):
      example[k_out] = example.pop(k_in)
    return example

  def load(self, split):
    """Creates TFRecordDataset for split with prebatching transforms applied."""
    sep = '/' if self._split_folder else ''
    pattern = os.path.join(self._folder, f'*{split}*{sep}*{self.FILE_EXT}*')
    files = tf.io.gfile.glob(pattern)
    files_ds = tf.data.Dataset.from_tensor_slices(files).shuffle(len(files))
    ds = files_ds.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    ds = ds.map(self._coder.decode, num_parallel_calls=tf.data.AUTOTUNE)
    if self._input_sequence_key != self._output_sequence_key:
      ds = ds.map(self.rename)
    return ds


@gin.configurable
class TFDSLoader:
  """A wrapper around tfds.load to make it a loader."""

  def __init__(self, name = gin.REQUIRED, data_dir = None):
    self._name = name
    self._data_dir = data_dir

  def load(self, split):
    return tfds.load(name=self._name, split=split, data_dir=self._data_dir)


@gin.configurable
class TSVLoader:
  """Creates tf.data.Dataset instances from TSV files with a header."""

  def __init__(self,
               folder,
               file_pattern = '*',
               field_delim = '\t',
               use_quote_delim = False):
    self._folder = folder
    self._file_pattern = file_pattern
    self._field_delim = field_delim
    self._use_quote_delim = use_quote_delim
    self._field_names = None

  def _list_files(self, split):
    pattern = os.path.join(self._folder, split, self._file_pattern)
    return tf.io.gfile.glob(pattern)

  @property
  def field_names(self):
    if self._field_names is None:
      filename = self._list_files('train')[0]
      with tf.io.gfile.GFile(filename, 'r') as f:
        header = f.readline().strip()
      self._field_names = header.split(self._field_delim)
    return self._field_names

  def _csv_dataset_fn(self, filenames):
    return tf.data.experimental.CsvDataset(
        filenames,
        record_defaults=[tf.string] * len(self.field_names),
        header=True,
        field_delim=self._field_delim,
        use_quote_delim=self._use_quote_delim)

  def load(self, split):
    """Creates CSVDataset for split."""
    files = self._list_files(split)
    files_ds = tf.data.Dataset.from_tensor_slices(np.array(files, dtype=str))
    ds = files_ds.interleave(
        self._csv_dataset_fn,
        cycle_length=16,
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE)
    return ds.map(
        lambda *ex: {k: v for k, v in zip(self.field_names, ex)},
        num_parallel_calls=tf.data.AUTOTUNE)


################################################################################
# Interfaces DEDAL loaders with newer SeqIO-based data pipelines.
################################################################################


class DedalLoaderFnCallable(typing_extensions.Protocol):

  def __call__(self):
    Ellipsis


@gin.configurable
class DedalLoaderDataSource(seqio.FunctionDataSource):
  """Wraps a SeqIO `DataSource` over a legacy DEDAL-style loader."""

  def __init__(
      self,
      loader_cls = gin.REQUIRED,
      splits = ('train', 'validation', 'test'),
  ):

    def dataset_fn(
        split,
        shuffle_files,
        seed = None,
    ):
      del shuffle_files  # Legacy: shuffling done by `builder.Builder`.
      del seed  # Legacy: non-deterministic data pipelines by default.
      return loader_cls().load(split)

    super().__init__(
        dataset_fn=dataset_fn,
        splits=splits,
        num_input_examples=None,
        caching_permitted=False)


@gin.configurable
class SeqIOLoader:
  """A wrapper around SeqIO to make it a loader."""

  def __init__(
      self,
      mixture_or_task_name = gin.REQUIRED,
      task_feature_lengths = gin.REQUIRED,
      feature_converter_factory = gin.REQUIRED,
      seed = None,
      train_key = 'train',
  ):
    self.mixture_or_task_name = mixture_or_task_name
    self._task_feature_lengths = task_feature_lengths
    self._feature_converter_factory = feature_converter_factory
    self._seed = seed
    self._train_key = train_key

  def load(self, split):
    is_train = split == self._train_key

    return seqio.get_dataset(
        mixture_or_task_name=self.mixture_or_task_name,
        task_feature_lengths=self._task_feature_lengths,
        feature_converter=self._feature_converter_factory(),
        dataset_split=split,
        shuffle=False,
        num_epochs=None if is_train else 1,
        seed=self._seed)
