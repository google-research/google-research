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

"""Builds a dataset with uniref sequences."""

import os
from typing import Mapping, Optional, Sequence, Union

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from dedal.data import serialization


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
class CSVLoader:
  """Creates tf.data.Dataset instances for CSV files."""

  def __init__(self,
               folder,
               fields,
               fields_to_use,
               header = True,
               file_ext = None):
    self._folder = folder
    self._field_names = list(fields.keys())
    self._field_dtypes = list(fields.values())
    self._field_name_to_idx = {
        k: self._field_names.index(k) for k in fields_to_use
    }
    self._header = header
    self._file_ext = file_ext

  def _csv_dataset_fn(self, filenames):
    return tf.data.experimental.CsvDataset(
        filenames, record_defaults=self._field_dtypes, header=self._header)

  def load(self, split):
    """Creates CSVDataset for split."""
    file_pattern = '.'.join(filter(None, ['*', self._file_ext]))
    pattern = os.path.join(self._folder, split, file_pattern)
    files = tf.io.gfile.glob(pattern)
    files_ds = tf.data.Dataset.from_tensor_slices(files).shuffle(len(files))
    ds = files_ds.interleave(
        self._csv_dataset_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    ds = ds.map(
        lambda *ex: {k: ex[i] for k, i in self._field_name_to_idx.items()},
        num_parallel_calls=tf.data.AUTOTUNE)
    return ds
