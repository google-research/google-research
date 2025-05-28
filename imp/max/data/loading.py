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

"""Modules for loading, decoding and processing raw data."""

import math
import os
from typing import Any, Callable

from dmvr import builders
from dmvr import sources
import jax
import numpy as np
import tensorflow as tf

from imp.max.core import constants


ParsingFeatureName = constants.ParsingFeatureName


# ----------------------------------------------------------------------
# ------------------ Tools for locating data tables. -------------------
# ----------------------------------------------------------------------
def is_glob(path):
  """Returns True if the path has glob tokens."""
  return any(s in path for s in {'*', '?'})


# TODO(hassanak): integrate this with DMVR's function
def get_vocab_path(vocab = constants.HOWTO100M_EN):
  """Return the vocabulary path for a given data & language."""

  vocab = vocab.lower()
  if vocab not in constants.ALL_VOCABULARIES:
    raise ValueError(
        f'Available vocabularies are: {constants.ALL_VOCABULARIES}.'
    )
  vocab_path = os.path.join(
      constants.VOCABULARY_DIR, vocab + constants.TXT_SUFFIX
  )
  return vocab_path


def get_latest_dir(base_dir,
                   replace = constants.Replace.LATEST):
  """Replaces instances of `replace` with the latest directory.

  For each instance of `replace`, the function replaces a single directory level
  with the last modified directory in the list of subdirectories.

  Example:
  ```
  '/path/to/{latest}/dir' -> '/path/to/123/dir'
  ```

  Args:
    base_dir: the base directory to use.
    replace: the substring to replace in base_dir with the latest directory.

  Returns:
    base_dir with any matched substrings replaced.

  Raises:
    ValueError: if no subdirectories are found when matching
  """
  paths = base_dir.split(replace)
  base_dir = paths[0]
  sub_dirs = paths[1:]

  for sub_dir in sub_dirs:
    next_sub_dirs = tf.io.gfile.listdir(base_dir)
    if not next_sub_dirs:
      raise ValueError(f'No matched subdirectories in {base_dir}')
    mtimes = [
        tf.io.gfile.stat(os.path.join(base_dir, s)).mtime_nsec
        for s in next_sub_dirs
    ]
    sorted_dirs = sorted(list(zip(next_sub_dirs, mtimes)),
                         key=lambda x: x[-1])
    latest_base_dir = sorted_dirs[-1][0]
    sub_dir = sub_dir[1:] if sub_dir.startswith('/') else sub_dir
    base_dir = os.path.join(base_dir, latest_base_dir, sub_dir)

  return base_dir


def get_sharded_files(table_path,
                      prop_data = 1.0,
                      prop_seed = None,
                      num_shards = None,
                      shard_index = None):
  """Get the final list of sharded files."""
  if is_glob(table_path):
    shards = tf.io.gfile.glob(table_path)
  else:
    shards = [table_path]

  shards_list = list(shards)

  if prop_data <= 0 or prop_data > 1:
    raise ValueError(
        f'The proportion of data must be in (0, 1] but is {prop_data}.')

  num_filenames = int(math.ceil(prop_data * len(shards)))
  num_init_shards = num_filenames
  if prop_seed:
    np.random.seed(prop_seed)
    np.random.shuffle(shards_list)
  shards_list = shards_list[:num_filenames]

  if num_shards is not None and shard_index is not None:
    if num_shards > num_init_shards:
      if (num_shards == jax.process_count()
          and shard_index == jax.process_index()):
        error_message = (f'A data table with total {num_init_shards} shards is '
                         f'being re-sharded on {num_shards!r} hosts. Try to '
                         'launch the experiment on a topology with at most '
                         f'{num_init_shards} hosts.')
      else:
        error_message = (f'num_shard ({num_shards}) may not be larger than '
                         f'initial number of shards {num_init_shards}')

      raise ValueError(error_message)
    split_shard_ids = np.array_split(np.arange(num_init_shards), num_shards)
    begin_loc = split_shard_ids[shard_index][0]
    end_loc = split_shard_ids[shard_index][-1] + 1
    shards_list = shards_list[begin_loc:end_loc]
  else:
    if not (num_shards is None and shard_index is None):
      raise ValueError('Either both num_shards and shard_index '
                       'or neither should be specified.')

  return shards_list


# ----------------------------------------------------------------------
# ----------------- Tools for reading from storage. --------------------
# ----------------------------------------------------------------------




class TFRecordSource(sources.Source):
  """Source for TFRecord data format."""

  def __init__(self,
               compression_type = None,
               num_parallel_reads = None,
               **kwargs):
    """Initializes the `TFRecordSource`.

    Args:
      compression_type: Whether TFRecords are compressed or not. acceptable
        values are 'ZLIB' and 'GZIP'. If `None`, it will be set to None.
      num_parallel_reads: number of processes for reading files in parallel
      **kwargs: Optional kwargs
    """
    super().__init__()
    self._compression_type = compression_type
    self._num_parallel_reads = (
        tf.data.experimental.AUTOTUNE or num_parallel_reads
    )
    del kwargs

  def load_and_decode_shard(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, shard):
    ds = tf.data.TFRecordDataset(
        shard,
        compression_type=self._compression_type,
        num_parallel_reads=self._num_parallel_reads)

    # TFRecords do not provide an index or key per example. Use shard path as
    # key, since it can be useful later for retrieval.
    key = shard.encode('utf-8') if isinstance(shard, str) else shard
    ds = ds.map(lambda example: (key, example))

    return ds


def get_source(source_type):
  if source_type.lower() == constants.TFRECORD:
    return TFRecordSource
  else:
    raise NotImplementedError


class ExampleCustomParserBuilder(builders.BaseParserBuilder):
  """Builder for the parser function from raw `tf.train.Example`."""

  def __init__(self):
    super().__init__()
    self._features = {}
    self._name_dict: dict[str, list[str]] = {}
    self._default_parse_fn = self._custom_parse_fn

  def override_parse_fn(
      self, parse_fn):
    """Overrides the annotated image extraction function."""
    self._default_parse_fn = parse_fn

  def parse_feature(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      feature_name,
      feature_type,
      output_name = None):
    """Parses the given feature when parsing the raw `tf.train.Example`.

    The same input feature can be added more than once with different
    `output_name` but always with the same `feature_type`. This is useful when
    multiple views (with different processings down the line) of the same data
    is needed.

    Args:
      feature_name: See base class.
      feature_type: See base class.
      output_name: See base class.

    Returns:
      This instance of `ExampleCustomParserBuilder`.

    Raises:
      ValueError: `output_name` is not unique.
      ValueError: Different `feature_type` for the same input feature.
    """

    # Validate name.
    output_name = output_name or feature_name
    for name_list in self._name_dict.values():
      if output_name in name_list:
        raise ValueError(f'Given output_name {output_name} is not unique.')

    if feature_name not in self._features:
      self._features[feature_name] = feature_type
    elif self._features[feature_name] != feature_type:
      raise ValueError('Different `feature_type` given for the same feature '
                       f'{feature_name}.')

    if feature_name not in self._name_dict:
      self._name_dict[feature_name] = []
    self._name_dict[feature_name].append(output_name)

    return self

  def _custom_parse_fn(self, record):
    """Your magic goes here."""

    _, (magic_data_1, magic_data_2) = tf.io.decode_proto(
        bytes=record,
        message_type='magical_message_type',
        descriptor_source='global://',
        field_names=['magic_1', 'magic_2'],
        output_types=[tf.string, tf.string])

    outputs = {'magic_1': magic_data_1, 'magic_2': magic_data_2}
    return outputs

  def _rename_output_dict(
      self, parsed_features):
    """Renames parsed features dict based on a canonical mapping."""

    output = {}
    for k, f in parsed_features.items():
      if k not in self._features:
        continue
      output_names = self._name_dict[k]
      for output_name in output_names:
        output[output_name]: tf.Tensor = tf.identity(f)

    return output

  def _parse_fn(self, raw_data):
    """Converts bytes of raw Example to a features dictionary."""

    parsed_features = self._default_parse_fn(raw_data)
    output = self._rename_output_dict(parsed_features)

    return output


