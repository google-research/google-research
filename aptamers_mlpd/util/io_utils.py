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

# Lint as: python3
"""IO related utilities for aptamers."""

import datetime
import itertools as it
import logging as stdlib_logging
import os.path
import sys

import numpy
import pandas

# Google Internal
import tensorflow as tf
import shards
import recordio
import gfile
import logging
import io_utils
import generationinfo
import sstable


def setup_output_dir(base_dir):
  """Create a new directory for saving script output.

  Generation info is automatically saved to 'geninfo.pbtxt'.

  Args:
    base_dir: string giving the base directory to which to place this directory.

  Returns:
    String giving the path of the new directory. This is a sub-directory of
    base_dir with a name given by a UTC timestamp.
  """
  save_dir = os.path.join(base_dir, datetime.datetime.utcnow().isoformat())
  # tensorflow doesn't like : in filenames
  save_dir = save_dir.replace(':', '-')
  logging.info('creating output directory %s', save_dir)
  gfile.MakeDirs(save_dir)
  generationinfo.to_file(os.path.join(save_dir, 'geninfo.pbtxt'))
  return save_dir


def read_train_dataframe_from_hdf5(input_pattern, val_fold=0, num_folds=5):
  """Reads the hdf5 tables for all folds except val_fold.

  Args:
    input_pattern: String with the path for the hdf5 with a '%d' for the fold.
    val_fold: Integer. The zero-based validation fold num, must be < num_folds.
    num_folds: Integer. The number of folds total, including the validation
      fold.
  Returns:
    A pandas dataframe containing the training data loaded from hdf5
  Raises:
    ValueError: if val_fold is <0 or >= num_folds.
  """
  if val_fold >= num_folds or val_fold < 0:
    raise ValueError('The validation fold number must be less than the number '
                     'of folds. Was %d out of %d folds.', val_fold, num_folds)

  paths = [input_pattern % n for n in range(num_folds) if n != val_fold]
  logging.info('The train paths are: %s', paths)
  return pandas.concat([io_utils.read_dataframe_from_hdf5(p) for p in paths])


def read_val_dataframe_from_hdf5(input_pattern, val_fold=0):
  """Reads the hdf5 tables for the validation fold only.

  Args:
    input_pattern: String with the path for the hdf5 with a '%d' for the fold.
    val_fold: Integer. The zero-based validation fold num.
  Returns:
    A pandas dataframe containing the validation data loaded from hdf5
  """
  logging.info('The validation path is : %s', input_pattern % val_fold)
  return io_utils.read_dataframe_from_hdf5(input_pattern % val_fold)


def write_protos_to_recordio(pb_list, filename):
  """Saves an iterable of protos into a recordIO file.

  Args:
    pb_list: List of protos.
    filename: string name of the file for the recordIO output.
  """
  with recordio.RecordWriter(filename, 'w') as f:
    for pb in pb_list:
      f.WriteRecord(pb.SerializeToString())


def read_protos_from_recordio(filename, pb_type):
  """Yields protos from a recordIO file.

  Args:
    filename: string name of the file of the recordIO.
    pb_type: the proto type
  Yields:
    Protos of the pb_type.
  """
  with recordio.RecordReader(filename) as f:
    for record in f:
      pb = pb_type.FromString(record)
      yield pb


def log_to_stderr_and_file(path, level=stdlib_logging.INFO, loggers=None):
  """Setup logging to the given path and stderr.

  We only log INFO and higher level logging messages to stderr. DEBUG only
  goes to the log file.

  Args:
    path: string path to which to log output.
    level: optional log level.
    loggers: optional loggers to setup. By default, use the root logger only.
  """
  if loggers is None:
    loggers = [stdlib_logging.getLogger()]

  task_log = gfile.GFile(path, 'a')
  formatter = logging.PythonFormatter()

  for handler in [
      stdlib_logging.StreamHandler(sys.stderr),
      stdlib_logging.StreamHandler(task_log)
  ]:
    handler.setLevel(level)
    handler.setFormatter(formatter)
    for logger in loggers:
      logger.addHandler(handler)


def load_sstable(pattern, value_type, has_known_shard_key=False):
  """Returns an SSTable with the given proto type as values.

  Args:
    pattern: The file pattern of the input. It may be a glob pattern or a
        sharded filename pattern.
    value_type: The protobuf class of the values.
    has_known_shard_key: If True, the sharded input can be loaded using
        ShardedSSTable. If False, a sharded file path is opened using
        MergedSSTable.

  Returns:
    An SSTable corresponding to the requested input path.

  Raises:
    ValueError: No SSTable could be found at the requested location.
  """
  if shards.IsShardedFileSpec(pattern):
    paths = shards.GenerateShardedFilenames(pattern)
  else:
    paths = gfile.Glob(pattern)

  if not paths:
    raise ValueError('No files found for SSTable %s' % pattern)
  elif len(paths) == 1:
    return sstable.SSTable(paths[0], wrapper=sstable.TableWrapper(value_type))
  elif has_known_shard_key:
    return sstable.ShardedSSTable(
        paths, wrapper=sstable.TableWrapper(value_type))
  else:
    return sstable.MergedSSTable(
        paths, wrapper=sstable.TableWrapper(value_type))


def _get_feature_dtype(key, tf_feature, int64_types, tolerant_byte=False):
  """Returns a tuple of (name, dtype) for the feature."""
  kind = tf_feature.WhichOneof('kind')
  if kind == 'bytes_list':
    if tolerant_byte:
      return (str(key), numpy.float64)
    else:
      raise ValueError('Unsupported non-index feature dtype.')
  elif kind == 'float_list':
    return (str(key), numpy.float64)
  elif kind == 'int64_list':
    if key in int64_types:
      return (str(key), numpy.int64)
    else:
      return (str(key), numpy.uint32)


def _get_feature_value(tf_feature):
  """Returns the encoded value of the feature."""
  kind = tf_feature.WhichOneof('kind')
  value_list = getattr(tf_feature, kind).value[:]
  value = value_list[0] if len(value_list) == 1 else value_list
  return value


def _make_validator(dtype):

  def validate_int(value):
    return numpy.iinfo(dtype).min <= value <= numpy.iinfo(dtype).max

  def validate(value):
    del value  # unused by validate
    return True

  if numpy.issubdtype(dtype, int):
    return validate_int
  else:
    return validate


def _batch_sstable_values(table, chunk_size):
  """Reads SSTables values out in batches using minimal memory.

  Args:
    table: Reference to a pywrap.sstable
    chunk_size: Number of values to read per iteration

  Yields:
    chunk_size values from the SSTable (final batch might be short)

  Raises:
    ValueError: chunk_size is not an int.
  """
  if not isinstance(chunk_size, int):
    raise ValueError('The chunk_size must be an int. Found %s.' % chunk_size)

  i = table.values()
  while True:
    # wrapping the chunk allows more efficient garbage collection
    wrapped_chunk = [list(it.islice(i, chunk_size))]
    if not wrapped_chunk[0]:
      break
    yield wrapped_chunk.pop()


def tfexample_sstable_to_dataframe(path,
                                   index='sequence',
                                   int64_types=None,
                                   max_size_per_df=None,
                                   num_batches=None,
                                   tolerant_byte=False):
  """Converts an SSTable of tf.Example protos into the corresponding DataFrame.

  Args:
    path: path to an SSTable containing tf.Example proto values.
    index: The column on which to index the dataframe.
    int64_types: A list of columns stored in int64_list in TF.Examples that need
        the full representational capacity of a 64-bit int. All other values
        are checked and casted to 32-bit ints.
    max_size_per_df: Integer. The maximum number of rows to include before
        yielding. A value of None means to yield the whole SSTable.
    num_batches: The maximum number of batches to write. If None, the whole
        SSTable is processed.
    tolerant_byte: optional boolean indicating whether to accept columns with
      bytes_list features.

  Yields:
    pandas.DataFrame containing the data in the SSTable.

  Raises:
    ValueError: The index value is not present in the tf.Examples, not all
        tf.Examples have the same attributes, or the representational
        capacity of the int columns is insufficient.
  """
  if int64_types is None:
    int64_types = ['cluster']
  table = load_sstable(
      path, value_type=tf.Example.FromString, has_known_shard_key=False)

  # Determine features and their types from the first entry in the table.
  first_example = table.values()[0]
  # The Features message is just a single map from name to Feature, so the below
  # selection extracts the mapping from string name to feature value.
  first_feature_map = first_example.features.ListFields()[0][1]
  all_keys = sorted(first_feature_map.keys())
  if len(all_keys) != len(set(all_keys)):
    raise ValueError(
        'Examples cannot have duplicated keys names: %s' % all_keys)
  if index not in all_keys:
    raise ValueError('Cannot index by %s for data with %s' % (index, all_keys))
  data_keys = [key for key in all_keys if key != index]
  data_dtypes = [
      _get_feature_dtype(key, first_feature_map[key], int64_types,
                         tolerant_byte)
      for key in data_keys
  ]
  data_validators = [_make_validator(dtype) for _, dtype in data_dtypes]
  keys_and_validators = list(zip(data_keys, data_validators))

  # Pre-allocate a numpy array of the correct type for all data.
  num_examples = len(table)
  if not max_size_per_df:
    max_size_per_df = num_examples
  logging.info('Writing out SSTables in iterations of size %d', max_size_per_df)
  batch_num = 0
  for rows in _batch_sstable_values(table, int(max_size_per_df)):
    batch_num += 1
    if num_batches and batch_num > num_batches:
      break
    logging.info('Starting on batch %d', batch_num)
    if tolerant_byte:
      data = []
    else:
      data = numpy.empty(len(rows), dtype=data_dtypes)
    index_data = []

    for i, example in enumerate(rows):
      keys_to_features = example.features.ListFields()[0][1]
      if sorted(keys_to_features) != all_keys:
        raise ValueError('All examples must have the same attribs: %s vs %s' %
                         (all_keys, sorted(keys_to_features)))
      row_data = []
      for key, validator in keys_and_validators:
        feature = keys_to_features[key]
        value = _get_feature_value(feature)
        if not validator(value):
          raise ValueError('Invalid value for %s: %s' % (example, feature))
        row_data.append(value)
      if tolerant_byte:
        data.append(tuple(row_data))
      else:
        data[i] = tuple(row_data)
      index_data.append(_get_feature_value(keys_to_features[index]))

    # Note: While the structured array does incur total copy upon conversion to
    # a pandas.DataFrame (https://github.com/pandas-dev/pandas/issues/9216), the
    # explicit specification of counts as uint32 rather than the inferred int64
    # makes the peak memory usage essentially equivalent and the steady-state
    # memory usage half as much as in an alternative implementation.
    if tolerant_byte:
      yield pandas.DataFrame(
          data, index=index_data, columns=[x[0] for x in data_dtypes])
    else:
      yield pandas.DataFrame(data, index=index_data)
