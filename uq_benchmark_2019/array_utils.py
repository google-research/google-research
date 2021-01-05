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
"""Utilities for manipulating, storing, and loading experiment data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import io
import threading

from absl import logging

import numpy as np
import six
from six.moves import range
import tensorflow.compat.v2 as tf
gfile = tf.io.gfile

_TFR_OPTIONS = tf.io.TFRecordOptions('GZIP')
SLICE = enum.Enum('SliceKey', ['ALL'])


def write_npz(output_dir, basename, stats_dict):
  """Write a dictionary of numpy arrays as an npz file.

  Args:
    output_dir: Directory for the output file.
    basename: Basename for output file path.
    stats_dict: Dictionary of strings to np.ndarrays.
  """
  bytesio = io.BytesIO()
  stats_dict = {
      k: np.stack(arr) if isinstance(arr, list) else arr
      for k, arr in six.iteritems(stats_dict)
  }
  np.savez_compressed(bytesio, **stats_dict)
  path = '%s/%s' % (output_dir, basename)
  logging.info('Recording stats to %s', path)
  with gfile.GFile(path, 'wb') as file_handle:
    file_handle.write(bytesio.getvalue())


def _dict_as_namedtuple(d):
  return collections.namedtuple('tup', list(d.keys()))(**d)


def load_npz(path, as_namedtuple=False):
  """Load dictionary of arrays from an npz file.

  Args:
    path: File path to npz file.
    as_namedtuple: If true, return the dictionary as a namedtuple.
  Returns:
    Dictionary (or namedtuple) of npz file contents.
  """
  with gfile.GFile(path) as fl:
    bytesio = io.BytesIO(fl.read())
  out = dict(np.load(bytesio))
  return _dict_as_namedtuple(out) if as_namedtuple else out


def stats_dict_to_tfexample(stats):
  """Converts a dictionary of numpy arrays to a tf.Example proto."""
  example = tf.train.Example()
  fm = example.features.feature
  for key, arr in six.iteritems(stats):
    arr = np.array(arr)
    if key.endswith('/shape'):
      raise ValueError('Invalid key: %s' % key)
    if arr.dtype in (np.float32, np.float64):
      fm[key].float_list.value.extend(arr.reshape([-1]))
      fm[key + '/shape'].int64_list.value.extend(arr.shape)
    elif arr.dtype in (np.int32, np.int64):
      fm[key].int64_list.value.extend(arr.reshape([-1]))
      fm[key + '/shape'].int64_list.value.extend(arr.shape)
    else:
      raise NotImplementedError('Unsupported array type %s for key=%s'
                                % (type(arr), key))
  return example


def tfexample_to_stats_dict(example):
  """Converts a tf.Example proto into a dictionary of numpy arrays."""
  out = {}
  fm = example.features.feature
  for key, value in six.iteritems(fm):
    if key.endswith('/shape'):
      continue
    arr = (value.int64_list.value or
           value.float_list.value or
           value.bytes_list.value)
    shape = fm[key + '/shape'].int64_list.value
    out[key] = np.array(arr).reshape(shape)

  return out


def load_stats_from_tfrecords(path, max_records=None, as_namedtuple=False,
                              gzip=False):
  """Loads data from a TFRecord table into a dictionary of np arrays.

  Args:
    path: Path to TFRecord file.
    max_records: Maximum number of records to read.
    as_namedtuple: If true, return the stats-dictionary as a namedtuple.
    gzip: Whether to use gzip compression.
  Returns:
    Dictionary (or namedtuple) of numpy arrays.
  """
  out = collections.defaultdict(list)
  if tf.executing_eagerly():
    itr = tf.data.TFRecordDataset(
        path, compression_type='GZIP' if gzip else None)
    parse_record = lambda x: tf.train.Example.FromString(x.numpy())
  else:
    tfr_options = _TFR_OPTIONS if gzip else None
    itr = tf.compat.v1.python_io.tf_record_iterator(path, tfr_options)
    parse_record = tf.train.Example.FromString
  for i, rec in enumerate(itr):
    if max_records and i >= max_records:
      break
    example = parse_record(rec)
    stats = tfexample_to_stats_dict(example)
    for key, array in six.iteritems(stats):
      out[key].append(array)
  out = {k: np.stack(arr) for k, arr in six.iteritems(out)}
  return _dict_as_namedtuple(out) if as_namedtuple else out


class StatsWriter(object):
  """Simple wrapper class to record stats-dictionaries in TFRecord tables."""

  def __init__(self, path, gzip=False):
    self._writer = tf.io.TFRecordWriter(path, _TFR_OPTIONS if gzip else None)

  def write(self, stats):
    tfexample = stats_dict_to_tfexample(stats)
    self._writer.write(tfexample.SerializeToString())

  def write_batch(self, stats_batch):
    batch_size, = set(len(x) for x in six.itervalues(stats_batch))
    for i in range(batch_size):
      stats_i = {k: v[i] for k, v in six.iteritems(stats_batch)}
      tfexample = stats_dict_to_tfexample(stats_i)
      self._writer.write(tfexample.SerializeToString())

  def __del__(self):
    self._writer.flush()
    self._writer.close()


def slice_structure(struct, keys):
  """Generalized (but limited) slice function on nested structures.

  This function offers limited numpy-style array slicing on nested structures
  of maps, lists, tuples, and arrays. Specifically, by assuming similar
  structures along each dictionary / list / tuple value, we can support
  select-all and index-list slicing (e.g. x[3, :, 1] or x[3, indices, 1]).

  For example,
    x = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
    slice_structure(x, [SLICE.ALL, 0])
  will yield `{'a': 1, 'b': 3, 'c': 5}`
  and
    slice_structure(x, [['a', 'c', 'b'], 0])
  yields `[1, 5, 3]`.

  Args:
    struct: Nested structure of dictionaries, lists, tuples, numpy arrays.
    keys: List of keys to apply at each successive depth;
        SLICE.ALL gathers all items.
  Returns:
    Nested structure with specified slices applied.
    Note: Structure elments are not necessarily copied in the process.
  """
  if not keys:
    return struct

  if keys[0] is SLICE.ALL:
    if isinstance(struct, dict):
      return {k: slice_structure(v, keys[1:]) for k, v in struct.items()}
    elif isinstance(struct, (list, tuple)):
      return type(struct)([slice_structure(struct_i, keys[1:])
                           for struct_i in struct])
    else:
      raise NotImplementedError('Unsupported type for ALL: %s.' % type(struct))
  # List-of-indices slicing.
  elif isinstance(keys[0], list):
    return [slice_structure(struct[k], keys[1:]) for k in keys[0]]
  # Simple get-element-at-index case.
  else:
    return slice_structure(struct[keys[0]], keys[1:])


class _MapResult(object):
  """Simple temporary container for threaded_map_structure() results.

  Note: We can't use a simple Python list (or other builtin mutable container)
  for this since tf.nest.map_structure will traverse the list and operate on
  its elements.

  Attributes:
    result: Equals None before calling the map-function;
        assigned to the function output afterwards.
  """

  def __init__(self):
    self.result = None

  def assign(self, x):
    """Assigns a value to a container attribute for later retrieval."""
    self.result = x


def threaded_map_structure(fn, *args):
  """Executes tf.nest.map_structure with parallel threads for each map call.

  Primarily useful for slow, non-compute functions (e.g. loading data from CNS).
  See tf.nest.map_structure for details.

  Args:
    fn: Function to map across leaf nodes in args structure.
    *args: Nested structures of arguments to map over.
  Returns:
    Parallel structure to the one in args with map results.
  """
  fn_nooutput = lambda result, *args_: result.assign(fn(*args_))
  def make_thread_fn(result, *args_):
    return threading.Thread(target=fn_nooutput, args=(result,) + args_)

  outputs = tf.nest.map_structure(lambda *_: _MapResult(), *args)
  threads = tf.nest.map_structure(make_thread_fn, outputs, *args)
  tf.nest.map_structure(lambda t: t.start(), threads)
  tf.nest.map_structure(lambda t: t.join(), threads)
  return tf.nest.map_structure(lambda x: x.result, outputs)
