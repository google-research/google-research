# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utility functions."""

import collections
import copy
import functools
import inspect
import logging as std_logging
import pprint
import random
import types
import uuid

from absl import logging
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

MIN_INT = np.iinfo(np.int64).min
MAX_INT = np.iinfo(np.int64).max
MIN_FLOAT = np.finfo(np.float32).min
MAX_FLOAT = np.finfo(np.float32).max


# TODO(ddohan): FrozenConfig type
class Config(dict):
  """a dictionary that supports dot and dict notation.

  Create:
    d = Config()
    d = Config({'val1':'first'})

  Get:
    d.val2
    d['val2']

  Set:
    d.val2 = 'second'
    d['val2'] = 'second'
  """
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __str__(self):
    return pprint.pformat(self)

  def __deepcopy__(self, memo):
    return self.__class__([(copy.deepcopy(k, memo), copy.deepcopy(v, memo))
                           for k, v in self.items()])


def get_logger(name='', with_absl=True, level=logging.INFO):
  """Creates a logger."""
  logger = std_logging.getLogger(name)
  if with_absl:
    logger.addHandler(logging.get_absl_handler())
    logger.propagate = False
  logger.setLevel(level)
  return logger


def get_random_state(seed_or_state):
  """Returns a np.random.RandomState given an integer seed or RandomState."""
  if isinstance(seed_or_state, int):
    return np.random.RandomState(seed_or_state)
  elif seed_or_state is None:
    # This returns the current global np random state.
    return np.random.random.__self__
  elif not isinstance(seed_or_state, np.random.RandomState):
    raise ValueError('Numpy RandomState or integer seed expected! Got: %s' %
                     seed_or_state)
  else:
    return seed_or_state


def set_seed(seed):
  """Sets global Numpy, Tensorboard, and random seed."""
  np.random.seed(seed)
  tf.set_random_seed(seed)
  random.seed(seed, version=1)


def to_list(values, none_to_list=True):
  """Converts `values` of any type to a `list`."""
  if (hasattr(values, '__iter__') and not isinstance(values, str) and
      not isinstance(values, dict)):
    return list(values)
  elif none_to_list and values is None:
    return []
  else:
    return [values]


def to_array(values):
  """Converts input values to a np.ndarray."""
  if tf.executing_eagerly() and tf.is_tensor(values):
    return values.numpy()
  else:
    return np.asarray(values)


def arrays_from_dataset(dataset):
  """Converts a tf.data.Dataset to nested np.ndarrays."""
  return tf.nest.map_structure(
      lambda tensor: np.asarray(tensor),  # pylint: disable=unnecessary-lambda
      tensors_from_dataset(dataset))


def dataset_from_tensors(tensors):
  """Converts nested tf.Tensors or np.ndarrays to a tf.Data.Dataset."""
  if isinstance(tensors, types.GeneratorType) or isinstance(tensors, list):
    tensors = tuple(tensors)
  return tf.data.Dataset.from_tensor_slices(tensors)


def random_choice(values, size=None, random_state=None, **kwargs):
  """Enables safer sampling from a list of values than `np.random.choice`.

  `np.random.choice` fails when trying to sample, e.g., from a list of

  indices instead of sampling from `values` directly.

  Args:
    values: An iterable of values.
    size: The sample size.
    random_state: An integer seed or a `np.random.RandomState`.
    **kwargs: Named arguments passed to `np.random.choice`.

  Returns:
    As single element from `values` if `size is None`, otherwise a list of
    samples from `values` of length `size`.
  """
  random_state = get_random_state(random_state)
  values = list(values)
  effective_size = 1 if size is None else size
  idxs = random_state.choice(range(len(values)), size=effective_size, **kwargs)
  samples = [values[idx] for idx in idxs]
  return samples[0] if size is None else samples


def random_shuffle(values, random_state=None):
  """Shuffles a list of `values` out-of-place."""
  return random_choice(
      values, size=len(values), replace=False, random_state=random_state)


def get_tokens(sequences, lower=False, upper=False):
  """Returns a sorted list of all unique characters of a list of sequences.

  Args:
    sequences: An iterable of string sequences.
    lower: Whether to lower-case sequences before computing tokens.
    upper: Whether to upper-case sequences before computing tokens.

  Returns:
    A sorted list of all characters that appear in `sequences`.
  """
  if lower and upper:
    raise ValueError('lower and upper must not be specified at the same time!')
  if lower:
    sequences = [seq.lower() for seq in sequences]
  if upper:
    sequences = [seq.upper() for seq in sequences]
  return sorted(set.union(*[set(seq) for seq in sequences]))


def tensors_from_dataset(dataset):
  """Converts a tf.data.Dataset to nested tf.Tensors."""
  tensors = list(dataset)
  if tensors:
    return tf.nest.map_structure(lambda *tensors: tf.stack(tensors), *tensors)
  # Return empty tensors if the dataset is empty.
  shapes_dtypes = zip(
      tf.nest.flatten(dataset.output_shapes),
      tf.nest.flatten(dataset.output_types))
  tensors = [
      tf.zeros(shape=[0] + shape.as_list(), dtype=dtype)
      for shape, dtype in shapes_dtypes
  ]
  return tf.nest.pack_sequence_as(dataset.output_shapes, tensors)


def hash_structure(structure):
  """Hashes a structure (n-d numpy array) of either ints or floats to  a string.

  Args:
    structure: A structure of ints that is castable to np.int32 (examples
      include an np.int32, np.int64, an int32 eager tf.Tensor,
      a list of python ints, etc.) or a structure of floats that is castable
      to np.float32. Here, we say that an array is castable if it can be
      converted to the target type, perhaps with some loss of precision
      (e.g. float64 to float32). See np.can_cast(..., casting='same_kind').

  Returns:
    A string hash for the structure. The hash will depend on the
    high-level type (int vs. float), but not the precision of such a type
    (int32 vs. int64).
  """
  array = np.asarray(structure)
  if np.can_cast(array, np.int32, 'same_kind'):
    return np.int32(array).tostring()
  elif np.can_cast(array, np.float32, 'same_kind'):
    return np.float32(array).tostring()
  raise ValueError('%s can not be safely cast to np.int32 or '
                   'np.float32' % str(structure))


def create_unique_id():
  """Creates a unique hex ID."""
  return uuid.uuid1().hex


def deduplicate_samples(samples, select_best=False):
  """De-duplicates Samples with identical structures.

  Args:
    samples: An iterable of `data.Sample`s.
    select_best: Whether to select the sample with the highest reward among
      samples with the same structure. Otherwise, the sample that occurs
      first will be selected.

  Returns:
    A list of Samples.
  """

  def _sort(to_sort):
    return (sorted(to_sort, key=lambda sample: sample.reward, reverse=True)
            if select_best else to_sort)

  return [_sort(group)[0] for group in group_samples(samples).values()]


def group_samples(samples, **kwargs):
  """Groups `data.Sample`s with identical structures using `group_by_hash`."""
  return group_by_hash(
      samples,
      hash_fn=lambda sample: hash_structure(sample.structure),
      **kwargs)


def group_by_hash(values, hash_fn=hash, store_index=False):
  """Groups values by their hash value.

  Args:
    values: An iterable of any objects.
    hash_fn: A function that is called to compute the hash of values.
    store_index: Whether to store the index of values or values in the returned
      dict.

  Returns:
    A `collections.OrderedDict` mapping hashes to values with that hash if
    `store_index=False`, or value indices otherwise. The length of the map
    corresponds to the number of unique hashes.
  """
  groups = collections.OrderedDict()
  for idx, value in enumerate(values):
    to_store = idx if store_index else value
    groups.setdefault(hash_fn(value), []).append(to_store)
  return groups


def get_instance(instance_or_cls, **kwargs):
  """Returns an instance given an instance or class reference.

  Enables passing both class references and class instances as (gin) configs.

  Args:
    instance_or_cls: An instance of class or reference to a class.
    **kwargs: Names arguments used for instantiation if `instance_or_cls` is a
      class.

  Returns:
    An instance of a class.
  """
  if (inspect.isclass(instance_or_cls) or inspect.isfunction(instance_or_cls) or
      isinstance(instance_or_cls, functools.partial)):
    return instance_or_cls(**kwargs)
  else:
    return instance_or_cls


def pd_option_context(width=999,
                      max_colwidth=999,
                      max_rows=200,
                      float_format='{:.3g}',
                      **kwargs):
  """Returns a Pandas context manager with changed default arguments."""
  return pd.option_context('display.width', width,
                           'display.max_colwidth', max_colwidth,
                           'display.max_rows', max_rows,
                           'display.float_format', float_format.format,
                           **kwargs)


def log_pandas(df_or_series, logger=logging.info, **kwargs):
  """Logs a `pd.DataFrame` or `pd.Series`."""
  with pd_option_context(**kwargs):
    for row in df_or_series.to_string().splitlines():
      logger(row)


def get_indices(valid_indices,
                selection,
                map_negative=True,
                validate=False,
                exclude=False):
  """Maps a `selection` to `valid_indices` for indexing an iterable.

  Supports selecting indices as by
   - a scalar: it[i]
   - a list of scalars: it[[i, j, k]]
   - a (list of) negative scalars: it[[i, -j, -k]]
   - slices: it[slice(-3, None)]

  Args:
    valid_indices: An iterable of valid indices that can be selected.
    selection: A scalar, list of scalars, or `slice` for selecting indices.
    map_negative: Whether to interpret `-i` as `len(valid_indices) + i`.
    validate: Whether to raise an `IndexError` if `selection` is not
    contained in `valid_indices`.
    exclude: Whether to return all `valid_indices` except the selected ones.

  Raises:
    IndexError: If `validate == True` and `selection` contains an index that is
      not contained in `valid_indices`.

  Returns:
    A list of indices.
  """
  def _raise_index_error(idx):
    raise IndexError(f'Index {idx} invalid! Valid indices: {valid_indices}')

  if isinstance(selection, slice):
    idxs = valid_indices[selection]
  else:
    idxs = []
    for idx in to_list(selection):
      if map_negative and isinstance(idx, int) and idx < 0:
        if abs(idx) <= len(valid_indices):
          idxs.append(valid_indices[idx])
        elif validate:
          _raise_index_error(idx)
      elif idx in valid_indices:
        idxs.append(idx)
      elif validate:
        _raise_index_error(idx)
  if exclude:
    idxs = [idx for idx in valid_indices if idx not in idxs]
  return idxs
