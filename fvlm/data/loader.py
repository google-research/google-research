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
import enum
import functools
import inspect
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import gin
import jax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# The import is needed to tfds can load the dataset, even though it is unused.
from utils import types
from utils.types import LoaderFn
from utils.types import LoaderFns
from utils.types import TextDict
from utils.types import Vocab


gin.external_configurable(
    tfds.decode.SkipDecoding, name='tfds.decode.SkipDecoding')


@gin.constants_from_enum
@enum.unique
class IteratorType(enum.Enum):
  """Dataset type for which dataset to use."""
  NUMPY = 'numpy'
  TFDS = 'tfds'


@gin.configurable
def compose_map_fns(map_fns):
  """Sequentially applies multiple map functions."""

  def composed_map_fn(x, vocab):
    for map_fn in map_fns:
      if isinstance(map_fn, str):
        # When passed in through xm.hyper flags, these do not get parsed
        # correctly. They come in a strings, e.g., '@prefix/function_name', and
        # are converted to the configured function here, and removing the '@'.
        if '()' in map_fn:
          # Call the function.
          map_fn = gin.get_configurable(map_fn.replace('@', '')[:-2])()
        else:
          map_fn = gin.get_configurable(map_fn.replace('@', ''))

      fn_args = set(inspect.signature(map_fn).parameters.keys())
      if 'vocab' in fn_args:
        map_fn = functools.partial(map_fn, vocab=vocab)
      x = map_fn(x)
    return x

  return composed_map_fn


@gin.configurable
def compose_filter_fns(filter_fns):
  """Sequentially applies multiple filter functions with logical and."""

  def composed_filter_fn(x):
    res = tf.constant(True)
    for filter_fn in filter_fns:
      res = tf.logical_and(res, filter_fn(x))
    return res

  return composed_filter_fn


@gin.configurable
def tfds_loader(name = gin.REQUIRED,
                split = gin.REQUIRED,
                data_dir = None,
                decoders = None,
                direct_read = False):
  """Prepares and returns a `tf.data.Dataset` from a TensorFlow Dataset.

  Args:
    name: the name of the TensorFlow Dataset.
    split: the split name (e.g. train, 'all')
    data_dir: the path to the data directory.
    decoders: custom TFDS decoders for the dataset.
    direct_read: Skip the read instruction that distribute the shards across
      hosts. This is useful for building multi-split datasets.

  Returns:
    a `tf.data.Dataset` instance.
  """
  if direct_read:
    read_instructions = split
  else:
    # Read separate slices of data in each replica. This is important to avoid
    # IO congestion for multi-host setup. Also, this is important to avoid
    # double-counting examples during eval.
    percent_per_replica = 100 // jax.host_count()
    read_instructions = tfds.core.ReadInstruction(
        split_name=split,
        from_=jax.host_id() * percent_per_replica,
        to=(jax.host_id() + 1) * percent_per_replica,
        unit='%')

  data = tfds.load(
      name=name,
      split=read_instructions,
      data_dir=data_dir,
      with_info=False,
      decoders=decoders)
  return data


@gin.configurable
def tfrecords_loader(
    file_pattern,
    shuffle_files = True,
    parse_fn = None
):
  """Prepares and returns a `tf.data.Dataset` from tfrecords files."""
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  # fetch file paths to python to split files per host.
  files = list(dataset)
  files = [
      f for i, f in enumerate(files) if i % jax.host_count() == jax.host_id()
  ]
  dataset = tf.data.Dataset.from_tensor_slices(files)
  if shuffle_files:
    dataset = dataset.shuffle(len(files))

  dataset = dataset.interleave(
      lambda f: tf.data.TFRecordDataset(f).prefetch(1),
      num_parallel_calls=tf.data.AUTOTUNE)
  if parse_fn is not None:
    dataset = dataset.map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def get_device_replica_batch_size(batch_size):
  """Calculate the per device and replica batch size from total batch size."""
  num_devices = jax.device_count()
  if batch_size % num_devices != 0:
    raise ValueError(
        f'batch_size is not divisible by the number of devices: {num_devices}')
  per_device_batch = batch_size // num_devices
  per_replica_batch = per_device_batch * jax.local_device_count()
  return per_device_batch, per_replica_batch


def local_devices_split(
    batch, per_device_batch, feature_names,
    label_names):
  """Prepares features and labels and splits them across local devices.

  Args:
    batch: A dictionary of features to reshape by the device dimension.
    per_device_batch: Per device batch size.
    feature_names: Features names to take from the batch.
    label_names: Label names to take from the batch.

  Returns:
    Reshaped features and reshaped labels dictionary with feature names
    and label names as keys. Return only the features if label names are None.
  """
  features = {
      feature_name: batch[feature_name] for feature_name in feature_names
  }
  local_device_count = jax.local_device_count()
  split_by_device = lambda x: x.reshape(
      (local_device_count, per_device_batch) + x.shape[1:])
  features = jax.tree.map(split_by_device, features)
  if label_names is None:
    return features
  labels = {label_name: batch[label_name] for label_name in label_names}
  labels = jax.tree.map(split_by_device, labels)
  return features, labels


@gin.configurable
def get_input(loader_fn = gin.REQUIRED,
              batch_size = gin.REQUIRED,
              feature_names = None,
              label_names = None,
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
              iterator_type = IteratorType.TFDS,
              ignore_errors = False,
              unbatch = False,
              vocab = None,
              return_dataset = False):
  """input function to obtain a data generator.

  Args:
    loader_fn: a function which returns a `tf.data.Dataset`.
    batch_size: the global batch size.
    feature_names: a sequence of feature names to select from the dataset.
    label_names: the label names to be passed to loss function/metrics.
    preprocess_fn: a function from tf.data.Dataset to tf.data.Dataset to do any
      pre-processing needed on the dataset object. NOTE: this is different from
        the map_fn that maps the dataset contents.
    postprocess_fn: a function from tf.data.Dataset to tf.data.Dataset to do any
      post-processing needed on the dataset object. NOTE: this is different from
        the map_fn that maps the dataset contents.
    filter_fn: the predicate function to filter the contents of the dataset.
    map_fn: a function that operates on individual items in the dataset. NOTE:
      this is called before batching hence it operates on a per example basis.
    batch_map_fn: like `map_fn` but it is called after batching. It is useful
      for processing a batch of data instead of individual data points (e.g.
      cutmix or mixup).
    flatten: whether to flatten the output of `map_fn`. It is used when the map
      function produces N examples (where N is the leading dimension) and we
      want to produce N separate elements from it. NOTE: this option is
        generally preferred over the use of `unbatch` because of performance
      issues. Please see:
      https://www.tensorflow.org/api_docs/python/tf/data/Dataset#unbatch
    repeat: whether to repeat the data indefinitely.
    shuffle: whether to shuffle the examples.
    shuffle_multiplier: Specifies the buffer size for shuffling which will be
      per_replica_batch * shuffle_multiplier.
    cache: whether to cache the dataset in memory.
    iterator_type: Type of iterator to return. Numpy for Colab demo, tfds for
      normal training and evaluation.
    ignore_errors: Set to true to ignore errors and drop samples that fail
      processing.
    unbatch: bool, set to True to unbatch data before making batches. Useful for
      data like VQA where there are multiple questions per image that need to be
      unbatched then batched into equal sizes.
    vocab: Optional vocab to use to process data. Passes in a vocab argument to
      the map function if needed.
    return_dataset: bool, set to True to return the dataset before batching and
      splitting to local devices.

  Returns:
    If return_dataset is True, a tf.data.Dataset after all the transformations.
    If return_dataset if False, returns a generator that yields a tuple of
    (features, labels) if labels is set, otherwise it yields the features as a
    list of values. Both features and labels are ready for `pmap` functions,
    which means they have the shape [num_local_devices, per_device_batch, ...].
  """
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
  per_device_batch, per_replica_batch = get_device_replica_batch_size(
      batch_size)
  num_parallel_calls = (1 if iterator_type == IteratorType.NUMPY else
                        tf.data.experimental.AUTOTUNE)

  if preprocess_fn:
    dataset = preprocess_fn(dataset)
  if cache:
    dataset = dataset.cache()
  if repeat:
    dataset = dataset.repeat()
  if shuffle:
    dataset = dataset.shuffle(per_replica_batch * shuffle_multiplier)
  if map_fn:
    fn_args = set(inspect.signature(map_fn).parameters.keys())
    if 'vocab' in fn_args:
      map_fn = functools.partial(map_fn, vocab=vocab)
    dataset = dataset.map(
        map_fn,
        num_parallel_calls=num_parallel_calls,
        deterministic=(iterator_type == IteratorType.NUMPY))
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

  dataset = dataset.batch(per_replica_batch, drop_remainder=True)
  if batch_map_fn:
    dataset = dataset.map(
        batch_map_fn,
        num_parallel_calls=num_parallel_calls,
        deterministic=(iterator_type == IteratorType.NUMPY))
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  if iterator_type == IteratorType.NUMPY:
    dataset = dataset.map(
        lambda x: [x[name] for name in feature_names],
        num_parallel_calls=num_parallel_calls,
        deterministic=True)
    return dataset.as_numpy_iterator()

  split_device_fn = functools.partial(
      local_devices_split,
      per_device_batch=per_device_batch,
      feature_names=feature_names,
      label_names=label_names)
  return map(split_device_fn, tfds.as_numpy(dataset))


def create_datasets(loader_fns):
  """Creates the initial datasets."""
  datasets = []
  for loader_fn in loader_fns:
    if isinstance(loader_fn, str):
      # When passed in through xm.hyper flags, these do not get parsed correctly
      # They come in a strings, e.g., '@prefix/function_name', and are converted
      # to the configured function here, and removing the '@' prefix.
      datasets.append(gin.get_configurable(loader_fn.replace('@', ''))())
    else:
      datasets.append(loader_fn())
  return datasets


@gin.configurable(denylist=['datasets'])
def static_sampling(datasets,
                    sampling_weights = None
                    ):
  """Statically samples datasets based on the sampling weights.

  Args:
    datasets: A list of datasets to mix together.
    sampling_weights: A list of weights to sample the datasets. The weights must
      be non-negative, and will be normalized later into sampling probabilities.
      It's optional and defaults to uniform sampling when it's None. Note that
      this is left only for backwards compatibility. It is preferred to set the
      argument in static_sampling in gin directly.

  Returns:
    The mixed dataset and None for the weights variable.
  """
  if sampling_weights:
    # Normalize sampling weights if they are not None.
    total_weights = sum(sampling_weights)
    sampling_weights = [float(x) / total_weights for x in sampling_weights]
  # Set stop_on_empty_dataset=True per recommended by the documentation:
  # https://www.tensorflow.org/api_docs/python/tf/data/experimental/sample_from_datasets
  dataset = tf.data.experimental.sample_from_datasets(
      datasets, weights=sampling_weights, stop_on_empty_dataset=True)
  dataset = types.DatasetReturnObject(dataset, None)
  return dataset


@gin.configurable
def get_input_mixture(
    loader_fns = gin.REQUIRED,
    batch_size = gin.REQUIRED,
    feature_names = gin.REQUIRED,
    sample_fn = static_sampling,
    label_names = None,
    sampling_weights = None,
    ):
  """Multi-dataset input function to obtain a data generator.

  Args:
    loader_fns: A list of functions that return tf.data.Datasets.
    batch_size: the global batch size.
    feature_names: a sequence of feature names to select from the dataset.
    sample_fn: Function to sample the multiple datasets with.
    label_names: the label names to be passed to loss function/metrics.
    sampling_weights: A list of weights to sample the datasets. The weights must
      be non-negative, and will be normalized later into sampling probabilities.
      It's optional and defaults to uniform sampling when it's None. Note that
      this is left only for backwards compatibility. It is preferred to set the
      argument in static_sampling in gin directly. dynamic sampling does not
      use this argument.

  Returns:
    A DatasetReturnType with a optional weights variable (for sampling) and
    A generator that yields a tuple of (features, labels) if labels is set,
    otherwise it yields the features as a list of values. The order of features
    and labels corresponds to the same order in `feature_names` and
    'label_names' respectively. both features and labels are ready for `pmap`
    functions (which means they have the shape [num_local_devices,
    per_device_batch, ...].
  """
  datasets = create_datasets(loader_fns)
  data_return_obj = sample_fn(datasets, sampling_weights=sampling_weights)
  dataset: tf.data.Dataset = data_return_obj.dataset

  # Calculate the batch size for each device.
  per_device_batch, per_replica_batch = get_device_replica_batch_size(
      batch_size)

  dataset = dataset.batch(
      per_replica_batch, drop_remainder=True)
  dataset = dataset.prefetch(
      tf.data.experimental.AUTOTUNE)

  split_device_fn = functools.partial(
      local_devices_split,
      per_device_batch=per_device_batch,
      feature_names=feature_names,
      label_names=label_names)
  data_return_obj.dataset = map(
      split_device_fn, tfds.as_numpy(dataset))
  return data_return_obj
