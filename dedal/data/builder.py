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

"""Abstract classes for input data pipelines."""

import abc
import functools
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import gin
import tensorflow as tf
import tensorflow_datasets as tfds

from dedal import multi_task
from dedal.data import transforms

MultiTaskTensor = multi_task.Backbone[tf.Tensor]
MultiTaskOptionalTensor = multi_task.Backbone[Optional[tf.Tensor]]
MetaData = Mapping[str, tf.Tensor]
FinalizedOutput = Tuple[
    tf.Tensor, MultiTaskTensor, MultiTaskOptionalTensor, MetaData]
MergedFinalizedOutput = Tuple[
    Sequence[tf.Tensor], MultiTaskTensor, MultiTaskOptionalTensor, MetaData]
Split = Union[str, tfds.Split]
MultiSplit = Sequence[Split]


def transform(inputs, transformations = ()):  # pylint: disable=g-bare-generic
  result = inputs
  for transform_fn in transformations:
    result = transform_fn(result)
  return result


@gin.configurable
class DatasetBuilder(abc.ABC):
  """Builds datasets."""

  def __init__(self,
               data_loader=None,
               ds_transformations = (),
               transformations = (),
               batched_transformations = (),
               labels = gin.REQUIRED,
               metadata = (),
               sequence_key = 'sequence',
               repeat = None,
               shuffle_buffer = 1024,
               drop_remainder = True,
               split = None,
               debug_mode = False):
    self._data_loader = data_loader
    self._ds_transform = functools.partial(
        transform, transformations=ds_transformations)
    self._transform = functools.partial(
        transform, transformations=transformations)
    self._batched_transform = functools.partial(
        transform, transformations=batched_transformations)
    self._sequence_key = sequence_key
    self._labels = labels
    self._metadata = metadata
    self._repeat = repeat  # None or negative, infinite. Set to 1 to not repeat.
    self._shuffle_buffer = shuffle_buffer  # None or negative, do not shuffle.
    self._drop_remainder = drop_remainder  # Might need to set to False in eval.
    # Used as default split in case no split is passed to the make function. If
    # `Sequence[Split]`, `self.split[0]` will be used as default. However, the
    # remainder of the list will be accessible and used by `TrainingLoop` to run
    # eval on multiple splits within the same job.
    self.split = split
    if debug_mode:
      tf.data.experimental.enable_debug_mode()

  def finalize(self, inputs):
    """Builds up the ground truth and weights and returns a tuple."""
    labels = [(x, None) if isinstance(x, str) else x for x in self._labels]
    targets = self._labels.pack([inputs[key] for (key, _) in labels])

    flat_weights = []
    for label_key, weight_key in labels:
      w = inputs.get(weight_key, None)
      batch_size = tf.shape(inputs[label_key])[0]
      flat_weights.append(tf.ones(batch_size) if w is None else w)
    weights = self._labels.pack(flat_weights)

    metadata = {key: inputs[key] for key in self._metadata}

    dummy = tf.constant([])
    return (inputs[self._sequence_key],
            targets.flatten(empty_value=dummy),
            weights.flatten(empty_value=dummy),
            metadata)

  def prepare(self, split):
    """Loads data split and applies transforms prior to batching."""
    # If no split is given, use `self.split` as default. If several splits are
    # specified by `self.split`, default to first one.
    split = self.split if split is None else split
    split = split if isinstance(split, str) else split[0]
    ds = self._data_loader.load(split)
    logging.info('%s dataset loaded.', split)
    ds = ds.apply(self._ds_transform)
    ds = ds.map(self._transform, num_parallel_calls=tf.data.AUTOTUNE)
    logging.info('%s dataset prepared.', split)
    return ds

  def build(self,
            input_ctx = None,
            split = None,
            global_batch_size = 32,
            for_train = True):
    """Builds (optionally distributed) `tf.data.Dataset` for `split`."""
    ds = self.prepare(split)

    ds = ds.repeat(self._repeat)
    if self._shuffle_buffer is not None and self._shuffle_buffer > 0:
      ds = ds.shuffle(self._shuffle_buffer)

    local_batch_size = global_batch_size
    if input_ctx is not None:
      local_batch_size = input_ctx.get_per_replica_batch_size(global_batch_size)
    ds = ds.batch(local_batch_size, drop_remainder=self._drop_remainder)

    ds = ds.map(self._batched_transform, num_parallel_calls=tf.data.AUTOTUNE)
    if for_train:
      ds = ds.map(self.finalize, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

  def make(
      self,
      split = None,
      batch_size = 32,
      strategy = None,
      for_train = True,
  ):
    """Returns a `tf.data.Dataset` to iterate over minibatches.

    Creates (optionally distributed) minibatches of elements provided by the
    subclass-specific method `prepare(split)`.

    Args:
      split: Specifies which split to use. If None, `self.split` will be used as
        default.
      batch_size: The (global) size of the batches.
      strategy: A `tf.distribute.Strategy` instance for distributed training.
        Set to `tf.distribute.OneDeviceStrategy` (or None) for training on a
        single accelerator or CPU.
      for_train: Whether to pack dataset elements into format expected by the
        custom training loop, discarding any metadata not specified in the
        metadata argument of the constructor and other non-essential components.

    Returns:
      A `tf.data.Dataset` instance, which might be distributed depending on the
      provided strategy.
    """
    def make_fn(
        input_ctx):
      ds = self.build(input_ctx, split, batch_size, for_train)
      logging.info('DatasetBuilder: %s dataset finalized.', split)
      return ds.prefetch(tf.data.AUTOTUNE)

    if strategy is not None:
      return strategy.distribute_datasets_from_function(make_fn)
    return make_fn(input_ctx=None)


@gin.configurable
class MultiDatasetBuilder:
  """Combines N `DatasetBuilder`s into a single, merged dataset builder.

  Attributes:
    builders: a sequence of N `DatasetBuilder` instances.
    switch: a `multi_task.SwitchBackbone` instance configuring the way the
      `FinalizedOutput`s of each builder will be combined. In brief, setting
      `switch.embeddings[l] = i` (resp. `switch.alignments[l] = i`) implies the
      l-th entry of `embeddings` (resp. `alignments`) for elements of the merged
      dataset will be taken from the i-th `DatasetBuilder` in `builders`. See
      the documentation for `multi_task.SwitchBackbone` for additional details.
    split: default split(s) to be built by `make` when no explicit input is
      given. Can be:
        + a `str`, specifying a single default split where all `DatasetBuilder`s
          in `builders` use the same split name,
        + a `Sequence[str]`, specifying T default splits where, for each default
          split, all `DatasetBuilder`s in `builders` use the same split name,
        + a `Sequence[Sequence[str]]` such that `len(split[i]) = len(builders)`
          for all `i`. This specifies T default splits (T=1 is possible), where
          each default split uses a different split name per `DatasetBuilder`.
      When specifying T > 1 default splits, `make` will use the first one when
      provided no input. However, this feature is mainly used to configure
      `TrainingLoop` to run T different eval splits within the same job when
      parallel, multi-job execution is not desired.
    n_datasets: the number of datasets (i.e. `DatasetBuilder`s) being combined.
  """

  def __init__(
      self,
      builders = gin.REQUIRED,
      switch = gin.REQUIRED,
      split = None):
    self.builders = builders
    self.switch = switch
    self.split = split

  @property
  def n_datasets(self):
    return len(self.builders)

  def _maybe_broadcast(self, v = None):
    """Broadcasts singletons to `Sequence`s of `n_datasets` elements."""
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
      if len(v) == self.n_datasets:
        return v
      elif len(v) == 1:
        v = v[0]
      else:
        raise ValueError('MultiDatasetBuilder._maybe_broadcast: arg has length '
                         f'{len(v)}, expected length {self.n_datasets}.')
    return self.n_datasets * [v]

  def merge_datasets(
      self,
      *finalized_examples,
  ):
    """Combines output elements from multiple `DatasetBuilder`s.

    Args:
      *finalized_examples: a list of `n_datasets` `FinalizedOutput`s as
        formatted by `DatasetBuilder.finalize`.

    Returns:
      A `MergedFinalizedOutput` tuple such that:
        + `inputs` is a list of `n_datasets` `tf.Tensor` representing the N
          input minibatches.
        + `y_true` and `weights` are `Backbone` objects representing the merged
          label and weight sets, respectively. The specific way the N individual
          `Backbone` containers are combined is configured by the `switch`
          attribute.
        + `metadata` is the `dict` union across all N individual datasets. Note
          that *the current implementation assumes no key collisions* will occur
          when taking the union.
    """
    inputs, targets, weights, metadata = tuple(zip(*finalized_examples))
    targets = self.switch.merge_flattened(targets)
    weights = self.switch.merge_flattened(weights)
    # NOTE(fllinares): could also add a prefix to prevent collisions, but then
    # configs for metrics should account for this behavior which would be less
    # modular.
    metadata = functools.reduce(lambda x, y: {**x, **y}, metadata)
    return inputs, targets, weights, metadata

  def make(
      self,
      split = None,
      batch_size = 32,
      strategy = None,
      for_train = True,
  ):
    """Returns a `tf.data.Dataset` to iterate over minibatches.

    Creates (optionally distributed) minibatches of elements provided by the
    subclass-specific method `prepare(split)` of several independent datasets.

    Args:
      split: Specifies which splits to use. Can be
        + a `str`, in which case all `DatasetBuilder`s will be passed the same
          `split` argument to their respective `make` methods,
        + a `Sequence[str]` of length `n_datasets`, in which case each
          `DatasetBuilder`'s `make` will be invoked with the corresponding
          `split` arg.
      batch_size: The (global) size of the batches. Can be either `int` or
        `Sequence[int]`, with identical semantics as `split`.
      strategy: A `tf.distribute.Strategy` instance for distributed training.
        Set to `tf.distribute.OneDeviceStrategy` (or None) for training on a
        single accelerator or CPU.
      for_train: Whether to pack dataset elements into format expected by the
        custom training loop, discarding any metadata not specified in the
        metadata argument of the constructor and other non-essential components.
        If `True`, `y_true`, `weights` and `metadata` will retain the same
        format of `DatasetBuilder.finalize`, while `inputs` will contain a list
        with `n_datasets` `tf.Tensor` instances. See `merge_datasets` above for
        additional details.

    Returns:
      A `tf.data.Dataset` instance, which might be distributed depending on the
      provided strategy.
    """
    # If no split is given, use `self.split` as default. If several splits are
    # specified by `self.split`, default to first one.
    if split is None:
      split = self.split if isinstance(self.split, str) else self.split[0]
    # Ensures `split` and `batch_size` are Sequences with `n_datasets` elements.
    splits = self._maybe_broadcast(split)
    batch_sizes = self._maybe_broadcast(batch_size)

    def make_fn(
        input_ctx):
      ds = []
      for builder, split, batch_size in zip(self.builders, splits, batch_sizes):
        ds.append(builder.build(input_ctx, split, batch_size, for_train))
      ds = tf.data.Dataset.zip(tuple(ds))
      if for_train:
        ds = ds.map(self.merge_datasets, num_parallel_calls=tf.data.AUTOTUNE)
      for i, split in enumerate(splits):
        logging.info('MultiDatasetBuilder (%d / %d): %s dataset finalized.',
                     i + 1, self.n_datasets, splits)
      return ds.prefetch(tf.data.AUTOTUNE)

    if strategy is not None:
      return strategy.distribute_datasets_from_function(make_fn)
    return make_fn(input_ctx=None)
