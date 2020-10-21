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

# Lint as: python3
"""Dataset utils for the Learned Interpreters framework."""

import collections

from typing import Any, Optional

from absl import logging
import dataclasses
import jax
import jax.numpy as jnp
import six
import tensorflow as tf
import tensorflow_datasets as tfds
import tree



@dataclasses.dataclass
class DatasetInfo:
  dataset: Any = None
  generator: Any = None
  environment: Any = None
  info: Optional[Any] = None  # info_lib.LearnedInterpretersDatasetInfo
  set_task: Any = None  # Callable[[TaskFn, int], Any] = None


def _default_padding_value(dtype):
  """Gets the default value for the given dtype for padding.

  Args:
    dtype: A tensorflow dtype.
  Returns:
    A default (zero) value for the given type.
  """
  if dtype == tf.string:
    return ' '
  elif dtype == tf.int64:
    return tf.constant(0, dtype=tf.int64)
  elif dtype == tf.int32:
    return tf.constant(0, dtype=tf.int32)
  elif dtype == tf.float32:
    return tf.constant(0.0, dtype=tf.float32)
  elif dtype == tf.float64:
    return tf.constant(0.0, dtype=tf.float64)
  elif dtype == tf.bool:
    return tf.constant(False, dtype=tf.bool)
  else:
    raise ValueError('Unexpected type.', dtype)


def verify_reasonable_dataset(dataset_name, info, config):
  """Verifies that the dataset configs are at least reasonable.

  For example, if the max_length is set too low such that every example would be
  filtered out, we catch that here.

  This lets us fail fast if we accidentally put in configs that will lead to all
  examples being filtered out, rather than silently succeeding but never making
  progress.

  Args:
    dataset_name: The name of the dataset being loaded.
    info: The dataset info object.
    config: The config for the model.
  """
  if dataset_name.startswith('control_flow_programs'):
    # TODO(dbieber): Move this logic into the dataset definition.
    length = info.program_generator_config.length
    tokens_per_statement = info.program_encoder.tokens_per_statement
    assert (
        not config.dataset.max_length
        or config.dataset.max_length >= tokens_per_statement * length)


def cannot_set_task(**kwargs):
  """Use this as the set_task fn when no curriculum is permitted."""
  del kwargs  # Unused.
  raise ValueError('The task cannot be changed. This is probably because the '
                   'data is being loaded from disk, rather than generated '
                   'at training time.')


def get_split(config):
  """Select the default split according to the config.

  Args:
    config: (ml_collections.ConfigDict) The experimental config.
  Returns:
    The TFDS split for the experimental setup indicated by the config.
  """
  splits = {
      'train': 'train[:70%]',
      'valid': 'train[70%:90%]',
      'test': 'train[90%:]',
  }
  if config.dataset.split == 'default':
    split_name = 'valid' if config.runner.mode.startswith('eval') else 'train'
    split = splits[split_name]
  elif config.dataset.split in splits:
    split = splits[config.dataset.split]
  else:
    raise ValueError('Unexpected split.')
  return split


def get_dataset(data_dir, config, dataset_name=None):
  """The training dataset for the code model for fault localization.

  Args:
    data_dir: The data directory to use with tfds.load.
    config: The config for the model.
    dataset_name: If set, use this dataset name in place of the one from the
      config.
  Returns:
    train_dataset: The tf.data.Dataset with batched examples.
    info: The DatasetInfo object containing the feature connectors and other
      info about the dataset.
  """
  dataset_name = dataset_name or config.dataset.name
  split = get_split(config)
  version = (
      None if config.dataset.version == 'default' else config.dataset.version)

  # If in interact mode, use an interactive dataset.
  if config.runner.mode == 'interact':
    dbuilder = tfds.builder(
        dataset_name, data_dir=data_dir, version=version)
    unused_split_generators = dbuilder._split_generators(dl_manager=None)  # pylint: disable=protected-access
    info = dbuilder.info
    info._builder.set_representation(config.dataset.representation)  # pylint: disable=protected-access
    assert config.dataset.batch_size == 1
    dataset = make_interactive_dataset(info, config)
    if config.dataset.batch:
      dataset = apply_batching(dataset, info, config)
    set_task = cannot_set_task
    return DatasetInfo(
        dataset=dataset,
        info=info,
        set_task=set_task
    )

  # Load the dataset.
  if config.dataset.in_memory:
    dbuilder = tfds.builder(
        dataset_name, data_dir=data_dir, version=version)
    unused_split_generators = dbuilder._split_generators(dl_manager=None)  # pylint: disable=protected-access
    dataset, set_task = dbuilder.as_in_memory_dataset(split='all')
    info = dbuilder.info
  else:
    name = dataset_name
    if version is not None:
      name = f'{name}:{version}'
    dataset, info = tfds.load(
        name=name, split=split,
        data_dir=data_dir,
        # batch_size=config.dataset.batch_size,
        with_info=True)
    set_task = cannot_set_task

  info._builder.set_representation(config.dataset.representation)  # pylint: disable=protected-access

  verify_reasonable_dataset(dataset_name, info, config)
  dataset = dataset.repeat()
  dataset = apply_filtering(dataset, info, config)
  if config.dataset.batch:
    dataset = apply_batching(dataset, info, config)
  return DatasetInfo(
      dataset=dataset,
      info=info,
      set_task=set_task,
  )


def apply_filtering(dataset, info, config):
  del info  # Unused.
  # TODO(dbieber): Reinstate filtering, but refactor it.
  # if config.dataset.max_length:
  #   dataset = dataset.filter(
  #       lambda x: x[info._builder.key('length')] <= config.dataset.max_length)  # pylint: disable=protected-access
  if config.dataset.max_examples:
    dataset = dataset.take(config.dataset.max_examples)
  return dataset


def apply_sharding(generator, stack_fn, shape_fn):
  """Shards a dataset with a device dimension.

  Args:
    generator: Yields pytrees of numpy arrays.
    stack_fn: Applied to each example before stacking.
    shape_fn: Applied to each example to determine which examples to group.
      Examples with the same shape are grouped.
  Returns:
    A new generator where each leaf now has a leading device axis.
  """
  def generator_fn():
    used_shapes = set()
    examples_by_shapes = collections.defaultdict(list)
    for example in generator:
      shapes = shape_fn(example)
      if shapes not in used_shapes and shapes not in examples_by_shapes:
        logging.info('New shape started: %s', shapes)
      examples_by_shapes[shapes].append(example)
      if len(examples_by_shapes[shapes]) == jax.local_device_count():
        stacked_examples = tree.map_structure(
            lambda *x: jnp.stack(x, axis=0),
            *[stack_fn(example) for example in examples_by_shapes[shapes]]
        )
        yield stacked_examples, examples_by_shapes[shapes]
        examples_by_shapes[shapes] = []
        if shapes not in used_shapes:
          logging.info('New shape finished: %s', shapes)
        used_shapes.add(shapes)
  return generator_fn()


def apply_batching(dataset, info, config):
  """Applies standard batching to the dataset."""
  del info  # Unused.
  padded_shapes = tree.map_structure(
      lambda items: [None] * len(items),
      tf.compat.v1.data.get_output_shapes(dataset))
  padding_values = tree.map_structure(
      _default_padding_value,
      tf.compat.v1.data.get_output_types(dataset))
  dataset = dataset.padded_batch(
      config.dataset.batch_size, padded_shapes, padding_values,
      drop_remainder=True)
  return dataset


def dataset_from_generator(generator_fn, info, config):
  """Creates a dataset from a given generator fn."""
  del config  # Unused.
  dtype = info.features.dtype
  shape = info.features.shape
  dataset = tf.data.Dataset.from_generator(generator_fn, dtype, shape)
  return dataset


def _example_from_string(code, info):
  example_dict = info._builder.generate_example_from_string(code)  # pylint: disable=protected-access
  encoded_example = info.features.encode_example(example_dict)
  decoded_example = info.features.decode_example(encoded_example)
  return decoded_example


def make_interactive_dataset(info, config):
  """Makes a dataset from interactively provided examples."""
  logging.info('Generating dataset interactively. batch_size=%d',
               config.dataset.batch_size)
  def generator_fn():
    while True:
      example_str = six.moves.input()
      if not example_str:
        break
      try:
        yield _example_from_string(example_str, info)
      except Exception as e:  # pylint: disable=broad-except
        logging.info('Encountered error in _example_from_string: %s', e)
  return dataset_from_generator(generator_fn, info, config)
