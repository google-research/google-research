# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Input pipeline for the SC09 dataset."""

import dataclasses
from typing import Dict, Optional

from clu import deterministic_data as data
import jax
import jax.numpy as jnp
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

from autoregressive_diffusion.experiments.audio.datasets import speech_commands09  # pylint: disable=unused-import

AUTOTUNE = tf.data.experimental.AUTOTUNE
Features = Dict[str, tf.Tensor]


@dataclasses.dataclass
class PrepareAudio:
  """Pre-process audio.

  Attributes:
    shift: Value by which to shift audio inputs.
    max_length: Maximum allowed audio length.
  """
  shift: int = 2 ** 15
  max_length: int = 16000

  def __call__(self, features):
    features['inputs'] = tf.cast(features.pop('audio'), tf.int32)
    features['inputs'] = tf.pad(
        features['inputs'],
        [(0, self.max_length - tf.shape(features['inputs'])[0])])
    features['inputs'] = features['inputs'] + self.shift
    features['inputs'] = tf.expand_dims(features['inputs'], axis=-1)
    features['inputs'] = tf.ensure_shape(
        features['inputs'], (self.max_length, 1))
    features['label'] = tf.cast(features['label'], tf.int32)
    return features


def get_split(rng,
              builder,
              split,
              batch_size,
              num_epochs = None,
              shuffle_buffer_size = None,
              repeat_after = False,
              cache = False):
  """Loads a audio dataset and shifts audio values to be positive.

  Args:
    rng: JAX PRNGKey random number generator state.
    builder: TFDS dataset builder instance.
    split: TFDS split to load.
    batch_size: Global batch size.
    num_epochs: Number of epochs. None to repeat forever.
    shuffle_buffer_size: Size of the shuffle buffer. If None, data is not
      shuffled.
    repeat_after: If True, the dataset is repeated infinitely *after* CLU.
    cache: If True, the dataset is cached prior to pre-processing.

  Returns:
    Audio datasets with `inputs` and `label` features. The former is shifted to
    be non-negative.
  """
  host_count = jax.process_count()
  if batch_size % host_count != 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the host'
                     f' count ({host_count}).')
  batch_size = batch_size // host_count
  device_count = jax.local_device_count()
  if batch_size % device_count != 0:
    raise ValueError(f'Local batch size ({batch_size}) must be divisible by the'
                     f' local device count ({device_count}).')
  batch_dims = [device_count, batch_size // device_count]

  host_split = data.get_read_instruction_for_host(
      split,
      dataset_info=builder.info,
      remainder_options=data.RemainderOptions.BALANCE_ON_PROCESSES)
  ds = data.create_dataset(
      builder,
      split=host_split,
      preprocess_fn=PrepareAudio(),
      cache=cache,
      batch_dims=batch_dims,
      rng=rng,
      num_epochs=num_epochs,
      pad_up_to_batches='auto',
      shuffle=shuffle_buffer_size and (shuffle_buffer_size > 0),
      shuffle_buffer_size=shuffle_buffer_size or 0)
  if repeat_after:
    ds = ds.repeat()
  return ds


# -----------------------------------------------------------------------------
# Main dataset prep routines.
# -----------------------------------------------------------------------------


def get_dataset(rng, config):
  """Load and return dataset of batched examples for use during training."""
  builder = tfds.builder(
      config.dataset.name,
      data_dir=config.dataset.get('data_dir'))

  splits = {
      'train': {
          'split': config.dataset.train_split,
          'batch_size': config.batch_size,
          'shuffle_buffer_size': 2 ** 15,
          'cache': True,
      },
      'eval': {
          'split': config.dataset.eval_split,
          'batch_size': config.eval_batch_size,
          'cache': True,
          'repeat_after': True,
          'num_epochs': 1
      },
      'test': {
          'split': config.dataset.test_split,
          'batch_size': config.eval_batch_size,
          'cache': True,
          'repeat_after': True,
          'num_epochs': 1
      },
  }

  ds, metadata = {}, {}
  for name, conf in splits.items():
    ds[name] = get_split(rng, builder, **conf)
    num_examples = builder.info.splits[conf['split']].num_examples
    num_batches, remainder = divmod(num_examples, conf['batch_size'])
    if remainder:
      num_batches += 1
    metadata[name] = {
        'num_classes': 2 ** 16,
        'sample_rate': 16000,
        'num_examples': num_examples,
        'num_batches': num_batches,
        'shape': jax.tree_map(
            lambda x: x.shape, next(ds[name].as_numpy_iterator())),
    }

  return ds, metadata
