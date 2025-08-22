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

"""Factory for data."""

from typing import Any, Iterable

from absl import logging
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from imp.max.core import constants
from imp.max.data import config as data_config
from imp.max.data.datasets import factories
from imp.max.utils import typing


class _GeneratorWithMetadata:
  """Constructs a generator that adds metadata to a given iterable.

  Attributes:
    iterable_dataset: An iterable that contains dictionaries as its elements.
      Each element should be a nested dictionary with the standard structure in
      the MAX data pipeline (see max.data.datasets.factories for more details.)
    metadata: An instance of max.config.data.Metadata that contains a data-wide
      metadata to be fed to the downstream model and objective modules.
      (See max.config.data.Metadata for more details.)
  """

  def __init__(self,
               iterable_dataset,
               metadata):
    self.iterable_dataset = iterable_dataset
    self.metadata = metadata

  def __iter__(self):
    for element in self.iterable_dataset:
      element[constants.DataFeatureType.METADATA] = self.metadata
      yield element


# TODO(b/243447569): improve documentation on DataLoader
class DataLoader:
  """A generic data loader based on DMVR with tf.data service support.

  Attributes:
    config: the dataloader config.
    factories: the underlying factory/factories.
    sampling_weights: the sampling weight for each factory.
    sampling_rates: the normalized sampling rate for each factory.
    metadata: A user-defined structured metadata for downstream modules that
      need custom data/model consumption.
    batch_size: the global batch size to use.
    microbatch_splits: the number of microbatch steps to be performed for a
      given dataset.
    per_host_batch_size: the local batch size across hosts.
    shuffle: if True, shuffle the dataset examples.
    shuffle_buffer_multiplier: a multiplier to set the shuffle buffer size.
    shuffle_buffer_size: Equivalent to `shuffle_buffer_multiplier *
      per_host_batch_size`. A larger shuffle buffer is helpful for certain tasks
      like contrastive learning, but requires more memory.
    num_epochs: the number of epochs to sample.
    drop_remainder: whether to drop the last batch if it is not evenly
      divisible. If we are not training, then we should return the final batch,
      which may have a smaller batch size than the specified batch_size. Note
      that num_epochs needs to be >0 for this to trigger, otherwise new examples
      will be returned indefinitely.
    use_data_service: if True, use a TF data service for this dataloader.
      Otherwise, use the host process. Note that batching and prefetching will
      be moved after the data service. This avoids a potentially huge memory
      usage increase, as each worker can work on individual examples instead of
      aggregating entire batches. Batching will be done on the current host.
    data_service_address: the data service address to use when use_data_service
      is True.
    data_service_sharding_policy: the data service sharding policy, see TF data
      service documentation.
    ignore_errors: if True (default), skip corrupted examples instead of raising
      an error.
    mode: (unused) an optional dataset mode.
    name: the name of this loader.
  """

  def __init__(self,
               config,
               mode = '',
               name = ''):
    if isinstance(config.dataset, tuple):
      dataset_configs = config.dataset
    else:
      dataset_configs = (config.dataset,)

    self.config = config
    self.factories = []
    for dataset_config in dataset_configs:
      factory_cls = factories.get_data_factory(dataset_config.factory)
      self.factories.append(
          factory_cls(**dataset_config.data.as_dict()).configure(
              dataset_config))
    self.sampling_weights = config.sampling_weights
    self.sampling_rates = tuple(
        x / sum(self.sampling_weights) for x in self.sampling_weights)
    self.metadata = config.metadata
    self.batch_size = config.batch_size
    self.microbatch_splits = config.microbatch_splits or 1
    # Under JAX multi-device, shard data to the number of hosts.
    if self.batch_size % jax.process_count() != 0:
      raise ValueError(
          'Batch size must be divisible by the number of processes. Instead, '
          f'received {self.batch_size=} and {jax.process_count()=}.',
      )
    self.per_host_batch_size = int(self.batch_size / jax.process_count())
    if self.per_host_batch_size % self.microbatch_splits != 0:
      raise ValueError(
          'Per-host batch size must be divisible by the number of microbatch '
          f'splits. Instead, received {self.per_host_batch_size=} and '
          f'{self.microbatch_splits=}.'
      )
    self.shuffle = config.shuffle
    self.shuffle_buffer_multiplier = config.shuffle_buffer_multiplier
    self.shuffle_buffer_size = int(
        self.per_host_batch_size * self.shuffle_buffer_multiplier)
    self.num_epochs = config.num_epochs
    self.drop_remainder = config.is_training
    self.use_data_service = config.use_data_service
    self.data_service_address = config.data_service_address
    if config.data_service_sharding_policy is not None:
      self.data_service_sharding_policy = config.data_service_sharding_policy
    else:
      self.data_service_sharding_policy = (
          tf.data.experimental.service.ShardingPolicy.DYNAMIC)
    self.ignore_errors = config.is_training
    self.mode = mode
    self.name = name

    logging.info(
        ('Global batch_size = %s distributed to per-host batch_size = %s with '
         'a total of %s microbatches per batch.'),
        self.batch_size,
        self.per_host_batch_size,
        self.microbatch_splits,
    )

    # If using the data service, prefetch is added after the pipeline.
    prefetch = config.prefetch if not self.use_data_service else 1

    # Tune factory for large-scale runs.
    for factory in self.factories:
      # Tune loader for large-scale runs.
      if config.tuning == constants.DataTuning.EFFICIENT:
        factory.tune(
            seed=config.seed,
            num_process_threads=8,
            num_parser_threads=8,
            num_postprocess_threads=2,
            parser_buffer_size=16,
            shuffle_buffer=self.shuffle_buffer_size,
            prefetch_buffer_size=prefetch,
        )
      elif config.tuning == constants.DataTuning.FAST:
        factory.tune(
            seed=config.seed,
            shuffle_buffer=self.shuffle_buffer_size,
            prefetch_buffer_size=prefetch,
        )
      else:
        raise ValueError(f'Unknown tuning option {config.tuning}')

  def add_metadata(self,
                   dataset):
    """Adds the metadata to the iterable elements of the dataset."""
    return _GeneratorWithMetadata(dataset, self.metadata)

  def __call__(self, **kwargs):
    """Builds the dataset."""

    del kwargs

    all_datasets = []
    for factory in self.factories:
      # Initialize tokenizer, if any
      if factory.tokenizer is not None:
        factory.tokenizer.initialize()

      dataset = factory.make_dataset(
          shuffle=self.shuffle,
          num_epochs=self.num_epochs,
          # If using the data service, batching is added after the pipeline.
          batch_size=(
              self.per_host_batch_size if not self.use_data_service else None),
          padded_batch=False,
          drop_remainder=self.drop_remainder,
          keep_key=False,
          override_preprocess_fn=None,
          ignore_processing_errors=self.ignore_errors,
          shuffle_shards=not self.use_data_service,
      )

      if self.use_data_service:
        if not self.data_service_address:
          raise ValueError(
              'Requested data service, but no address specified for '
              f'{self.config=}')

        # Each unique dataset should share a job name so that each graph
        # only needs to be constructed once.
        # Note: sharding should be handled by the ShardingPolicy, it is not
        # recommended to pre-apply sharding on the filenames.
        job_name = f'shared_job_{self.name}'
        logging.info('Using data service %s at address %s, policy %s',
                     job_name,
                     self.data_service_address,
                     self.data_service_sharding_policy)

        dataset = dataset.apply(
            tf.data.experimental.service.distribute(
                processing_mode=self.data_service_sharding_policy,
                service=self.data_service_address,
                job_name=job_name,
                max_outstanding_requests=None,
                compression='AUTO'))

        # Batching is done after the data service to avoid large memory
        # inefficiencies (up to 5x memory reduction).
        dataset = dataset.batch(self.per_host_batch_size,
                                drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(self.config.prefetch)

      all_datasets.append(dataset)

    if len(all_datasets) > 1:
      all_datasets = [ds.unbatch() for ds in all_datasets]
      try:
        dataset = tf.data.experimental.sample_from_datasets(
            all_datasets, self.sampling_rates)
        dataset = dataset.batch(
            batch_size=self.per_host_batch_size,
            drop_remainder=self.drop_remainder)
      except TypeError as e:
        raise TypeError(
            'Two mismatching datasets are configured. Please make sure you '
            'configure the exact same modalities with the same resulting '
            'dimensionalities.') from e

    else:
      dataset = all_datasets[0]

    dataset = tfds.as_numpy(dataset)

    if self.metadata is not None:
      dataset = self.add_metadata(dataset)

    return dataset


def create_data(
    config,
):
  """Create stack of data loaders to be fed to executors."""

  data_loaders = ()
  for loader_config in config.loaders:
    if isinstance(loader_config.dataset, tuple):
      name = '_'.join([dataset.name for dataset in loader_config.dataset])
      subset = '_'.join(
          [dataset.data.table for dataset in loader_config.dataset])
    else:
      name = loader_config.dataset.name
      subset = loader_config.dataset.data.table

    logging.info('Creating dataset %s', name)
    loader = DataLoader(config=loader_config, name=name)()

    data_loaders += ({
        'loader': loader,
        'name': name,
        'serving': loader_config.serving,
        'interval': loader_config.interval,
        'subset': subset,
        'config': loader_config,
    },)
    logging.info('Dataset %s created successfully.', name)

  return data_loaders
