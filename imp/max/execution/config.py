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

"""Configuration of the experiment pipeline."""

import dataclasses
import os
import pprint
import time
from typing import Any, Type, TypeVar

from absl import logging
import flax
import jax
import pyglove as pg
from vizier import pyglove as pg_vizier

from imp.max.config import base
from imp.max.config import validators
from imp.max.core import constants
from imp.max.core import utils
from imp.max.data import config as data_config
from imp.max.evaluation import config as eval_config
from imp.max.modeling import config as mdl_config
from imp.max.optimization import config as opt_config
from imp.max.utils import typing


_ExperimentT = TypeVar('_ExperimentT', bound='Experiment')
ExperimentT = Type[_ExperimentT]

Mode = constants.Mode
Task = constants.Task


@dataclasses.dataclass
class Partitioning(base.Config):
  """Base configuration for any experiment."""

  num_partitions: int = 1
  num_experts: int | None = None
  model_parallel_submesh: typing.HardwareMesh | None = None
  params_on_devices: bool = True


@dataclasses.dataclass
class Execution(base.Config):
  """Base configuration for any experiment."""

  partitioning: Partitioning = dataclasses.field(default_factory=Partitioning)


@validators.lock
@dataclasses.dataclass
class Experiment(base.Config):
  """Base configuration for an experiment.

  Attributes:
    name: a unique name for the experiment.
    path: a directory to save experimental results to.
    mode: the experiment mode for training or evaluation.
    task: the experiment task for pretraining or finetuning.
    model: the model config.
    data: the data config for dataset preparation.
    execution: the execution hardware config, e.g., for partitioning.
    evaluation: the evaluation config for reporting metrics.
    optimization: the optimization config for optimizers and training steps.
    search_algorithm: the search algorithm to use for Vizier search.
    max_num_trials: the maximum number of trials to run in Vizier.
  """

  name: str | None = None
  path: str | None = None
  mode: Mode | None = None
  task: Task | None = None
  model: mdl_config.Model | None = None
  data: data_config.ExperimentData | None = None
  execution: Execution | None = None
  evaluation: eval_config.Evaluation | None = None
  optimization: opt_config.Optimization | None = None
  search_algorithm: str | None = None
  max_num_trials: int | None = None

  def _update_data_shards(self):
    """Updates shard info based on the distributed system info."""
    if self.data is None or jax.process_count() == 1:
      return

    for loader in self.data.loaders:
      if isinstance(loader.dataset, tuple):
        loader_datasets = loader.dataset
      else:
        loader_datasets = (loader.dataset,)

      name = '_'.join([dataset.name for dataset in loader_datasets])
      is_training = all(dataset.is_training for dataset in loader_datasets)

      logging.info(
          'Dataset %s: shuffle=%s, training=%s, checkpointing=%s, '
          'num_epochs=%s', name, loader.shuffle, is_training,
          self.data.checkpointing, loader.num_epochs)
      if (loader.shuffle and is_training and not self.data.checkpointing
          and loader.num_epochs == -1):
        # Create a random seed based on the current time and process to
        # make a unique shuffle per host.
        seed = hash(f'{time.time()}_{jax.process_index()}')
        seed = int(seed % 2**30)  # Truncate to fit in int32
        loader.seed = seed
        for i, dataset in enumerate(loader_datasets):
          dataset.data.prop_seed = seed + i + 1
        logging.info('Enabling seed on %s: using seed %s on host %s',
                     name, seed, jax.process_index())
      else:
        logging.info('Disabling seed on %s', name)

      if not loader.shuffle and not loader.use_data_service:
        # If using the data service, it will handle sharding.
        for dataset in loader_datasets:
          dataset.data.num_shards = jax.process_count()
          dataset.data.shard_index = jax.process_index()
          logging.info('Sharding files on %s: %d:%d',
                       dataset.name,
                       dataset.data.num_shards,
                       dataset.data.shard_index)
      else:
        logging.info('Disabling file sharding on %s', name)

  def _validate_objectives(self):
    """Validates the optimization objectives."""
    if self.optimization is None or self.data is None:
      return

    loss = self.optimization.loss
    loaders = self.data.loaders

    if len(loss) > 1 and len(loss) != len(loaders):
      raise AssertionError(
          f'The length of optimization losses ({len(loss)}) should be equal to '
          f'the length of the data loaders ({len(loaders)}) and should be in '
          f'the same order. Got {loss} and {loaders}')

  def _validate_unique_dataset_names(self):
    """Validates that the dataset names are unique."""
    if self.data is None:
      return

    def _get_dataset_names(loader):
      """Fetches the dataset names from the loader."""
      if isinstance(loader.dataset, tuple):
        return tuple(dataset.name for dataset in loader.dataset)
      else:
        return (loader.dataset.name,)

    dataset_names = [_get_dataset_names(loader) for loader in self.data.loaders]
    if len(dataset_names) != len(set(dataset_names)):
      raise AssertionError(f'Dataset names should be unique: {dataset_names}')

  def __post_init__(self):
    """Validates the config."""
    self._update_data_shards()
    if self.mode is Mode.TRAIN:
      # TODO(b/235616229): add relevant tests wrt experiment configuration
      self._validate_objectives()
      self._validate_unique_dataset_names()

  def with_search_space(self):
    """Returns a copy of this config with applied search space params."""
    raise NotImplementedError()


class ExperimentParser:
  """Parses an experiment config via PyGlove to be used in a multi-job search.

  This parser sets up a search space for hyperparameter search. It later is
  used to interface with the Vizier API to get back the next set of
  hyperparameters after a trial has completed. This works with multiple
  workers to accelerate the search. Each worker should be assigned a unique
  group number for scheduling.

  See the PyGlove tutorial for more details:
  https://github.com/google/pyglove
  """

  def __init__(self, config, study_name, group):
    """Initializes the search executor.

    Args:
      config: the experiment config to use, which should internally define a
        search space.
      study_name: the name of the Vizier study.
      group: the assigned group ID for the current process, which should
        correspond to a unique ID for the current work unit. This allows a new
        set of hyperparameters to be assigned to each work unit. Train and eval
        jobs within the same work unit should share the same ID so they can
        share the same hyperparameters.
    """
    self.config = config
    self.study_name = study_name
    self.group = group

  def __iter__(self):
    """Yields a concrete config and a callable feedback function."""

    base_path = self.config.path

    pg_vizier.init(self.study_name)

    search_algorithm = self.config.search_algorithm or 'DEFAULT'
    max_num_trials = self.config.max_num_trials

    # Grab the search space from the flattened experiment config.
    # Note: this implementation constructs the search space by searching
    # for any PyGlove primitives like `pg.oneof`. It is possible to pass
    # the config directly into the sampler below by wrapping it in an
    # annotated function, but for easier interpretability we reconstruct
    # only the relevant search space parameters and save it here.
    search_config = self.config.with_search_space()
    params = search_config.as_flat_dict(sep='.')
    search_space = pg.Dict({
        k: v for k, v in params.items()
        if isinstance(v, pg.hyper.HyperPrimitive)
    })

    if not search_space.to_json():
      raise ValueError('Search space is empty!')

    logging.info('Search space: %s', search_space)
    search_space_debug_file = os.path.join(base_path, 'search_space.txt')
    utils.safe_write(search_space_debug_file, str(search_space))

    # The algorithm will be supplied by Vizier
    # For a list of available algorithms, see `pg_vizier.BuiltinAlgorithm`.
    algorithm: Any = pg_vizier.BuiltinAlgorithm(search_algorithm)

    # Run the main loop and exit once the max number of trials are exhausted.
    for example, feedback in pg.sample(
        search_space,
        algorithm=algorithm,
        num_examples=max_num_trials,
        group=self.group,
        backend='vizier',
        name=self.study_name):

      # Override the current config with the concrete search values
      # suggested by the algorithm.
      overrides = flax.traverse_util.unflatten_dict(
          {tuple(k.split('.')): v for k, v in example.items()})
      concrete_config = self.config.copy_and_override(overrides)
      concrete_config.path = os.path.join(base_path, str(feedback.id))

      logging.info('Using overrides, %s', pprint.pformat(overrides))

      # Run a single experiment with the suggested parameters.
      yield concrete_config, feedback
