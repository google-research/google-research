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

"""Running the pipeline."""

import inspect
import pprint
import types

from absl import flags
from absl import logging
import jax
import tensorflow as tf

from imp.max.config import registry
from imp.max.data.datasets import dataloader
from imp.max.execution import config as exec_config
from imp.max.execution import initialization

Registrar = registry.Registrar


def define_flags():
  """Defines all flags for experiment jobs."""
  flags.DEFINE_string(
      'config_name', '',
      'Define the name of the experiment config to use.')
  flags.DEFINE_string(
      'config_overrides', '{}', 'Config overrides.')
  flags.DEFINE_string(
      'vizier_study', None, 'The name of the vizier study.')
  flags.DEFINE_integer(
      'vizier_tuner_group', None,
      'The identifier for the tuner group that current process belongs to. '
      'If None, all processes will be working on different trials. '
      'When specified, paired training and eval processes should use '
      'the same tuner group, which will get the same trial during tuning. Only '
      'one process should report the measurement and signal the completion or '
      'stopping of the training. See flag `vizier_metrics_from` for details.')
  flags.DEFINE_string(
      'tf_data_service_address', None, 'Address for TF data service.')
  flags.DEFINE_string(
      'coordinator_address',
      None,
      help='IP address:port for multi-host GPU coordinator.')
  flags.DEFINE_integer(
      'process_count', None, help='Number of processes for multi-host GPU.')
  flags.DEFINE_integer(
      'process_index', None, help='Index of this process.')


def initialize_devices():
  """Initializes all device-related functions."""
  # Make sure TF does not have visibility to accelerators
  tf.config.set_visible_devices([], 'GPU')

  # This should be only touched on multi-host settings
  jax.distributed.initialize(
      flags.FLAGS.coordinator_address,
      flags.FLAGS.process_count,
      flags.FLAGS.process_index,
  )

  # Wait until existing functions have completed any side-effects.
  jax.effects_barrier()

  # Log initialization results
  logging.info(
      'Initialized process %s-out-of-%s containing %s-out-of-%s devices',
      jax.process_index(),
      jax.process_count(),
      jax.local_device_count(),
      jax.device_count(),
  )


def ensure_registered(config_package):
  """Ensures at least one experiment config is registered.

  Args:
    config_package: Python package containing all experiment configs. This is
      used to check if any of the configs in the package are registered.
  """

  all_registered_names = set(Registrar.class_names())

  package_class_names = set()
  for _, obj in inspect.getmembers(config_package, inspect.isclass):
    try:
      package_class_names.add(obj.name)
    except AttributeError:
      pass

  if not all_registered_names.intersection(package_class_names):
    raise ValueError(
        f'None of the classes in {config_package} are registered.')


def _run_experiment(config):
  """Runs a single experiment job given a concrete config."""

  logging.info('Experiment path: %s', config.path)
  logging.info('Experiment config: %s', pprint.pformat(config.as_dict()))
  # TODO(hassanak): find a safe way to resolve the existing issues
  # config.export(config.path, filename=f'config.{config.mode.value}.yaml')

  # fetch the model class bound to the model config and construct it
  model = Registrar.get_class_by_name(config.model.name)(
      **config.model.as_dict())

  # construct the data loader
  dataloaders = dataloader.create_data(config.data)

  # TODO(b/234949870): deprecate init_override and merge with checkpointing
  # construct the init_override function (if any)
  init_override = initialization.create_init_fn(config.model.init_override)

  # construct the executor and run the experiment
  executor_cls = Registrar.get_class_by_name(config.name)
  executor_cls(model, dataloaders, config, init_override).run(config.mode)


def run(config_name,
        config_overrides,
        vizier_study = None,
        vizier_tuner_group = None,
        tf_data_service_address = None):
  """Main function to launch a single or a study experiment.

  Args:
    config_name: The name for which the experiment config class is registered
      with in the registry.
    config_overrides: A set of json-dumped configs passed from the command
      line to override the experiment config values. This is usually useful
      for overriding the experiment path.
    vizier_study: The name of the Vizier study, in case of hyperparam search.
    vizier_tuner_group: The assigned group ID for the current process, which
      should correspond to a unique ID for the current work unit. This allows
      a new set of hyperparameters to be assigned to each work unit. Train and
      eval jobs within the same work unit should share the same ID so they can
      share the same hyperparameters.
    tf_data_service_address: the data service address to use.
  """

  # Log the entire experiment config
  logging.info('Launching experiment: %s', config_name)

  # fetch the full experiment config
  config = Registrar.get_config_by_name(config_name)()
  config.override_from_str(config_overrides)
  config.data.update_data_service_address(tf_data_service_address)

  # Write the base config to the top directory
  logging.info('Using base path: %s', config.path)
  tf.io.gfile.makedirs(config.path)
  # TODO(hassanak): find a safe way to resolve the existing issues
  # config.export(config.path, filename=f'config.{config.mode.value}.yaml')

  if vizier_study:
    configs_iterator = exec_config.ExperimentParser(
        config=config,
        study_name=vizier_study,
        group=vizier_tuner_group)

    for concrete_config, feedback in configs_iterator:
      logging.info('Start working on trial %d (group=%r)...', feedback.id,
                   vizier_tuner_group)
      _run_experiment(concrete_config)
      # TODO(b/234045631): add support for returning rewards to vizier
      feedback(0)
  else:
    _run_experiment(config)
