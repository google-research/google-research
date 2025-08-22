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

"""Config Utilities."""

import collections
import functools
import importlib
import pathlib
import typing
from typing import Any, Optional, Protocol

from absl import flags
from etils import epath
from ml_collections import config_dict
from ml_collections.config_flags import config_flags



@typing.runtime_checkable
class ConfigProtocol(Protocol):
  """A pickleable protocol for configurations."""

  def get_config(self):
    Ellipsis



def get_configurable(module, config, **kwargs):
  """Initializes a configurable object from a module or dictionary."""
  config = config.to_dict()
  name = config.pop('name')

  if isinstance(module, dict):
    fn = module.get(name, None)
  else:
    fn = getattr(module, name, None)

  if fn is not None:
    return fn(**config, **kwargs)
  else:
    raise ValueError(f'Unable to find config {name} on {module}.')


FLAGS = flags.FLAGS


class ConfigHelper:
  """A helper to simplify some operations relater to a config file."""

  def __init__(
      self, path_to_project, config_flag_name = 'config'
  ):
    """Builds absolute and relative paths to a config file.

    Given a config file flag name, determine paths to the config file specified
    by this flag which are useful both for the launcher (absolute) and for the
    experiment (relative).

    Args:
      path_to_project: Path to a project directory. Can be an absolute one
        (e.g., obtained from the launch script by calling
        `os.path.realpath(__file__)`) or relative to the launcher script (e.g.,
        just `'.'`).
      config_flag_name: Name of the config file flag.

    Raises:
      RuntimeError: if config file is not found.
    """
    self.config_flag_name = config_flag_name

    # A path passed to a config flag.
    config_path = epath.Path(
        config_flags.get_config_filename(FLAGS[config_flag_name])
    )

    project_root = path_to_project.parent.resolve()

    if config_path.is_absolute():
      self.absolute_path = config_path
    else:
      self.absolute_path = epath.Path(pathlib.Path.cwd()) / config_path

    if not self.absolute_path.exists():
      raise RuntimeError(f'Config {self.absolute_path} does not exist.')

    try:
      self.relative_to_project = self.absolute_path.relative_to(project_root)
    except ValueError:
      raise NotImplementedError(
          f'Config file should be inside the project, got {self.absolute_path}'
      ) from None

  def get_config_flags(self):
    """Provide config-related flags for the experiment to run.

    Returns:
      a {flag: value} dictionary containing information related to config file
      and individual values.
    """
    args = collections.OrderedDict()

    # Given the config file `--config=...` it is possible to modify individual
    # parameters by passing e.g. `--config.foo=...`.
    args |= {
        name: value
        for name, value in FLAGS.flag_values_dict().items()
        if name.startswith(f'{self.config_flag_name}.')
    }
    return args

  @functools.cached_property
  def module(self):
    """Gets config module."""
    # Loading here from file location avoids importing parent packages.
    # Complex configs might include other packages which are not related to the
    # parameter sweeps.
    spec = importlib.util.spec_from_file_location(
        self.config_flag_name, self.absolute_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pytype: disable=attribute-error
    if not isinstance(module, ConfigProtocol):
      raise RuntimeError(f"Config {module} doesn't conform to ConfigProtocol")
    return module

