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

"""Configuration for baseline model."""
import dataclasses as dc
from typing import Callable, Optional, Tuple, Type, TypeVar

from wildfire_perc_sim import datasets

ConfigType = TypeVar('ConfigType', bound='ProjectConfig')  # pylint: disable=invalid-name


@dc.dataclass
class ScheduleConfig:
  """Config for specifying a schedule."""
  init_value: float = 0.1
  warmup_epochs: float = 5.0
  warmup_init_value: float = 0.0
  end_value: float = 0.0
  steps_per_epoch: int = 2_000


@dc.dataclass
class LossConfig:
  weight_observation: float = 1.0
  weight_hidden_state: float = 1.0
  weight_kl_divergence: float = 1.0


@dc.dataclass
class OptConfig:
  schedule: ScheduleConfig = dc.field(default_factory=ScheduleConfig)
  weight_decay: Optional[float] = None


@dc.dataclass
class ModelConfig:
  observation_channels: int = 2
  latent_dim: int = 128
  field_shape: Tuple[int, int] = (64, 64)
  hidden_state_channels: int = 9
  stage_sizes: Tuple[int, int, int, int] = (2, 2, 2, 2)
  decoder_num_starting_filters: int = 512


@dc.dataclass
class ProjectConfig:
  """Base class for project configs.

  This class exists to simplify support of easily specifying complex configs
  from command line, by providing a registry of parameterized parsers to
  assign values to the entire nested config using simple preset values.

  For example config class:

  ```
  from ml_collections import config_flags

  @dc.dataclass
  class MyConfig(ProjectConfig):
     train: TrainConfig
     model: BackboneConfig


  config_flags.DEFINE_config_dataclass(
    'cfg', train_lib.Config(),
    'Configuration flags', parse_fn=MyConfig.parse_config)

  @MyConfig.register_parser('my_config')
  def config(cls, my_param=123, other_param=312):
    config = cls()
    config.train.param =  my_param
    ...
    return config

  Then this setup can be automatically populated  simpy providing
  --cfg=my_config:123:345  where 123 will be passed as my_param and 345
  as my_other_param. The value populated by the parser can be overridden
  by providing --cfg.train.param = 'something' in the command line.

  Some special parsers (such as 'xm') are defined on  ProjectConfig which
  enables them to be available to all config obects that inherit from
  ProjectConfig.
  """
  global_init_rng: int = 12345

  _registry = {}

  @classmethod
  def parse_config(cls, config):
    """Parses config and returns an instance of cls."""
    name, *args = config.split(':')
    available = []
    # Now we try to see if any superclasses of cls have registered config
    # name 'name'.
    for each in cls.mro():
      available.extend([name for (k, name) in cls._registry if k is each])
      if (each, name) in cls._registry:
        return validated_config(cls._registry[each, name](cls, *args))
    raise ValueError(f'{config} is not a valid config. Available {available}')

  @classmethod
  def register_parser(
      cls, name
  ):
    """Registers flag parser with a given name.

    This is a decorator that can be used to decorate functions that can parse
    configs. The parser will be invoked if we pass

    --cfg={name}[:colon-separated-values].

    This allows to easily create multiple parameterized parsers predicated on
    different names. The decorated function should accept as the first argument
    the class type that it will  need to instantiate, while the remaining
    positional arguments    is the config that's passed in the command line.

    For example:

    @register_parser(Config, 'simple')
    def simple_parser(cls, param1, param2=''):
      cfg = cls()
      cfg.something=param1 + param2
      ...
      return cfg

    and we call this --cfg=simple:foo:bar, param1 will be initialized to 'foo'
    and param2 will be initialized to 'bar' and the config flag value will
    default to cfg.something='foobar'

    Args:
      name: string

    Returns:
      wrapper
    """
    def wrapper(
        fun):
      cls._registry[(cls, name)] = fun
      return fun

    return wrapper


class _InvalidConfigError(AttributeError):
  pass


def validated_config(value):
  """Recursibvely validates config to make sure there are no undeclared fields.

  Args:
    value: config value to validate.

  Returns:
    validated value (currently unchanged).

  Raises:
    AttributeError: if value contains any public attributes that are not
    dataclass fields.
  """
  assert dc.is_dataclass(value)
  legal_names = {f.name for f in dc.fields(value)}
  for field, field_value in vars(value).items():
    if field not in legal_names:
      raise _InvalidConfigError(
          f'Illegal {field=}')
    if dc.is_dataclass(field_value):
      try:
        validated_config(field_value)
      except _InvalidConfigError as e:
        raise _InvalidConfigError(f'{e} within field {field}') from e
  return value


@dc.dataclass
class TrainConfig:
  """Model training configuration."""
  log_every: int = 100
  eval_every: int = 1000
  save_every: int = 5000
  distributed: bool = False
  output_dir: str = '/tmp'
  num_batches_to_eval: int = -1
  aux_batches_to_eval: int = 10
  restore_model_from: str = ''
  keep_every_n_steps: int = 0
  reset_params: str = 'none'   # Acceptable values: PARAMETER_GROUPS.keys()
  num_train_steps: int = 50000
  backward_observation_length: int = 5
  forward_observation_length: int = 10


@dc.dataclass
class ExperimentConfig(ProjectConfig):
  data: datasets.DatasetConfig = dc.field(
      default_factory=datasets.DatasetConfig
  )
  loss: LossConfig = dc.field(default_factory=LossConfig)
  opt: OptConfig = dc.field(default_factory=OptConfig)
  model: ModelConfig = dc.field(default_factory=ModelConfig)
  train: TrainConfig = dc.field(default_factory=TrainConfig)
