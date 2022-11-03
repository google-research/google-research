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

"""Utils to create default configs for create SNLDS models."""


class ConfigDict(dict):
  """Configuration dictionary that allows the `.` access.

  Example:
  ```python
  config = ConfigDict()
  config.test_number = 1
  ```
  The content could be access by
  ```python
  print(config.test_number)  # 1 will be returned.
  ```
  """

  def __init__(self, *args, **kwargs):
    super(ConfigDict, self).__init__(*args, **kwargs)
    for arg in args:
      if isinstance(arg, dict):
        for k, v in arg.iteritems():
          self[k] = v

    if kwargs:
      for k, v in kwargs.iteritems():
        self[k] = v

  def __getattr__(self, attr):
    return self.get(attr)

  def __setattr__(self, key, value):
    self.__setitem__(key, value)

  def __setitem__(self, key, value):
    super(ConfigDict, self).__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super(ConfigDict, self).__delitem__(key)
    del self.__dict__[key]


def get_data_config(batch_size):
  data_config = ConfigDict()
  data_config.batch_size = batch_size
  return data_config


def get_cross_entropy_config(
    decay_rate=0.99,
    decay_steps=20000,
    initial_value=1.e+4,
    kickin_steps=5000,
    use_entropy_annealing=False):
  """Create default config for cross entropy regularization."""
  config = ConfigDict()
  config.decay_rate = decay_rate
  config.decay_steps = decay_steps
  config.initial_temperature = initial_value
  config.kickin_steps = kickin_steps
  config.use_entropy_annealing = use_entropy_annealing
  return config


def get_distribution_config(
    cov_mat=None,
    triangular_cov=True,
    trainable_cov=False,
    raw_sigma_bias=0.,
    sigma_min=1.e-5,
    sigma_scale=0.05):
  """Create default config for a multivariate normal gaussian distribution."""
  config = ConfigDict()
  config.cov_mat = cov_mat
  config.use_triangular_cov = triangular_cov
  config.use_trainable_cov = trainable_cov
  config.raw_sigma_bias = raw_sigma_bias
  config.sigma_min = sigma_min
  config.sigma_scale = sigma_scale
  return config


def get_learning_rate_config(
    flat_learning_rate=True,
    inverse_annealing_lr=False,
    learning_rate=1.e-3,
    decay_alpha=1.e-2,
    decay_steps=20000,
    warmup_steps=5000,
    warmup_start_lr=1.e-5):
  """Create default config for learning rate."""
  config = ConfigDict()
  config.flat_learning_rate = flat_learning_rate
  config.inverse_annealing_lr = inverse_annealing_lr
  config.learning_rate = learning_rate
  config.decay_alpha = decay_alpha
  config.decay_steps = decay_steps
  config.warmup_steps = warmup_steps
  config.warmup_start_lr = warmup_start_lr
  return config


def get_temperature_config(
    decay_steps=20000,
    decay_rate=0.99,
    initial_temperature=1.e+3,
    minimal_temperature=1.,
    kickin_steps=10000,
    use_temperature_annealing=False):
  """Create default config for temperature annealing."""
  config = ConfigDict()
  config.decay_steps = decay_steps
  config.decay_rate = decay_rate
  config.initial_temperature = initial_temperature
  config.minimal_temperature = minimal_temperature
  config.kickin_steps = kickin_steps
  config.use_temperature_annealing = use_temperature_annealing
  return config
