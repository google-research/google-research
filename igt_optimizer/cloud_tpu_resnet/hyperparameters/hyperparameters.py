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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Generating hyperparameters dictionary from multiple sources."""

import os
import warnings
import six
import tensorflow.compat.v1 as tf
import yaml


# TODO(amangu): Add tests for this class.
class _Hyperparameters(object):
  """_Hyperparameters class to generate final hparams from various inputs."""

  def __init__(self, default_hparams_file, specific_hparams_file, input_flags,
               hparams_overrides):
    """Initialze and load parameter dictionary with different input sources.

    Args:
      default_hparams_file: YAML storing default values of all hyperparameters.
      specific_hparams_file: YAML file storing accelerator specific values of
        hyperparameters to override the default values.
      input_flags: Command line flags values for hyperparameters. [This is for
        backward compatibility, so that users passing hyperparameters as regular
        flags should not run into trouble].
      hparams_overrides: A kv string representing which hyperparameters need to
        be override from the command-line.

    Raises:
      ValueError: Raised when 'default_hparams_file' is not readable.
    """
    if not tf.io.gfile.exists(default_hparams_file):
      raise ValueError(
          'Expected a valid path to a YAML file, which represents the default '
          'hyperparameters file. {}'.format(default_hparams_file))

    self._params = {}
    self._params_source = {}
    self._default_hparams_file = default_hparams_file
    self._specific_hparams_file = specific_hparams_file
    self._input_flags = input_flags
    self._hparams_overrides = hparams_overrides

  def get_parameters(self, log_params):
    """Returns the dictionary loaded with final values of all hyperparameters.

    Args:
      log_params: Bool to specify if the hyperparameters final value need to be
        logged or not.

    Returns:
      Python dictionary with all the final hyperparameters.
    """
    self._params, self._params_source = load_from_file(
        self._params, self._params_source, self._default_hparams_file)
    self._params, self._params_source = load_from_file(
        self._params, self._params_source, self._specific_hparams_file)
    self._params, self._params_source = load_from_input_flags(
        self._params, self._params_source, self._input_flags)
    self._params, self._params_source = load_from_hparams_overrides(
        self._params, self._params_source, self._hparams_overrides)

    if log_params:
      self.log_parameters()

    return self._params

  def log_parameters(self):
    """Log the hyperparameters value along with the source of those values."""
    params_log = ''

    for k in self._params:
      params_log += k + ': \t' + str(self._params[k])
      params_log += ' \t[' + self._params_source[k] + ']\n'
    tf.logging.info('\nModel hyperparameters [source]:\n%s', params_log)


def load_from_file(params, params_source, file_path):
  """Given a path to a YAML file, read the file and load it to dictionary.

  Args:
    params: Python dictionary of hyperparameters.
    params_source: Python dictionary to record source of hyperparameters.
    file_path: Python string containing path to file.

  Returns:
    Python dict of hyperparameters.
  """
  if file_path is None:
    return params, params_source

  if not tf.io.gfile.exists(file_path):
    warnings.warn('Could not read Hyperparameter file : ' + file_path,
                  RuntimeWarning)
    return params, params_source

  with tf.gfile.Open(file_path, 'r') as f:
    overrides = yaml.safe_load(f)
  for key, value in six.iteritems(overrides):
    params[key] = value
    params_source[key] = os.path.basename(file_path)

  return params, params_source


# TODO(amangu): Once global hyperparameter flags will be removed, we won't need
# this function. Remove this functions after implementing this.
def load_from_input_flags(params, params_source, input_flags):
  """Update params dictionary with input flags.

  Args:
    params: Python dictionary of hyperparameters.
    params_source: Python dictionary to record source of hyperparameters.
    input_flags: All the flags with non-null value of overridden
      hyperparameters.

  Returns:
    Python dict of hyperparameters.
  """
  if params is None:
    raise ValueError(
        'Input dictionary is empty. It is expected to be loaded with default '
        'values')

  if not isinstance(params, dict):
    raise ValueError(
        'The base parameter set must be a Python dict, was: {}'.format(
            type(params)))

  for key in params:
    flag_value = input_flags.get_flag_value(key, None)

    if flag_value is not None:
      params[key] = flag_value
      params_source[key] = 'Command-line flags'

  return params, params_source


# TODO(amangu): Add tests to verify different dtypes of params.
def load_from_hparams_overrides(params, params_source, hparams_overrides):
  """Given a dictionary of hyperparameters and a list of overrides, merge them.

  Args:
    params: Python dict containing a base hyperparameters set.
    params_source: Python dictionary to record source of hyperparameters.
    hparams_overrides: Python list of strings. This is a set of k=v overrides
      for the hyperparameters in `params`; if `k=v1` in `params` but `k=v2` in
      `hparams_overrides`, the second value wins and the value for `k` is `v2`.

  Returns:
    Python dict of hyperparameters.
  """
  if params is None:
    raise ValueError(
        'Input dictionary is empty. It is expected to be loaded with default '
        'values')

  if not isinstance(params, dict):
    raise ValueError(
        'The base hyperparameters set must be a Python dict, was: {}'.format(
            type(params)))

  if hparams_overrides is None:
    return params, params_source

  if isinstance(hparams_overrides, six.string_types):
    hparams_overrides = [hparams_overrides]

  if not isinstance(hparams_overrides, list):
    raise ValueError(
        'Expected that hparams_overrides would be `None`, a single string, or a'
        ' list of strings, was: {}'.format(type(hparams_overrides)))

  for kv_pair in hparams_overrides:
    if not isinstance(kv_pair, six.string_types):
      raise ValueError(
          'Expected that hparams_overrides would contain Python list of strings,'
          ' but encountered an item: {}'.format(type(kv_pair)))
    key, value = kv_pair.split('=')
    parser = type(params[key])
    if parser is bool:
      params[key] = value not in ('0', 'False', 'false')
    else:
      params[key] = parser(value)
    params_source[key] = 'Command-line `hparams` flag'

  return params, params_source


def get_hyperparameters(default_hparams_file,
                        specific_hparams_file,
                        input_flags,
                        hparams_overrides,
                        log_params=True):
  """Single function to get hparams for any model using different sources.

  Args:
    default_hparams_file: YAML storing default values of all hyperparameters.
    specific_hparams_file: YAML file storing accelerator specific values of
      hyperparameters to override the default values.
    input_flags: Command line flags values for hyperparameters. [This is for
      backward compatibility, so that users passing hyperparameters as regular
      flags should not run into trouble].
    hparams_overrides: A kv string representing which hyperparameters need to be
      override from the command-line.
    log_params: Bool to specify if the hyperparameters final value need to be
      logged or not.

  Returns:
    Python dictionary with all the final hyperparameters.
  """
  parameter = _Hyperparameters(default_hparams_file, specific_hparams_file,
                               input_flags, hparams_overrides)

  return parameter.get_parameters(log_params)
