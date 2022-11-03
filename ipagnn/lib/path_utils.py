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

"""Utilities for constructing run_dir paths used by the launcher and runner."""

import json
import math
import os

from absl import logging  # pylint: disable=unused-import
import tensorflow as tf


PARAMETERS_PLACEHOLDER = '<parameters>'


def _abbrev_key(key):
  key = key.split('.')[-1]
  return ''.join(part[0] for part in key.split('_'))


def _sig_figs(value, sigfigs):
  if value == 0:
    return 0
  first_digit = int(math.floor(math.log10(abs(value))))
  return round(value, sigfigs - 1 - first_digit)


def _abbrev_value(value):
  if isinstance(value, bool):
    return 'T' if value else 'F'
  if isinstance(value, float):
    return _sig_figs(value, 4)
  if isinstance(value, str) and value.startswith('/cns'):
    return os.path.basename(value).replace('*', '').replace('-', '')
  return value


def parameters_to_str(parameters):
  """Convenience method which flattens the parameters to a string.

  Args:
    parameters (dict): Parameters for this run.
  Returns:
    The parameters string.
  """
  name = ['{}={}'.format(_abbrev_key(k), _abbrev_value(v))
          for k, v in sorted(parameters.items())]
  parameters_str = ','.join(name)
  # Escape some special characters
  replace_str = {
      '\n': ',',
      ':': '=',
      '\'': '',
      '"': '',
  }
  for c, new_c in replace_str.items():
    parameters_str = parameters_str.replace(c, new_c)
  for c in ('\\', '/', '[', ']', '(', ')', '{', '}', '%'):
    parameters_str = parameters_str.replace(c, '-')
  if len(parameters_str) > 170:
    raise ValueError('Parameters string is too long.', parameters_str,
                     len(parameters_str))
  return parameters_str


def wid_parameters_to_str(wid, parameters):
  """Convenience method which flattens the parameters to a string.

  Used as mapping function for the WorkUnitCustomiser.

  Args:
    wid (int): Worker id.
    parameters (dict): Parameters for this run.
  Returns:
    The parameter string.
  """
  if not parameters:
    return str(wid)
  else:
    parameters_str = parameters_to_str(parameters)
    return '{}-{}'.format(wid, parameters_str)


def expand_run_dir(run_dir, xm_parameters=None):
  """Expands the run_dir into a concrete run directory."""
  parameters = json.loads(xm_parameters or '{}')
  if PARAMETERS_PLACEHOLDER in run_dir:
    run_dir = run_dir.replace(
        PARAMETERS_PLACEHOLDER, parameters_to_str(parameters))
  return expand_glob(run_dir)


def expand_glob(path):
  """Uniquely expands a glob if present in the provided path."""
  paths = tf.io.gfile.glob(path)
  if not paths:  # New path or nonexistant path.
    return path
  elif len(paths) == 1:  # Glob matched precisely one path.
    return paths[0]
  else:
    raise ValueError('path was ambiguous', path)
