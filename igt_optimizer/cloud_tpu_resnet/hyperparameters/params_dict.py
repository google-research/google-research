# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A parameter dictionary class which supports the nest structure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import six
import tensorflow.compat.v1 as tf
import yaml


class ParamsDict(object):
  """A hyperparameter container class."""

  RESERVED_ATTR = ['_locked', '_restrictions']

  def __init__(self, default_params=None, restrictions=None):
    """Instantiate a ParamsDict.

    Instantiate a ParamsDict given a set of default parameters and a list of
    restrictions. Upon initialization, it validates itself by checking all the
    defined restrictions, and raise error if it finds inconsistency.

    Args:
      default_params: a Python dict or another ParamsDict object including the
        default parameters to initialize.
      restrictions: a list of strings, which define a list of restrictions to
        ensure the consistency of different parameters internally. Each
        restriction string is defined as a binary relation with a set of
        operators, including {'==', '!=',  '<', '<=', '>', '>='}.
    """
    self._locked = False
    self._restrictions = []
    if restrictions:
      self._restrictions = restrictions
    if default_params is None:
      default_params = {}
    self.override(default_params, is_strict=False)
    self.validate()

  def _set(self, k, v):
    if isinstance(v, dict):
      self.__dict__[k] = ParamsDict(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __setattr__(self, k, v):
    """Sets the value of the existing key.

    Note that this does not allow directly defining a new key. Use the
    `override` method with `is_strict=False` instead.

    Args:
      k: the key string.
      v: the value to be used to set the key `k`.

    Raises:
      KeyError: if k is not defined in the ParamsDict.
    """
    if k not in ParamsDict.RESERVED_ATTR:
      if k not in self.__dict__.keys():
        raise KeyError('The key `%{}` does not exist. '
                       'To extend the existing keys, use '
                       '`override` with `is_strict` = True.'.format(k))
      if self._locked:
        raise ValueError('The ParamsDict has been locked. '
                         'No change is allowed.')
    self._set(k, v)

  def __getattr__(self, k):
    """Gets the value of the existing key.

    Args:
      k: the key string.

    Returns:
      the value of the key.

    Raises:
      KeyError: if k is not defined in the ParamsDict.
    """
    if k not in self.__dict__.keys():
      raise KeyError('The key `{}` does not exist. '.format(k))
    return self.__dict__[k]

  def override(self, override_params, is_strict=True):
    """Override the ParamsDict with a set of given params.

    Args:
      override_params: a dict or a ParamsDict specifying the parameters to be
        overridden.
      is_strict: a boolean specifying whether override is strict or not. If
        True, keys in `override_params` must be present in the ParamsDict. If
        False, keys in `override_params` can be different from what is currently
        defined in the ParamsDict. In this case, the ParamsDict will be extended
        to include the new keys.
    """
    if self._locked:
      raise ValueError('The ParamsDict has been locked. No change is allowed.')
    if isinstance(override_params, ParamsDict):
      override_params = override_params.as_dict()
    self._override(override_params, is_strict)  # pylint: disable=protected-access

  def _override(self, override_dict, is_strict=True):
    """The implementation of `override`."""
    for k, v in six.iteritems(override_dict):
      if k in ParamsDict.RESERVED_ATTR:
        raise KeyError('The key `%{}` is internally reserved. '
                       'Can not be overridden.')
      if k not in self.__dict__.keys():
        if is_strict:
          raise KeyError('The key `{}` does not exist. '
                         'To extend the existing keys, use '
                         '`override` with `is_strict` = True.'.format(k))
        else:
          self._set(k, v)
      else:
        if isinstance(v, dict):
          self.__dict__[k]._override(v, is_strict)  # pylint: disable=protected-access
        elif isinstance(v, ParamsDict):
          self.__dict__[k]._override(v.as_dict(), is_strict)  # pylint: disable=protected-access
        else:
          self.__dict__[k] = copy.deepcopy(v)

  def lock(self):
    """Makes the ParamsDict immutable."""
    self._locked = True

  def as_dict(self):
    """Returns a dict representation of ParamsDict.

    For the nested ParamsDict, a nested dict will be returned.
    """
    params_dict = {}
    for k, v in six.iteritems(self.__dict__):
      if k not in ParamsDict.RESERVED_ATTR:
        if isinstance(v, ParamsDict):
          params_dict[k] = v.as_dict()
        else:
          params_dict[k] = copy.deepcopy(v)
    return params_dict

  def validate(self):
    """Validate the parameters consistency based on the restrictions.

    This method validates the internal consistency using the pre-defined list of
    restrictions. A restriction is defined as a string which specfiies a binary
    operation. The supported binary operations are {'==', '!=', '<', '<=', '>',
    '>='}. Note that the meaning of these operators are consistent with the
    underlying Python immplementation. Users should make sure the define
    restrictions on their type make sense.

    For example, for a ParamsDict like the following
    ```
    a:
      a1: 1
      a2: 2
    b:
      bb:
        bb1: 10
        bb2: 20
      ccc:
        a1: 1
        a3: 3
    ```
    one can define two restrictions like this
    ['a.a1 == b.ccc.a1', 'a.a2 <= b.bb.bb2']

    What it enforces are:
     - a.a1 = 1 == b.ccc.a1 = 2
     - a.a2 = 2 <= b.bb.bb2 = 20

    Raises:
      KeyError: if any of the following happens
        (1) any of parameters in any of restrictions is not defined in
            ParamsDict,
        (2) any inconsistency violating the restriction is found.
      ValueError: if the restriction defined in the string is not supported.
    """

    def _get_kv(dotted_string, params_dict):
      tokenized_params = dotted_string.split('.')
      v = params_dict
      for t in tokenized_params:
        v = v[t]
      return tokenized_params[-1], v

    def _get_kvs(tokens, params_dict):
      if len(tokens) != 2:
        raise ValueError('Only support binary relation in restriction.')
      stripped_tokens = [t.strip() for t in tokens]
      left_k, left_v = _get_kv(stripped_tokens[0], params_dict)
      right_k, right_v = _get_kv(stripped_tokens[1], params_dict)
      return left_k, left_v, right_k, right_v

    params_dict = self.as_dict()
    for restriction in self._restrictions:
      if '==' in restriction:
        tokens = restriction.split('==')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v != right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '!=' in restriction:
        tokens = restriction.split('!=')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v == right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '<' in restriction:
        tokens = restriction.split('<')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v >= right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '<=' in restriction:
        tokens = restriction.split('<=')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v > right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '>' in restriction:
        tokens = restriction.split('>')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v <= right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '>=' in restriction:
        tokens = restriction.split('>=')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v < right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      else:
        raise ValueError('Unsupported relation in restriction.')


def read_yaml_to_params_dict(file_path):
  """Reads a YAML file to a ParamsDict."""
  with tf.gfile.Open(file_path, 'r') as f:
    params_dict = yaml.safe_load(f)
    return ParamsDict(params_dict)


def save_params_dict_to_yaml(params, file_path):
  """Saves the input ParamsDict to a YAML file."""
  with tf.gfile.Open(file_path, 'w') as f:

    def _my_list_rep(dumper, data):
      # u'tag:yaml.org,2002:seq' is the YAML internal tag for sequence.
      return dumper.represent_sequence(
          u'tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, _my_list_rep)
    yaml.dump(params.as_dict(), f, default_flow_style=False)


def override_params_dict(params, dict_or_string_or_yaml_file, is_strict):
  """Override a given ParamsDict using a dict or a YAML file.

  Args:
    params: a ParamsDict object to be overridden.
    dict_or_string_or_yaml_file: a Python dict or a YAML string or path to a
      YAML file specifying the parameters to be overridden.
    is_strict: a boolean specifying whether override is strict or not.

  Returns:
    params: the overridden ParamsDict object.

  Raises:
    ValueError: if failed to override the parameters.
  """
  if not dict_or_string_or_yaml_file:
    return params
  if isinstance(dict_or_string_or_yaml_file, dict):
    params.override(dict_or_string_or_yaml_file, is_strict)
  elif isinstance(dict_or_string_or_yaml_file, six.string_types):
    params_dict = yaml.safe_load(dict_or_string_or_yaml_file)
    if isinstance(params_dict, dict):
      params.override(params_dict, is_strict)
    else:
      with tf.gfile.Open(dict_or_string_or_yaml_file) as f:
        params.override(yaml.safe_load(f), is_strict)
  else:
    raise ValueError('Unknown input type to parse.')
  return params
