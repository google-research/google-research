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

# Lint as: python3
"""Hyperparameter values."""

import json
import numbers
import six


def _cast_to_type_if_compatible(name, param_type, value):
  """Cast hparam to the provided type, if compatible.

  Args:
    name: Name of the hparam to be cast.
    param_type: The type of the hparam.
    value: The value to be cast, if compatible.

  Returns:
    The result of casting `value` to `param_type`.
  Raises:
    ValueError: If the type of `value` is not compatible with param_type.
      * If `param_type` is a string type, but `value` is not.
      * If `param_type` is a boolean, but `value` is not, or vice versa.
      * If `param_type` is an integer type, but `value` is not.
      * If `param_type` is a float type, but `value` is not a numeric type.
  """
  fail_msg = ("Could not cast hparam '%s' of type '%s' from value %r" %
              (name, param_type, value))

  # Some callers use None, for which we can't do any casting/checking. :(
  if issubclass(param_type, type(None)):
    return value

  # Avoid converting a non-string type to a string.
  if (issubclass(param_type, (six.string_types, six.binary_type)) and
      not isinstance(value, (six.string_types, six.binary_type))):
    raise ValueError(fail_msg)

  # Avoid converting a number or string type to a boolean or vice versa.
  if issubclass(param_type, bool) != isinstance(value, bool):
    raise ValueError(fail_msg)

  # Avoid converting float to an integer (the reverse is fine).
  if (issubclass(param_type, numbers.Integral) and
      not isinstance(value, numbers.Integral)):
    raise ValueError(fail_msg)

  # Avoid converting a non-numeric type to a numeric type.
  if (issubclass(param_type, numbers.Number) and
      not isinstance(value, numbers.Number)):
    raise ValueError(fail_msg)

  return param_type(value)


class HParams(object):
  """Class to hold a set of hyperparameters as name-value pairs.

  A `HParams` object holds hyperparameters used to build and train a model,
  such as the number of hidden units in a neural net layer or the learning rate
  to use when training.
  You first create a `HParams` object by specifying the names and values of the
  hyperparameters.
  To make them easily accessible the parameter names are added as direct
  attributes of the class.  A typical usage is as follows:
  ```python
  # Create a HParams object specifying names and values of the model
  # hyperparameters:
  hparams = HParams(learning_rate=0.1, num_hidden_units=100)
  # The hyperparameter are available as attributes of the HParams object:
  hparams.learning_rate ==> 0.1
  hparams.num_hidden_units ==> 100
  ```
  Hyperparameters have type, which is inferred from the type of their value
  passed at construction type.   The currently supported types are: integer,
  float, string, and list of integer, float, or string.
  You can override hyperparameter values by calling the
  [`parse()`](#HParams.parse) method, passing a string of comma separated
  `name=value` pairs.  This is intended to make it possible to override
  any hyperparameter values from a single command-line flag to which
  the user passes 'hyper-param=value' pairs.  It avoids having to define
  one flag for each hyperparameter.
  The syntax expected for each value depends on the type of the parameter.
  See `parse()` for a description of the syntax.
  Example:
  ```python
  # Define a command line flag to pass name=value pairs.
  # For example using argparse:
  import argparse
  parser = argparse.ArgumentParser(description='Train my model.')
  parser.add_argument('--hparams', type=str,
                      help='Comma separated list of "name=value" pairs.')
  args = parser.parse_args()
  ...
  def my_program():
    # Create a HParams object specifying the names and values of the
    # model hyperparameters:
    hparams = tf.HParams(learning_rate=0.1, num_hidden_units=100,
                         activations=['relu', 'tanh'])
    # Override hyperparameters values by parsing the command line
    hparams.parse(args.hparams)
    # If the user passed `--hparams=learning_rate=0.3` on the command line
    # then 'hparams' has the following attributes:
    hparams.learning_rate ==> 0.3
    hparams.num_hidden_units ==> 100
    hparams.activations ==> ['relu', 'tanh']
    # If the hyperparameters are in json format use parse_json:
    hparams.parse_json('{"learning_rate": 0.3, "activations": "relu"}')
  ```
  """

  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self, hparam_def=None, model_structure=None, **kwargs):
    """Create an instance of `HParams` from keyword arguments.

    The keyword arguments specify name-values pairs for the hyperparameters.
    The parameter types are inferred from the type of the values passed.
    The parameter names are added as attributes of `HParams` object, so they
    can be accessed directly with the dot notation `hparams._name_`.
    Example:
    ```python
    # Define 3 hyperparameters: 'learning_rate' is a float parameter,
    # 'num_hidden_units' an integer parameter, and 'activation' a string
    # parameter.
    hparams = tf.HParams(
        learning_rate=0.1, num_hidden_units=100, activation='relu')
    hparams.activation ==> 'relu'
    ```
    Note that a few names are reserved and cannot be used as hyperparameter
    names.  If you use one of the reserved name the constructor raises a
    `ValueError`.
    Args:
      hparam_def: Serialized hyperparameters, encoded as a hparam_pb2.HParamDef
        protocol buffer. If provided, this object is initialized by
        deserializing hparam_def.  Otherwise **kwargs is used.
      model_structure: An instance of ModelStructure, defining the feature
        crosses to be used in the Trial.
      **kwargs: Key-value pairs where the key is the hyperparameter name and the
        value is the value for the parameter.

    Raises:
      ValueError: If both `hparam_def` and initialization values are provided,
        or if one of the arguments is invalid.
    """
    del hparam_def
    # Register the hyperparameters and their type in _hparam_types.
    # This simplifies the implementation of parse().
    # _hparam_types maps the parameter name to a tuple (type, bool).
    # The type value is the type of the parameter for scalar hyperparameters,
    # or the type of the list elements for multidimensional hyperparameters.
    # The bool value is True if the value is a list, False otherwise.
    self._hparam_types = {}
    self._model_structure = model_structure
    for name, value in six.iteritems(kwargs):
      self.add_hparam(name, value)

  def add_hparam(self, name, value):
    """Adds {name, value} pair to hyperparameters.

    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter. Can be one of the following types:
        int, float, string, int list, float list, or string list.

    Raises:
      ValueError: if one of the arguments is invalid.
    """
    # Keys in kwargs are unique, but 'name' could the name of a pre-existing
    # attribute of this object.  In that case we refuse to use it as a
    # hyperparameter name.
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError('Multi-valued hyperparameters cannot be empty: %s' %
                         name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    setattr(self, name, value)

  def set_hparam(self, name, value):
    """Set the value of an existing hyperparameter.

    This function verifies that the type of the value matches the type of the
    existing hyperparameter.
    Args:
      name: Name of the hyperparameter.
      value: New value of the hyperparameter.

    Raises:
      ValueError: If there is a type mismatch.
    """
    param_type, is_list = self._hparam_types[name]
    if isinstance(value, list):
      if not is_list:
        raise ValueError(
            'Must not pass a list for single-valued parameter: %s' % name)
      setattr(self, name,
              [_cast_to_type_if_compatible(name, param_type, v) for v in value])
    else:
      if is_list:
        raise ValueError('Must pass a list for multi-valued parameter: %s.' %
                         name)
      setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

  def override_from_dict(self, values_dict):
    """Override hyperparameter values, parsing new values from a dictionary.

    Args:
      values_dict: Dictionary of name:value pairs.

    Returns:
      The `HParams` instance.
    Raises:
      ValueError: If `values_dict` cannot be parsed.
    """
    for name, value in values_dict.items():
      self.set_hparam(name, value)
    return self

  # @deprecation.deprecated(None, 'Use `override_from_dict`.')
  def set_from_map(self, values_map):
    """DEPRECATED. Use override_from_dict."""
    return self.override_from_dict(values_dict=values_map)

  def set_model_structure(self, model_structure):
    self._model_structure = model_structure

  def get_model_structure(self):
    return self._model_structure

  def to_json(self, indent=None, separators=None, sort_keys=False):
    """Serializes the hyperparameters into JSON.

    Args:
      indent: If a non-negative integer, JSON array elements and object members
        will be pretty-printed with that indent level. An indent level of 0, or
        negative, will only insert newlines. `None` (the default) selects the
        most compact representation.
      separators: Optional `(item_separator, key_separator)` tuple. Default is
        `(', ', ': ')`.
      sort_keys: If `True`, the output dictionaries will be sorted by key.

    Returns:
      A JSON string.
    """
    return json.dumps(
        self.values(),
        indent=indent,
        separators=separators,
        sort_keys=sort_keys)

  def parse_json(self, values_json):
    """Override hyperparameter values, parsing new values from a json object.

    Args:
      values_json: String containing a json object of name:value pairs.

    Returns:
      The `HParams` instance.
    Raises:
      ValueError: If `values_json` cannot be parsed.
    """
    values_map = json.loads(values_json)
    return self.override_from_dict(values_map)

  def values(self):
    """Return the hyperparameter values as a Python dictionary.

    Returns:
      A dictionary with hyperparameter names as keys.  The values are the
      hyperparameter values.
    """
    return {n: getattr(self, n) for n in self._hparam_types.keys()}

  def get(self, key, default=None):
    """Returns the value of `key` if it exists, else `default`."""
    if key in self._hparam_types:
      # Ensure that default is compatible with the parameter type.
      if default is not None:
        param_type, is_param_list = self._hparam_types[key]
        type_str = 'list<%s>' % param_type if is_param_list else str(param_type)
        fail_msg = ("Hparam '%s' of type '%s' is incompatible with "
                    'default=%s' % (key, type_str, default))

        is_default_list = isinstance(default, list)
        if is_param_list != is_default_list:
          raise ValueError(fail_msg)

        try:
          if is_default_list:
            for value in default:
              _cast_to_type_if_compatible(key, param_type, value)
          else:
            _cast_to_type_if_compatible(key, param_type, default)
        except ValueError as e:
          raise ValueError('%s. %s' % (fail_msg, e))

      return getattr(self, key)

    return default

  def __contains__(self, key):
    return key in self._hparam_types

  def __str__(self):
    return str(sorted(self.values().items()))

  def __repr__(self):
    return '%s(%s)' % (type(self).__name__, self.__str__())

  @staticmethod
  def _get_kind_name(param_type, is_list):
    """Returns the field name given parameter type and is_list.

    Args:
      param_type: Data type of the hparam.
      is_list: Whether this is a list.

    Returns:
      A string representation of the field name.
    Raises:
      ValueError: If parameter type is not recognized.
    """
    if issubclass(param_type, bool):
      # This check must happen before issubclass(param_type, six.integer_types),
      # since Python considers bool to be a subclass of int.
      typename = 'bool'
    elif issubclass(param_type, six.integer_types):
      # Setting 'int' and 'long' types to be 'int64' to ensure the type is
      # compatible with both Python2 and Python3.
      typename = 'int64'
    elif issubclass(param_type, (six.string_types, six.binary_type)):
      # Setting 'string' and 'bytes' types to be 'bytes' to ensure the type is
      # compatible with both Python2 and Python3.
      typename = 'bytes'
    elif issubclass(param_type, float):
      typename = 'float'
    else:
      raise ValueError('Unsupported parameter type: %s' % str(param_type))

    suffix = 'list' if is_list else 'value'
    return '_'.join([typename, suffix])
