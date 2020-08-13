# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python2, python3
"""ParameterContainer is a class for managing hyper-parameters.

Key capabilities:


1. CONVERTS A PYTHON DICTIONARY TO A CLASS, WHERE THE DICTIONARY KEYS ARE
  CONVERTED TO ATTRIBUTES. For example:

  params_dict = {'key1': 'value1',
                 'nested': {'key2': 'val2',
                            'key3': 'val3'}}

  pc = ParameterContainer(params_dict)
  assert pc.key1 == 'value1'
  assert pc.nested.key2 == 'val2'
  assert pc.nested.key3 == 'val3'


2. ALLOWS EXTENSION AND OVERRIDING. For example:

  extension_dict = {'nested': {'key2': 'new_val2'},
                    'key4': 'val4',
                    'nested2': {'key5': 'val5'}}

  pc.override(extension_dict)
  assert pc.key1 == 'value1'
  assert pc.nested.key2 == 'new_val2'
  assert pc.nested.key3 == 'val3'
  assert pc.key4 == 'val4'
  assert pc.nested2.key5 == 'val5'


3. ALLOWS OVERRIDING VALUES WITH USER-SPECIFIED PARAMETERS. For example:

  user_params = {nested2': {'key5': 'user_val5'},
                 'key4': 'user_val4'}

  pc.override(extension_dict, is_custom=True)
  assert pc.key1 == 'value1'
  assert pc.nested.key2 == 'new_val2'
  assert pc.nested.key3 == 'val3'
  assert pc.key4 == 'user_val4'
  assert pc.nested2.key5 == 'user_val5'

  The differences between is_custom=True and the default is_custom=False are:

  a. Once a parameter was set to be custom, its value becomes immutable.
  b. On can later query pc.is_custom(parameter_name) to know if it was set by
     the user or not. The idea is that user-specified parameters are set in
     stone. For example, we may decide to choose the weight decay of
     convolutions depending on whether we are using batch-norm or not. But if
     the user explicitly specified the weight decay, we will always use the
     latter.


4. LOADS THE PARAMETERS FROM JSON format.
  The override method can receive a filepath to a json file or a
  literal json contents instead of a dictionary.


5. HAS A LOCK METHOD.
  While we do want to allow adjustments in the parameters at the early stages
  (e. g. the batch-norm-dependent weight decay mentioned above), we would like
  the parameters to be read-only. The lock() method will make the
  ParameterContainer read-only. That is, an exception will be thrown when
  attempting to modify values (of course you can hack around it, but you
  shouldn't).

  Before lock() has been called, values can be modified directly:
  pc.nested.key3 = 'another_val3'
  assert pc.nested.key3 == 'another_val3'

  After calling lock(), the lines above will throw an exception.


WHY NOT USE PROTOCOL BUFFERS?
Protocol buffers have four main disadvantages compared to ParameterContainer:

- They require at least one separate file, probably two.
  The intended usage pattern of ParameterContainer is quickly defining a new
  network architecture and training setup, which will mostly reuse existing
  hyper-parameters, but may override a few values and add a few new specific
  keys.
  With Protocol Buffers, extension would require an additional proto file, and
  overriding would require an additional pbtxt file, because proto extensions
  cannot override default values of their parent. We would have to reference
  the pbtxt file somewhere, parse it and override the parameter values, which is
  not elegant and error prone.

- When merging protocol buffers, repeated fields are concatenated rather than
  replaced. This is usually not what we want - if a parameter is a list of
  values, and the user overrides it, they most likely want to replace the list
  by their own list.

- When using protocol buffer extensions, passing a literal string in a user flag
  is somewhat cluttered. For example:
  [my_namespace.MyExtensionName.ext] { my_parameter_name: "value" }

- Protocol buffers do not distinguish between overriding by new defaults (say,
  I have a network architecture where I know the weight decay should be
  different than the common value), and overriding by user values. In both
  cases, querying has_ will return true.


WHY NOT USE HPARAMS?
HParams is a wrapper over protocol buffers, often used with TensorFlow, which
also implements a parser for user-specified values. It also resolves the merging
issue of repeating fields, which are replaced instead of being concatenated.
However:

- It is still a protocol buffer, so it requires extra files for each new
  network architecture / training setup.
- Extensions are not supported, nor is nesting, which means that all parameters
  live in the one long list, and cannot be grouped into meaningful groups.


WHY JSON?
- Writing our own format / parser sounds like a bad idea.
- Json can be parsed from c++ too, unlike other python-specific serializations.
- Json is the preferred (and default) format.

ParameterContainer works well with polymorphism. If a base class contains a
ParameterContainer object, subclasses may call its override() method to add or
override parameters, but they cannot remove parameters. Therefore one can be
sure that if the base class' ParameterContainer has some key, all subclasses
will have it too.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import copy
import json
import six

# Datatypes that should be overridden without further nesting.
PRIMITIVES = (int, float, bool, six.string_types)


def parse_object(object_or_json_or_file, allow_dict=False, allow_list=False):
  """Parses an allowed object from several alternative representations.

  Parsing is attempted in the order specified in the argument definition below,
  until it succeeds. The parsing order matters because ", ', {, } are all valid
  filename characters, and because a single quoted string is a valid json.

  Args:
    object_or_json_or_file: Any of the following: * A dictionary, or * a list,
      or * a string with a json serialization of a dictionary, or * a string
      with a python dictionary literal, or * a string with a path to a file with
      a json serialization of a dictionary
    allow_dict: Whether to allow the parsed object to be a dict.
    allow_list: Whether to allow the parsed object to be a list.

  Returns:
    A dictionary or list, which is object_or_json_or_file itself if the latter
      dictionary or list, or the parse result otherwise.

  Raises:
    ValueError: If the json could not be parsed as a an allowed object.
    OSError: If the file did not open.

  """
  if not allow_dict and not allow_list:
    raise ValueError('At least one of allow_dict or allow_list must be True.')

  if not object_or_json_or_file:
    return {}

  if isinstance(object_or_json_or_file, six.string_types):
    # first, attempt to parse the string as a JSON dict
    try:
      object_or_json_or_file = json.loads(object_or_json_or_file)
    except ValueError as literal_json_parsing_error:
      # then try to parse as a python dict literal
      #
      # Note that parsing as JSON above is still required, since null literal
      # (without quotes) is not valid python but is valid json (it is mapped to
      # None by json.loads).
      try:
        object_or_json_or_file = ast.literal_eval(object_or_json_or_file)
        # Looking for a literal dict, no need for isinstance.
        if type(object_or_json_or_file) == dict:  # pylint: disable=unidiomatic-typecheck
          if not allow_dict:
            raise ValueError(
                'object_or_json_or_file parsed as a dictionary, but allow_dict=False.'
            )
          return object_or_json_or_file
        # Looking for a literal list, no need for isinstance.
        elif type(object_or_json_or_file) == list:  # pylint: disable=unidiomatic-typecheck
          if not allow_list:
            raise ValueError(
                'object_or_json_or_file parsed as a list, but allow_list=False.'
            )
          return object_or_json_or_file
      # These are the only exceptions ever raised by literal_eval.
      except SyntaxError as e:
        python_parsing_error = e
      except ValueError as e:
        python_parsing_error = e
      else:
        python_parsing_error = None

      try:
        # then try to use as a path to a JSON file
        f = open(object_or_json_or_file)
        object_or_json_or_file = json.load(f)
        f.close()
      except ValueError as json_file_parsing_error:
        raise ValueError('Unable to parse the content of the json file %s. '
                         'Parsing error: %s.' %
                         (object_or_json_or_file, str(json_file_parsing_error)))
      except OSError as file_error:
        max_file_error_len = 50
        if len(str(file_error)) > max_file_error_len:
          file_error_str = str(file_error)[:max_file_error_len] + '...'
        else:
          file_error_str = file_error.message
        message = ('Unable to parse override parameters either as a literal '
                   'JSON or as a python dictionary literal or as the name of '
                   'a file that exists.\n\n'
                   'GFile error: %s\n\n'
                   'JSON parsing error: %s\n\n'
                   'Python dict parsing error: %s\n\n'
                   'Override parameters:\n%s.\n' %
                   (file_error_str, str(literal_json_parsing_error),
                    str(python_parsing_error), object_or_json_or_file))
        if '{' in object_or_json_or_file or '}' in object_or_json_or_file:
          message += _debug_message(object_or_json_or_file)

        raise ValueError(message)

  if isinstance(object_or_json_or_file, dict):
    if not allow_dict:
      raise ValueError(
          'object_or_json_or_file parsed as a dictionary, but allow_dict=False.'
      )
  elif isinstance(object_or_json_or_file, list):
    if not allow_list:
      raise ValueError(
          'object_or_json_or_file parsed as a list, but allow_list=False.')
  else:
    raise ValueError(
        'object_or_json_or_file did not parsed to a supported type. Found: %s' %
        object_or_json_or_file)
  return object_or_json_or_file


def parse_dict(dict_or_json_or_file):
  return parse_object(dict_or_json_or_file, allow_dict=True)


def _get_key_and_indices(maybe_key_with_indices):
  """Extracts key and indices from key in format 'key_name[index0][index1]'."""
  patterns = maybe_key_with_indices.split('[')
  if len(patterns) == 1:
    return (maybe_key_with_indices, None)
  # For each index ensure that the brackets are closed and extract number
  indices = []
  for split_pattern in patterns[1:]:
    # Remove surrounding whitespace.
    split_pattern = split_pattern.strip()
    if split_pattern[-1] != ']':
      raise ValueError(
          'ParameterName {} has bad format. Supported format: key_name, '
          'key_name[index0], key_name[index0][index1], ...'.format(
              maybe_key_with_indices))
    try:
      indices.append(int(split_pattern[:-1]))
    except ValueError:
      raise ValueError(
          'Only integer indexing allowed for ParameterName. '
          'Faulty specification: {}'.format(maybe_key_with_indices))
  return patterns[0], indices


class ParameterContainer(object):
  """Helper class that provides a class-like interface to a dictionary.

  It can be initialized with a dictionary of custom parameters, which can
  be passed either as a dict, a JSON string, or a path to a JSON file.

  ParameterContainer supports nesting of dictionaries. For example, the
  dictionary

  {
      'key1': 'value1',
      'nested': {
          'key2': 'val2',
          'key3': 'val3'
      }
  }

  will be converted to a ParameterContainer pc that has the attributes
  pc.key1, pc.nested.key2, and pc.nested.key3. pc.nested is a ParameterContainer
  itself.
  """

  SPECIAL_ATTRS = ['_lock', '_custom_params']
  # Allow integration with static python type checking. See
  # https://opensource.google/projects/pytype. Since ParameterContainer
  # dynamically sets its attributes, pytype needs guidance to know that pc.key1
  # is valid.
  HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, default_params=None, custom_params=None):
    if default_params is None:
      default_params = {}
    self._custom_params = set()
    self._lock = False
    if not custom_params:
      custom_params = {}

    self.override(default_params)
    self.override(custom_params, True)

  @classmethod
  def from_defaults_and_overrides(cls, defaults, overrides, **override_kwargs):
    """Creates a new object from defaults and overrides.

    Args:
      defaults: A dictionary or another type of object that ParameterContainer's
        consructor accepts, containing the default arguments.
      overrides: A dictionary or another type of object that
        ParameterContainer's consructor accepts, containing overrides for the
        default arguments.
      **override_kwargs: Keyword arguments that are passed to the `override`
        method when `overrides` override `defaults`.

    Returns:
      A ParameterContainer object constructed from `defaults` and overridden by
      `overrides`.
    """
    params = cls(defaults)
    params.override(overrides, **override_kwargs)
    return params

  def override(self,
               params,
               is_custom=False,
               is_strict=False,
               strictness_depth=None,
               allow_custom_override=False):
    """Extends fields and overrides values of the ParameterContainer.

    Args:
      params: A dictionary, where the keys are strings and the values are
        objects that are json-serializable or dictionaries where the keys
        are strings and the values are objects that are json-serializable or
        other dictionaries... (and so on), OR:
        A string, a json-serialization of that dictionary, OR:
        A string, a filepath to a json-serialization of that dictionary, OR:
        A ParameterContainer. In the latter case, the 'custom' properties of
        the keys in 'params' will propagate to self.
      is_custom: A boolean, any overridden / extended parameter will be marked
        as 'custom', and will become immutable.
      is_strict: If true, an exception will be thrown if a key in
        params_dict (or in its nested sub-ductionaries) does not already exist
        in the ParameterContainer. In other words, if is_strict is True, only
        overriding is allowed, but extension is forbidden.
      strictness_depth: An integer or None. If is_strict is true,
        strictness_depth states for how many levels of depth the strictness
        will hold. For example, if strictness_depth is 1, strictness is imposed
        only the first level keys (as opposed to nested keys). None means that
        strictness will be imposed all the way through, at all levels.
      allow_custom_override: if true, no exception are thrown if custom params
        are being overridden.

      The ParameterContainer remembers, for any parameter, if the last update
      was with is_custom True or False.

    Raises:
      ValueError: If one of the keys in the dictionary is not a string, or if
        param_dict was not successfully parsed from the literal json or json
        file.
      AttributeError: If one of the keys in 'params_dict' cannot be updated,
        because of one of the following reasons:
        1. is_strict=True and the key does not exist in the ParameterContainer.
        2. The key has already been updated with a custom value, which made it
           immutable.
        3. The ParameterContainer (or the relevant nested ParameterContainer is
           locked.
    """
    if strictness_depth is not None:
      if not is_strict:
        raise ValueError('strictness_depth can only be specified when is_strict'
                         ' is true')
      if strictness_depth <= 0:
        raise ValueError(
            'strictness_depth must be positive, not %d' % strictness_depth)

    if isinstance(params, ParameterContainer):
      params_dict = params.as_dict()
      if is_custom:
        self._override(
            params_dict,
            is_custom=True,
            is_strict=is_strict,
            strictness_depth=strictness_depth,
            allow_custom_override=allow_custom_override)
      else:
        self._override(
            params_dict,
            is_custom=False,
            is_strict=is_strict,
            strictness_depth=strictness_depth,
            allow_custom_override=allow_custom_override)
        self._override(
            params.as_dict(custom_only=True),
            is_custom=True,
            allow_custom_override=allow_custom_override)
    else:
      self._override(
          params,
          is_custom=is_custom,
          is_strict=is_strict,
          strictness_depth=strictness_depth,
          allow_custom_override=allow_custom_override)

  def _override_list_element(self, indices, nested_list, params_dict, is_custom,
                             is_strict, strictness_depth,
                             allow_custom_override):
    """Recursively overrides the item at list index indices[0]."""
    if indices:
      if not isinstance(nested_list, list):
        raise AttributeError(
            'Nested List mismatch: Can only override list with list.')
      nested_list[indices[0]] = self._override_list_element(
          indices[1:], nested_list[indices[0]], params_dict, is_custom,
          is_strict, strictness_depth, allow_custom_override)
    elif isinstance(nested_list, PRIMITIVES):
      if not isinstance(params_dict, PRIMITIVES):
        raise AttributeError(
            'Nested List mismatch: Can only override primitive with primitive.')
      nested_list = params_dict
    else:
      if isinstance(nested_list, dict):
        nested_list = ParameterContainer(nested_list)
      # Using _override to bypass strictness_depth > 0 test in override.
      # pylint: disable=protected-access
      nested_list._override(params_dict, is_custom, is_strict, strictness_depth,
                            allow_custom_override)
      # pylint: enable=protected-access
    return nested_list

  def _override(self,
                params_dict,
                is_custom=False,
                is_strict=None,
                strictness_depth=None,
                allow_custom_override=False):
    """Extends fields and overrides values of the ParameterContainer."""
    self._raise_if_locked()
    params_dict = parse_dict(params_dict)
    if strictness_depth == 0:
      is_strict = False
    if strictness_depth is not None and strictness_depth > 0:
      strictness_depth -= 1
    for k, v in six.iteritems(params_dict):
      if not isinstance(k, six.string_types):
        raise ValueError('The keys in the dictionary must be strings, \'%s\' '
                         'encountered (type: %s).' % (str(k), type(k)))
      k, indices = _get_key_and_indices(k)
      if indices is not None and not isinstance(self.__dict__[k], list):
        raise ValueError('Only parameters with list values can use '
                         'indices (key=%s, index0=%d)' % (k, indices[0]))
      # If v is a dict k needs to be overridden recursively.
      if isinstance(v, dict):
        if hasattr(self, k) and isinstance(self.__dict__[k],
                                           ParameterContainer):
          # Using _override to bypass strictness_depth > 0 test in override.
          # pylint: disable=protected-access
          self.__dict__[k]._override(v, is_custom, is_strict, strictness_depth,
                                     allow_custom_override)
          # pylint: enable=protected-access
        else:
          # self.__dict__[k] is not a ParameterContainer.
          if hasattr(self, k):
            if isinstance(self.__dict__[k], list) and indices is not None:
              # If k's value is a list and indices are specified, recursively
              # override the value at indices[0].
              self.__dict__[k] = self._override_list_element(
                  indices, self.__dict__[k], v, is_custom, is_strict,
                  strictness_depth, allow_custom_override)
            else:
              # No known handling strategy for the paramater configuration.
              raise TypeError(
                  'Parameter {} cannot be overridden with {}'.format(k, v))
          elif is_strict:
            # Only existing keys can be overridden if is_strict.
            raise AttributeError('Parameter not recognized: %s' % k)
          else:
            # Unknown parameter, create empty container at k and override it.
            self.__dict__[k] = ParameterContainer()
            # Using _override to bypass strictness_depth > 0 test in override.
            # pylint: disable=protected-access
            self.__dict__[k]._override(v, is_custom, is_strict,
                                       strictness_depth, allow_custom_override)
            # pylint: enable=protected-access

      else:
        if not allow_custom_override:
          self._raise_if_custom(k)
        if is_strict or k in self.__dict__:
          self.__setattr__(k, v, allow_custom_override, indices=indices)
        else:
          if isinstance(v, StringEnum):
            self.__dict__[k] = copy.deepcopy(v)
          else:
            self.__dict__[k] = v
      if is_custom:
        self._custom_params.add(k)

  def _raise_if_custom(self, key):
    if hasattr(self, '_custom_params') and key in self._custom_params:
      raise AttributeError(
          'Parameter %s is immutable because it has a custom value.' % key)

  def _raise_if_locked(self):
    if hasattr(self, '_lock') and self._lock:
      raise AttributeError('Cannot set parameter: the ParameterContainer '
                           'is locked.')

  def _set_nested_attribute(self, nested_list, indices, value):
    if indices:
      if not isinstance(nested_list, list):
        raise AttributeError(
            'Nested List mismatch: Can only override list with list.')
      nested_list[indices[0]] = self._set_nested_attribute(
          nested_list[indices[0]], indices[1:], value)
      return nested_list
    return value

  def __setattr__(self, key, value, allow_custom_override=False, indices=None):
    self._raise_if_locked()
    if not allow_custom_override:
      self._raise_if_custom(key)

    if key not in self.__dict__ and key not in ParameterContainer.SPECIAL_ATTRS:
      raise AttributeError('Parameter not recognized: %s' % key)

    if key in self.__dict__ and isinstance(self.__dict__[key], StringEnum):
      self.__dict__[key].assign(value)
    else:
      if indices is not None:
        self.__dict__[key] = self._set_nested_attribute(self.__dict__[key],
                                                        indices, value)
      else:
        self.__dict__[key] = value

  def get(self, key, default_value=None):
    v = self.__dict__.get(key, default_value)
    if isinstance(v, StringEnum):
      return v.value
    else:
      return v

  def is_custom(self, key):
    return key in self._custom_params

  def _nested_list_to_dict(self, nested_list):
    """Returns a (possibly nested) list of all parameters in the list.

    Args:
      nested_list: The list to convert.

    Returns:
      A copy of nested_list where each ParamContainer is converted to a dict
      and each StringEnum converted to its value.
    """
    output_list = []
    for element in nested_list:
      if isinstance(element, ParameterContainer):
        output_list.append(element.as_dict())
      elif isinstance(element, StringEnum):
        output_list.append(element.value)
      elif isinstance(element, list):
        output_list.append(self._nested_list_to_dict(element))
      else:
        output_list.append(element)
    return output_list

  def as_dict(self, custom_only=False):
    """Returns a (possibly nested) dictionary with all / custom parameters.

    Args:
      custom_only: A boolean, if true, only the custom parameters will be
        retured.

    Returns:
      A (possibly nested) dictionary with all / custom parameters.
    """
    params = {}
    for k, v in six.iteritems(self.__dict__):
      if (k in ParameterContainer.SPECIAL_ATTRS or
          k not in self._custom_params and custom_only):
        continue
      if isinstance(v, ParameterContainer):
        params[k] = v.as_dict(custom_only)
      elif isinstance(v, StringEnum):
        if k in self._custom_params or v.value is not None:
          params[k] = v.value
      elif isinstance(v, list):
        params[k] = self._nested_list_to_dict(v)
      else:
        params[k] = v
    return params

  def __repr__(self):
    # Will output for example ParameterContainer{'a': 1}.
    return 'ParameterContainer' + self.as_dict().__repr__()

  def __str__(self):
    return self.__repr__()

  def lock(self):
    """The ParameterContainer can be set to read-only via .lock().

    By definition, this cannot be reverted.
    """
    for v in self.__dict__.values():
      if isinstance(v, ParameterContainer):
        v.lock()
    self._lock = True


def get_params_of_indicated_type(params):
  """Gets the parameters of the type indicated in params.type itself.

  For example, if params is

  params = ParameterContainer({
      'type': 'MOMENTUM',

      'ADAM': {
          'beta1': 0.9,
          'beta2': 0.999,
      },

      'MOMENTUM': {
          'momentum': 0.9,
      },
  })

  then

  get_params_of_indicated_type(params) is

  ParameterContainer({'momentum': 0.9}).

  Args:
    params: A ParameterContainer of the structure illustrated above.

  Returns:
    A ParameterContainer which is:
    - params.get('params.type') if params.type is present
    - ParameterContainer() if params.type is not present
    - None if params.type is None.

  Raises:
    ValueError: If params does not have match the pattern illustrated above.
  """
  try:
    type_ = params.type
  except AttributeError:
    raise ValueError("`params` must have a 'type' attribute.")

  if type_ is None:
    return None

  params_of_type = params.get(type_, ParameterContainer())

  return params_of_type


def import_params(defaults, overrides, strictness_depth=None):
  """Overrides one dictionary's value with the other's.

  The choice of the name import_params is because a typical usage pattern is
  importing the parameters dictionary from some other file and possibly
  overriding some of them.

  Args:
    defaults: A dictionary
    overrides: A dictionary
    strictness_depth: An integer or None.

  Returns:
    A dictionary. For ant key that is in `override_params` its value will
    override the respective value in `base_params`. This includes nested
    dictionaries, the same way as it is done with ParameterContainer.override
    (see above).

  Raises:
    ValueError: If `overrides` (or a nested dictionary within) has a key that
      `defaults` does not have
  """
  base = ParameterContainer(defaults)
  try:
    base.override(overrides, is_strict=True, strictness_depth=strictness_depth)
  except AttributeError as e:
    raise ValueError(str(e))
  return base.as_dict()


def extend_params(params, more_params):
  """Extends dictionary with new values.

  Args:
    params: A dictionary
    more_params: A dictionary

  Returns:
    A dictionary which combines keys from both dictionaries.

  Raises:
    ValueError: if dicts have the same key.
  """
  for yak in more_params:
    if yak in params:
      raise ValueError('Key "%s" is already in dict' % yak)
  params.update(more_params)
  return params


class StringEnum(object):
  """Class for defining and storing enum value inside ParameterContainer.

  Example:
    params = ParameterContainer({
        'v': utils.StringEnum(
            [
                # Comment about a particular value goes here.
                'a',
                # Another option to consider.
                'b',
                # Also a valid value.
                'c'
            ], default_value='b')
    })
    ...
    params.v               # equals to 'b'
    params.v.value_a       # equals to 'a'

    params.v = 'a'                   # OK
    params.override({'v': 'a'}       # OK
    params.v = 'xxx'                 # raises ValueError
    params.override({'v': 'xxx'})    # raiese ValueError

    params.v == 'a'                  # OK
    params.v == 'xxx'                # raises ValueError
  """

  def __init__(self, values, default_value):
    self._values = values
    self.assign(default_value)
    self._setup_values()

  def assign(self, value):
    """Assigns value with checking correctness."""
    if not isinstance(value, six.string_types):
      raise ValueError('Can\'t assign to a non-string')
    if value not in self._values:
      raise ValueError('Expected one of ["%s"], found: "%s"' %
                       ('","'.join(self._values), value))
    self._value = value

  @property
  def value(self):
    """Returns current value of the enum."""
    return self._value

  def _setup_values(self):
    """Setups values constants."""
    for v in self._values:
      setattr(self, 'value_' + v, v)

  def __eq__(self, other):
    """Compares enum value against a string."""
    if not isinstance(other, six.string_types):
      raise ValueError('Can\'t compare with a non-string')
    if other not in self._values:
      raise ValueError('Expected one of ["%s"], found: "%s"' %
                       ('","'.join(self._values), other))
    return self._value == other

  def __ne__(self, other):
    """Compares enum value against a string."""
    return not self.__eq__(other)

  def __hash__(self):
    """Hash to be used for the enum."""
    return self._value.__hash__()

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return '\'%s\' of StringEnum([\'%s\'])' % (
        self.value, '\', \''.join(self._values))


def _debug_message(object_or_json_or_file):
  """Create a debug message for badly formatted json params."""
  debug_info = 'Debug Info:\n'
  counts = {
      c: object_or_json_or_file.count(c)
      for c in ['\'', '"', '{', '}', ':', ',']
  }
  # Expect to see same number of { and }.
  if counts['{'] != counts['}']:
    debug_info += ('Expected counts of \'{\' and \'}\' differ %d != %d\n' %
                   (counts['{'], counts['}']))
  # Usually there is a : for each two delimiters.
  if (counts[':'] * 2 != counts['\''] and counts[':'] * 2 != counts['"'] and
      counts[':'] * 2 != counts['\''] + counts['"']):
    debug_info += (
        'Expected counts of ":" (%d) to be half of string delimiters: '
        '\' (%d) and \" (%d)\n' % (counts[':'], counts['\''], counts['"']))
  # ' and " should come in pairs.
  for c in ['\'', '"']:
    if counts[c] % 2:
      debug_info += ('Expected even number of delimiter %s but notice %d' %
                     (c, counts[c]))

  debug_info += 'Char  Count\n'
  for c, count in counts.items():
    debug_info += '\n%s ==> %d' % (c, count)

  return debug_info
