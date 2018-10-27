# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Base class and helper functions for configurable modules."""
import copy
import inspect
import json
import math
import pprint
import os
import time

import tensorflow as tf

__all__ = ['load_config', 'save_config']


class Config(dict):
  """a dictionary that supports dot and dict notation.

  Create:
    d = Config()
    d = Config({'val1':'first'})

  Get:
    d.val2
    d['val2']

  Set:
    d.val2 = 'second'
    d['val2'] = 'second'
  """
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __str__(self):
    return pprint.pformat(self)

  def __deepcopy__(self, memo):
    return self.__class__([(copy.deepcopy(k, memo), copy.deepcopy(v, memo))
                           for k, v in self.items()])


def to_config(mapping):
  out = Config(copy.deepcopy(mapping))
  for k, v in out.iteritems():
    if isinstance(v, dict):
      out[k] = to_config(v)
  return out


def to_config(mapping):
  out = Config(copy.deepcopy(mapping))
  for k, v in out.iteritems():
    if isinstance(v, dict):
      out[k] = to_config(v)
  return out


def load_config(config_file, config=None):
  """Load config from file and string with priority given to string."""
  parsed_config = {}
  if config:
    parsed_config = parse_config_string(config)

  if config_file:
    with tf.gfile.GFile(config_file, 'r') as f:
      loaded_config = json.load(f)

    # Merge config
    parsed_config = merge(loaded_config, parsed_config)

  tf.logging.info('Loaded config:\n%s', parsed_config)
  return parsed_config


def save_config(output_dir, config):
  """Write out a config as %output_dir%/config.json.

  Args:
    output_dir: string directory to save config in.
    config: dict to save.

  Raises:
    ValueError: If the file already exists.
  """
  filename = os.path.join(output_dir, 'config.json')
  if tf.gfile.Exists(filename):
    raise ValueError('Config file %s already exists' % filename)

  # Write to file atomically
  content = json.dumps(config)
  if not tf.gfile.IsDirectory(os.path.dirname(filename)):
    tf.gfile.MakeDirs(os.path.dirname(filename))
  # Write to tmp file first and then rename it to make the change atomic.
  tmp_file = '{}.tmp.{}'.format(filename, str(time.time()))
  with tf.gfile.GFile(tmp_file, 'w') as f:
    f.write(content)
  tf.gfile.Rename(tmp_file, filename, overwrite=True)


#############################################
# Utilities for dealing with nested configs #
#############################################


def merge(*args, **kwargs):
  """Merge together an iterable of configs in order.

  The first instance of a key determines its type. The last instance of a key
  determines its value.

  Example:

  args[0] = {
    'layers': 1,
    'dropout': 0.5
  }
  kwargs = {
    dropout': 0
  }

  Final result:
  {
    'layers': 1,
    'dropout': 0.0  # note the type change
  }

  Args:
    *args: List of dict-like objects to merge in order.
    **kwargs: Any additional values to add. Treated like as a final additional
      dict to merge.

  Returns:
    dict resulting from merging all configs together.

  Raises:
    TypeError: if there is a type mismatch between the same key across dicts.
  """
  assert args
  config = copy.deepcopy(args[0])
  configs = list(args)
  configs.append(kwargs)
  for c in configs[1:]:
    for k, v in c.iteritems():
      if isinstance(v, dict):
        v = copy.deepcopy(v)
      if k in config:
        value_type = type(config[k])

        if not isinstance(v, value_type):
          v = _convert_type(v, value_type)

        if isinstance(v, dict):
          config[k] = merge(config[k], v)
        else:
          config[k] = v
      else:
        config[k] = v

  return copy.deepcopy(config)


def flatten_config(config):
  """Flatten a nested dict. {'test': {'config': 3}} --> {'test.config': 3}."""
  result = {}
  for key, val in config.iteritems():
    if isinstance(val, dict):
      flattened = flatten_config(val)
      for inner_key, inner_val in flattened.iteritems():
        newkey = '%s.%s' % (key, inner_key)
        result[newkey] = inner_val
    else:
      result[key] = val
  return result


def unflatten_dict(flat_dict):
  """Convert a flattened dict to a nested dict.

  Inverse of flatten_config.

  Args:
    flat_dict: A dictionary to unflatten.

  Returns:
    A dictionary with all keys containing `.` split into nested dicts.
    {'a.b.c': 1} --> {'a': {'b': {'c': 1}}}
  """
  result = {}
  for key, val in flat_dict.iteritems():
    parts = key.split('.')
    cur = result
    for part in parts[:-1]:
      if part not in cur:
        cur[part] = {}
      cur = cur[part]
    cur[parts[-1]] = val
  return result


def config_to_string(config):
  """Convert a config to a string that can be passed in at command line."""
  flattened = flatten_config(config)
  parts = []
  for key, val in flattened.iteritems():
    parts.append('%s=%s' % (key, val))
  # Sort to make sure it is deterministic
  return ','.join(sorted(parts))


def _check_boolean(string):
  """Attempt to convert a string to a boolean variable if matches."""
  if string in ['True', 'true']:
    return True
  elif string in ['False', 'false']:
    return False
  else:
    return string


def _try_numeric(string):
  """Attempt to convert a string to an int then a float.


  Args:
    string: String to convert

  Returns:
    Attempts, in order, to return an integer, a float, and finally the original
    string.
  """
  try:
    float_val = float(string)
    if math.floor(float_val) == float_val:
      return int(float_val)
    return float_val
  except ValueError:
    return string


def smart_split(s, delimiter=','):
  """Split e.g., "a=5,b=5,2,c=4", into ['a=5', 'b=5,2', 'c=4']."""
  tokens = []
  for token in s.split(delimiter):
    if token.find('=') != -1:  # has '=' sign, so it's the right split
      tokens.append(token)
    else:  # no '=' sign, incorrect split, so merge
      tokens[len(tokens) - 1] += delimiter + token
  return tokens


def parse_config_string(string):
  """Parse a config string such as one produced by `config_to_string`.

  example:

  A config of:
  ```
  {
    'model': {
      'fn': 'RNN'
    }
    'train_steps': 500
  }
  ```

  Yields a serialized string of: `model.fn=RNN,train_steps=500`

  Args:
   string: String to parse.

  Returns:
    dict resulting from parsing the string. Keys are split on `.`s.

  """
  result = {}
  for entry in smart_split(string, delimiter=','):
    try:
      key, val = entry.split('=')
    except ValueError:
      raise ValueError('Error parsing entry %s' % entry)
    val = _check_boolean(val)
    val = _try_numeric(val)
    result[key] = val
  return copy.deepcopy(unflatten_dict(result))


def _convert_type(val, tp):
  """Attempt to convert given value to type.

  This is used when trying to convert an input value to fit the desired type.

  Args:
    val: Value to convert.
    tp: Type to convert to.

  Returns:
    Value after type conversion.

  Raises:
    ValueError: If the conversion fails.
  """
  if tp in [int, float, str, unicode, bool, tuple, list]:
    in_type = type(val)
    cast = tp(val)
    if in_type(cast) != val:
      raise TypeError(
          'Type conversion between %s (%s) and %s (%s) loses information.' %
          (val, type(val), cast, tp))
    return cast
  raise ValueError('Cannot convert %s (type %s) to type %s' % (val, type(val),
                                                               tp))


def all_subclasses(cls):
  """Return all subclasses of specified class.

  Used to fetch subclasses of Configurable.

  Args:
    cls: A class to find all subclasses of.

  Returns:
    List of subclasses of `cls`.
  """
  return cls.__subclasses__() + [
      g for s in cls.__subclasses__() for g in all_subclasses(s)
  ]


class Configurable(object):
  """Mixin that specifies that a class is configurable.

  Provides a default build_config method that recursively loads configs.
  """

  def __init__(self, config=None):
    config = config or {}
    self.config = self.build_config(**config)

  @staticmethod
  def _config():
    return {}

  @classmethod
  def parse(cls, string):
    """Parse a config string to a config dict."""
    partial_config = parse_config_string(string)
    config = cls.build_config(**partial_config)
    return config

  @classmethod
  def build_config(cls, path=None, **kwargs):
    """Build the default config and merge in given kwargs.

    Starts by getting the modules default config using `cls._config()`. It then
    iterates through each key and recursively merges in any corresponding values
    in `kwargs`.

    If a value is a subclass of Configurable, then its `build_config` method is
    recursively called. `kwargs` may set a different subconfig by setting its
    `fn` value.

    Example:

    Model._config = {
       # DummyConfigurableClass is any Configurable class.
       # Replaced by RNNLayer from the provided config.
      'encoder': DummyConfigurableClass,
      'dropout': 0.5
    }

    # RNNLayer specifies that it has 1 layer by default
    RNNLayer._config = {
      'layers': 1
    }

    # The user passes in a config specifying that we should use RNNLayer
    # as the encoder with a dropout of 0.1
    **kwargs = {
      'encoder': {
        'fn': 'RNNLayer',
      },
        'dropout': 0.1
    }

    # After merging in the provided config into the base config (and loading
    #  the defaults from RNNLayer), the final config for the experiment is:
    {
      'encoder': {
        'fn': 'RNNLayer',
        'layers': 1
      },
      'dropout': 0.1
    }

    Args:
      path: Path from config root to current point in tree. This is used to
        provide more informative errors.
      **kwargs: A nested dictionary specifying values to override.

    Returns:
      A fully specified config, where all sub Configurable objects have been
      expanded to their full config.

    Raises:
      ValueError: If a config contains a class that isn't configurable, or if
      **kwargs contains a value that isn't in the original _config.

    """
    path = path or []
    defaults = cls._config()
    config = copy.deepcopy(defaults)

    for key, val in config.iteritems():
      merge_val = {}  # Used if default value is a class or dictionary
      if key in kwargs:
        merge_val = kwargs[key]

      if inspect.isclass(val):
        # Case 1: Is a class & is configurable
        if not issubclass(val, Configurable):
          raise ValueError(
              'Configs can only contain Configurable classes for now')
        if inspect.isclass(merge_val) and issubclass(merge_val, Configurable):
          # If the new value is a configurable class,
          #  replace current values entirely
          config[key] = merge_val.build_config(path=path + [key])  # pylint: disable=no-member
        else:
          if 'fn' in merge_val and merge_val['fn'] != val.__name__:
            # Load the proper class
            # TODO(ddohan): Eventually we are going to hit namespace issues
            # Perhaps we should allow specifying a namespace/scope
            val = cls.load(merge_val)
          config[key] = val.build_config(path=path + [key], **merge_val)
      elif isinstance(val, dict):
        # Case 2: Is a dictionary. Merge in any shared values
        new_val = copy.deepcopy(val)  # Make a copy before modifying
        new_val.update(merge_val)
        config[key] = new_val
      else:
        # Case 3: Is a primitive type
        if key in kwargs:
          # tp = config.get_type(key)
          tp = type(config[key])
          new_val = kwargs[key]
          if not isinstance(new_val, tp):
            new_val = _convert_type(new_val, tp)

          if new_val != val:
            # Log out any non-default values
            tf.logging.info('Overriding %s from %s (tp %s) to %s (tp %s)',
                            '.'.join(path + [key]), val, type(val), new_val,
                            type(new_val))
          config[key] = new_val

    # Throw an error for any values provided that weren't used
    for key, val in kwargs.iteritems():
      # The special case of `fn` is ugly...
      if key not in config and key != 'fn':
        tf.logging.info('Provided key:val %s:%s is not in the config.', key,
                        val)

    config['fn'] = cls.__name__

    # Use a frozen config whenever possible to reduce error surface

    x = copy.deepcopy(config)
    x = Config(x)
    return x

  @staticmethod
  def load(config):
    """Load subclass of Configurable by name or config."""
    if isinstance(config, str):
      fn = config
    elif isinstance(config, dict):
      if 'fn' not in config:
        raise ValueError('Config must contain `fn` field to lookup class.')
      fn = config['fn']
      if inspect.isclass(fn):
        if not issubclass(fn, Configurable):
          raise ValueError(
              'Provided a fn to use that is not an instance of Configurable. '
              'Was %s' % fn)
        return fn
    else:
      raise ValueError('Must pass in a string or a dictionary')
    match = None
    subclasses = all_subclasses(Configurable)
    for subclass in subclasses:
      if subclass.__name__ == fn:
        if match is not None:
          raise ValueError('Multiple matches for %s', fn)
        match = subclass

    if match is None:
      raise ValueError('No subclass of Configurable with name %s found', fn)
    return match

  @staticmethod
  def initialize(config, **kwargs):
    """Initialize instance of Configurable module from given config."""
    return Configurable.load(config)(config=config, **kwargs)
