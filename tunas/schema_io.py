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

"""Utilities for schema serialization and deserialization.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from typing import Any, ByteString, Callable, Dict, Iterable, Optional, Sequence, Text, Tuple, Type, TypeVar, Union

import six
from six.moves import map
from six.moves import zip

from tunas import schema

# Primitive types (like integers or strings) that are supported in both Python
# and JSON.
_PRIMITIVE_TYPES = (int, float) + six.string_types

# We don't have a good way to identify namedtuples to the Python type system,
# except that they're subclasses of tuple.
_NamedTupleTypeVar = TypeVar('_NamedTupleTypeVar', bound=Tuple[Any, Ellipsis])

# Registration interface for namedtuple serialization and deserialization.
# typing.Type[] is not currently supported inside type annotation comments,
# so we use the annotation Any instead.
_NAMEDTUPLE_NAME_TO_CLASS = {}  # type: Dict[Text, Any]
_NAMEDTUPLE_CLASS_TO_NAME = {}  # type: Dict[Any, Text]
_NAMEDTUPLE_CLASS_TO_DEFAULTS = {}  # type: Dict[Any, Dict[Text, Any]]


def register_namedtuple(
    name,
    deprecated_names = None,
    defaults = None,
):
  """Register a namedtuple class for serialization/deserialization.

  Namedtuples that are registered can be serialized and deserialized using
  the utilities in this file.

  Example usage:

    @schema_io.register_namedtuple('package.C')
    class C(collections.namedtuple('C', ['field1'])):
      pass

    # Later in the code
    serialized = schema_io.serialize(C('foo'))    # returns a serialized string
    restored = schema_io.deserialize(serialized)  # returns a namedtuple

  Args:
    name: String, globally unique identifier for the registered class.
    deprecated_names: Optional list of Strings constaining deprecated names for
      the registered class.
    defaults: Optional list of default argument values. This makes it possible
      to add new fields to a namedtuple while preserving backwards-compatibility
      for old objects which are loaded from disk.

  Returns:
    A class decorator.
  """
  def decorator(cls):
    """Register a new class instance."""
    if name in _NAMEDTUPLE_NAME_TO_CLASS:
      raise ValueError('Duplicate name in registry: {:s}'.format(name))
    if cls in _NAMEDTUPLE_CLASS_TO_NAME:
      raise ValueError('Duplicate class in registry: {:s}'.format(name))
    if not issubclass(cls, tuple) or not hasattr(cls, '_fields'):
      raise ValueError(
          'Cannot register class {}.{} because it is not a namedtuple'
          .format(cls.__module__, cls.__name__))

    _NAMEDTUPLE_NAME_TO_CLASS[name] = cls
    _NAMEDTUPLE_CLASS_TO_NAME[cls] = name

    if deprecated_names:
      for deprecated_name in deprecated_names:
        if deprecated_name in _NAMEDTUPLE_NAME_TO_CLASS:
          raise ValueError(
              'Duplicate name registered: {:s}'.format(deprecated_name))
        _NAMEDTUPLE_NAME_TO_CLASS[deprecated_name] = cls

    if defaults:
      for field in sorted(defaults.keys()):
        if field not in cls._fields:
          raise ValueError(
              'Field {} appears in defaults but not in class {}.{}'
              .format(field, cls.__module__, cls.__name__))
      _NAMEDTUPLE_CLASS_TO_DEFAULTS[cls] = dict(defaults)

    return cls
  return decorator


def namedtuple_class_to_name(cls):
  if cls not in _NAMEDTUPLE_CLASS_TO_NAME:
    raise KeyError(
        'Namedtuple class {}.{} is not registered. Did you forget to use a '
        '@schema_io.register_namedtuple() decorator?'
        .format(cls.__module__, cls.__name__))
  return _NAMEDTUPLE_CLASS_TO_NAME[cls]


def namedtuple_name_to_class(name):
  if name not in _NAMEDTUPLE_NAME_TO_CLASS:
    raise KeyError(
        'Namedtuple name {} is not registered. Did you forget to use a '
        '@schema_io.register_namedtuple() decorator?'
        .format(repr(name)))
  return _NAMEDTUPLE_NAME_TO_CLASS[name]


def _to_json(structure):
  """Convert a nested datastructure to pure JSON."""
  if structure is None or isinstance(structure, _PRIMITIVE_TYPES):
    return structure
  elif isinstance(structure, schema.OneOf):
    result = ['oneof']
    result.append(['choices', _to_json(structure.choices)])
    result.append(['tag', _to_json(structure.tag)])
    return result
  elif isinstance(structure, list):
    result = ['list']
    result.extend(map(_to_json, structure))
    return result
  elif isinstance(structure, tuple) and hasattr(structure, '_fields'):
    result = ['namedtuple:' + namedtuple_class_to_name(structure.__class__)]
    result.extend(zip(structure._fields, map(_to_json, structure)))
    return result
  elif isinstance(structure, tuple):
    result = ['tuple']
    result.extend(map(_to_json, structure))
    return result
  elif isinstance(structure, dict):
    result = ['dict']
    for k in sorted(structure):
      result.append((_to_json(k), _to_json(structure[k])))
    return result
  else:
    raise ValueError('Unrecognized type: {}'.format(type(structure)))


def _namedtuple_from_json(
    cls, kv_pairs):
  """Convert a JSON data structure to a namedtuple."""
  # Start with a list of default keyword arguments.
  if cls in _NAMEDTUPLE_CLASS_TO_DEFAULTS:
    kwargs = dict(_NAMEDTUPLE_CLASS_TO_DEFAULTS[cls])
  else:
    kwargs = dict()

  # Add all the user-provided key-value pairs.
  for key, value in kv_pairs:
    if key not in cls._fields:
      raise ValueError(
          'Invalid field: {} for class: {}, permitted values: {}'
          .format(key, cls, cls._fields))
    kwargs[key] = value

  # Make sure we've provided all the arguments we need.
  for field in cls._fields:
    if field not in kwargs:
      raise ValueError(
          'Missing field: {} for class: {}'.format(field, cls))

  # Now wrap the key-value pairs in a namedtuple.
  return cls(**kwargs)


def _from_json(structure):
  """Converted a pure JSON data structure to one with namedtuples and OneOfs."""
  if structure is None or isinstance(structure, _PRIMITIVE_TYPES):
    return structure
  elif isinstance(structure, list):
    assert structure
    typename = structure[0]
    structure = structure[1:]

    if typename == 'dict':
      return {_from_json(k): _from_json(v) for (k, v) in structure}
    elif typename.startswith('namedtuple:'):
      cls = namedtuple_name_to_class(typename[len('namedtuple:'):])
      kv_pairs = [(_from_json(k), _from_json(v)) for (k, v) in structure]
      return _namedtuple_from_json(cls, kv_pairs)
    elif typename == 'oneof':
      keys = tuple(_from_json(k) for (k, v) in structure)
      assert keys == ('choices', 'tag'), keys
      return schema.OneOf(*(_from_json(v) for (k, v) in structure))
    elif typename == 'list':
      return list(map(_from_json, structure))
    elif typename == 'tuple':
      return tuple(map(_from_json, structure))
    else:
      raise ValueError('Unsupported __type: {}'.format(typename))
  else:
    raise ValueError('Unrecognized JSON type: {}'.format(type(structure)))


def serialize(structure):
  """Serialize a nested data structure to a string.

  Args:
    structure: A recursive data structure, possibly consisting of integers,
        strings, tuples, dictionaries, and namedtuples. Namedtuples must be
        registered with the @register_namedtuple decorator above.

  Returns:
    A json-serialized string.
  """
  return json.dumps(_to_json(structure), sort_keys=True, indent=2)


def deserialize(serialized):
  """Convert a serialized string to a nested data structure.

  Args:
    serialized: A json-serialized string returned by serialize().

  Returns:
    A (possibly nested) data structure.
  """
  return _from_json(json.loads(serialized))
