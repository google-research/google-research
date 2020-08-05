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

"""Utilities for manipulating search space definitions and model specs.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v1 as tf
from typing import Any, Callable, Generic, List, Optional, Text, Tuple, TypeVar, Union


_T = TypeVar('_T')


class OneOf(Generic[_T]):
  """Class representing a choice of one of N possible values.

  """

  def __init__(self,
               choices,
               tag,
               mask = None):
    """Class initializer.

    Args:
      choices: List of possible choices.
      tag: Short, human-readable string. This string can be used to store
          metadata about the kind of choice we're making (e.g., searching over
          operations vs. searching over filter sizes).
      mask: Optional tf.Tensor indicating the selected choice.
    """
    if mask is not None and not isinstance(mask, tf.Tensor):
      raise ValueError('mask must be an instance of tf.Tensor or None')

    self._choices = choices
    self._tag = tag
    self._mask = mask

  # The return value of this property should have type List[_T], but Pytype has
  # trouble mixing @property with generics.
  @property
  def choices(self):  # pytype: disable=invalid-annotation
    return self._choices

  @property
  def tag(self):
    return self._tag

  @property
  def mask(self):
    return self._mask

  def __repr__(self):
    fields = [
        'choices={:s}'.format(repr(self.choices)),
        'tag={:s}'.format(repr(self.tag)),
    ]
    if self.mask is not None:
      fields.append('mask={:s}'.format(repr(self.mask)))
    return 'OneOf({:s})'.format(', '.join(fields))

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False

    return (self._choices == other.choices
            and self._tag == other.tag
            and self._mask == other.mask)


_TuplePath = Tuple[Union[Text, int]]


def _map_oneofs_helper(
    func,
    value,
    path):
  """Walk over `value` and recursively apply `func` to all OneOf nodes."""
  if isinstance(value, OneOf):
    new_choices = []
    for i, choice in enumerate(value.choices):
      new_path = path + ('choices', i)
      new_choices.append(_map_oneofs_helper(func, choice, new_path))
    new_value = OneOf(choices=new_choices, mask=value.mask, tag=value.tag)
    return func(path, new_value)
  elif isinstance(value, (list, tuple)):
    # This case handles lists and tuples. It also handles namedtuples,
    # which subclass 'tuple'.
    new_elements = []
    if hasattr(value, '_fields'):  # value is a namedtuple
      for k, v in zip(value._fields, value):
        new_elements.append(_map_oneofs_helper(func, v, path + (k,)))
      return value.__class__(*new_elements)
    else:
      for k, v in enumerate(value):
        new_elements.append(_map_oneofs_helper(func, v, path + (k,)))
      return value.__class__(new_elements)
  elif isinstance(value, dict):
    # Value could be an ordinary Python dictionary or an OrderedDict. If
    # it's an OrderedDict we ignore the existing order and instead sort
    # elements by key. We do this for compatibility with tf.nest.map_structure.
    new_items = []
    for k in sorted(value):
      new_items.append((k, _map_oneofs_helper(func, value[k], path + (k,))))
    return value.__class__(new_items)
  elif isinstance(value, (int, float, str, bytes)):  # six.string_types
    return value
  else:
    raise ValueError(
        'Unsupported value: {} of type: {}'.format(value, type(value)))


def map_oneofs_with_tuple_paths(
    func,
    structure):
  """Walk through `structure` and apply `func` to each OneOf object.

  We will visit elements of `structure` using a pre-order traversal. If Node X
  is a child of Node Y in `structure` then we will visit X before Y. Nodes are
  visited in a deterministic order.

  Args:
    func: A function that takes as input a pair `(path, oneof)`, where path
        is a tuple of strings and integers indicating the current position
        within the object, and `oneof` is a OneOf object. It should return a
        new value for `oneof`.
    structure: A data structure composed of OneOf objects and collections such
        as dicts, lists, and namedtuples.

  Returns:
    A copy of `structure` with new values substituted for its OneOf nodes.
  """

  prefix = ()  # type: Tuple[Union[int, str]]
  return _map_oneofs_helper(func, structure, prefix)


def map_oneofs_with_paths(
    func,
    structure):
  """Walk through `structure` and apply `func` to each OneOf object.

  We will visit elements of `structure` using a pre-order traversal. If Node X
  is a child of Node Y in `structure` then we will visit X before Y. Nodes are
  visited in a deterministic order.

  Args:
    func: A function that takes as input a pair `(path, oneof)`, where path
        is a slash-separated string indicating the current position within the
        object, and `oneof` is a OneOf object. It should return a new value for
        `oneof`.  structure: A data structure composed of OneOf objects and
        collections such as dicts, lists, and namedtuples.
    structure: A data structure composed of OneOf objects and collections such
        as dicts, lists, and namedtuples.

  Returns:
    A copy of `structure` with new values substituted for its OneOf nodes.
  """
  def visit(path, oneof):
    string_path = '/'.join(map(str, path))
    return func(string_path, oneof)
  return map_oneofs_with_tuple_paths(visit, structure)


def map_oneofs(
    func,
    structure):
  """Walk through `structure` and apply `func` to each OneOf object.

  We will visit elements of `structure` using a pre-order traversal. If Node X
  is a child of Node Y in `structure` then we will visit X before Y. Nodes are
  visited in a deterministic order.

  Args:
    func: A function that takes a OneOf object as input. It should return a new
        value for the OneOf object.
    structure: A data structure composed of OneOf objects and collections such
        as dicts, lists, and namedtuples.

  Returns:
    A copy of `structure` with new values substituted for its OneOf nodes.
  """
  def visit(path, oneof):
    del path
    return func(oneof)
  return map_oneofs_with_tuple_paths(visit, structure)
