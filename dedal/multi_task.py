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

"""Generic Multi task architecture."""

import copy
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TypeVar, Generic

import gin


T = TypeVar('T')


class NamedLists(dict, Generic[T]):
  """A generic architecture for multi tasks with potentially several levels."""

  def __init__(self, layers):
    layers = {k: list(v) for k, v in layers.items()}
    super().__init__(layers)

  def __getattr__(self, attr):
    return self[attr]

  @property
  def levels(self):
    return list(self.values())

  @property
  def size(self):
    return sum(len(x) for x in self.values())

  def constant_copy(self, value):
    """Returns a copy of the structure with only the same value everywhere."""
    return NamedLists(
        layers={k: [value for _ in v] for k, v in self.items()})

  def copy(self):
    """Returns a copy of the NamedLists."""
    return NamedLists(copy.deepcopy(super().copy()))

  def pack(self, values, default_value=None):
    """Packs the values in a NamedLists with the same structure as self."""
    result = self.constant_copy(default_value)
    it = result.__iter__()
    for val in values:
      next(it)
      it._level[it._idx] = val  # pylint: disable=protected-access
    return result

  def flatten(self, empty_value=None):
    result = {}
    for name, values in self.items():
      for i, value in enumerate(values):
        result[f'{name}/{i}'] = value
      if not values:  # special case for empty list to keep the structure.
        result[name + '/'] = empty_value
    return result

  @staticmethod
  def unflatten(values):
    """Unflatten a dict of values that have been previously flattened."""
    result = dict()
    for name, value in values.items():
      idx = name.rfind('/')
      key = name[:idx]
      if key not in result:
        result[key] = []
      if idx != len(name) - 1:
        result[key].append(value)
    return NamedLists(result)

  class _Iterator:
    """Iterator on NamedLists."""

    def __init__(self, container):
      self._level_iter = iter(container.values())
      self._level = None
      self._idx = -1

    def __next__(self):
      self._idx += 1
      if self._level is None or self._idx >= len(self._level):
        self._level = next(self._level_iter)  # Might raise StopIteration here.
        self._idx = -1
        return self.__next__()
      return self._level[self._idx]

  def __iter__(self):
    return NamedLists._Iterator(self)

  @property
  def shape(self):
    return tuple(len(level) for level in self.levels)


@gin.configurable
class Backbone(NamedLists, Generic[T]):
  """A specific case of NamedList that is used in sequence alignments."""

  def __init__(self,
               embeddings = (),
               alignments = ()):
    super().__init__(layers=dict(embeddings=embeddings, alignments=alignments))

  @classmethod
  def constant_from_shape(cls, value, shape):
    return cls(
        embeddings=[value for _ in range(shape[0])],
        alignments=[value for _ in range(shape[1])])


@gin.configurable
class SwitchNamedLists(NamedLists[int]):
  """Provides methods to merge N compatible `NamedLists`.

  A `SwitchNamedLists` instance is a `NamedLists[int]` with values in [0, N)
  whose structure matches that of the desired merged `NamedLists` and elements
  indicate from which of the N input `NamedLists` the corresponding output value
  should be taken. That is,
    `output.key[l] = inputs[self.key[l]].key[l]`,
  where `inputs` is a sequence of N `NamedLists`.

  The N input `NamedLists` are assumed to be compatible in the sense that they
  have the same keys and the total number of elements they contain equals the
  number of elements in the `SwitchSeqAlign` instance. That is,
    `self.size == sum(inputs_i.size for inputs_i in inputs)`
  must hold true.
  """

  @property
  def n(self):
    """Returns the number of `NamedLists` being "switched over"."""
    return max(max(l) for l in self.values()) + 1  # Assumes elems in [0, n).

  def filter(self, inputs, i):
    """Removes elements from `NamedLists` not belonging to i-th input.

    Primarily used to remove "dummy" values e.g. from model output.

    Args:
      inputs: a `NamedLists` with structure identical to `self`.
      i: an int between 0 and N-1, both inclusive, where N is the number of
        `NamedLists` to be merged.

    Returns:
      A `NamedLists` defined as
        `output.key = [v for v, j in zip(inputs.key, self.key) if j == i]`.
      That is, for each key, only those elements in the list for which `self`
      takes value `i` at the matching position will be kept.
    """
    flags = self.get_selector(i)
    layers = {}
    for k in self.keys():
      layers[k] = [v for v, flag in zip(inputs[k], flags[k]) if flag]
    return NamedLists(layers)

  def merge(self, inputs):
    """Merges a sequence of N compatible `NamedLists`.

    Args:
      inputs: a sequence of N `NamedLists` with the same keys as `self`
        satisfying `self.size == sum(inputs_i.size for inputs_i in inputs)`.

    Returns:
      a `NamedLists` instance such that
        `output.key[l] = inputs[self.key[l]].key[l]`
      for each key in `self`.
    """
    inputs = [list(inputs_i) for inputs_i in inputs]
    offsets = len(inputs) * [0]
    outputs = []
    for i in list(self):  # Needed to appease AutoGraph?
      outputs.append(inputs[i][offsets[i]])
      offsets[i] += 1
    return self.pack(outputs)

  def merge_flattened(
      self, inputs):
    """Merges a sequence of N compatible, flattened `NamedLists`.

    Args:
      inputs: a sequence of N `Mapping[str, T]` corresponding to N `NamedLists`
        that have been flattened. These must have the same keys as `self` and
        satisfy
          `self.size == sum(unflatten(inputs_i).size for inputs_i in inputs)`.

    Returns:
      a `NamedLists` instance such that
        `output.key[l] = inputs[self.key[l]].key[l]`
      for each key in `self`, flattened to `Mapping[str, T]`.
    """
    return self.merge([Backbone.unflatten(m_i) for m_i in inputs]).flatten()

  def get_selector(self, i):
    """Returns `NamedLists` of bools flagging elements from i-th input.

    Args:
      i: an int between 0 and N - 1, both inclusive, where N is the number of
        `NamedLists` to be merged.

    Returns:
      a `NamedLists[bool]` such that `output.key[l] = self.key[l] == i`.
    """
    return self.pack([j == i for j in self])


@gin.configurable
class SwitchBackbone(SwitchNamedLists):
  """A specific case of SwitchNamedLists that is used in sequence alignments."""

  def __init__(self,
               embeddings = (),
               alignments = ()):
    super().__init__(layers=dict(embeddings=embeddings, alignments=alignments))

  @classmethod
  def constant_like(cls, container, value = 0):
    return cls(
        embeddings=[value for _ in container.embeddings],
        alignments=[value for _ in container.alignments])
