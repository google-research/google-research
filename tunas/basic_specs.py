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
"""Operation and filters spec definition.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

from typing import Any, NamedTuple, Sequence

from tunas import schema
from tunas import schema_io

OP_TAG = 'op_indices'
FILTERS_TAG = 'filters_indices'


@schema_io.register_namedtuple(
    'basic_specs.ZeroSpec',
    deprecated_names=['basic_specs.ZeroSPec'])
class ZeroSpec(collections.namedtuple('ZeroSpec', [])):
  """NamedTuple representing a trivial layer whose output is all zeros.

  The layer output has the same shape as the input.
  """
  pass


@schema_io.register_namedtuple('basic_specs.Block')
class Block(collections.namedtuple('Block', ['layers', 'filters'])):
  """Group of layers that share the same input/output filter sizes.

  Attributes:
    layers: A list of layers.
    filters: int or schema.OneOf with integer-valued choices specifying the
        number of filters to use for the layers within the block.
  """
  pass


@schema_io.register_namedtuple('basic_specs.ConvTowerSpec')
class ConvTowerSpec(collections.namedtuple(
    'ConvTowerSpec', ['blocks', 'filters_base'])):
  """Namedtuple representing a convolution tower.

  Attributes:
    blocks: A list of basic_specs.Block objects.
    filters_base: Positive integer. The number of filters in each layer of the
        model must be a multiple of this value.
  """
  pass


@schema_io.register_namedtuple('basic_specs.FilterMultiplier')
class FilterMultiplier(NamedTuple('FilterMultiplier', [('scale', float)])):
  """Namedtuple representing a relative filter size.

  The size must be relative to some other layer of the network; the exact layer
  is context-dependent. For example, we might specify the number of filters in
  the middle of an inverted bottleneck layer relative to the number of input
  filters.

  Attributes:
    scale: Ratio of the number of filters in the specified layer to the number
      of filters in our reference layer.
  """
  pass


def block(layers, filters):
  filters = schema.OneOf([filters], tag=FILTERS_TAG)
  return Block(layers=layers, filters=filters)
