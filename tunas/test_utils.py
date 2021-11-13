# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Utility functions for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from typing import TypeVar, Union

import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import schema


def random_one_hot(size):
  index = tf.random_uniform((), 0, size, tf.int32)
  return tf.one_hot(index, size)


_T = TypeVar('_T')


def with_random_op_masks(model_spec):
  """Assign random one-hot masks OneOf whose tags are basic_specs.OP_TAG."""
  def update(oneof):
    if oneof.tag == basic_specs.OP_TAG:
      mask = random_one_hot(len(oneof.choices))
      return schema.OneOf(choices=oneof.choices, tag=oneof.tag, mask=mask)
    else:
      return oneof
  return schema.map_oneofs(update, model_spec)


def with_random_masks(model_spec):
  """Assign random one-hot masks OneOf."""
  def update(oneof):
    mask = random_one_hot(len(oneof.choices))
    return schema.OneOf(choices=oneof.choices, tag=oneof.tag, mask=mask)
  return schema.map_oneofs(update, model_spec)


def with_random_pruning(model_spec):
  """Pick a random value for each OneOf and prune away the remaining choices."""
  def update(oneof):
    index = random.randrange(len(oneof.choices))
    return schema.OneOf(choices=[oneof.choices[index]], tag=oneof.tag)
  return schema.map_oneofs(update, model_spec)


def _get_shape_to_name_map(name_shape_pairs):
  result = dict()
  for name, shape in name_shape_pairs:
    key = tuple(shape)
    result.setdefault(key, []).append(name)
  return result


class ModelTest(tf.test.TestCase):
  """"Test class with common utility functions for model or model builder."""

  def assert_same_shapes(self, lhs, rhs):
    """Assert that each shape appears the same number of times in lhs as in rhs.

    Args:
      lhs: A list of (name, shape) pairs, where each name is a string and
          each shape is a list or tuple of integers.
      rhs: A list of (name, shape) pairs, where each name is a string and
          each shape is a list or tuple of integers.
    """
    lhs_shape_map = _get_shape_to_name_map(lhs)
    rhs_shape_map = _get_shape_to_name_map(rhs)
    errors = []
    all_shapes = set()
    all_shapes.update(lhs_shape_map.keys())
    all_shapes.update(rhs_shape_map.keys())
    for shape in all_shapes:
      lhs_names = lhs_shape_map.get(shape, [])
      rhs_names = rhs_shape_map.get(shape, [])
      lhs_count = len(lhs_names)
      rhs_count = len(rhs_names)
      if lhs_count != rhs_count:
        errors.append(
            'shape {} appears {:d} times in lhs but {:d} times in rhs, '
            'with lhs names: {}, rhs names: {}'
            .format(shape, lhs_count, rhs_count, lhs_names, rhs_names))

    if errors:
      self.fail('Found shape mismatches:\n{:s}'.format('\n'.join(errors)))
