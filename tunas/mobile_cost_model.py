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
"""Utilities for cost model features for mobile image models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Iterable, Optional, Sequence, Text, TypeVar

import six
import tensorflow.compat.v1 as tf

from tunas import cost_model_lib
from tunas import mobile_search_space_v3
from tunas import schema


def _kron_if_not_none(lhs, rhs):
  if lhs is None:
    return rhs
  elif rhs is None:
    return lhs
  else:
    return cost_model_lib.kron(lhs, rhs)


def _maybe_get_mask(value):
  if isinstance(value, schema.OneOf):
    return cost_model_lib.get_mask(value)
  else:
    return None


def _reduce_kron(values):
  result = None
  for value in values:
    result = _kron_if_not_none(result, value)
  return result


def coupled_tf_features(model_spec):
  """Cost model features for mobile_search_space_v3."""
  features = []
  assert len(model_spec.blocks) >= 1, model_spec

  input_filters_feature = None
  for block in model_spec.blocks:
    assert block.layers
    output_filters_feature = _maybe_get_mask(block.filters)

    for choice_spec in block.layers:
      features.append(
          _reduce_kron([
              output_filters_feature,
              input_filters_feature,
              _get_layer_features(choice_spec)]))
      input_filters_feature = output_filters_feature

  # Eliminate features where there's only one possible choice.
  pruned_features = []
  for feature in features:
    if feature is not None and int(feature.shape[0]) > 1:
      pruned_features.append(feature)

  if pruned_features:
    return tf.concat(pruned_features, axis=0)
  else:
    return tf.ones(shape=[1], dtype=tf.float32)


def _get_layer_features(layer_spec):
  """Get a 1D feature tensor for a layer in the V3 search space."""
  if isinstance(layer_spec, (float, int) + six.string_types):
    return tf.ones(shape=[1], dtype=tf.float32)
  elif isinstance(layer_spec, (list, tuple)):  # can also be a namedtuple
    result = _reduce_kron(map(_get_layer_features, layer_spec))
    return tf.ones(shape=[1], dtype=tf.float32) if result is None else result
  elif isinstance(layer_spec, dict):
    values = [layer_spec[k] for k in sorted(layer_spec)]
    result = _reduce_kron(map(_get_layer_features, values))
    return tf.ones(shape=[1], dtype=tf.float32) if result is None else result
  if isinstance(layer_spec, schema.OneOf):
    result = []
    for i, choice in enumerate(layer_spec.choices):
      choice_features = _get_layer_features(choice)
      result.append(layer_spec.mask[i] * choice_features)
    return tf.concat(result, axis=0)
  else:
    raise ValueError('Unsupported object type: {}'.format(layer_spec))


_T = TypeVar('_T')
_U = TypeVar('_U')


def _assert_correct_oneof_count(indices,
                                model_spec):
  """Ensure the length of indices matches the number of OneOfs in model_spec."""

  # We use an object with static member fields to maintain internal state so
  # that the elements inside can be updated within a nested function.
  class State(object):
    count = 0  # Total number of oneofs in 'model_spec'

  # Count the number of elements in model_spec.
  def update_count(oneof):
    del oneof  # Unused
    State.count += 1

  schema.map_oneofs(update_count, model_spec)
  if State.count != len(indices):
    raise ValueError('Wrong number of indices. Expected: {} but got: {}'.format(
        State.count, len(indices)))


def _with_constant_masks(indices, model_spec):
  """Assign constant one-hot masks to the OneOf nodes in model_spec."""
  _assert_correct_oneof_count(indices, model_spec)

  # We use an object with static member fields to maintain internal state so
  # that the elements inside can be updated within a nested function.
  class State(object):
    position = 0  # Current position within 'indices'

  def update_mask(path, oneof):
    """Add a one-hot mask to 'oneof' whose value is derived from 'indices'."""
    index = indices[State.position]
    State.position += 1

    if index < 0 or index >= len(oneof.choices):
      raise ValueError(
          'Invalid index: {:d} for path: {:s} with {:d} choices'.format(
              index, path, len(oneof.choices)))

    mask = tf.one_hot(index, len(oneof.choices))
    return schema.OneOf(oneof.choices, oneof.tag, mask)

  return schema.map_oneofs_with_paths(update_mask, model_spec)


def estimate_cost(indices, ssd):
  """Estimate the cost of a given architecture based on its indices.

  Args:
    indices: List of integers encoding an architecture in the search space.
    ssd: The name of the search space definition to use for the cost model.

  Returns:
    The estimated cost for the specified network architecture.
  """
  with tf.Graph().as_default():
    model_spec = mobile_search_space_v3.get_search_space_spec(ssd)
    model_spec = _with_constant_masks(indices, model_spec)
    features = coupled_tf_features(model_spec)
    cost = cost_model_lib.estimate_cost(features, ssd)

    with tf.Session() as sess:
      return float(sess.run(cost))
