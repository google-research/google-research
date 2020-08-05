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
"""Common utility functions for basic elements in search space.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from typing import Any, Optional, Dict, List, Sequence, Text, Tuple, TypeVar, Union
import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import schema


# List of possible reward functions to use during a search.
RL_REWARD_MNAS = 'mnas'
RL_REWARD_MNAS_HARD = 'mnas_hard'
RL_REWARD_ABS = 'abs'
RL_REWARDS = (
    RL_REWARD_MNAS,
    RL_REWARD_MNAS_HARD,
    RL_REWARD_ABS
)


def normalize_strides(
    strides
):
  """Normalize strides of the same format.

  Args:
    strides: An integer or pair of integers.

  Returns:
    A pair of integers.

  Raises:
    ValueError: Input strides is neither an integer nor a pair of integers.
  """
  if isinstance(strides, (list, tuple)) and len(strides) == 2:
    return tuple(strides)
  elif isinstance(strides, int):
    return (strides, strides)
  else:
    raise ValueError(
        'Strides - {} is neither an integer nor a pair of integers.'.format(
            strides))


def scale_filters(filters, multiplier, base):
  """Scale `filters` by `factor`and round to the nearest multiple of `base`.

  Args:
    filters: Positive integer. The original filter size.
    multiplier: Positive float. The factor by which to scale the filters.
    base: Positive integer. The number of filters will be rounded to a multiple
        of this value.

  Returns:
    Positive integer, the scaled filter size.
  """
  round_half_up = int(filters * multiplier / base + 0.5)
  result = int(round_half_up * base)
  return max(result, base)


def tf_scale_filters(filters,
                     multiplier,
                     base):
  """Similar to `scale_filters`, but with Tensor instead of numeric inputs.

  Args:
    filters: Scalar int32 Tensor. The original filter size.
    multiplier: Scalar float32 Tensor. The factor by which to scale `filters`.
    base: Scalar int32 Tensor. The number of filters will be rounded to a
        multiple of this value.

  Returns:
    Scalar int32 Tensor. The scaled filter size.
  """
  filters = tf.convert_to_tensor(filters, dtype=tf.int32)
  base = tf.convert_to_tensor(base, dtype=filters.dtype)

  multiplier = tf.convert_to_tensor(multiplier, dtype=tf.float32)
  float_filters = tf.cast(filters, multiplier.dtype)
  float_base = tf.cast(base, multiplier.dtype)

  round_half_up = tf.cast(
      float_filters * multiplier / float_base + 0.5, tf.int32)
  round_half_up_float = tf.cast(round_half_up, multiplier.dtype)
  result = tf.cast(round_half_up_float * float_base, filters.dtype)
  return tf.math.maximum(result, base)


def make_divisible(v, divisor):
  """Alternate filter scaling, compatible with the one used by MobileNet V3."""
  new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def tf_make_divisible(v,
                      divisor):
  """Analogue of make_divisible() that operates on tf.Tensor objects."""
  v = tf.convert_to_tensor(v, preferred_dtype=tf.int32)
  divisor = tf.convert_to_tensor(divisor, dtype=v.dtype)

  new_v = tf.cast(v, tf.float32) + tf.cast(divisor, tf.float32)/2
  new_v = tf.cast(new_v, v.dtype) // divisor * divisor
  new_v = tf.maximum(new_v, divisor)

  # Condition is equivalent to (new_v * 0.9*v), but works with integer inputs.
  new_v = tf.where_v2(10*new_v < 9*v, new_v + divisor, new_v)
  return new_v


_T = TypeVar('_T')


def _validate_genotype_dict(model_spec,
                            genotype):
  """Verify that the tag counts in `genotype` match those in `model_spec`."""
  # Count the number of times each tag appears in ConvTowerSpec.
  tag_counts = collections.Counter()
  def update_tag_counts(oneof):
    tag_counts[oneof.tag] += 1
  schema.map_oneofs(update_tag_counts, model_spec)

  # Report any size mismatches we come across.
  bad_tags = set(genotype) - set(tag_counts)
  if bad_tags:
    raise ValueError(
        'Tag(s) appear in genotype but not in model_spec: {:s}'
        .format(', '.join(bad_tags)))

  for tag in genotype:
    if len(genotype[tag]) != tag_counts[tag]:
      raise ValueError(
          'Tag {:s} appears {:d} times in genotype but {:d} times in '
          'model_spec'.format(tag, len(genotype[tag]), tag_counts[tag]))


def _validate_genotype_sequence(model_spec,
                                genotype):
  """Verify that the number of OneOfs in `genotype` matches `model_spec`."""
  # Note: Conceptually, we just need oneof_count to be an integer. But we need
  # to be able to update its value from within the update_count() function, and
  # storing it inside a dictionary makes that easier.
  oneof_count = {'value': 0}
  def update_count(oneof):
    del oneof  # Unused
    oneof_count['value'] += 1
  schema.map_oneofs(update_count, model_spec)

  if len(genotype) != oneof_count['value']:
    raise ValueError(
        'Genotype contains {:d} oneofs but model_spec contains {:d}'
        .format(len(genotype), oneof_count['value']))


def prune_model_spec(model_spec,
                     genotype,
                     path_dropout_rate = 0.0,
                     training = None,
                     prune_filters_by_value = False):
  """Creates a representation for an architecture with constant ops.

  Args:
    model_spec: Nested data structure containing schema.OneOf objects.
    genotype: A dictionary mapping tags to sequences of integers. Or a sequence
        of integers containing the selections for all the OneOf nodes in
        model_spec.
    path_dropout_rate: Float or scalar float Tensor between 0 and 1. If greater
        than zero, we will randomly zero out skippable operations during
        training with this probability. Cannot be used with an rl controller.
        Should be set to 0 at evaluation time.
    training: Boolean. True during training, false during evaluation/inference.
        Can be None if path_dropout_rate is zero.
    prune_filters_by_value: Boolean. If true, treat genotype[FILTERS_TAG] as a
        list of values rather than a list of indices.

  Returns:
    A pruned version of `model_spec` with all unused options removed.
  """
  if path_dropout_rate != 0.0:
    if basic_specs.OP_TAG not in genotype:
      raise ValueError(
          'If path_dropout_rate > 0 then genotype must contain key {:s}.'
          .format(basic_specs.OP_TAG))
    if training is None:
      raise ValueError(
          'If path_dropout_rate > 0 then training cannot be None.')

  # Create a mutable copy of 'genotype'. This will let us modify the copy
  # without updating the original.
  genotype_is_dict = isinstance(genotype, dict)
  if genotype_is_dict:
    genotype = {key: list(value) for (key, value) in genotype.items()}
    _validate_genotype_dict(model_spec, genotype)
  else:  # genotype is a list/tuple of integers
    genotype = list(genotype)
    _validate_genotype_sequence(model_spec, genotype)

  # Everything looks good. Now prune the model.
  zero_spec = basic_specs.ZeroSpec()
  def update_spec(oneof):
    """Visit a schema.OneOf node in `model_spec`, return an updated value."""
    if genotype_is_dict and oneof.tag not in genotype:
      return oneof

    if genotype_is_dict:
      selection = genotype[oneof.tag].pop(0)
      if oneof.tag == basic_specs.FILTERS_TAG and prune_filters_by_value:
        selection = oneof.choices.index(selection)
    else:
      selection = genotype.pop(0)

    # If an operation is skippable (i.e., it can be replaced with a ZeroSpec)
    # then we optionally apply path dropout during stand-alone training.
    # This logic, if enabled, will replace a standard RL controller.
    mask = None
    if (path_dropout_rate != 0.0
        and training
        and oneof.tag == basic_specs.OP_TAG
        and zero_spec in oneof.choices):
      keep_prob = 1.0 - path_dropout_rate
      # Mask is [1] with probability `keep_prob`, and [0] otherwise.
      mask = tf.cast(tf.less(tf.random_uniform([1]), keep_prob), tf.float32)
      # Normalize the mask so that the expected value of each element 1.
      mask = mask / keep_prob

    return schema.OneOf([oneof.choices[selection]], oneof.tag, mask)

  return schema.map_oneofs(update_spec, model_spec)


def scale_conv_tower_spec(
    model_spec,
    multipliers,
    base = None):
  """Scale all the filters in `model_spec`, rounding to multiples of `base`.

  Args:
    model_spec: A ConvTowerSpec namedtuple.
    multipliers: float or list/tuple of floats, the possible filter multipliers.
    base: Positive integer, all filter sizes must be a multiple of this value.

  Returns:
    A new basic_specs.ConvTowerSpec.
  """
  if base is None:
    base = model_spec.filters_base

  if isinstance(multipliers, (int, float)):
    multipliers = (multipliers,)

  def update(oneof):
    """Compute version of `oneof` whose filters have been scaled up/down."""
    if oneof.tag != basic_specs.FILTERS_TAG:
      return oneof

    all_filters = set()
    for filters in oneof.choices:
      if isinstance(filters, basic_specs.FilterMultiplier):
        # Skip scaling because the filter sizes are relative, not absolute.
        all_filters.add(filters)
      else:
        for mult in multipliers:
          all_filters.add(scale_filters(filters, mult, base))

    return schema.OneOf(sorted(all_filters), basic_specs.FILTERS_TAG)

  result = schema.map_oneofs(update, model_spec)
  return basic_specs.ConvTowerSpec(result.blocks, base)


def tf_argmax_or_zero(oneof):
  """Returns zero or the index with the largest value across axes of mask.

  Args:
    oneof: A schema.OneOf objective.

  Returns:
    A scalar int32 tensor.
  """
  if oneof.mask is None:
    if len(oneof.choices) != 1:
      raise ValueError(
          'Expect pruned structure with one choice when mask is None. '
          'Got {} number of choices in structure.'.format(len(oneof.choices)))
    return tf.constant(0, tf.int32)
  else:
    return tf.argmax(oneof.mask, output_type=tf.int32)


def tf_indices(model_spec):
  """Extract `indices` from `model_spec` as Tensors.

  Args:
    model_spec: Nested data structure containing schema.OneOf objects.

  Returns:
    `indices`, a rank-1 integer Tensor.
  """
  indices = []

  def visit(oneof):
    index = tf_argmax_or_zero(oneof)
    indices.append(index)

  schema.map_oneofs(visit, model_spec)
  return tf.stack(indices)


def parse_list(string, convert_fn):
  """Parse a (possibly empty) colon-separated list of values."""
  string = string.strip()
  if string:
    return [convert_fn(piece) for piece in string.split(':')]
  else:
    return []


def reward_for_single_cost_model(
    quality,
    rl_reward_function,
    estimated_cost,
    rl_cost_model_target,
    rl_cost_model_exponent):
  """Compute reward based on quality and cost of a single cost model.

  Args:
    quality: quality of the model. For example, validation accuracy.
    rl_reward_function: name of the reward function.
    estimated_cost: estimated cost value.
    rl_cost_model_target: the target value for cost.
    rl_cost_model_exponent: a hyperparameter to balance cost and accuracy
        in the reward function.

  Returns:
    A dictionary containing the following keys:
      rl_reward: reward value.
      rl_cost_ratio: a ratio between estimated cost and cost target.
      rl_cost_adjustment: how much reward has been adjusted by cost.
  """
  rl_cost_ratio = estimated_cost / rl_cost_model_target
  if rl_reward_function == RL_REWARD_MNAS:
    # reward = accuracy * (T/T0)^beta
    rl_cost_adjustment = tf.pow(rl_cost_ratio, rl_cost_model_exponent)
    rl_reward = quality * rl_cost_adjustment
  elif rl_reward_function == RL_REWARD_MNAS_HARD:
    # reward = accuracy * min((T/T0)^beta, 1)
    rl_cost_adjustment = tf.pow(rl_cost_ratio, rl_cost_model_exponent)
    rl_cost_adjustment = tf.minimum(rl_cost_adjustment, 1.)
    rl_reward = quality * rl_cost_adjustment
  elif rl_reward_function == RL_REWARD_ABS:
    # reward = accuracy + beta * abs(T/T0 - 1)
    rl_cost_adjustment = rl_cost_model_exponent * tf.abs(rl_cost_ratio - 1)
    rl_reward = quality + rl_cost_adjustment
  else:
    raise ValueError('Unsupported rl_reward_function: {}'.format(
        rl_reward_function))

  return {
      'rl_reward': rl_reward,
      'rl_cost_ratio': rl_cost_ratio,
      'rl_cost_adjustment': rl_cost_adjustment
  }
