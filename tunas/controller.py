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
"""Utilities for sampling network architectures from a search space.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
from typing import Any, Optional, Text, Tuple, Union

import numpy as np
import six
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tunas import basic_specs
from tunas import schema


def _replace_sample_with_probability(
    value,
    log_prob,
    replace_prob,
    replace_value):
  """Replace `value` with `replace_value` with probability `replace_prob`."""
  if replace_prob is None:
    return (value, log_prob)

  replace_prob = tf.convert_to_tensor(replace_prob, tf.float32)
  should_replace = tf.less(tf.random_uniform(()), replace_prob)
  new_value = tf.where_v2(should_replace, replace_value, value)
  new_log_prob = tf.where_v2(
      should_replace,
      tf.log(replace_prob),
      tf.log(1-replace_prob) + log_prob)
  return (new_value, new_log_prob)


def independent_sample(
    structure,
    increase_ops_probability = None,
    increase_filters_probability = None,
    hierarchical = True,
    name = None,
    temperature = 1.0):
  """Generate a search space specification for an RL controller model.

  Each OneOf value is sampled independently of every other; hence the name.

  Args:
    structure: Nested data structure containing OneOf objects to search over.
    increase_ops_probability: Scalar float Tensor or None. If not None, we will
        randomly enable all possible operations instead of just the selected
        operations with this probability.
    increase_filters_probability: Scalar float Tensor or None. If not None, we
        will randomly use the largest possible filter sizes with this
        probability.
    hierarchical: Boolean. If true, the values of the outputs `sample_log_prob`
        and `entropy` will only take into account subgraphs that are enabled at
        the current training step.
    name: Optional name for the newly created TensorFlow scope.
    temperature: Positive scalar controlling the temperature to use when
        sampling from the RL controller.

  Returns:
    A tuple (new_structure, dist_info) where `new_structure` is a copy
    of `structure` annotated with mask tensors, and `dist_info` is a
    dictionary containing information about the sampling distribution
    which contains the following keys:
      - entropy: Scalar float Tensor, entropy of the current probability
            distribution.
      - logits_by_path: OrderedDict of rank-1 Tensors, sample-independent logits
            for each OneOf in `structure`. Names are derived from OneOf paths.
      - logits_by_tag: OrderedDict of rank-1 Tensors, sample-independent logits
            for each OneOf in `structure`. Names are derived from OneOf tags.
      - sample_log_prob: Scalar float Tensor, log-probability of the current
            sample associated with `new_structure`.
  """
  with tf.variable_scope(name, 'independent_sample'):
    temperature = tf.convert_to_tensor(temperature, tf.float32)
    dist_info = {
        'entropy': tf.constant(0, tf.float32),
        'logits_by_path': collections.OrderedDict(),
        'logits_by_tag': collections.OrderedDict(),
        'sample_log_prob': tf.constant(0, tf.float32),
    }
    tag_counters = collections.Counter()

    entropies = dict()
    log_probs = dict()
    is_active = dict()
    def visit(tuple_path, oneof):
      """Visit a OneOf node in `structure`."""
      string_path = '/'.join(map(str, tuple_path))
      num_choices = len(oneof.choices)

      logits = tf.get_variable(
          name='logits/' + string_path,
          initializer=tf.initializers.zeros(),
          shape=[num_choices],
          dtype=tf.float32)
      logits = logits / temperature

      tag_name = '{:s}_{:d}'.format(oneof.tag, tag_counters[oneof.tag])
      tag_counters[oneof.tag] += 1

      dist_info['logits_by_path'][string_path] = logits
      dist_info['logits_by_tag'][tag_name] = logits

      dist = tfp.distributions.OneHotCategorical(
          logits=logits, dtype=tf.float32)
      entropies[tuple_path] = dist.entropy()

      sample_mask = dist.sample()
      sample_log_prob = dist.log_prob(sample_mask)
      if oneof.tag == basic_specs.OP_TAG:
        sample_mask, sample_log_prob = _replace_sample_with_probability(
            sample_mask, sample_log_prob, increase_ops_probability,
            tf.constant([1.0/num_choices]*num_choices, tf.float32))
      elif oneof.tag == basic_specs.FILTERS_TAG:
        # NOTE: While np.argmax() was originally designed to work with integer
        # filter sizes, it will also work with any object type that supports
        # "less than" and "greater than" operations.
        sample_mask, sample_log_prob = _replace_sample_with_probability(
            sample_mask, sample_log_prob, increase_filters_probability,
            tf.one_hot(np.argmax(oneof.choices), len(oneof.choices)))

      log_probs[tuple_path] = sample_log_prob
      for i in range(len(oneof.choices)):
        tuple_subpath = tuple_path + ('choices', i)
        is_active[tuple_subpath] = tf.greater(tf.abs(sample_mask[i]), 1e-6)

      return schema.OneOf(choices=oneof.choices,
                          tag=oneof.tag,
                          mask=sample_mask)

    new_structure = schema.map_oneofs_with_tuple_paths(visit, structure)

    assert six.viewkeys(entropies) == six.viewkeys(log_probs)
    for path in entropies:
      path_is_active = tf.constant(True)
      if hierarchical:
        for i in range(len(path) + 1):
          if path[:i] in is_active:
            path_is_active = tf.logical_and(path_is_active, is_active[path[:i]])

      path_is_active = tf.cast(path_is_active, tf.float32)
      dist_info['entropy'] += entropies[path] * path_is_active
      dist_info['sample_log_prob'] += log_probs[path] * path_is_active

    return (new_structure, dist_info)
