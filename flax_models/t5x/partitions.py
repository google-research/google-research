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

# Lint as: python3
"""Utilities for constructing PyTrees of PartitionSpecs."""

import re
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax

# Sentinels
_unmatched = object()

# Partition spec
Spec = jax.interpreters.sharded_jit.PartitionSpec
# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
  """Return True if regexes in qs match any window of strings in tuple ks."""
  # compile regexes and force complete match
  qts = tuple(map(lambda x: re.compile(x + '$'), qs))
  for i in range(len(ks) - len(qs) + 1):
    matches = [x.match(y) for x, y in zip(qts, ks[i:])]
    if matches and all(matches):
      return True
  return False


def _replacement_rules(rules):

  def replace(key, val):
    for rule, replacement in rules:
      if _match(rule, key):
        return replacement
    return val

  return replace


def _get_partition_rules(num_partitions):
  # TODO(b/112340395, b/124017683): pytype fails on __new__ for classes that
  # inherit from `tuple`, such as sharded_jit.PartitionSpec (Spec) used below.
  # pytype: disable=wrong-arg-count
  return [
      (('state', 'step'), None),
      (('state', 'param_states'), None),  # Don't shard the Adafactor state.
      (('encoder_relative_posemb',), None),
      (('decoder_relative_posemb',), None),
      (('shared_embedding',), Spec(num_partitions, 1)),
      ((r'LayerNorm_\d+', '(bias|scale)'), None),
      ((r'encoder(decoder)?_norm', '(bias|scale)'), None),
      ((r'SelfAttention_\d+', '(query|key|value)', 'kernel'),
       Spec(1, num_partitions)),
      ((r'SelfAttention_\d+', 'out', 'kernel'), Spec(num_partitions, 1)),
      ((r'MultiHeadDotProductAttention_\d+', '(query|key|value)', 'kernel'),
       Spec(1, num_partitions)),
      ((r'MultiHeadDotProductAttention_\d+', 'out', 'kernel'),
       Spec(num_partitions, 1)),
      ((r'MlpBlock_\d+', r'DenseGeneral_\d+', 'bias'), None),
      ((r'MlpBlock_\d+', r'wi(_\d+)?', 'kernel'), Spec(1, num_partitions)),
      ((r'MlpBlock_\d+', 'wo', 'kernel'), Spec(num_partitions, 1)),
  ]
  # pytype: enable=wrong-arg-count


def set_partitions(num_partitions, in_dict):
  rules = _get_partition_rules(num_partitions)
  replace = _replacement_rules(rules)
  initd = {k: _unmatched for k in flatten_dict(in_dict)}
  result = {k: replace(k, v) for k, v in initd.items()}
  assert _unmatched not in result.values(), 'Incomplete partition spec.'
  return unflatten_dict(result)
