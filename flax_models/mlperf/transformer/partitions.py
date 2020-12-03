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
"""Utilities for constructing PyTrees of PartitionSpecs.
"""

import re
from flax import traverse_util
import jax


# Sentinels
_unmatched = object()

# Partition spec
spec = jax.interpreters.sharded_jit.PartitionSpec
# For specifying empty leaf dict `{}`
empty_dict = object()


def match(qs, ks):
  """Return True if regexes in qs match any window of strings in tuple ks."""
  # compile regexes and force complete match
  qts = tuple(map(lambda x: re.compile(x + '$'), qs))
  for i in range(len(ks)-len(qs)+1):
    matches = [x.match(y) for x, y in zip(qts, ks[i:])]
    if matches and all(matches):
      return True
  return False


lmatch = lambda q, k: match(q, k[:len(q)])
rmatch = lambda q, k: match(q, k[-len(q):])


def mark_empty(x):
  if isinstance(x, dict):
    return {k: mark_empty(v) for k, v in x.items()} if x else empty_dict
  else:
    return x


def unmark_empty(x):
  return jax.tree_map(lambda z: {} if z == empty_dict else z, x)


flatten_dict = lambda x: traverse_util.flatten_dict(mark_empty(x))
unflatten_dict = lambda x: unmark_empty(traverse_util.unflatten_dict(x))


def replacement_rules(rules):
  def replace(key, val):
    for rule, replacement in rules:
      if match(rule, key):
        return replacement
    return val
  return replace


def set_partitions(rules, in_dict):
  replace = replacement_rules(rules)
  initd = {k: _unmatched for k in flatten_dict(in_dict)}
  result = {k: replace(k, v) for k, v in initd.items()}
  assert _unmatched not in result.values(), 'Incomplete partition spec.'
  return unflatten_dict(result)
