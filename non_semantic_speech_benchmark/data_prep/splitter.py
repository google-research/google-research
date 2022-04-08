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

"""Functions for splitting samples into train/dev/test."""

import collections
import json
from absl import logging
import numpy as np
import tensorflow as tf


CANONICAL_SPLIT_NUM = 666


def _compute_split_boundaries(split_probs, n_items):
  """Computes boundary indices for each of the splits in split_probs.

  Args:
    split_probs: List of (split_name, prob),
                 e.g. [('train', 0.6), ('dev', 0.2), ('test', 0.2)]
    n_items: Number of items we want to split.

  Returns:
    The item indices of boundaries between different splits. For the above
    example and n_items=100, these will be
    [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)].
  """
  if len(split_probs) > n_items:
    raise ValueError('Not enough items for the splits. There are {splits} '
                     'splits while there are only {items} items'.format(
                         splits=len(split_probs), items=n_items))
  total_probs = sum(p for name, p in split_probs)
  if abs(1 - total_probs) > 1E-8:
    raise ValueError('Probs should sum up to 1. probs={}'.format(split_probs))
  split_boundaries = []
  sum_p = 0.0
  for name, p in split_probs:
    prev = sum_p
    sum_p += p
    split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

  # Guard against rounding errors.
  split_boundaries[-1] = (
      split_boundaries[-1][0], split_boundaries[-1][1], n_items)

  return split_boundaries


def get_splits_by_group(items_and_groups, split_probs, split_number,
                        split_method='inter',
                        fail_on_intra_split_too_small=False):
  """Split items to train/dev/test, based on the item groups.

  If the split_method is `inter` then all items from the same group go to
  the same split, and if the split_method is `intra`, then every group
  gets a separate train/dev/test split.

  Args:
    items_and_groups: Sequence of (item_id, group_id) pairs.
    split_probs: List of (split_name, prob), e.g.
      [('train', 0.6), ('dev', 0.2), ('test', 0.2)]
    split_number: Generated splits should change with split_number.
    split_method: String indecating whether this is an `inter` or
      `intra` split.
    fail_on_intra_split_too_small: If `True`, fail if intra speaker split is too
      small.

  Returns:
    Dictionary. Either in the form of {split name -> set(ids)} for inter speaker
      tasks, or {group -> {split name -> set(ids)}} for intra speaker tasks.
  """
  if split_method == 'inter':
    return _get_inter_splits_by_group(
        items_and_groups, split_probs, split_number)
  elif split_method == 'intra':
    return _get_intra_splits_by_group(
        items_and_groups, split_probs, split_number,
        fail_on_intra_split_too_small)
  else:
    raise ValueError('Got an unknown split_method %s' % split_method)


def _get_intra_splits_by_group(items_and_groups, split_probs, split_number,
                               fail_on_intra_split_too_small):
  """For each group, split items from that group to train/dev/test.

  Args:
    items_and_groups: Sequence of (item_id, group_id) pairs.
    split_probs: List of (split_name, prob), e.g.
      [('train', 0.6), ('dev', 0.2), ('test', 0.2)].
    split_number: Generated splits should change with split_number.
    fail_on_intra_split_too_small: If `True`, fail if intra speaker split is too
      small.

  Returns:
    Dictionary that looks like {group -> {split name -> set(ids)}}.
  """
  rng = np.random.RandomState(split_number)
  split_to_ids = collections.defaultdict(dict)
  groups_to_items = collections.defaultdict(list)
  for i, g in items_and_groups:
    groups_to_items[g].append(i)

  for group, items in groups_to_items.items():
    if len(items) < len(split_probs):
      if fail_on_intra_split_too_small:
        raise ValueError(f'Failed on {group} with {len(items)} items. '
                         f'{split_probs}')
      logging.info('Skipping speaker %s since they only have %d items',
                   group, len(items))
      continue
    split_to_ids[group] = collections.defaultdict(set)
    rng.shuffle(items)
    split_boundaries = _compute_split_boundaries(split_probs, len(items))
    for split_name, i_start, i_end in split_boundaries:
      for i in range(i_start, i_end):
        split_to_ids[group][split_name].add(items[i])

  return split_to_ids


def _get_inter_splits_by_group(items_and_groups, split_probs, split_number):
  """Split items to train/dev/test, so all items in group go into same split.

  Args:
    items_and_groups: Sequence of (item_id, group_id) pairs.
    split_probs: List of (split_name, prob),
                 e.g. [('train', 0.6), ('dev', 0.2), ('test', 0.2)]
    split_number: Generated splits should change with split_number.

  Returns:
    Dictionary that looks like {split name -> set(ids)}.
  """
  groups = sorted(set(group_id for item_id, group_id in items_and_groups))
  rng = np.random.RandomState(split_number)
  rng.shuffle(groups)

  split_boundaries = _compute_split_boundaries(split_probs, len(groups))
  group_id_to_split = {}
  for split_name, i_start, i_end in split_boundaries:
    for i in range(i_start, i_end):
      group_id_to_split[groups[i]] = split_name

  split_to_ids = collections.defaultdict(set)
  for item_id, group_id in items_and_groups:
    split = group_id_to_split[group_id]
    split_to_ids[split].add(item_id)

  return split_to_ids


def write_split_file(filename, data_splits):
  """Write data splits to a file in json format.

  Args:
    filename: The filename to write to.
    data_splits: {split_name: set(ids)}

  Returns:
    Nothing. Output is written to disk.
  """
  data_splits = {k: list(v) for k, v in data_splits.items()}
  with tf.io.gfile.GFile(filename, 'w') as f:
    json.dump(data_splits, f)


def read_split_file(filename):
  """Read data splits written by write_split_file().

  Args:
    filename: The filename to read from.

  Returns:
    Dictionary {split_name: set(ids)}
  """
  with tf.io.gfile.GFile(filename, 'r') as f:
    data_splits = json.load(f)
  return {k: set(v) for k, v in data_splits.items()}
