# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Axuiliary functions for Beam pipelines to process Pfam databases."""

import random
from typing import Iterable, Iterator, Optional

import apache_beam as beam
from apache_beam import pvalue

from dedal.preprocessing import types


# Type aliases
SingleValue = types.SingleValue
Value = types.Value
Record = types.Record
KeyRecordPair = types.KeyRecordPair
KeyGroupedRecordsPair = types.KeyGroupedRecordsPair


def change_key(key_record, new_key_field):
  """Swaps previous key of a `key_record` pair by `element[new_key_field]`."""
  _, record = key_record
  return record[new_key_field], record


def rename_key_record_fields(
    key_record,
    field_name_mapping,
):
  """Renames `record` fields of a `key_record` pair keeping the same key."""
  key, record = key_record

  new_record: Record = {}
  for field_name, field_value in record.items():
    # Discards fields that are not defined in `field_name_mapping`.
    if field_name in field_name_mapping:
      new_field_name = field_name_mapping[field_name]
      new_record[new_field_name] = field_value

  return key, new_record


def rename_record_fields(
    record,
    field_name_mapping,
):
  """Renames fields of an unkeyed `record`."""
  new_record: Record = {}
  for field_name, field_value in record.items():
    # Discards fields that are not defined in `field_name_mapping`.
    if field_name in field_name_mapping:
      new_field_name = field_name_mapping[field_name]
      new_record[new_field_name] = field_value
  return new_record


def drop_key_record_fields(
    key_record,
    fields_to_drop,
):
  """Removes all of `fields_to_drop` fields from a `key_record` pair."""
  key, record = key_record
  for field_name in fields_to_drop:
    record.pop(field_name)
  return key, record


def drop_record_fields(
    record,
    fields_to_drop,
):
  """Removes all of `fields_to_drop` fields from an unkeyed `record`."""
  for field_name in fields_to_drop:
    record.pop(field_name)
  return record


def merge_key_into_record(
    key_record,
    key_field = 'key',
):
  """Drops key from a `key_record` pair, saving the key as a new field."""
  key, record = key_record
  record[key_field] = key
  return record


def is_in_pcollection(
    key_grouped_records,
    pcol_name,
):
  """Returns True if there are >= 1 elements from `PCollection` `pcol_name`."""
  _, grouped_records = key_grouped_records
  return bool(len(grouped_records[pcol_name]))


def append_to_record(record, k, v):
  """Adds `v` to `record` with key `k`, promoting field to var-len if needed."""
  # `Value`s can be either singleton `SingleValue`s or lists of
  # `SingleValue`s. Conversion to list happens whenever the first
  # attempt to rewrite the value of a key `k` occurs.
  if k not in record:
    record[k] = v
  else:
    record[k]: list[SingleValue] = (record[k] if isinstance(record[k], list)
                                    else [record[k]])
    if isinstance(v, list):
      record[k].extend(v)
    else:
      record[k].append(v)


def flatten_grouped_records(
    key_grouped_records,
):
  """Flattens the result of `CoGroupByKey`, removing `PCollection` names."""
  key, grouped_records = key_grouped_records
  new_record: Record = {}
  for records in grouped_records.values():  # Ignores `PCollection` names.
    for record in records:
      for k, v in record.items():
        append_to_record(new_record, k, v)  # Promotes to var-len when needed.
  return key, new_record


class Combinations(beam.PTransform):
  """Exhaustive enumeration of all `Record` pairs."""

  def __init__(
      self,
      groupby_field = None,
      key_field = 'key',
      num_samples = None,
      suffixes = ('x', 'y')):
    self.groupby_field = groupby_field
    self.key_field = key_field
    self.num_samples = num_samples
    self.suffixes = suffixes

  def merge_pair(self, record_x, record_y):
    """Merges `record_x` and `record_y` into a single `Record`."""
    new_record: Record = {}
    for k in set(record_x) & set(record_y):
      new_record[f'{k}_{self.suffixes[0]}'] = record_x[k]
      new_record[f'{k}_{self.suffixes[1]}'] = record_y[k]
    return new_record

  def yield_unique_pairs(
      self,
      key_grouped_records,
  ):
    """Returns an iterator over all unique pairs of merged `Record`s."""
    _, grouped_records = key_grouped_records
    for record_x in grouped_records[self.suffixes[0]]:
      for record_y in grouped_records[self.suffixes[1]]:
        # Arbitrarily use lexicographic order of keys to ensure each pair of
        # elements is generated only once. It also skips pairs formed by the
        # same element twice.
        if record_x[self.key_field] < record_y[self.key_field]:
          yield self.merge_pair(record_x, record_y)

  def expand(
      self,
      pcol,
  ):

    def key_fn(record):
      key = '1' if self.groupby_field is None else record[self.groupby_field]
      return key, record

    pcol = pcol | 'AddKey' >> beam.Map(key_fn)
    paired_pcol = (
        {suffix: pcol for suffix in self.suffixes}
        | 'GroupByKey' >> beam.CoGroupByKey()
        | 'ReshuffleBeforePairing' >> beam.Reshuffle()
        | 'EnumPairs' >> beam.FlatMap(self.yield_unique_pairs))
    if self.num_samples is not None:
      paired_pcol = (
          paired_pcol
          | 'Downsample' >> beam.combiners.Sample.FixedSizeGlobally(
              self.num_samples)
          | 'UnravelSamples' >> beam.FlatMap(lambda x: x))
    return paired_pcol | 'ReshuffleAfterPairing' >> beam.Reshuffle()


class SubsampleOuterProduct(beam.PTransform):
  """Enumeration of a random subsample of all `Record` pairs."""

  def __init__(
      self,
      avg_num_samples,
      groupby_field = None,
      key_field = 'key',
      suffixes = ('x', 'y')):
    self.avg_num_samples = avg_num_samples
    self.groupby_field = groupby_field
    self.key_field = key_field
    self.suffixes = suffixes

  def merge_pair(self, record_x, record_y):
    """Merges `record_x` and `record_y` into a single `Record`."""
    new_record: Record = {}
    for k in set(record_x) & set(record_y):
      new_record[f'{k}_{self.suffixes[0]}'] = record_x[k]
      new_record[f'{k}_{self.suffixes[1]}'] = record_y[k]
    return new_record

  def yield_pairs_subsample(
      self,
      record_x,
      records_y,
      records_pey_key,
  ):
    rng = random.Random(hash(record_x[self.key_field]))

    if self.groupby_field is None:
      # Total number of records.
      num_records = sum(records_pey_key.values())
      # Total number of record pairs.
      num_pairs = num_records * num_records
      # Probability to keep a pair consisting of `record_y` and any other
      # record.
      keep_prob = self.avg_num_samples / num_records / num_records
    else:
      groupby_field = record_x[self.groupby_field]
      # Number of pairs consisting of `record_x` and another record with the
      # same `groupby_field` key.
      num_pairs_x = records_pey_key[groupby_field] - 1
      # Total number of records pairs with the same `groupby_field` key.
      num_pairs = sum(x * (x - 1) for x in records_pey_key.values())
      # Probability to keep a pair consisting of `record_x` and another
      # record with the same `groupby_field` key.
      ratio_x = num_pairs_x / (num_pairs_x + 1)  # Note: zero for singletons!
      keep_prob = self.avg_num_samples * ratio_x / num_pairs

    for record_y in records_y:
      # If a `groupby_field` was specified, ensures only pairs with the same
      # `groupby_field` key are generated.
      groupby_cond = (self.groupby_field is None or
                      record_y[self.groupby_field] == groupby_field)
      # Skips pairs consisting of the same record twice.
      self_cond = record_x[self.key_field] != record_y[self.key_field]
      if groupby_cond and self_cond and rng.uniform(0, 1) <= keep_prob:
        yield self.merge_pair(record_x, record_y)

  def expand(
      self,
      pcol,
  ):

    def key_fn(record):
      key = '1' if self.groupby_field is None else record[self.groupby_field]
      return key, record

    num_records = (
        pcol
        | 'AddKey' >> beam.Map(key_fn)
        | 'CountByKey' >> beam.combiners.Count.PerKey())
    paired_pcol = (
        pcol
        | 'YieldPairsSubsample' >> beam.FlatMap(
            self.yield_pairs_subsample,
            pvalue.AsIter(pcol),
            pvalue.AsDict(num_records)))
    return paired_pcol | 'ReshuffleAfterPairingRecords' >> beam.Reshuffle()
