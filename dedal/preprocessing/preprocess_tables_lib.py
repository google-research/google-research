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

"""Pipelines to preprocess the Pfam MySQL database."""

from __future__ import annotations

import bisect
import collections
import copy
import functools
from typing import Callable, Iterable, Iterator, List

import apache_beam as beam

from dedal.preprocessing import schemas
from dedal.preprocessing import schemas_lib
from dedal.preprocessing import stockholm_lib
from dedal.preprocessing import types
from dedal.preprocessing import utils


# Type aliases
Key = types.Key
Value = types.Value
Record = types.Record
KeyRecordPair = types.KeyRecordPair
GroupedRecords = types.GroupedRecords
KeyGroupedRecordsPair = types.KeyGroupedRecordsPair

# Constants
HMMER_REG_TABLES = ('pfam_seed', 'pfam_reg', 'uniprot_reg')
OTHER_REG_TABLES = ('other_reg',)
REG_TABLES = HMMER_REG_TABLES + OTHER_REG_TABLES
HMMER_REGION_FIELD_NAMES = ('seq_start', 'seq_end', 'pfam_acc')
OTHER_REGION_FIELD_NAMES = ('seq_start', 'seq_end', 'type_id')
SEQUENCE_FIELD_NAMES = ('sequence', 'description', 'species', 'taxonomy')


class ReadPfamSQLData(beam.PTransform):
  """Reads data from Pfam MySQL tables."""

  def __init__(
      self,
      pfam_seed_path,
      pfam_reg_path,
      uniprot_reg_path,
      other_reg_path,
      uniprot_seq_path,
  ):
    self.pfam_seed_path = pfam_seed_path
    self.pfam_reg_path = pfam_reg_path
    self.uniprot_reg_path = uniprot_reg_path
    self.other_reg_path = other_reg_path
    self.uniprot_seq_path = uniprot_seq_path

  def expand(
      self,
      root,
  ):
    # Defines particular cases of `schemas_lib.ReadFromTable` for clarity and
    # re-use.
    read_from_pfam_region_table_cls = functools.partial(
        schemas_lib.ReadFromTable,
        fields_to_keep=HMMER_REGION_FIELD_NAMES)
    read_from_pfam_other_region_table_cls = functools.partial(
        schemas_lib.ReadFromTable,
        fields_to_keep=OTHER_REGION_FIELD_NAMES)
    read_from_pfam_sequence_table_cls = functools.partial(
        schemas_lib.ReadFromTable,
        fields_to_keep=SEQUENCE_FIELD_NAMES)
    # Reads and formats Pfam region-like databases.
    reg_tables = {
        'pfam_seed':
            root | 'PfamSeedRead' >> read_from_pfam_region_table_cls(
                file_pattern=self.pfam_seed_path,
                schema_cls=schemas.PfamARegSeed,
                key_field='pfamseq_acc'),
        'pfam_reg':
            root | 'PfamRegRead' >> read_from_pfam_region_table_cls(
                file_pattern=self.pfam_reg_path,
                schema_cls=schemas.PfamARegFullSignificant,
                key_field='pfamseq_acc'),
        'uniprot_reg':
            root | 'UniProtRegRead' >> read_from_pfam_region_table_cls(
                file_pattern=self.uniprot_reg_path,
                schema_cls=schemas.UniProtRegFull,
                key_field='uniprot_acc'),
    }
    other_reg_tables = {
        'other_reg':
            root | 'OtherRegRead' >> read_from_pfam_other_region_table_cls(
                file_pattern=self.other_reg_path,
                schema_cls=schemas.OtherReg,
                key_field='pfamseq_acc'),
    }
    # Reads and formats Pfam sequence-like databases.
    seq_tables = {
        'uniprot_seq':
            root | 'UniProtSeqRead' >> read_from_pfam_sequence_table_cls(
                file_pattern=self.uniprot_seq_path,
                schema_cls=schemas.UniProt,
                key_field='uniprot_acc')
    }
    return {**reg_tables, **other_reg_tables, **seq_tables}


def aggregate_hits(
    key_grouped_records,
    region_pcol_names,
    region_field_names,
    output_key,
):
  """Aggregates all Pfam hits per sequence (keyed by `pfamseq_acc`)."""
  key, grouped_records = key_grouped_records

  visited_hits = set()
  hits = collections.defaultdict(list)
  for pcol_name in region_pcol_names:
    for record in grouped_records[pcol_name]:
      # Skips duplicate hits.
      hit = tuple(record[k] for k in region_field_names)
      if hit not in visited_hits:
        # Inserts hit in increasing order of starting position.
        insort_idx = bisect.bisect_right(
            a=hits[f'{output_key}_seq_start'], x=record['seq_start'])
        for field in region_field_names:
          hits[f'{output_key}_{field}'].insert(insort_idx, record[field])
        # Updates registry of hits seen for the sequence with Pfam accession
        # `pfamseq_acc` (`key`).
        visited_hits.add(hit)

  # Inserts `hits` into `grouped_records` keeping the `PCollection`'s structure
  # unchanged.
  grouped_records[output_key]: Iterable[Record] = []
  for values in zip(*hits.values()):
    grouped_records[output_key].append({k: v for k, v in zip(hits, values)})
  return key, grouped_records


def rekey_by_hit(
    key_grouped_records,
    target_region_pcol_name,
    region_pcol_names = (),
):
  """Unravels Pfam regions in the same protein as distinct `Record`s."""
  old_key, grouped_records = key_grouped_records

  # Gathers and flattens all fields that are shared across all hits in the
  # sequence with Pfam accession `pfamseq_acc` (`old_key`).
  shared_record: Record = {}
  for pcol_name, records in grouped_records.items():
    # Discards leftover, raw, unprocessed fields from region database entries.
    if pcol_name not in region_pcol_names:
      for record in records:
        for k, v in record.items():
          utils.append_to_record(shared_record, k, v)

  # Gathers and flattens the hit-specific fields, unravelling the set of hits
  # from the Pfam region database `target_region_pcol_name` that occur in the
  # sequence with Pfam accession `pfamseq_acc` as individual (key, value) pairs.
  for record in grouped_records[target_region_pcol_name]:
    new_key = f"{old_key}/{record['seq_start']}-{record['seq_end']}"

    new_record: Record = copy.deepcopy(shared_record)
    for k, v in record.items():
      utils.append_to_record(new_record, k, v)

    yield new_key, new_record


def add_clan_accessions(
    key_record,
    clan_membership,
    hit_key = 'hmm_hit',
):
  """Lifts all fields containing Pfam accessions to new clan-valued fields."""
  key, record = key_record

  field_keys = ('pfam_acc', f'{hit_key}_pfam_acc')
  for field_key in field_keys:
    new_field_key = field_key.replace('pfam', 'clan')

    if field_key in record:
      # Unifies the treatment of scalar and variable-length fields.
      values = record[field_key]
      if not isinstance(values, List):
        values = [values]

      for value in values:
        # `pfam_acc`s that are *not* in the `clan_membership` dict get assigned
        # by default a `clan_acc` equal to `pfam_acc`. These are "singleton"
        # Pfam families that do not belong to any Pfam clan as of yet.
        new_value = clan_membership.get(value, value)
        utils.append_to_record(record, new_field_key, new_value)

  return key, record


def apply_quality_control(
    key_element,
    var_len_field_names = (),
    gap_chars = ('.', '-'),
):
  """Applies a sequence of sanity-checks to the output records."""
  key, element = key_element

  # First, checks that fields that are not annotated as being variable length
  # do not contain duplicates.
  passed_qc = True
  for k, v in element.items():
    if k not in var_len_field_names and isinstance(v, list):
      passed_qc = False

  # Second, verifies that `sequence` and `gapped_sequence` are non-empty.
  if passed_qc:
    sequence = element.get('sequence', '')
    gapped_sequence = element.get('gapped_sequence', '')
    if not sequence or not gapped_sequence:
      passed_qc = False

  # Third, tests if the reported hit endpoints are well-formed and within
  # the bounds of the sequence.
  if passed_qc:
    seq_start, seq_end = element['seq_start'], element['seq_end']
    if seq_start > seq_end or seq_start < 1 or seq_end > len(sequence):
      passed_qc = False

  # Finally, makes sure that the gapped sequence coincides with the subsequence
  # of the UniProtKB sequence delimited by the reported hit endpoints.
  if passed_qc:
    # Removes gap characters from the gapped sequence.
    for gap_char in gap_chars:
      gapped_sequence = gapped_sequence.replace(gap_char, '')
      gapped_sequence = gapped_sequence.upper()
    passed_qc = sequence[seq_start - 1:seq_end] == gapped_sequence

  element['passed_qc'] = passed_qc

  return key, element


def build_preprocess_pfam_pipeline(
    pfam_seed_path,
    pfam_reg_path,
    uniprot_reg_path,
    other_reg_path,
    uniprot_seq_path,
    clan_membership_path,
    stockholm_path,
    output_path,
):
  """1a) returns a pipeline to preprocess Pfam entries from Pfam MySQL data.

  Args:
    pfam_seed_path: The path to `pfamA_reg_seed.txt`.
    pfam_reg_path: The path to `pfamA_reg_full_significant.txt`.
    uniprot_reg_path: The path to `uniprot_reg_full.txt`.
    other_reg_path: The path to `other_reg.txt`.
    uniprot_seq_path: The path to `uniprot.txt`.
    clan_membership_path: The path to `clan_membership.txt`.
    stockholm_path: The path to `Pfam-A.seed`.
    output_path: The path prefix to the output files.

  Returns:
  A beam.Pipeline.
  """

  def pipeline(root):
    # Reads and formats data from Pfam region and sequence tables.
    tables = (
        root
        | 'ReadTableData' >> ReadPfamSQLData(
            pfam_seed_path=pfam_seed_path,
            pfam_reg_path=pfam_reg_path,
            uniprot_reg_path=uniprot_reg_path,
            other_reg_path=other_reg_path,
            uniprot_seq_path=uniprot_seq_path))
    # Reads and formats Pfam clan membership list, linking `pfam_acc` accessions
    # to `clan_acc` accessions. This results in ~10,000 (`pfam_acc`, `clan_acc`)
    # pairs to be used as side inputs in subsequent steps in the pipeline.
    clan_membership = (
        root
        | 'ClanMembershipRead' >> schemas_lib.ReadFromTable(
            file_pattern=clan_membership_path,
            schema_cls=schemas.ClanMembership,
            key_field='pfam_acc',
            fields_to_keep=('clan_acc',))
        | 'ClanMembershipFlatten' >> beam.Map(
            lambda key_record: (key_record[0], key_record[1]['clan_acc'])))
    # Reads and formats Pfam alignments in Stockholm format. These will be
    # cross-referenced with hits reported by the Pfam tables `reg_tables` for
    # use as additional fields.
    alignments = (
        root
        | 'AlignmentRead' >> stockholm_lib.ReadFromStockholm(
            file_pattern=stockholm_path))

    # Merges all Pfam region tables with the Pfam sequence tables.
    # 1. Entries are cross-referenced by `pfamseq_acc`.
    # 2. Only those present in the target region table (`pfam_seed` by default)
    #    are kept.
    # 3. For each remaining sequence, hits reported in *any* Pfam region table
    #    are aggregated and sorted.
    # 4. The set of hits in the target region table (`pfam_seed` by default) are
    #    unraveled with new unique keys, carrying all metadata from the sequence
    #    in which the occur.
    # 5. Clan membership information is explicitly added to complement family
    #    membership.
    merged_tables = (
        tables
        | 'JoinPfamTables' >> beam.CoGroupByKey()
        | 'KeepSequencesInPfamSeed' >> beam.Filter(
            functools.partial(
                utils.is_in_pcollection,
                pcol_name='pfam_seed'))
        | 'AggregateHMMERMatchHits' >> beam.Map(
            functools.partial(
                aggregate_hits,
                region_pcol_names=HMMER_REG_TABLES,
                region_field_names=HMMER_REGION_FIELD_NAMES,
                output_key='hmm_hit'))
        | 'AggregateOtherRegionHits' >> beam.Map(
            functools.partial(
                aggregate_hits,
                region_pcol_names=OTHER_REG_TABLES,
                region_field_names=OTHER_REGION_FIELD_NAMES,
                output_key='other_hit'))
        | 'ReshuffleAfterAggregatingHits' >> beam.Reshuffle()
        | 'ReKeyByHit' >> beam.FlatMap(
            functools.partial(
                rekey_by_hit,
                target_region_pcol_name='pfam_seed',
                region_pcol_names=REG_TABLES))
        | 'AddClanAccessions' >> beam.Map(
            add_clan_accessions,
            clan_membership=beam.pvalue.AsDict(clan_membership),
            hit_key='hmm_hit'))
    # Merges alignment information from the Stockholm file into the
    # `PCollection` with the Pfam table data. Next, a "quality control" bool
    # flag is computed for each entry based on a series of basic cross-database
    # consistency checks.
    merged_tables_with_alignments = (
        {'merged_tables': merged_tables, 'alignments': alignments}
        | 'MergeAlignments' >> schemas_lib.JoinTables()
        | 'QualityControl' >> beam.Map(
            apply_quality_control,
            var_len_field_names=schemas.ParsedPfamRow.var_len_field_names()))

    # Writes hits with metadata to tab-delimited output files.
    _ = (
        merged_tables_with_alignments
        | 'MergeKey' >> beam.Map(utils.merge_key_into_record)
        | 'WriteToTable' >> schemas_lib.WriteToTable(
            file_path_prefix=output_path,
            schema_cls=schemas.ParsedPfamRow))

  return pipeline


def build_preprocess_uniprot_pipeline(
    pfam_seed_path,
    pfam_reg_path,
    uniprot_reg_path,
    other_reg_path,
    uniprot_seq_path,
    clan_membership_path,
    output_path,
):
  """1b) Returns a pipeline to preprocess UniProt entries from Pfam MySQL data.

  Args:
    pfam_seed_path: The path to `pfamA_reg_seed.txt`.
    pfam_reg_path: The path to `pfamA_reg_full_significant.txt`.
    uniprot_reg_path: The path to `uniprot_reg_full.txt`.
    other_reg_path: The path to `other_reg.txt`.
    uniprot_seq_path: The path to `uniprot.txt`.
    clan_membership_path: The path to `clan_membership.txt`.
    output_path: The path prefix to the output files.

  Returns:
  A beam.Pipeline.
  """

  def pipeline(root):
    # Reads and formats data from Pfam region and sequence tables.
    tables = (
        root
        | 'ReadTableData' >> ReadPfamSQLData(
            pfam_seed_path=pfam_seed_path,
            pfam_reg_path=pfam_reg_path,
            uniprot_reg_path=uniprot_reg_path,
            other_reg_path=other_reg_path,
            uniprot_seq_path=uniprot_seq_path))
    # Reads and formats Pfam clan membership list, linking `pfam_acc` accessions
    # to `clan_acc` accessions. This results in ~10,000 (`pfam_acc`, `clan_acc`)
    # pairs to be used as side inputs in subsequent steps in the pipeline.
    clan_membership = (
        root
        | 'ClanMembershipRead' >> schemas_lib.ReadFromTable(
            file_pattern=clan_membership_path,
            schema_cls=schemas.ClanMembership,
            key_field='pfam_acc',
            fields_to_keep=('clan_acc',))
        | 'ClanMembershipFlatten' >> beam.Map(
            lambda key_record: (key_record[0], key_record[1]['clan_acc'])))

    # Merges all Pfam region tables with the Pfam sequence tables.
    # 1. Entries are cross-referenced by `pfamseq_acc`.
    # 2. Only those present in the UniProt sequence table are kept.
    # 3. For each remaining sequence, hits reported in *any* Pfam region table
    #    are aggregated and sorted.
    # 4. Clan membership information is explicitly added to complement family
    #    membership.
    merged_tables = (
        tables
        | 'JoinPfamTables' >> beam.CoGroupByKey()
        | 'KeepSequencesInUniProt' >> beam.Filter(
            functools.partial(
                utils.is_in_pcollection,
                pcol_name='uniprot_seq'))
        | 'AggregateHMMERMatchHits' >> beam.Map(
            functools.partial(
                aggregate_hits,
                region_pcol_names=HMMER_REG_TABLES,
                region_field_names=HMMER_REGION_FIELD_NAMES,
                output_key='hmm_hit'))
        | 'AggregateOtherRegionHits' >> beam.Map(
            functools.partial(
                aggregate_hits,
                region_pcol_names=OTHER_REG_TABLES,
                region_field_names=OTHER_REGION_FIELD_NAMES,
                output_key='other_hit'))
        | 'ReshuffleAfterAggregatingHits' >> beam.Reshuffle()
        | 'FlattenGroupedRecords' >> beam.Map(utils.flatten_grouped_records)
        | 'AddClanAccessions' >> beam.Map(
            add_clan_accessions,
            clan_membership=beam.pvalue.AsDict(clan_membership),
            hit_key='hmm_hit'))

    # Writes hits with metadata to tab-delimited output files.
    _ = (
        merged_tables
        | 'MergeKey' >> beam.Map(utils.merge_key_into_record)
        | 'WriteToTable' >> schemas_lib.WriteToTable(
            file_path_prefix=output_path,
            schema_cls=schemas.ParsedUniProtRow))

  return pipeline
