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

"""`beam.PTransform` to read from alignment files in Stockholm format."""

from __future__ import annotations

import functools
from typing import ClassVar

import apache_beam as beam

from dedal.preprocessing import types
from dedal.preprocessing import utils


# Type aliases
KeyRecordPair = types.KeyRecordPair

# Constants
STOCKHOLM_GS_FIELD_NAMES = {
    'AC': 'accession',
    'DR': 'database_references',
}


def rekey_by_accession(key_record):
  """Re-keys Pfam domains by their `pfamseq_acc` and endpoints."""
  old_key, record = key_record

  accession_field_name = STOCKHOLM_GS_FIELD_NAMES['AC']
  accession = record[accession_field_name]
  # `AC` field should not be duplicated in the Stockholm file.
  if isinstance(accession, list):
    raise ValueError(f'{old_key} has duplicated #=GS AC annotations.')
  if not isinstance(accession, str):
    raise ValueError(f'{accession} must be a string.')

  pfamseq_acc = accession.split('.')[0]
  region_str = old_key.split('/')[1]

  return f'{pfamseq_acc}/{region_str}', record


class Latin1Coder(beam.coders.Coder):
  """A coder used for reading and writing strings as `latin-1`."""

  def encode(self, value):
    return value.encode('latin-1')

  def decode(self, value):
    return value.decode('latin-1')

  def is_deterministic(self):
    return True


class ReadFromStockholm(beam.PTransform):
  """Generates key-value pairs from alignments in Pfam's Stockholm format."""

  # Format: `"#=GS <seqname> <feature> <free text>"` where:
  # + <seqname> has regex `r'\w+/\d+-\d+'`.
  # + <feature> has regex `r'[A-Z]{2}'`.
  # + <free text> has regex `r'\S.*'`.
  GS_PATTERN: ClassVar[str] = r'^#=GS\s+(\w+/\d+-\d+)\s+([A-Z]{2})\s+(\S.*)$'
  # Format: `"<seqname> <aligned sequence>"` where:
  # + <seqname> has regex `r'\w+/\d+-\d+'`.
  # + <aligned sequence> has regex `r'[a-zA-z\.-]+'`.
  ALI_PATTERN: ClassVar[str] = r'^(\w+/\d+-\d+)\s+([a-zA-z\.-]+)$'

  def __init__(self, file_pattern):
    self.file_pattern = file_pattern

  def expand(self, root):
    lines = (
        root
        | 'ReadFromText' >> beam.io.ReadFromText(
            file_pattern=self.file_pattern,
            coder=Latin1Coder()))
    alignments = (
        lines
        | 'AlignmentRegex' >> beam.Regex.all_matches(regex=self.ALI_PATTERN)
        | 'AlignmentDiscardGroup0' >> beam.Map(lambda matches: matches[1:])
        | 'AlignmentsToKeyValue' >> beam.Map(
            lambda matches: (matches[0], {'gapped_sequence': matches[1]})))
    annotations = (
        lines
        | 'AnnotationsRegex' >> beam.Regex.all_matches(regex=self.GS_PATTERN)
        | 'AnnotationsDiscardGroup0' >> beam.Map(lambda matches: matches[1:])
        | 'AnnotationsToKeyValue' >> beam.Map(
            lambda matches: (matches[0], {matches[1]: matches[2]}))
        | 'RenameElementFields' >> beam.Map(
            functools.partial(
                utils.rename_key_record_fields,
                field_name_mapping=STOCKHOLM_GS_FIELD_NAMES,
            )))
    return (
        {'alignments': alignments, 'annotations': annotations}
        | 'JoinDatabases' >> beam.CoGroupByKey()
        | 'FlattenJoinedElements' >> beam.Map(utils.flatten_grouped_records)
        | 'ReKeyByAccession' >> beam.Map(rekey_by_accession))
