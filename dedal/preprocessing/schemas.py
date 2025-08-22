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

"""Implements dataclasses representing tables from the Pfam database."""

from __future__ import annotations

import dataclasses
from typing import List

from dedal.preprocessing import schemas_lib


# ------------------------------------------------------------------------------
# PFAM-DEFINED SCHEMAS
# See https://pfam-docs.readthedocs.io/en/latest/pfam-database.html for a
# detailed description of the fields in each table.
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PfamARegSeed(schemas_lib.TableRowMixin):
  """Represents an entry from the `pfamA_reg_seed.txt` table."""
  pfam_acc: str  # Note: called `pfamA_acc` in the original table.
  pfamseq_acc: str
  seq_start: int
  seq_end: int
  cigar: str
  tree_order: str
  seq_version: int
  md5: str
  source: str


@dataclasses.dataclass(frozen=True)
class PfamSeq(schemas_lib.TableRowMixin):
  """Represents an entry from the `pfamseq.txt` table."""
  pfamseq_acc: str
  pfamseq_id: str
  seq_version: int
  crc64: str
  md5: str
  description: str
  evidence: int
  length: int
  species: str
  taxonomy: str
  is_fragment: int
  sequence: str
  updated: str
  created: str
  ncbi_taxid: int
  genome_seq: str
  auto_architecture: str
  treefam_acc: str


@dataclasses.dataclass(frozen=True)
class UniProt(schemas_lib.TableRowMixin):
  """Represents an entry from the `uniprot.txt` table."""
  uniprot_acc: str
  uniprot_id: str
  seq_version: int
  crc64: str
  md5: str
  description: str
  evidence: int
  length: int
  species: str
  taxonomy: str
  is_fragment: int
  sequence: str
  updated: str
  created: str
  ncbi_taxid: int
  ref_proteome: int
  complete_proteome: int
  treefam_acc: str
  rp15: int
  rp35: int
  rp55: int
  rp75: int


@dataclasses.dataclass(frozen=True)
class PfamARegFullSignificant(schemas_lib.TableRowMixin):
  """Represents an entry from the `pfamA_reg_full_significant.txt` table."""
  auto_pfam_reg_full: int  # Called `auto_pfamA_reg_full` in the og. table.
  pfam_acc: str  # Called `pfamA_acc` in the original table.
  pfamseq_acc: str
  seq_start: int
  seq_end: int
  ali_start: int
  ali_end: int
  model_start: int
  model_end: int
  domain_bits_score: float
  domain_eval_score: float
  sequence_bits_score: float
  sequence_evalue_score: float
  cigar: str
  in_full: int
  tree_order: str
  domain_order: str


@dataclasses.dataclass(frozen=True)
class UniProtRegFull(schemas_lib.TableRowMixin):
  """Represents an entry from the `uniprot_reg_full.txt` table."""
  auto_uniprot_reg_full: int
  pfam_acc: str  # Called `pfamA_acc` in the original table.
  uniprot_acc: str
  seq_start: int
  seq_end: int
  ali_start: int
  ali_end: int
  model_start: int
  model_end: int
  domain_bits_score: float
  domain_eval_score: float
  sequence_bits_score: float
  sequence_evalue_score: float
  in_full: int


@dataclasses.dataclass(frozen=True)
class OtherReg(schemas_lib.TableRowMixin):
  """Represents an entry from the `other_reg.txt` table."""
  region_id: int
  pfamseq_acc: str
  seq_start: int
  seq_end: int
  type_id: str
  source_id: str
  score: str
  orientation: str


@dataclasses.dataclass(frozen=True)
class ClanMembership(schemas_lib.TableRowMixin):
  """Represents an entry from the `clan_membership.txt` table."""
  clan_acc: str
  pfam_acc: str


# ------------------------------------------------------------------------------
# CUSTOM SCHEMAS
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedPfamRow(schemas_lib.TableRowMixin):
  """Represents a row of the prepreocessed Pfam output table."""
  key: str
  pfam_acc: str
  clan_acc: str
  seq_start: int
  seq_end: int
  sequence: str
  gapped_sequence: str
  description: str
  species: str
  taxonomy: str
  accession: str
  passed_qc: bool
  database_references: List[str] = dataclasses.field(default_factory=list)
  hmm_hit_seq_start: List[int] = dataclasses.field(default_factory=list)
  hmm_hit_seq_end: List[int] = dataclasses.field(default_factory=list)
  hmm_hit_pfam_acc: List[str] = dataclasses.field(default_factory=list)
  hmm_hit_clan_acc: List[str] = dataclasses.field(default_factory=list)
  other_hit_seq_start: List[int] = dataclasses.field(default_factory=list)
  other_hit_seq_end: List[int] = dataclasses.field(default_factory=list)
  other_hit_type_id: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class ParsedUniProtRow(schemas_lib.TableRowMixin):
  """Represents a row of the prepreocessed Pfam output table."""
  key: str
  sequence: str
  description: str
  species: str
  taxonomy: str
  hmm_hit_seq_start: List[int] = dataclasses.field(default_factory=list)
  hmm_hit_seq_end: List[int] = dataclasses.field(default_factory=list)
  hmm_hit_pfam_acc: List[str] = dataclasses.field(default_factory=list)
  hmm_hit_clan_acc: List[str] = dataclasses.field(default_factory=list)
  other_hit_seq_start: List[int] = dataclasses.field(default_factory=list)
  other_hit_seq_end: List[int] = dataclasses.field(default_factory=list)
  other_hit_type_id: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class DatasetSplits(schemas_lib.TableRowMixin):
  key: str
  split: str


@dataclasses.dataclass(frozen=True)
class PairwiseAlignmentRow(schemas_lib.TableRowMixin):
  """Represents a row of the pairwise alignments output table."""
  extended_key_x: str
  extended_key_y: str
  key_x: str
  key_y: str
  pfam_acc_x: str
  pfam_acc_y: str
  clan_acc_x: str
  clan_acc_y: str
  sequence_x: str
  sequence_y: str
  ali_start_x: int
  ali_start_y: int
  states: str
  percent_identity: float
  bos_x: bool
  bos_y: bool
  eos_x: bool
  eos_y: bool
  maybe_confounded: bool
  fallback: bool
  shares_clans_in_flanks: bool = False
  shares_coiled_coil_in_flanks: bool = False
  shares_disorder_in_flanks: bool = False
  shares_low_complexity_in_flanks: bool = False
  shares_sig_p_in_flanks: bool = False
  shares_transmembrane_in_flanks: bool = False


@dataclasses.dataclass(frozen=True)
class PairwiseHomologyRow(schemas_lib.TableRowMixin):
  """Represents a row of the pairwise homology detection output table."""
  extended_key_x: str
  extended_key_y: str
  key_x: str
  key_y: str
  pfam_acc_x: str
  pfam_acc_y: str
  clan_acc_x: str
  clan_acc_y: str
  sequence_x: str
  sequence_y: str
  homology_label: int
  percent_identity: float
  bos_x: bool
  bos_y: bool
  eos_x: bool
  eos_y: bool
  maybe_confounded: bool
  shares_clans: bool = False
  shares_coiled_coil: bool = False
  shares_disorder: bool = False
  shares_low_complexity: bool = False
  shares_sig_p: bool = False
  shares_transmembrane: bool = False


NUM_FLANK_SEEDS = 5

FLANK_FIELDS = []
for p in ('n', 'c'):
  for i in range(NUM_FLANK_SEEDS):
    FLANK_FIELDS.append((f'{p}_flank_seed_key_{i + 1}',
                         'str',
                         dataclasses.field(default='')))
    FLANK_FIELDS.append((f'{p}_flank_seed_sequence_{i + 1}',
                         'str',
                         dataclasses.field(default='')))
    FLANK_FIELDS.append((f'{p}_flank_seed_hmm_hit_clan_acc_{i + 1}',
                         'List[str]',
                         dataclasses.field(default_factory=list)))
    FLANK_FIELDS.append((f'{p}_flank_seed_other_hit_type_id_{i + 1}',
                         'List[str]',
                         dataclasses.field(default_factory=list)))


ExtendedParsedPfamRow = dataclasses.make_dataclass(
    cls_name='ExtendedParsedPfamRow',
    fields=FLANK_FIELDS,
    bases=(ParsedPfamRow,),
    frozen=True,
    namespace={'__module__': __name__})
