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

"""Pipelines to fetch random "chunks" from UniProt to build extended domains."""

import functools
import random
from typing import Callable, Dict, Iterator, List

import apache_beam as beam
from apache_beam import pvalue
import numpy as np

from dedal.preprocessing import schemas
from dedal.preprocessing import schemas_lib
from dedal.preprocessing import tasks_lib
from dedal.preprocessing import types


# Type aliases
Record = types.Record
KeyRecordPair = types.KeyRecordPair
GroupedRecords = types.GroupedRecords
Array = np.ndarray
PRNG = random.Random


# Constants
FLANK_SEED_FIELDS = (
    'key', 'sequence', 'hmm_hit_seq_start', 'hmm_hit_seq_end',
    'hmm_hit_clan_acc', 'other_hit_seq_start', 'other_hit_seq_end',
    'other_hit_type_id')
SPLITS = ('train', 'iid_validation', 'ood_validation', 'iid_test', 'ood_test')
SPLIT_BINS = (0.0, 0.75, 0.80, 0.85, 0.925, 1.01)
NUM_SPLITS = len(SPLITS)
OOD_SPLITS = ('ood_validation', 'ood_test')
PREFIXES = ('n', 'c')  # Indices of the two flanks (N-terminus, C-terminus).
SUFFIXES = ('x', 'y')  # Indices of the two regions in the pairs.


def by_split(region, _, split_key = 'split'):
  """Partitions `PCollection` by dataset split (see `SPLITS`)."""
  return SPLITS.index(region[split_key])


def by_rng(flank_seed, _, global_seed = 0):
  rng = random.Random(global_seed + hash(flank_seed['key'][::-1]))
  return np.digitize(rng.uniform(0.0, 1.0), SPLIT_BINS) - 1


def get_pfamseq_acc(region):
  """Retrieves pfamseq_acc from the `key` field of a `ParsedPfamRow` element."""
  return region['key'].split('/')[0]


def dispatch_flank_seeds(
    flank_seed,
    region_keys,
    clan_acc_denylist,
    split_to_pfamseq_acc,
    target_split,
    dispatch_to = 100,
    global_seed = 0
):
  """If `flank_seed` is "valid", assigns it to `dispatch_to` Pfam regions."""
  # Avoids generating flank seeds from UniProt sequences ocurring in the other
  # data splits.
  pfamseq_acc_denylist = set(
      pfamseq_acc
      for split, pfamseq_acc in split_to_pfamseq_acc.items()
      if split != target_split)
  if flank_seed['key'] in pfamseq_acc_denylist:
    return
  # Avoids generating flank seeds from UniProt sequences containing annotations
  # from out-of-distribution clans.
  if set(flank_seed['hmm_hit_clan_acc']) & set(clan_acc_denylist):
    return

  # Assigns the UniProt sequence as a "flank seed" to up to `dispatch_to` Pfam
  # regions by sampling region keys uniformly at random without replacement.
  rng = random.Random(global_seed + hash(flank_seed['key']))
  for region_key in rng.sample(region_keys, k=dispatch_to):
    yield region_key, flank_seed


def append_flank_seed(
    rng,
    region,
    flank_seed,
    prefix,
    flank_len,
    flank_cnt,
    margin,
    min_overlap,
):
  """Adds `flank_seed` and its metadata to `region`."""
  kwargs = []
  flank_seed_sequence = flank_seed['sequence'][margin:-margin]
  flank_seed_len = len(flank_seed_sequence)

  offset = rng.randint(0, flank_seed_len - flank_len)
  flank_start = 1 + margin + offset
  flank_end = flank_start + flank_len - 1

  kwargs += [(f'{prefix}_flank_seed_key_{flank_cnt + 1}',
              f"{flank_seed['key']}/{flank_start}-{flank_end}")]
  kwargs += [(f'{prefix}_flank_seed_sequence_{flank_cnt + 1}',
              flank_seed_sequence[offset:offset + flank_len])]

  for ann_type, id_name in zip(('hmm_hit', 'other_hit'),
                               ('clan_acc', 'type_id')):
    hit_start = np.asarray(flank_seed[f'{ann_type}_seq_start'])
    hit_end = np.asarray(flank_seed[f'{ann_type}_seq_end'])
    hit_ids = np.asarray(flank_seed[f'{ann_type}_{id_name}'])

    overlaps = tasks_lib.interval_overlaps(
        start=flank_start,
        end=flank_end,
        ref_starts=hit_start,
        ref_ends=hit_end)
    indices = overlaps > min_overlap

    kwargs += [(f'{prefix}_flank_seed_{ann_type}_{id_name}_{flank_cnt + 1}',
                list(set(hit_ids[indices])))]

  region.update(dict(kwargs))

  return region


def merge_flank_seeds_into_region(
    grouped_records,
    max_len = 511,
    margin = 10,
    min_overlap = 1,
    global_seed = 0,
):
  """Attempts to find up to `schemas.NUM_FLANK_SEEDS` "valid" flank "seeds"."""
  region = grouped_records['regions'][0]
  flank_seeds = grouped_records['flank_seeds']

  rng = random.Random(global_seed + hash(region['key']))

  reg_len = region['seq_end'] - region['seq_start'] + 1
  max_ctx_len = max(0, max_len - reg_len)
  n_len = c_len = max_ctx_len

  if n_len <= c_len:
    min_flank_len, max_flank_len = n_len, c_len
    prefix_min, prefix_max = 'n', 'c'
  else:
    min_flank_len, max_flank_len = c_len, n_len
    prefix_min, prefix_max = 'c', 'n'

  min_flank_cnt, max_flank_cnt = 0, 0

  rng.shuffle(flank_seeds)
  for flank_seed in flank_seeds:

    if region['clan_acc'] in flank_seed['hmm_hit_clan_acc']:
      continue

    flank_seed_sequence = flank_seed['sequence'][margin:-margin]
    flank_seed_len = len(flank_seed_sequence)

    if (flank_seed_len >= max_flank_len > 0 and
        max_flank_cnt < schemas.NUM_FLANK_SEEDS):
      region = append_flank_seed(
          rng=rng,
          region=region,
          flank_seed=flank_seed,
          prefix=prefix_max,
          flank_len=max_flank_len,
          flank_cnt=max_flank_cnt,
          margin=margin,
          min_overlap=min_overlap)
      max_flank_cnt += 1
    elif (flank_seed_len >= min_flank_len > 0 and
          min_flank_cnt < schemas.NUM_FLANK_SEEDS):
      region = append_flank_seed(
          rng=rng,
          region=region,
          flank_seed=flank_seed,
          prefix=prefix_min,
          flank_len=min_flank_len,
          flank_cnt=min_flank_cnt,
          margin=margin,
          min_overlap=min_overlap)
      min_flank_cnt += 1

    # Early termination condition: we found the flanks we were looking for.
    if ((max_flank_len == 0 or max_flank_cnt == schemas.NUM_FLANK_SEEDS) and
        (min_flank_len == 0 or min_flank_cnt == schemas.NUM_FLANK_SEEDS)):
      break

  return region


def build_uniprot_flanks_pipeline(
    pfam_file_pattern,
    uniprot_file_pattern,
    dataset_splits_path,
    output_path,
    max_len = 511,
    min_overlap = 1,
    dispatch_to = 100,
    global_seed = 0,
):
  """2) Returns a pipeline to pre-select flank "seeds" for each Pfam region.

  Note: the desired number of flanks "seeds" for each Pfam region is defined by
  `schemas.NUM_FLANK_SEEDS`. In rare cases, some regions may have a smaller
  number of valid "seeds" found.

  Args:
    pfam_file_pattern: The file path from which to read preprocessed Pfam data.
      This is assumed to be the result of the `preprocess_pfam_tables.py`
      pipeline.
    uniprot_file_pattern: The file path from which to read preprocessed UniProt
      data. This is assumed to be the result of the
      `preprocess_uniprot_tables.py` pipeline.
    dataset_splits_path: The path to the key, split mapping file.
    output_path: The path prefix to the output files.
    max_len: The maximum length of sequences to be included in the output
      dataset.
    min_overlap: The minimum number of residues in a sequence that need to
      overlap with a region annotation in order for the annotation to be applied
      to the sequence.
    dispatch_to: Number of Pfam regions each UniProt sequence is dispatched to
      for generating flank "seeds".
    global_seed: A global seed for the PRNG.

  Returns:
  A beam.Pipeline.
  """
  def pipeline(root):
    # Reads all preprocessed Pfam regions, keeping all its fields.
    regions = (
        root
        | 'ReadParsedPfamData' >> tasks_lib.ReadParsedPfamData(
            file_pattern=pfam_file_pattern,
            dataset_splits_path=dataset_splits_path,
            fields_to_keep=None,
            max_len=None,
            filter_by_qc=False))

    # Reads all preprocessed UniProt sequences. We will use random chunks from
    # these as flank "seeds" for the extended Pfam regions.
    sequences = (
        root
        | 'ReadParsedUniProtData' >> schemas_lib.ReadFromTable(
            file_pattern=uniprot_file_pattern,
            schema_cls=schemas.ParsedUniProtRow,
            key_field='key',
            skip_header_lines=1,
            fields_to_keep=FLANK_SEED_FIELDS)
        | 'RemoveKey' >> beam.Values())

    # Retrieves the set of all Pfam sequence accessions, keyed by the dataset
    # split they belong to. As a precaution, any UniProt sequences that contain
    # Pfam regions in the other data splits will be deny-listed when generating
    # flank "seeds" for a given data split.
    split_to_pfamseq_acc = (
        regions
        | 'PfamseqAccs' >> beam.Map(lambda x: (x['split'], get_pfamseq_acc(x))))

    # Retrieves the set of Pfam clan accessions that occur in any of the
    # out-of-distribution data splits. To be on the safe side, we will filter
    # any UniProt sequences that contain annotations from any of these clans
    # when generating flank seeds.
    clan_acc_denylist = (
        regions
        | 'ClanAccs' >> beam.Map(lambda x: (x['split'], x['clan_acc']))
        | 'KeepOODSplits' >> beam.Filter(lambda x: x[0] in OOD_SPLITS)
        | 'KeepClanAccsOnly' >> beam.Values()
        | 'KeepUniqueClanAccs' >> beam.Distinct())

    # Partitions the set of preprocessed Pfam regions based on the dataset split
    # to which they were assigned. Thus, `regions[split]` is a `PCollection`
    # containing all Pfam regions for the `split` dataset.
    by_split_fn = functools.partial(by_split, split_key='split')
    regions = regions | 'SplitRegs' >> beam.Partition(by_split_fn, NUM_SPLITS)
    regions = dict(zip(SPLITS, regions))
    # To prevent any form of leakage, the UniProt sequences from which flanks
    # will also be assigned to a unique split each. We do this at random
    # "on-the-fly".
    by_rng_fn = functools.partial(by_rng, global_seed=global_seed)
    sequences = sequences | 'SplitSeqs' >> beam.Partition(by_rng_fn, NUM_SPLITS)
    sequences = dict(zip(SPLITS, sequences))

    extended_regions = []
    for split in SPLITS:
      keyed_regions = (
          regions[split]
          | f'ReKeyRegions_{split}' >> beam.Map(lambda x: (x['key'], x)))
      region_keys = keyed_regions | f'Keys_{split}' >> beam.Keys()
      flank_seeds = (
          sequences[split]
          | f'DispatchFlankSeeds_{split}' >> beam.FlatMap(
              dispatch_flank_seeds,
              region_keys=pvalue.AsList(region_keys),
              clan_acc_denylist=pvalue.AsList(clan_acc_denylist),
              split_to_pfamseq_acc=pvalue.AsDict(split_to_pfamseq_acc),
              target_split=split,
              dispatch_to=dispatch_to,
              global_seed=global_seed))
      extended_regions.append(
          {'regions': keyed_regions, 'flank_seeds': flank_seeds}
          | f'GroupRegionsWithFlanksSeeds_{split}' >> beam.CoGroupByKey()
          | f'ReshuffleAfterGroupingFlankSeeds_{split}' >> beam.Reshuffle()
          | f'RemoveKeyAfterGroupingFlankSeeds_{split}' >> beam.Values()
          | f'MergeFlankSeedsIntoRegion_{split}' >> beam.Map(
              functools.partial(
                  merge_flank_seeds_into_region,
                  max_len=max_len,
                  min_overlap=min_overlap,
                  global_seed=global_seed)))
    extended_regions = extended_regions | 'MergeSplits' >> beam.Flatten()

    # Writes preprocessed Pfam regions alongside the newly generated flank
    # "seeds" to tab-delimited output files.
    _ = (
        extended_regions
        | 'WriteToTable' >> schemas_lib.WriteToTable(
            file_path_prefix=output_path,
            schema_cls=schemas.ExtendedParsedPfamRow))

  return pipeline
