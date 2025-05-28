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

"""CoVID-specific utilities for the spectre paper."""

import os

import immutabledict
import numpy as np
import pandas as pd
import tree

from covid_vhh_design import helper
from covid_vhh_design import utils


PARENT_NAME = 'SARS_VHH72'
PARENT_SEQ = (
    'QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTISRDN'
    'AKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSS')

COV1 = 'CoV1'
COV2 = 'CoV2'
COV1_WT = 'SARS-CoV1_RBD'
COV2_WT = 'SARS-CoV2_RBD'
COV_WT = (COV1_WT, COV2_WT)
COV2_DELTA = 'SARS-CoV2-Delta_RBD'
COV2_OMICRON = 'SARS-CoV2-Omicron_RBD'


TARGET_SHORT_NAME_MAPPING = immutabledict.immutabledict({
    COV1_WT: 'SARS-CoV-1',
    COV2_WT: 'SARS-CoV-2',
    COV2_DELTA: 'Delta',
    COV2_OMICRON: 'Omicron',
    'SARS-CoV2_RBD_G502D': 'SARS-CoV-2_G502D',
    'SARS-CoV2_RBD_N439K': 'SARS-CoV-2_N439K',
    'SARS-CoV2_RBD_N501D': 'SARS-CoV-2_N501D',
    'SARS-CoV2_RBD_N501F': 'SARS-CoV-2_N501F',
    'SARS-CoV2_RBD_N501Y': 'SARS-CoV-2_N501Y',
    'SARS-CoV2_RBD_N501Y+K417N+E484K': 'SARS-CoV-2_N501Y+K417N+E484K',
    'SARS-CoV2_RBD_N501Y+K417T+E484K': 'SARS-CoV-2_N501Y+K417T+E484K',
    'SARS-CoV2_RBD_R408I': 'SARS-CoV-2_R408I',
    'SARS-CoV2_RBD_S477N': 'SARS-CoV-2_S477N',
    'SARS-CoV2_RBD_V367F': 'SARS-CoV-2_V367F',
})

DBR = 'DBR'

# Maps a group to a sequence design method: Parent, Baseline, ML, or Shuffled.
DESIGN_BY_GROUP = immutabledict.immutabledict((
    ('parent', utils.PARENT),
    # Baseline round 0
    ('singles', utils.BASELINE),
    # Baseline COVID round 1
    ('cdr12_singles_IMGT', utils.BASELINE),
    ('cdr3_singles_IMGT', utils.BASELINE),
    ('cdr123_best_IMGT', utils.BASELINE),
    ('cdr123_singles_IMGT', utils.BASELINE),
    ('shuffled_parent_CDR1', utils.SHUFFLED),
    ('shuffled_parent_CDR2', utils.SHUFFLED),
    ('shuffled_parent_CDR3', utils.SHUFFLED),
    # Baseline CoVID Round 2
    ('baseline_r0', utils.BASELINE),
    ('baseline_r1', utils.BASELINE),
    # MBO CoVID round 0
    ('cdr3_multies1', utils.ML),
    ('cdr3_multies2', utils.ML),
    ('cdr3_random', utils.ML),
    ('cdrh12_multies1', utils.ML),
    ('cdrh12_multies2', utils.ML),
    ('cdrh12_random', utils.ML),
    ('mutant', utils.ML),
    # MBO CoVID round 1
    ('mbo_BD', utils.ML),
    ('mbo_CDR12', utils.ML),
    ('mbo_CDR123', utils.ML),
    ('mbo_CDR3', utils.ML),
    # MBO CoVID round 2
    ('mbo_cnn', utils.ML),
    ('mbo_dbr', utils.ML),
    ('mbo_lgb', utils.ML),
))

# Maps a group to machine learning model, e.g., LGB, CNN or VAE.
MODEL_BY_GROUP = immutabledict.immutabledict((
    # MBO CoVID round 0
    ('cdr3_multies1', utils.VAE),
    ('cdr3_multies2', utils.VAE),
    ('cdr3_random', utils.VAE_RANDOM),
    ('cdrh12_multies1', utils.VAE),
    ('cdrh12_multies2', utils.VAE),
    ('cdrh12_random', utils.VAE_RANDOM),
    ('mutant', utils.VAE_RANDOM),
    # MBO CoVID round 1
    ('mbo_BD', utils.LGB),
    ('mbo_CDR12', utils.LGB),
    ('mbo_CDR123', utils.LGB),
    ('mbo_CDR3', utils.LGB),
    # MBO CoVID round 2
    ('mbo_cnn', utils.CNN),
    ('mbo_dbr', DBR),
    ('mbo_lgb', utils.LGB),
))

ROUND_MAPPING = immutabledict.immutabledict({
    0: 'Round 1',
    1: 'Round 2',
    2: 'Round 3',
})


ROUND0_SOURCE_GROUP_MAPPING = immutabledict.immutabledict({
    'singles': 'Singles',
    'cdrh12_multies1': 'VAE-1 / CDR12',
    'cdrh12_multies2': 'VAE-2 / CDR12',
    'cdrh12_random': 'VAE-R / CDR12',
    'cdr3_multies1': 'VAE-1 / CDR3',
    'cdr3_multies2': 'VAE-2 / CDR3',
    'cdr3_random': 'VAE-R / CDR3',
})


# cdr(12|3)_singles: singles of round 0 sequences that mutated CDR12 or CDR3.
# cdr123_best: recombinations of round 0 sequences that mutate CDR12 and CDR3.
# cdr123_singles: single mutations on top of cdr123_best.
# For more details: https://chat.google.com/room/AAAA7eHa1F0/39JK0C8uwU4
ROUND1_SOURCE_GROUP_MAPPING = immutabledict.immutabledict({
    'singles': 'Singles',
    'cdr12_singles_IMGT': 'Singles (R1 / CDR12)',
    'cdr3_singles_IMGT': 'Singles (R1 / CDR3)',
    'cdr123_best_IMGT': 'Recombinants',
    'cdr123_singles_IMGT': 'Singles of recombinants',
    'mbo_CDR12': 'LGB / CDR12',
    'mbo_CDR3': 'LGB / CDR3',
    'mbo_CDR123': 'LGB / CDR123',
    'mbo_BD': 'LGB / Binding domain',
    'shuffled_parent_CDR1': 'Shuffled / CDR1',
    'shuffled_parent_CDR2': 'Shuffled / CDR2',
    'shuffled_parent_CDR3': 'Shuffled / CDR3',
})

ROUND2_SOURCE_GROUP_MAPPING = immutabledict.immutabledict({
    'baseline_r0': 'Baseline (Round 1)',
    'baseline_r1': 'Baseline (Round 2)',
    'mbo_lgb': 'LGB',
    'mbo_cnn': 'CNN',
})

ALLOWED_POS = ('27', '28', '29', '30', '35', '36', '37', '38', '40', '42', '49',
               '52', '55', '56', '57', '58', '59', '62', '63', '64', '65', '66',
               '68', '69', '80', '87', '96', '101', '105', '106', '107', '108',
               '109', '110', '111', '111A', '111B', '112C', '112B', '112A',
               '112', '113', '114', '115', '116', '117')


# Contains all data associated with the paper.
DATA_DIR = 'gs://gresearch/covid_vhh_design'


def load_df(basename, compression = 'infer'):
  """Loads a DataFrame from the data directory."""
  return helper.read_csv(
      os.path.join(DATA_DIR, basename), compression=compression
  )


def load_aligned_parent_seq(offset_ipos = 1):
  """Loads the IMGT aligned parent sequence."""
  return load_df('parent_seq.csv').assign(
      pos=lambda df: df['imgt'],
      ipos=lambda df: np.arange(len(df)) + offset_ipos,
  )


def load_alphaseq_data(round_idx):
  """Loads raw alphaseq data from disk."""
  return load_df(f'round{round_idx}.csv.gz')


def annotate_alphaseq_data(raw_data):
  return utils.annotate_alphaseq_data(
      raw_data=raw_data,
      parent_seq=PARENT_SEQ,
      parent_df=load_aligned_parent_seq(offset_ipos=0),
      allowed_pos=ALLOWED_POS,
      model_by_group=MODEL_BY_GROUP,
      design_by_group=DESIGN_BY_GROUP,
      target_short_name_mapping=TARGET_SHORT_NAME_MAPPING,
  )


def filter_alphaseq_data(
    raw_data):
  """Drops replica 1 in 2nd library, DBR sequences, and invalid mutations."""
  raw_data[1] = raw_data[1][raw_data[1]['replica'] != 1].copy()
  raw_data[2] = raw_data[2][raw_data[2]['source_group'] != 'mbo_dbr'].copy()
  return tree.map_structure(lambda df: df[~df['has_invalid_mutation']].copy(),
                            raw_data)
