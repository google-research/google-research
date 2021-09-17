# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# pylint: disable=line-too-long
# Line-too-long disabled because of the example command.
r"""Run generate input files for, and then train and test, hmmer or phmmer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tempfile
from typing import Callable, Text, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import blundell_constants
import generate_hmmer_files
import hmmer
import pandas as pd
import phmmer
import util as classification_util

FAMILY_ACCESSION_DF_KEY = 'family_accession'

TEST_FOLD = 'test'
DEV_FOLD = 'dev'
TRAIN_FOLD = 'train'
FOLD_ENUM_VAULES = [TRAIN_FOLD, DEV_FOLD, TEST_FOLD]

CLUSTERED_SPLIT = 'clustered'
RANDOM_SPLIT = 'random'
RANDOM_ONE_SHOT_LEARNING_SPLIT = 'random_one_shot_learning'
CLUSTERED_ONE_SHOT_LEARNING_SPLIT = 'clustered_one_shot_learning'
SPLIT_ENUM_VALUES = [
    RANDOM_SPLIT, CLUSTERED_SPLIT, RANDOM_ONE_SHOT_LEARNING_SPLIT,
    CLUSTERED_ONE_SHOT_LEARNING_SPLIT
]

SPLIT_ENUM_TO_DATASET_SPLIT = {
    RANDOM_SPLIT: RANDOM_SPLIT,
    CLUSTERED_SPLIT: CLUSTERED_SPLIT,
    RANDOM_ONE_SHOT_LEARNING_SPLIT: RANDOM_SPLIT,
    CLUSTERED_ONE_SHOT_LEARNING_SPLIT: CLUSTERED_SPLIT
}

_HMMER_STR = 'hmmer'
_PHMMER_STR = 'phmmer'
_BINARY_CHOICE_ENUM_VALUES = [_HMMER_STR, _PHMMER_STR]

flags.DEFINE_enum(
    'test_fold',
    DEV_FOLD,
    FOLD_ENUM_VAULES,
    'Which fold to run on.',
)

flags.DEFINE_enum(
    'split',
    RANDOM_SPLIT,
    SPLIT_ENUM_VALUES,
    'Which split to run on.',
)

flags.DEFINE_enum(
    'hmmer_or_phmmer',
    _HMMER_STR,
    _BINARY_CHOICE_ENUM_VALUES,
    'Whether to run hmmer or phmmer',
)

FLAGS = flags.FLAGS

_FOLD_TO_TEST_PROTOS_RANDOM_SPLIT_PATH_MAPPING = {
    TRAIN_FOLD: blundell_constants.RANDOM_SPLIT_SEED_TRAINING_DATA_PATH,
    DEV_FOLD: blundell_constants.RANDOM_SPLIT_SEED_DEV_DATA_PATH,
    TEST_FOLD: blundell_constants.RANDOM_SPLIT_SEED_TEST_DATA_PATH,
}

_FOLD_TO_TEST_PROTOS_CLUSTERED_SPLIT_PATH_MAPPING = {
    TRAIN_FOLD: blundell_constants.CLUSTERED_SPLIT_SEED_TRAINING_DATA_PATH,
    DEV_FOLD: blundell_constants.CLUSTERED_SPLIT_SEED_DEV_DATA_PATH,
    TEST_FOLD: blundell_constants.CLUSTERED_SPLIT_SEED_TEST_DATA_PATH,
}

# Uses same data as random split, with additional filtering based on
# family size.
_FOLD_TO_TEST_PROTOS_RANDOM_ONE_SHOT_LEARNING_SPLIT_PATH_MAPPING = _FOLD_TO_TEST_PROTOS_RANDOM_SPLIT_PATH_MAPPING

# Uses same data as clustered split, with additional filtering based on
# family size.
_FOLD_TO_TEST_PROTOS_CLUSTERED_ONE_SHOT_LEARNING_SPLIT_PATH_MAPPING = _FOLD_TO_TEST_PROTOS_CLUSTERED_SPLIT_PATH_MAPPING


def train_and_test_paths_from(split, fold):
  """Return train_path, test_path for running hmmer/phmmer on a split and fold.

  Args:
    split: string. One of SPLIT_ENUM_VALUES.
    fold: string. One of FOLD_ENUM_VALUES.

  Returns:
    str, str. The first is the path to columnio files containing protein protos
    for the train data, and the second is the same, except for the test data.

  Raises:
    ValueError: if split is not in SPLIT_ENUM_VALUES.
    ValueError: if fold is not in FOLD_ENUM_VALUES.
  """
  if split == RANDOM_SPLIT:
    train_proteins_protos_path = (
        blundell_constants.RANDOM_SPLIT_SEED_TRAINING_DATA_PATH)

    if fold in _FOLD_TO_TEST_PROTOS_RANDOM_SPLIT_PATH_MAPPING:
      test_proteins_protos_path = (
          _FOLD_TO_TEST_PROTOS_RANDOM_SPLIT_PATH_MAPPING[fold])
    else:
      raise ValueError('Unknown fold {}. Expected one of {}'.format(
          fold, FOLD_ENUM_VAULES))

  elif split == CLUSTERED_SPLIT:
    train_proteins_protos_path = (
        blundell_constants.CLUSTERED_SPLIT_SEED_TRAINING_DATA_PATH)

    if fold in _FOLD_TO_TEST_PROTOS_CLUSTERED_SPLIT_PATH_MAPPING:
      test_proteins_protos_path = (
          _FOLD_TO_TEST_PROTOS_CLUSTERED_SPLIT_PATH_MAPPING[fold])
    else:
      raise ValueError('Unknown fold {}. Expected one of {}'.format(
          fold, FOLD_ENUM_VAULES))

  elif split == RANDOM_ONE_SHOT_LEARNING_SPLIT:
    train_proteins_protos_path = (
        blundell_constants.RANDOM_SPLIT_SEED_TRAINING_DATA_PATH)

    if fold in _FOLD_TO_TEST_PROTOS_RANDOM_ONE_SHOT_LEARNING_SPLIT_PATH_MAPPING:
      test_proteins_protos_path = (
          _FOLD_TO_TEST_PROTOS_RANDOM_ONE_SHOT_LEARNING_SPLIT_PATH_MAPPING[fold]
      )
    else:
      raise ValueError('Unknown fold {}. Expected one of {}'.format(
          fold, FOLD_ENUM_VAULES))
  elif split == CLUSTERED_ONE_SHOT_LEARNING_SPLIT:
    train_proteins_protos_path = (
        blundell_constants.CLUSTERED_SPLIT_SEED_TRAINING_DATA_PATH)

    if fold in _FOLD_TO_TEST_PROTOS_CLUSTERED_ONE_SHOT_LEARNING_SPLIT_PATH_MAPPING:
      test_proteins_protos_path = (
          _FOLD_TO_TEST_PROTOS_CLUSTERED_ONE_SHOT_LEARNING_SPLIT_PATH_MAPPING[
              fold])
    else:
      raise ValueError('Unknown fold {}. Expected one of {}'.format(
          fold, FOLD_ENUM_VAULES))
  else:
    raise ValueError('Unknown split {}. Expected one of {}.'.format(
        split, SPLIT_ENUM_VALUES))

  return train_proteins_protos_path, test_proteins_protos_path


def _large_fams_and_one_unaligned_seq_from_small_fams(
    df, dataset_split):
  """Filters a DataFrame to make amenable for one-shot learning.

  Specifically:
    - Take all training elements of large families, including alignments.
    - Take only the first element of the small families. Discard alignments,
      because the alignments include a lot of information from the rest of the
      family, which we are explicitly trying to exclude for one-shot learning.

  Args:
    df: pd.DataFrame with columns 'family_accession', 'aligned_sequence', and
      'sequence'.
    dataset_split: The name of the data split.
  Returns:
    pd.DataFrame.
  """
  names_of_small_families, names_of_large_families = (
      classification_util.names_of_small_and_large_families(dataset_split))

  large_families = df[df[FAMILY_ACCESSION_DF_KEY].isin(names_of_large_families)]
  small_families = df[df[FAMILY_ACCESSION_DF_KEY].isin(names_of_small_families)]

  fewshot_small_families = (
      small_families.groupby(FAMILY_ACCESSION_DF_KEY).head(1))

  # Remove alignment info.
  fewshot_small_families.aligned_sequence = fewshot_small_families.sequence

  return pd.concat([large_families, fewshot_small_families], ignore_index=True)


def custom_train_proto_postprocessing_fn(
    split):
  """Creates a function for postprocessing training data.

  Returns None for any split except ONE_SHOT_LEARNING_SPLIT.
  See _large_fams_and_one_unaligned_seq_from_small_fams for more information.

  Args:
    split: one of SPLIT_ENUM_VALUES.

  Returns:
    None or function from DataFrame to DataFrame.
  """
  if split not in [
      RANDOM_ONE_SHOT_LEARNING_SPLIT, CLUSTERED_ONE_SHOT_LEARNING_SPLIT
  ]:
    return None

  dataset_split = SPLIT_ENUM_TO_DATASET_SPLIT[split]
  return functools.partial(
      _large_fams_and_one_unaligned_seq_from_small_fams,
      dataset_split=dataset_split)


def generate_input_files(split, fold):
  """Write files for hmmer and phmmer for a given split and fold.

  Args:
    split: string. One of SPLIT_ENUM_VALUES.
    fold: string. One of FOLD_ENUM_VALUES.

  Returns:
    string. Path to a directory where the input files for hmmer and phmmer
    input.
  """
  train_proteins_protos_path, test_proteins_protos_path = (
      train_and_test_paths_from(split=split, fold=fold))

  output_directory = tempfile.mkdtemp(
      'train_and_test_files_{split}_{fold}'.format(split=split, fold=fold))
  logging.info('Output directory for fasta files is %s', output_directory)

  generate_hmmer_files.run(
      train_proteins_protos_path=train_proteins_protos_path,
      test_proteins_protos_path=test_proteins_protos_path,
      output_directory=output_directory,
      custom_train_proto_postprocessing_fn=custom_train_proto_postprocessing_fn(
          split=split))
  return output_directory


def run_hmmer(split, fold, fasta_files_root_dir):
  """Run hmmer on aligned train files, and test on unaligned sequences.

  Args:
    split: string. One of SPLIT_ENUM_VALUES.
    fold: string. One of FOLD_ENUM_VALUES.
    fasta_files_root_dir: string. Return value of generate_input_files.
  """
  train_align_dir = os.path.join(fasta_files_root_dir,
                                 generate_hmmer_files.HMMER_TRAINSEQS_DIRNAME)
  test_sequence_file = os.path.join(fasta_files_root_dir,
                                    generate_hmmer_files.TESTSEQS_FILENAME)

  hmm_output_dir = tempfile.mkdtemp('hmm_output_dir_{split}_{fold}'.format(
      split=split, fold=fold))
  _, output_file_name = tempfile.mkstemp(
      prefix='hmmer_output_{split}_{fold}'.format(split=split, fold=fold),
      suffix='.csv')
  logging.info('Saving computed hmms to %s and final output to %s',
               hmm_output_dir, output_file_name)

  hmmer.run_hmmer_on_data(
      train_align_dir=train_align_dir,
      hmm_dir=hmm_output_dir,
      test_sequence_file=test_sequence_file,
      parsed_output=os.path.abspath(output_file_name))


def run_phmmer(split, fold, fasta_files_root_dir):
  """Run phmmer given unaligned train files, and unaligned test sequences.

  Args:
    split: string. One of SPLIT_ENUM_VALUES.
    fold: string. One of FOLD_ENUM_VALUES.
    fasta_files_root_dir: string. Return value of generate_input_files.
  """
  train_align_path = os.path.join(
      fasta_files_root_dir, generate_hmmer_files.PHMMER_TRAINSEQS_FILENAME)
  test_sequence_file = os.path.join(fasta_files_root_dir,
                                    generate_hmmer_files.TESTSEQS_FILENAME)
  _, output_file_name = tempfile.mkstemp(
      prefix='phmmer_output_{split}_{fold}'.format(split=split, fold=fold),
      suffix='.csv')

  logging.info('Saving final output to %s', output_file_name)
  phmmer.write_phmmer_predictions(
      train_sequence_file=train_align_path,
      test_sequence_file=test_sequence_file,
      parsed_output=os.path.abspath(output_file_name))


def run(split, fold, hmmer_or_phmmer):
  """Run phmmer or hmmer given a fold and split.

  Args:
    split: string. One of SPLIT_ENUM_VALUES.
    fold: string. One of FOLD_ENUM_VALUES.
    hmmer_or_phmmer: string. One of _BINARY_CHOICE_ENUM_VALUES.

  Raises:
    ValueError: if hmmer_or_phmmer is not one of _BINARY_CHOICE_ENUM_VALUES.
  """
  logging.info('Generating input files for hmmer/phmmer.')
  fasta_files_root_dir = generate_input_files(split=split, fold=fold)

  if hmmer_or_phmmer == _HMMER_STR:
    logging.info('Running hmmer.')
    run_hmmer(split=split, fold=fold, fasta_files_root_dir=fasta_files_root_dir)
  elif hmmer_or_phmmer == _PHMMER_STR:
    logging.info('Running phmmer.')
    run_phmmer(
        split=split, fold=fold, fasta_files_root_dir=fasta_files_root_dir)
  else:
    raise ValueError(
        'Unexpected value for hmmer_or_phmmer {}'.format(hmmer_or_phmmer))


def main(_):
  run(split=FLAGS.split,
      fold=FLAGS.test_fold,
      hmmer_or_phmmer=FLAGS.hmmer_or_phmmer)


if __name__ == '__main__':
  app.run(main)
