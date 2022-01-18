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

r"""Generate fasta-formatted files for phmmer.py and hmmer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Callable, Text, Union

from absl import logging
import pandas as pd
import pfam_utils
import tensorflow.compat.v1 as tf
import tqdm


HMMER_TRAINSEQS_DIRNAME = 'train_alignments'

TESTSEQS_FILENAME = 'all_testseqs.fasta'

PHMMER_TRAINSEQS_FILENAME = 'all_trainseqs.fasta'

TMP_TABLE_NAME = 'seed_train'
FAMILY_ACCESSION_KEY = 'family_accession'


def _fasta_yielder(name_for_family, dataframe_for_family,
                   write_aligned_sequences):
  """Yield fasta entries.

  Fasta entries take the form:
  >sequence_name_family_name
  sequence

  Args:
    name_for_family: string. accession id.
    dataframe_for_family: pandas DataFrame, containing column sequence_name.  If
      write_aligned_sequences is True, the DataFrame must contain column
      aligned_sequence. If write_aligned_sequences is False, the DataFrame must
      contain column sequence.
    write_aligned_sequences: whether to write aligned sequence data, or to write
      unaligned sequence data.

  Yields:
    string. Fasta entry with two lines.
  """
  for protein in dataframe_for_family.itertuples():
    if write_aligned_sequences:
      sequence_to_write = protein.aligned_sequence
    else:
      sequence_to_write = protein.sequence

    yield ('>' + protein.sequence_name + '_' + name_for_family + '\n' +
           sequence_to_write + '\n')


def _write_groups_to_fasta_files(all_families, output_directory,
                                 write_aligned_sequences):
  """Write a fasta file for each family containing aligned sequences.

  The output directory and all parents are recursively created.

  Args:
    all_families: pandas DataFrame containing columns FAMILY_ACCESSION_KEY and
      sequence_name.  If write_aligned_sequences is True, the DataFrame must
      contain column aligned_sequence. If write_aligned_sequences is False, the
      DataFrame must contain column sequence.
    output_directory: string. Directory where fasta files should be written.
    write_aligned_sequences: whether to write aligned sequence data, or to write
      unaligned sequence data.
  """
  tf.io.gfile.MakeDirs(output_directory)
  logging.info('Writing all families to separate files as aligned sequences.')

  grouped_families = all_families.groupby(FAMILY_ACCESSION_KEY)
  for grouping_key, dataframe_for_family in tqdm.tqdm(grouped_families):
    output_file_name = os.path.join(output_directory, grouping_key) + '.fasta'
    with tf.io.gfile.GFile(output_file_name, 'w') as output_file:
      for entry in _fasta_yielder(grouping_key, dataframe_for_family,
                                  write_aligned_sequences):
        output_file.write(entry)


def _write_seqs_to_fasta_file(all_families, output_file_name,
                              write_aligned_sequences):
  """Write a fasta file containing all sequences.

  The output directory and all parents are recursively created.

  Args:
    all_families: pandas DataFrame containing columns family_accession and
      sequence_name.  If write_aligned_sequences is True, the DataFrame must
      contain column aligned_sequence. If write_aligned_sequences is False, the
      DataFrame must contain column sequence.
    output_file_name: string. File where the fasta file should be written.
    write_aligned_sequences: whether to write aligned sequence data, or to write
      unaligned sequence data.
  """
  print(list(all_families))
  with tf.io.gfile.GFile(output_file_name, 'w') as output_file:
    for i in tqdm.tqdm(all_families.index):
      if write_aligned_sequences:
        sequence_to_write = all_families.loc[i, 'aligned_sequence']
      else:
        sequence_to_write = all_families.loc[i, 'sequence']
      output_file.write('>' + all_families.loc[i, 'sequence_name'] + '_' +
                        all_families.loc[i, 'family_accession'] + '\n' +
                        sequence_to_write + '\n')


def run(train_proteins_protos_path,
        test_proteins_protos_path,
        output_directory,
        custom_train_proto_postprocessing_fn = None):
  """Write the fasta files needed to run phmmer.py and hmmer.py.

  Args:
    train_proteins_protos_path:  Path to train input protos. Globs are allowed.
      ColumnIO of blundel_pb2.PFamProtein only.
    test_proteins_protos_path:  Path to test input protos. Globs are allowed.
      ColumnIO of blundel_pb2.PFamProtein only.
    output_directory: string. Directory where fasta files should be written.
    custom_train_proto_postprocessing_fn: function that optionally processes
      the training data after it's read from train_proteins_protos_path, but
      before it's written as a FASTA. If None, no transformation is done.
  """
  connection = pfam_utils.connection_with_tmp_table(TMP_TABLE_NAME,
                                                    train_proteins_protos_path)
  all_train_families = pfam_utils.get_all_rows_from_table(
      connection=connection, table_name=TMP_TABLE_NAME)
  if custom_train_proto_postprocessing_fn:
    logging.info('Applying custom train proto postprocessing function.')
    all_train_families = custom_train_proto_postprocessing_fn(
        all_train_families)

  logging.info('Making training data for hmmer: write train aligned sequences '
               'to family specific fasta files.')
  aligned_train_sequences_output_dir = os.path.join(output_directory,
                                                    HMMER_TRAINSEQS_DIRNAME)
  _write_groups_to_fasta_files(
      all_families=all_train_families,
      output_directory=aligned_train_sequences_output_dir,
      write_aligned_sequences=True)

  logging.info('Making sequence database (training data for phmmer): '
               'write train sequences to a single unaligned fasta file.')
  output_train_file = os.path.join(output_directory, PHMMER_TRAINSEQS_FILENAME)
  _write_seqs_to_fasta_file(
      all_families=all_train_families,
      output_file_name=output_train_file,
      write_aligned_sequences=False)

  logging.info('Making the test data for both hmmer and phmmer:'
               'write test sequences to a single unaligned fasta file.')
  connection = pfam_utils.connection_with_tmp_table(TMP_TABLE_NAME,
                                                    test_proteins_protos_path)
  all_test_families = pfam_utils.get_all_rows_from_table(
      connection, TMP_TABLE_NAME)

  output_test_file = os.path.join(output_directory, TESTSEQS_FILENAME)
  _write_seqs_to_fasta_file(
      all_families=all_test_families,
      output_file_name=output_test_file,
      write_aligned_sequences=False)
