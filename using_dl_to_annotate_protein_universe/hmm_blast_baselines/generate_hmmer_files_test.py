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

"""Tests for generate_hmmer_files.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import generate_hmmer_files
import pfam_utils
import tensorflow.compat.v1 as tf

TEST_FIXTURE_TABLE_NAME = "test_table_fixture"

FLAGS = flags.FLAGS


class TestGenerateHmmerTrainFilesTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          write_aligned_sequences=True,
          expected_output_first_file=">FIRST_ONE/1-2_PF12345.6\nA\n",
          expected_output_second_file=">GOOD_SEQ/5-9_PF98765.4\nYYYY\n"
          ">BEST_SEQ/1-5_PF98765.4\nP.YY\n"),
      dict(
          write_aligned_sequences=False,
          expected_output_first_file=">FIRST_ONE/1-2_PF12345.6\nA\n",
          expected_output_second_file=">GOOD_SEQ/5-9_PF98765.4\nYYYY\n"
          ">BEST_SEQ/1-5_PF98765.4\nPYY\n"),
  )
  def testWriteGroupsToFastaFiles(self, write_aligned_sequences,
                                  expected_output_first_file,
                                  expected_output_second_file):
    actual_dir_name = os.path.join(
        absltest.get_default_test_tmpdir(),
        generate_hmmer_files.HMMER_TRAINSEQS_DIRNAME,
        "write_aligned_" + str(write_aligned_sequences))
    all_rows = pfam_utils.get_all_rows_from_table(self.connection,
                                                  TEST_FIXTURE_TABLE_NAME)
    generate_hmmer_files._write_groups_to_fasta_files(
        all_families=all_rows,
        output_directory=actual_dir_name,
        write_aligned_sequences=write_aligned_sequences)

    actual_file_names = tf.io.gfile.ListDir(actual_dir_name)

    expected_file_names = ["PF12345.6.fasta", "PF98765.4.fasta"]
    self.assertListEqual(actual_file_names, expected_file_names)

    with tf.io.gfile.Open(
        os.path.join(actual_dir_name, actual_file_names[0]),
        "r") as first_fasta_file_written:
      self.assertEqual(first_fasta_file_written.read(),
                       expected_output_first_file)

    with tf.io.gfile.Open(
        os.path.join(actual_dir_name, actual_file_names[1]),
        "r") as second_fasta_file_written:
      self.assertEqual(second_fasta_file_written.read(),
                       expected_output_second_file)

  @parameterized.parameters(
      dict(
          write_aligned_sequences=True,
          expected_output_single_file=">FIRST_ONE/1-2_PF12345.6\nA\n"
          ">GOOD_SEQ/5-9_PF98765.4\nYYYY\n"
          ">BEST_SEQ/1-5_PF98765.4\nP.YY\n"),
      dict(
          write_aligned_sequences=False,
          expected_output_single_file=">FIRST_ONE/1-2_PF12345.6\nA\n"
          ">GOOD_SEQ/5-9_PF98765.4\nYYYY\n"
          ">BEST_SEQ/1-5_PF98765.4\nPYY\n"),
  )
  def testWriteSeqsToFastaFile(self, write_aligned_sequences,
                               expected_output_single_file):

    tmp_dir = os.path.join(absltest.get_default_test_tmpdir(),
                           generate_hmmer_files.HMMER_TRAINSEQS_DIRNAME,
                           "write_aligned_" + str(write_aligned_sequences))
    output_file_name = os.path.join(tmp_dir, "test.fasta")
    all_rows = pfam_utils.get_all_rows_from_table(self.connection,
                                                  TEST_FIXTURE_TABLE_NAME)
    generate_hmmer_files._write_seqs_to_fasta_file(
        all_families=all_rows,
        output_file_name=output_file_name,
        write_aligned_sequences=write_aligned_sequences)

    with tf.io.gfile.Open(output_file_name, "r") as fasta_file_written:
      self.assertEqual(fasta_file_written.read(), expected_output_single_file)


if __name__ == "__main__":
  absltest.main()
