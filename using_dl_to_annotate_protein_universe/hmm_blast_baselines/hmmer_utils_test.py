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

"""Tests for module inference.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import hmmer_utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import test_util
import util

FLAGS = flags.FLAGS

# Made by running hmmsearch --tblout pfam_output/PF00131.19.txt
# pfam_hmm/PF00131.19.hmm testseqs.fasta
_HMMER_TBLOUT = """
#                                                                          --- full sequence ---- --- best 2 domain ---- --- domain number estimation ----
# target name                   accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target
#           ------------------- ---------- -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------
MT4_CANLF/1-62_PF00131.19       -          PF00131.19           -            1.3e-15   60.9  58.4   1.4e-15   60.8  58.4   1.0   1   0   0   1   1   1   1 -
E4X7F8_OIKDI/453-561_PF05033.15 -          PF00131.19           -                0.6   13.9   5.0       0.6   13.9   5.0   2.3   2   0   0   2   2   2   0 -
#
# Program:         hmmsearch
# Version:         3.1b2 (February 2015)
# Pipeline mode:   SEARCH
# Query file:      pfam_hmm/PF00131.19.hmm
# Target file:     testseqs.fasta
# Option settings: hmmsearch --tblout pfam_output/PF00131.19.txt pfam_hmm/PF00131.19.hmm testseqs.fasta
# Date:            Sat Oct 20 12:26:56 2018
# [ok]
"""

_HMMER_TBLOUT_NO_OUTPUT = """
#                                                                          --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----
# target name                   accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target
#           ------------------- ---------- -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------
#
# Program:         hmmsearch
# Version:         3.1b2 (February 2015)
# Pipeline mode:   SEARCH
# Query file:      pfam_hmm/PF00131.19.hmm
# Target file:     testseqs.fasta
# Option settings: hmmsearch --tblout pfam_output/PF00131.19.txt pfam_hmm/PF00131.19.hmm testseqs.fasta
# Date:            Sat Oct 20 12:26:56 2018
# [ok]
"""


class HMMerUtilsTest(parameterized.TestCase):

  def testGetFamilyNameFromUnderscores(self):
    # Sequence name contains underscores.
    actual = hmmer_utils.get_family_name_from('V2R_HUMAN/54-325_PF00001.20')
    expected = 'PF00001.20'
    self.assertEqual(actual, expected)

  def testGetFamilyNameFromNoUnderscores(self):
    # Sequence name has no underscores.
    actual = hmmer_utils.get_family_name_from('Q12345_alanine')
    expected = 'alanine'
    self.assertEqual(actual, expected)

  def testGetSequenceNameFrom(self):
    actual = hmmer_utils.get_sequence_name_from('V2R_HUMAN/54-325_PF00001.20')
    expected = 'V2R_HUMAN/54-325'
    self.assertEqual(actual, expected)

  def testFormatAsCsvHmmerOutput(self):
    hmmer_output = hmmer_utils.HMMEROutput(
        sequence_name='MT4_CANLF/1-62',
        predicted_label='PF00131.19',
        true_label='PF12345.6',
        score=60.9,
        domain_evalue=1.3e-15,
    )

    # order should match hmmer_utils.HMMER_OUTPUT_CSV_COLUMN_HEADERS.
    # (That is, util.PREDICTION_FILE_COLUMN_NAMES + [DATAFRAME_SCORE_NAME_KEY].)
    expected = 'MT4_CANLF/1-62,PF12345.6,PF00131.19,60.9,1.3e-15'
    actual = hmmer_output.format_as_csv()

    self.assertEqual(actual, expected)

  def testParseHmmOutput(self):
    actual_hmmsearch = list(
        hmmer_utils.parse_hmmer_output(_HMMER_TBLOUT, 'PF00131.19'))
    print(actual_hmmsearch)
    expected_hmmsearch = [
        hmmer_utils.HMMEROutput(
            sequence_name='MT4_CANLF/1-62',
            predicted_label='PF00131.19',
            true_label='PF00131.19',
            score=60.9,
            domain_evalue=1.3e-15,
        ),
        hmmer_utils.HMMEROutput(
            sequence_name='E4X7F8_OIKDI/453-561',
            predicted_label='PF00131.19',
            true_label='PF05033.15',
            score=13.9,
            domain_evalue=0.6,
        ),
    ]

    self.assertEqual(actual_hmmsearch, expected_hmmsearch)
    # Test that for the output file with no hits, the sentinel value is written.
    actual_no_output = list(
        hmmer_utils.parse_hmmer_output(_HMMER_TBLOUT_NO_OUTPUT, 'PF00131.19'))
    print(actual_no_output)
    expected_no_output = [
        hmmer_utils.HMMEROutput(
            sequence_name='no_sequence/0-0',
            predicted_label='PF00131.19',
            true_label='PF00000.0',
            score=hmmer_utils.NO_SEQUENCE_MATCH_SCORE_SENTINEL,
            domain_evalue=hmmer_utils.NO_SEQUENCE_MATCH_DOMAIN_EVALUE_SENTINEL,
        )
    ]

    self.assertEqual(actual_no_output, expected_no_output)

  def testFilterFastaFileBySequenceName(self):
    fasta_file_contents = ('>A0A0F7V1V9_TOXGV/243-280_PF10417.9\n'
                           'MIREVEKNGGKQVCPANWRRGEKMMHASFEGVKNYLGQ\n'
                           '>Q1QPP6_NITHX/169-202_PF10417.9\n'
                           'ALQATMSGQKLAPANWQPGETLLLPADEKTQKDT\n'
                           '>Q2K9A0_RHIEC/165-202_PF10417.9\n'
                           'SIQLTAKHQVATPANWNQGEDVIITAAVSNDDAIARFG\n')
    input_fasta_file_name = test_util.tmpfile('input_fasta')
    with tf.io.gfile.GFile(input_fasta_file_name, 'w') as input_fasta_file:
      input_fasta_file.write(fasta_file_contents)

    actual = list(
        hmmer_utils.filter_fasta_file_by_sequence_name(
            input_fasta_file_name, ['Q2K9A0_RHIEC/165-202']))
    expected = [('>Q2K9A0_RHIEC/165-202_PF10417.9\n'
                 'SIQLTAKHQVATPANWNQGEDVIITAAVSNDDAIARFG\n')]
    self.assertEqual(actual, expected)

  def testAllSequenceNamesFromFastaFile(self):
    fasta_file_contents = ('>A0A0F7V1V9_TOXGV/243-280_PF10417.9\n'
                           'MIREVEKNGGKQVCPANWRRGEKMMHASFEGVKNYLGQ\n'
                           '>Q1QPP6_NITHX/169-202_PF10417.9\n'
                           'ALQATMSGQKLAPANWQPGETLLLPADEKTQKDT\n'
                           '>Q2K9A0_RHIEC/165-202_PF10417.9\n'
                           'SIQLTAKHQVATPANWNQGEDVIITAAVSNDDAIARFG\n')
    input_fasta_file_name = test_util.tmpfile('input_fasta')
    with tf.io.gfile.GFile(input_fasta_file_name, 'w') as input_fasta_file:
      input_fasta_file.write(fasta_file_contents)

    actual = hmmer_utils.all_sequence_names_from_fasta_file(
        input_fasta_file_name)
    expected = [
        'A0A0F7V1V9_TOXGV/243-280', 'Q1QPP6_NITHX/169-202',
        'Q2K9A0_RHIEC/165-202'
    ]

    np.testing.assert_array_equal(actual, expected)

  def testSequencesWithNoPrediction(self):
    hmmer_predictions = pd.DataFrame(
        [['A4YXG4_BRASO/106-134', 'PF00001.1', 'PF00001.1', 0.5]],
        columns=util.PREDICTION_FILE_COLUMN_NAMES)
    all_sequence_names = ['A_DIFFERENTSEQNAME/1-2', 'A4YXG4_BRASO/106-134']
    actual = hmmer_utils.sequences_with_no_prediction(
        all_sequence_names=all_sequence_names,
        hmmer_predictions=hmmer_predictions)
    expected = {'A_DIFFERENTSEQNAME/1-2'}

    self.assertEqual(actual, expected)

  def testYieldTopElByScoreForEachSequenceName(self):
    input_df = pd.DataFrame([
        ['SAME_SEQ_NAME', 'PF00264.20', 'PF000264.20', 10.1, 1e-3],
        ['SAME_SEQ_NAME', 'PF00264.20', 'PF00001.3', -65432.0, 100.],
    ],
                            columns=hmmer_utils.HMMER_OUTPUT_CSV_COLUMN_HEADERS)
    actual = pd.concat(
        list(
            hmmer_utils.yield_top_el_by_score_for_each_sequence_name(input_df)))
    expected = pd.DataFrame(
        [['SAME_SEQ_NAME', 'PF00264.20', 'PF000264.20', 10.1, 1e-3]],
        columns=hmmer_utils.HMMER_OUTPUT_CSV_COLUMN_HEADERS)

    self.assertLen(actual, 1)
    self.assertCountEqual(
        actual.to_dict('records'), expected.to_dict('records'))

  # pylint: disable=line-too-long
  # Disable line-too-long because phmmer output strings are long, and we
  # want to paste them verbatim.
  @parameterized.named_parameters(
      dict(
          testcase_name=('multiple seq outputs, no repeats, same families, all '
                         'identifiers have predictions'),
          phmmer_output="""#                                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----
# target name                accession  query name                   accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target
#        ------------------- ----------         -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------
OPSB_HUMAN/51-303_PF00001.20 -          OPSB_HUMAN/51-303_PF00001.20 -            2.5e-61  193.4  64.0   1.6e-22   66.2   0.9   8.7   8   1   0   8   8   8   8 -
OPS3_DROME/75-338_PF00001.20 -          OPS3_DROME/75-338_PF00001.20 -            2.1e-69  219.9  65.6   3.2e-21   62.0   0.3   9.8  10   1   0  10  10  10  10 -
#
# Program:         phmmer
# Version:         3.1b2 (February 2015)
# Pipeline mode:   SEARCH
# Query file:      /storage/hmm_train/PF00001.20.fasta
# Target file:     /storage/hmm_train/PF00001.20.fasta
# Option settings: phmmer -o /dev/null --tblout /dev/stdout -E 10 /storage/hmm_train/PF00001.20.fasta /storage/hmm_train/PF00001.20.fasta
# Date:            Wed Nov 21 09:26:50 2018
# [ok]
""",
          all_identifiers=[
              'OPSB_HUMAN/51-303_PF00001.20',
              'OPS3_DROME/75-338_PF00001.20',
          ],
          expected=[
              hmmer_utils.HMMEROutput(
                  sequence_name='OPSB_HUMAN/51-303',
                  true_label='PF00001.20',
                  predicted_label='PF00001.20',
                  score=193.4,
                  domain_evalue=2.5e-61,
              ),
              hmmer_utils.HMMEROutput(
                  sequence_name='OPS3_DROME/75-338',
                  true_label='PF00001.20',
                  predicted_label='PF00001.20',
                  score=219.9,
                  domain_evalue=2.1e-69,
              )
          ],
      ),
      dict(
          testcase_name=('one seq output, no repeats, same families, some '
                         'identifiers do not have predictions'),
          phmmer_output="""#                                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----
# target name                accession  query name                   accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target
#        ------------------- ----------         -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------
OPSB_HUMAN/51-303_PF00001.20 -          OPSB_HUMAN/51-303_PF00001.20 -            2.5e-61  193.4  64.0   1.6e-22   66.2   0.9   8.7   8   1   0   8   8   8   8 -
#
# Program:         phmmer
# Version:         3.1b2 (February 2015)
# Pipeline mode:   SEARCH
# Query file:      /storage/hmm_train/PF00001.20.fasta
# Target file:     /storage/hmm_train/PF00001.20.fasta
# Option settings: phmmer -o /dev/null --tblout /dev/stdout -E 10 /storage/hmm_train/PF00001.20.fasta /storage/hmm_train/PF00001.20.fasta
# Date:            Wed Nov 21 09:26:50 2018
# [ok]
""",
          expected=[
              hmmer_utils.HMMEROutput(
                  sequence_name='OPSB_HUMAN/51-303',
                  true_label='PF00001.20',
                  predicted_label='PF00001.20',
                  score=193.4,
                  domain_evalue=2.5e-61,
              ),
              hmmer_utils.HMMEROutput(
                  sequence_name='THIS_ISNOTSEEN/1-111',
                  true_label='PF09876.5',
                  predicted_label='PF00000.0',
                  score=hmmer_utils.NO_SEQUENCE_MATCH_SCORE_SENTINEL,
                  domain_evalue=hmmer_utils.NO_SEQUENCE_MATCH_DOMAIN_EVALUE_SENTINEL,
              )
          ],
          all_identifiers=[
              'OPSB_HUMAN/51-303_PF00001.20',
              'THIS_ISNOTSEEN/1-111_PF09876.5',
          ]),
      dict(
          testcase_name=('one seq output, with repeats, all identifiers have '
                         'predictions'),
          phmmer_output="""#                                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----
# target name                accession  query name                   accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target
#        ------------------- ----------         -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------
SSR1_HUMAN/75-323_PF00001.20  -          CXCR5_HUMAN/68-322_PF00001.20 -              1e-06   14.4   0.9     1e-06   14.4   0.9   2.0   2   0   0   2   2   2   2 -
CX3C1_RAT/49-294_PF00001.20   -          CXCR5_HUMAN/68-322_PF00001.20 -            1.3e-06   14.1   8.7   4.5e-06   12.3   1.7   2.6   3   0   0   3   3   3   2 -

#
# Program:         phmmer
# Version:         3.1b2 (February 2015)
# Pipeline mode:   SEARCH
# Query file:      /storage/hmm_train/PF00001.20.fasta
# Target file:     /storage/hmm_train/PF00001.20.fasta
# Option settings: phmmer -o /dev/null --tblout /dev/stdout -E 10 /storage/hmm_train/PF00001.20.fasta /storage/hmm_train/PF00001.20.fasta
# Date:            Wed Nov 21 09:26:50 2018
# [ok]
""",
          expected=[
              hmmer_utils.HMMEROutput(
                  sequence_name='CXCR5_HUMAN/68-322',
                  true_label='PF00001.20',
                  predicted_label='PF00001.20',
                  score=14.4,
                  domain_evalue=1e-6,
              ),
              hmmer_utils.HMMEROutput(
                  sequence_name='CXCR5_HUMAN/68-322',
                  true_label='PF00001.20',
                  predicted_label='PF00001.20',
                  score=14.1,
                  domain_evalue=1.3e-6,
              )
          ],
          all_identifiers=[
              'CXCR5_HUMAN/68-322_PF00001.20',
          ]),
      dict(
          testcase_name=('no hits for any input sequence'),
          phmmer_output="""
#                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----
# target name        accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target
#------------------- ---------- -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------
#
# Program:         phmmer
# Version:         3.1b2 (February 2015)
# Pipeline mode:   SEARCH
# Query file:      -
# Target file:     all_trainseqs.fasta
# Option settings: phmmer -o /dev/null --tblout /dev/stdout -E 10.0 - all_trainseqs.fasta
# Date:            Mon Oct 29 12:45:57 2018
# [ok]
""",
          all_identifiers=[
              'CXCR5_HUMAN/68-322_PF00001.20',
          ],
          expected=[
              hmmer_utils.HMMEROutput(
                  sequence_name='CXCR5_HUMAN/68-322',
                  true_label='PF00001.20',
                  predicted_label=hmmer_utils.NO_SEQUENCE_MATCH_FAMILY_NAME_SENTINEL,
                  score=hmmer_utils.NO_SEQUENCE_MATCH_SCORE_SENTINEL,
                  domain_evalue=hmmer_utils.NO_SEQUENCE_MATCH_DOMAIN_EVALUE_SENTINEL,
              ),
          ],
      ),
  )
  def testParsePhmmer(self, phmmer_output, all_identifiers, expected):
    actual = hmmer_utils.parse_phmmer_output(phmmer_output, all_identifiers)
    self.assertEqual(actual, expected)
  # pylint: enable=line-too-long

if __name__ == '__main__':
  absltest.main()
