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

"""Tests for utils.py."""


import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from protenn import utils


def make_tmpfile(filename):
  return os.path.join(
      tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir()),
      filename,
  )


class TestTensorUtils(parameterized.TestCase):

  @parameterized.parameters(
      dict(input_iterable=[], batch_size=1, expected=[]),
      dict(input_iterable=[], batch_size=2, expected=[]),
      dict(input_iterable=[1], batch_size=1, expected=[[1]]),
      dict(input_iterable=[1], batch_size=2, expected=[[1]]),
      dict(input_iterable=[1, 2], batch_size=1, expected=[[1], [2]]),
      dict(input_iterable=[1, 2], batch_size=2, expected=[[1, 2]]),
      dict(input_iterable=[1, 2, 3], batch_size=2, expected=[[1, 2], [3]]),
      dict(
          input_iterable=[1, 2, 3, 4], batch_size=2, expected=[[1, 2], [3, 4]]),
  )
  def testBatchIterable(self, input_iterable, batch_size, expected):
    actual = list(utils.batch_iterable(input_iterable, batch_size))

    self.assertEqual(actual, expected)

  def testSparseToOneHot(self):
    seq = 'AY'
    expected_output = [[
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0.
    ],
                       [
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 1.
                       ]]
    self.assertListEqual(expected_output,
                         utils.residues_to_one_hot(seq).tolist())

  @parameterized.named_parameters(
      dict(
          testcase_name='pad by nothing',
          input_one_hot=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ]]),
          pad_length=1,
          expected=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ]])),
      dict(
          testcase_name='pad with one element',
          input_one_hot=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ]]),
          pad_length=2,
          expected=np.array([[
              1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0.
          ],
                             [
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0.
                             ]])),
  )
  def testPadOneHotSameLength(self, input_one_hot, pad_length, expected):
    actual = utils.pad_one_hot(input_one_hot, pad_length)

    self.assertTrue(
        np.allclose(actual, expected),
        msg='Actual: ' + str(actual) + '\nExpected: ' + str(expected))

  def testFamilyToClanMapping(self):
    input_tsv_file_name = make_tmpfile('clans_pfam35.tsv')
    with tf.io.gfile.GFile(input_tsv_file_name, 'w') as f:
      # pylint: disable=line-too-long
      f.write(
          'PF00001\tCL0192\tGPCR_A\t7tm_1\t7 transmembrane receptor (rhodopsin'
          ' family)\n'
      )
      f.write(
          'PF00002\tCL0192\tGPCR_A\t7tm_2\t7 transmembrane receptor (Secretin'
          ' family)\n'
      )
      f.write(
          'PF00004\tCL0023\tP-loop_NTPase\tAAA\tATPase family associated with'
          ' various cellular activities (AAA)\n'
      )
      # pylint: enable=line-too-long

    actual_mapping = utils.family_to_clan_mapping(
        model_cache_path=os.path.dirname(input_tsv_file_name)
    )

    expected_mapping = {
        'PF00001': 'CL0192',
        'PF00002': 'CL0192',
        'PF00004': 'CL0023',
    }
    self.assertDictEqual(actual_mapping, expected_mapping)

  def testFamilyToClanMappingLiftedClanSemantics(self):
    input_tsv_file_name = make_tmpfile('clans_pfam35.tsv')
    with tf.io.gfile.GFile(input_tsv_file_name, 'w') as f:
      # pylint: disable=line-too-long
      f.write(
          'PF00001\tCL0192\tGPCR_A\t7tm_1\t7 transmembrane receptor (rhodopsin'
          ' family)\n'
      )
      # Notice empty string for clan value.
      f.write('PF99999\t\tWEIRD_FAM\tUNUSED\tUNUSED\n')
      # pylint: enable=line-too-long

    actual_mapping = utils.family_to_clan_mapping(
        model_cache_path=os.path.dirname(input_tsv_file_name),
        use_lifted_clan_semantics=True,
    )

    expected_mapping = {
        'PF00001': 'CL0192',
        'PF99999': 'PF99999',
    }
    self.assertDictEqual(actual_mapping, expected_mapping)

  def test_biologist_and_programmer_range(self):
    programmer_start = 0
    programmer_end = 42

    # Convert back and forth and assert we got what we started with.
    biologist_start, biologist_end = utils.programmer_range_to_biologist_range(
        programmer_start, programmer_end
    )
    programmer_start_again, programmer_end_again = (
        utils.biologist_range_to_programmer_range(
            biologist_start, biologist_end
        )
    )

    self.assertEqual(programmer_start, programmer_start_again)
    self.assertEqual(programmer_end, programmer_end_again)

  @parameterized.named_parameters(
      dict(
          testcase_name='identical ranges',
          seq1_start=1,
          seq1_end=10,
          seq2_start=1,
          seq2_end=10,
          expected=True,
      ),
      dict(
          testcase_name='both midpoints inside others',
          seq1_start=1,
          seq1_end=10,
          seq2_start=3,
          seq2_end=12,
          expected=True,
      ),
      dict(
          testcase_name='seq1 midpoint in seq2, not other way around',
          seq1_start=1,
          seq1_end=10,
          seq2_start=3,
          seq2_end=1000,
          expected=True,
      ),
      dict(
          testcase_name='seq2 midpoint in seq1, not other way around',
          seq1_start=3,
          seq1_end=1000,
          seq2_start=1,
          seq2_end=10,
          expected=True,
      ),
      dict(
          testcase_name='not overlapping',
          seq1_start=1,
          seq1_end=10,
          seq2_start=999,
          seq2_end=1000,
          expected=False,
      ),
  )
  def test_midpoint_in_either_range(
      self, seq1_start, seq1_end, seq2_start, seq2_end, expected
  ):
    actual = utils.midpoint_in_either_range(
        seq1_start, seq1_end, seq2_start, seq2_end
    )

    self.assertEqual(actual, expected)

  def test_get_known_nested_domains(self):
    test_file = self.create_tempfile(
        'nested_domains_pfam35.txt', content='PF00001\tPF09876\n'
    )
    input_family_to_clan = {
        'PF00001': 'CL0192',
        'PF09876': 'PF09876',
    }
    expected = {
        ('PF00001', 'PF09876'),
        ('CL0192', 'PF09876'),
        ('PF09876', 'PF00001'),
        ('PF09876', 'CL0192'),
    }

    actual = utils.get_known_nested_domains(
        model_cache_path=os.path.dirname(test_file.full_path),
        family_to_clan=input_family_to_clan,
    )

    self.assertSetEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='overlap',
          input_label1=('PF00001', (1, 10)),
          input_label2=('PF00002', (1, 10)),
          input_known_nested_domains=[],
          expected_answer=True,
      ),
      dict(
          testcase_name='non-overlap',
          input_label1=('PF00001', (1, 10)),
          input_label2=('PF00002', (100, 200)),
          input_known_nested_domains=[],
          expected_answer=False,
      ),
      dict(
          testcase_name='overlap in nested domains',
          input_label1=('PF00001', (1, 10)),
          input_label2=('PF00002', (1, 10)),
          input_known_nested_domains=[('PF00001', 'PF00002')],
          expected_answer=False,
      ),
      dict(
          testcase_name='non-overlap in nested domains',
          input_label1=('PF00001', (1, 10)),
          input_label2=('PF00002', (100, 200)),
          input_known_nested_domains=[('PF00001', 'PF00002')],
          expected_answer=False,
      ),
  )
  def test_labels_should_be_competed(
      self,
      input_label1,
      input_label2,
      input_known_nested_domains,
      expected_answer,
  ):
    actual = utils._labels_should_be_competed(
        input_label1, input_label2, input_known_nested_domains
    )
    self.assertEqual(actual, expected_answer)

  @parameterized.named_parameters(
      dict(
          testcase_name='competition between two labels same clan',
          input_first_label=('fam1', (1, 3)),
          input_second_label=('fam2', (1, 2)),
          expected_kept_label=('fam1', (1, 3)),
      ),
      dict(
          testcase_name='competition between two labels same clan other order',
          input_first_label=('fam2', (1, 2)),
          input_second_label=('fam1', (1, 3)),
          expected_kept_label=('fam1', (1, 3)),
      ),
      dict(
          testcase_name=(
              'competition between two labels same clan, same length, take'
              ' smaller accession'
          ),
          input_first_label=('fam2', (1, 2)),
          input_second_label=('fam1', (1, 2)),
          expected_kept_label=('fam1', (1, 2)),
      ),
  )
  def test_choose_label_to_keep_by_length(
      self, input_first_label, input_second_label, expected_kept_label
  ):
    actual = utils._choose_label_to_keep_by_length(
        input_first_label, input_second_label
    )
    self.assertEqual(actual, expected_kept_label)

  @parameterized.named_parameters(
      dict(
          testcase_name='same clan, one starts with PF the other CL',
          input_first_label=('PF00001', (1, 2)),
          input_second_label=('CL0192', (1, 2)),
          input_family_to_clan={
              'PF00001': 'CL0192',
          },
          expected_kept_label=('PF00001', (1, 2)),
      ),
      dict(
          testcase_name=(
              'same clan, one starts with PF the other CL, other order'
          ),
          input_first_label=('CL0192', (1, 2)),
          input_second_label=('PF00001', (1, 2)),
          input_family_to_clan={
              'PF00001': 'CL0192',
          },
          expected_kept_label=('PF00001', (1, 2)),
      ),
      dict(
          testcase_name='different clan, one is longer',
          input_first_label=('PF00001', (1, 2)),
          input_second_label=('PF09876', (1, 999)),
          input_family_to_clan={
              'PF00001': 'CL0192',
          },
          expected_kept_label=('PF09876', (1, 999)),
      ),
  )
  def test_choose_label_to_keep_by_competing(
      self,
      input_first_label,
      input_second_label,
      input_family_to_clan,
      expected_kept_label,
  ):
    actual = utils._choose_label_to_keep_by_competing(
        input_first_label, input_second_label, input_family_to_clan
    )
    self.assertEqual(actual, expected_kept_label)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty input list',
          input_labels_to_compete=[],
          input_known_nested_domains={},
          input_family_to_clan={},
          expected_competed_labels=[],
      ),
      dict(
          testcase_name='single label, no competition',
          input_labels_to_compete=[('fam1', (1, 2))],
          input_known_nested_domains={},
          input_family_to_clan={'fam1': 'clan'},
          expected_competed_labels=[('fam1', (1, 2))],
      ),
      dict(
          testcase_name='competition between two labels but theyre nested',
          input_labels_to_compete=[('fam1', (1, 3)), ('fam2', (1, 2))],
          input_known_nested_domains={('fam1', 'fam2')},
          input_family_to_clan={},
          expected_competed_labels=[('fam1', (1, 3)), ('fam2', (1, 2))],
      ),
      dict(
          testcase_name=(
              'competition between three labels same clan, longest is actual'
              ' clan label'
          ),
          input_labels_to_compete=[
              ('fam1', (1, 2)),
              ('fam2', (1, 3)),
              ('CLAN', (1, 4)),
          ],
          input_known_nested_domains={},
          input_family_to_clan={
              'fam1': 'CLAN',
              'fam2': 'CLAN',
          },
          # Prefer the family label even though clan label is longer
          expected_competed_labels=[('fam2', (1, 3))],
      ),
  )
  def test_compete_clan_labels_by_length(
      self,
      input_labels_to_compete,
      input_known_nested_domains,
      input_family_to_clan,
      expected_competed_labels,
  ):
    actual_competed_labels = utils.compete_clan_labels(
        input_labels_to_compete,
        input_known_nested_domains,
        input_family_to_clan,
    )

    self.assertCountEqual(actual_competed_labels, expected_competed_labels)


if __name__ == '__main__':
  absltest.main()
