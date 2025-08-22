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

"""Tests for protenn.per_residue_sparse."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import scipy.sparse
from protenn import per_residue_sparse


class PerResidueSparseTest(parameterized.TestCase):

  def test_true_label_to_coo(self):
    input_ground_truth = [(4, 987), (13, 987), (2, 1234)]
    actual = per_residue_sparse.true_label_to_coo(input_ground_truth)[:10]
    expected = [(4, 987, 1.0), (13, 987, 1.0), (2, 1234, 1.0)]
    self.assertListEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name=' empty inputs',
          input_ijv_tuples=[],
          input_vocab=np.array([]),
          input_applicable_label_dict={},
          expected=[],
      ),
      dict(
          testcase_name=' one input, nothing implied',
          input_ijv_tuples=[
              (0, 0, 1),
          ],
          input_vocab=np.array(['PF00001']),
          input_applicable_label_dict={},
          expected=[(0, 0, 1)],
      ),
      dict(
          testcase_name=' one input, something implied',
          input_ijv_tuples=[
              (0, 0, 1),
          ],
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_applicable_label_dict={'PF00001': 'CL0192'},
          # Second tuple gets added because it's implied by the first.
          expected=[(0, 0, 1), (0, 1, 1)],
      ),
      dict(
          testcase_name=' clan already has prediction, clan prediction weaker',
          input_ijv_tuples=[(0, 0, 1), (0, 1, 0.5)],
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_applicable_label_dict={'PF00001': 'CL0192'},
          # Expect that, because the family label is larger than the clan label,
          # the second tuple's last entry is 1, not .5.
          expected=[(0, 0, 1), (0, 1, 1)],
      ),
      dict(
          testcase_name=(
              ' clan already has prediction, clan prediction stronger'
          ),
          input_ijv_tuples=[(0, 0, 0.5), (0, 1, 1.0)],
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_applicable_label_dict={'PF00001': 'CL0192'},
          # Expect that, because the clan label is larger than the family label,
          # the second tuple's last entry is .5, not 1.
          expected=[(0, 0, 0.5), (0, 1, 1)],
      ),
      dict(
          testcase_name=' two inputs, clan label implied by both',
          input_ijv_tuples=[(0, 0, 0.5), (0, 1, 1.0)],
          input_vocab=np.array(['PF00001', 'PF00002', 'CL0192']),
          input_applicable_label_dict={
              'PF00001': 'CL0192',
              'PF00002': 'CL0192',
          },
          # Expect that the clan gets the maximum of either labels.
          expected=[(0, 0, 0.5), (0, 1, 1.0), (0, 2, 1.0)],
      ),
      dict(
          testcase_name=(
              ' two inputs at different indexes, clan label implied by both'
          ),
          input_ijv_tuples=[(0, 0, 0.5), (1, 0, 1.0)],
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_applicable_label_dict={'PF00001': 'CL0192'},
          # Expect that the clan label is applied to both indexes.
          expected=[(0, 0, 0.5), (1, 0, 1.0), (0, 1, 0.5), (1, 1, 1.0)],
      ),
  )
  def test_normalize_ijv_tuples(
      self, input_ijv_tuples, input_vocab, input_applicable_label_dict, expected
  ):
    actual = per_residue_sparse.normalize_ijv_tuples(
        input_ijv_tuples, input_vocab, input_applicable_label_dict
    )
    self.assertCountEqual(actual, expected)

  def test_dense_to_sparse_coo_list_of_tuples(self):
    input_dense = np.arange(9).reshape(3, 3)

    actual = per_residue_sparse.dense_to_sparse_coo_list_of_tuples(input_dense)
    expected = [
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 3),
        (1, 1, 4),
        (1, 2, 5),
        (2, 0, 6),
        (2, 1, 7),
        (2, 2, 8),
    ]
    self.assertListEqual(actual, expected)

  def test_np_matrix_to_array(self):
    input_array = np.arange(9).reshape(3, 3)
    input_matrix = scipy.sparse.coo_matrix(input_array).todense()
    actual = per_residue_sparse.np_matrix_to_array(input_matrix)

    expected = input_array
    np.testing.assert_allclose(actual, expected)

  def test_ijv_tuples_to_sparse_coo(self):
    # This is np.arange(9).reshape(3, 3).
    input_ijv_list = [
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 3),
        (1, 1, 4),
        (1, 2, 5),
        (2, 0, 6),
        (2, 1, 7),
        (2, 2, 8),
    ]
    input_sequence_length = 3
    input_num_classes = 3

    actual = per_residue_sparse.ijv_tuples_to_sparse_coo(
        input_ijv_list, input_sequence_length, input_num_classes
    )
    expected_num_nonzero = len(input_ijv_list)

    self.assertEqual(actual.count_nonzero(), expected_num_nonzero)
    self.assertEqual(actual.todense()[0, 1], 1)
    self.assertEqual(actual.todense()[0, 2], 2)

  def test_ijv_tuples_to_sparse_coo_empty_input(self):
    input_ijv_list = []
    input_sequence_length = 3
    input_num_classes = 3

    actual = per_residue_sparse.ijv_tuples_to_sparse_coo(
        input_ijv_list, input_sequence_length, input_num_classes
    )
    expected_num_nonzero = len(input_ijv_list)

    self.assertEqual(actual.count_nonzero(), expected_num_nonzero)

  def test_ijv_tuples_to_dense(self):
    # Identity matrix with 0, 1, 2 along diagonal instead of ones.
    input_ijv_list = [
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 3),
        (1, 1, 4),
        (1, 2, 5),
        (2, 0, 6),
        (2, 1, 7),
        (2, 2, 8),
    ]
    input_sequence_length = 3
    input_num_classes = 3

    actual = per_residue_sparse.ijv_tuples_to_dense(
        input_ijv_list, input_sequence_length, input_num_classes
    )

    expected = np.arange(9).reshape(3, 3)
    np.testing.assert_equal(actual, expected)

  def test_ijv_tuples_to_dense_empty_input(self):
    input_ijv_list = []
    input_sequence_length = 3
    input_num_classes = 3

    actual = per_residue_sparse.ijv_tuples_to_dense(
        input_ijv_list, input_sequence_length, input_num_classes
    )

    expected = np.zeros(shape=(3, 3))
    np.testing.assert_equal(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name=' all false',
          input_boolean_condition=np.array([False, False]),
          expected=np.empty(shape=(0, 2)),
      ),
      dict(
          testcase_name=' all true',
          input_boolean_condition=np.array([True, True, True]),
          expected=np.array([[0, 3]]),
      ),
      dict(
          testcase_name=' one true',
          input_boolean_condition=np.array([False, True, False]),
          expected=np.array([[1, 2]]),
      ),
      dict(
          testcase_name=' one true region',
          input_boolean_condition=np.array([False, True, True, False]),
          expected=np.array([[1, 3]]),
      ),
      dict(
          testcase_name=' two true regions',
          input_boolean_condition=np.array(
              [False, True, True, False, True, True]
          ),
          expected=np.array([[1, 3], [4, 6]]),
      ),
  )
  def test_contiguous_regions_1d(self, input_boolean_condition, expected):
    actual = per_residue_sparse.contiguous_regions_1d(input_boolean_condition)
    np.testing.assert_equal(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name=' no activations',
          input_activations=[],
          input_sequence_length=3,
          input_vocab=np.array(['PF00001', 'PF00002', 'CL0192']),
          input_reporting_threshold=0.8,
          expected={},
      ),
      dict(
          testcase_name=' one activation, below threshold',
          # .3 is below reporting threshold.
          input_activations=[(0, 0, 0.3)],
          input_sequence_length=3,
          input_vocab=np.array(['PF00001', 'PF00002', 'CL0192']),
          input_reporting_threshold=0.8,
          expected={},
      ),
      dict(
          testcase_name=' one activation, above threshold',
          # .99 is above reporting threshold.
          input_activations=[(0, 0, 0.99)],
          input_sequence_length=3,
          input_vocab=np.array(['PF00001', 'PF00002', 'CL0192']),
          input_reporting_threshold=0.8,
          expected={
              'PF00001': [(1, 1)],
          },
      ),
      dict(
          testcase_name=' two contiguous regions DO GET merged',
          # The two residues should get merged into one region for PF00001.
          input_activations=[(0, 0, 0.99), (1, 0, 0.99)],
          input_sequence_length=3,
          input_vocab=np.array(['PF00001']),
          input_reporting_threshold=0.8,
          expected={
              'PF00001': [(1, 2)],
          },
      ),
      dict(
          testcase_name=' two NONcontiguous regions DO NOT get merged',
          input_activations=[(0, 0, 0.99), (3, 0, 0.99)],
          input_sequence_length=5,
          input_vocab=np.array(['PF00001']),
          input_reporting_threshold=0.8,
          expected={
              'PF00001': [(1, 1), (4, 4)],
          },
      ),
      dict(
          testcase_name=(
              ' two contiguous regions belonging to different families DO NOT'
              ' GET merged'
          ),
          input_activations=[(0, 0, 0.99), (1, 1, 0.99)],
          input_sequence_length=3,
          input_vocab=np.array(['PF00001', 'PF00002']),
          input_reporting_threshold=0.8,
          expected={
              'PF00001': [(1, 1)],
              'PF00002': [(2, 2)],
          },
      ),
  )
  def test_contiguous_regions_2d(
      self,
      input_activations,
      input_sequence_length,
      input_vocab,
      input_reporting_threshold,
      expected,
  ):
    actual = per_residue_sparse.contiguous_regions_2d(
        activations=input_activations,
        sequence_length=input_sequence_length,
        vocab=input_vocab,
        reporting_threshold=input_reporting_threshold,
    )
    self.assertDictEqual(actual, expected)

  def test_filter_domain_calls_by_length(self):
    input_domain_calls = {'CL0036': [(1, 2), (3, 300)], 'PF00001': [(20, 500)]}
    input_min_length = 42
    actual = per_residue_sparse.filter_domain_calls_by_length(
        input_domain_calls, input_min_length
    )
    expected = {'CL0036': [(3, 300)], 'PF00001': [(20, 500)]}
    self.assertDictEqual(actual, expected)

  def test_activations_to_domain_calls(self):
    input_activations_class_0 = [(i, 0, 0.4) for i in range(50)]
    input_activations_class_1 = [(i, 1, 1.0) for i in range(3)]
    input_activations = input_activations_class_0 + input_activations_class_1
    input_sequence_length = 200
    input_vocab = np.array(['CLASS_0', 'CLASS_1'])
    input_reporting_threshold = 0.3
    input_min_domain_call_length = 50

    actual = per_residue_sparse.activations_to_domain_calls(
        input_activations,
        input_sequence_length,
        input_vocab,
        input_reporting_threshold,
        input_min_domain_call_length,
    )
    expected = {'CLASS_0': [(1, 50)]}
    self.assertEqual(actual, expected)

  def test_num_labels_in_dense_label_dict(self):
    input_dense_label_dict = {
        'CL1234': [(1, 2), (3, 4)],
        'PF00001': [(100, 200)],
    }
    actual = per_residue_sparse.num_labels_in_dense_label_dict(
        input_dense_label_dict
    )
    expected = 3

    self.assertEqual(actual, expected)

  def test_flatten_dict_of_domain_calls(self):
    input_dict_of_calls = {'CL0036': [(29, 252), (253, 254), (256, 257)]}
    expected = [
        ('CL0036', (29, 252)),
        ('CL0036', (253, 254)),
        ('CL0036', (256, 257)),
    ]
    actual = per_residue_sparse.flatten_dict_of_domain_calls(
        input_dict_of_calls
    )

    self.assertListEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
