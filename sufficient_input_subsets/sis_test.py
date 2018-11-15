# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for sufficient_input_subsets.sis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sufficient_input_subsets import sis

# Function that returns the L2 norm over each set of coordinates in the batch.
_F_L2 = lambda batch_coords: np.linalg.norm(batch_coords, ord=2, axis=-1)

# Function that returns the sum over each array in the batch.
_F_SUM = lambda batch: np.array([np.sum(arr) for arr in batch])

# Function that computes the dot product between a known vector ([1, 2, 0, 1])
# and each array in the batch (analagous to linear regression).
_LINREGRESS_THETA = np.array([1, 2, 0, 1])
_F_LINREGRESS = lambda bt: np.array([np.dot(_LINREGRESS_THETA, b) for b in bt])


class SisTest(parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(sis)

  def _assert_backselect_stack_equal(self, expected_backselect_stack,
                                     actual_backselect_stack):
    if not expected_backselect_stack:  # expected empty stack
      self.assertEqual(expected_backselect_stack, actual_backselect_stack)
      return

    expected_idxs, expected_values = zip(*expected_backselect_stack)
    actual_idxs, actual_values = zip(*actual_backselect_stack)

    try:
      np.testing.assert_allclose(expected_idxs, actual_idxs)
      np.testing.assert_allclose(expected_values, actual_values)
    except AssertionError:
      raise AssertionError(
          'Backselect stacks not equal. Expected %s, got %s.' %
          (str(expected_backselect_stack), str(actual_backselect_stack)))

  def _assert_equal_sis_result(self, expected_sis_result, actual_sis_result):
    if expected_sis_result is None:  # no SISResult
      self.assertEqual(expected_sis_result, actual_sis_result)
      return

    try:
      np.testing.assert_array_equal(expected_sis_result.sis,
                                    actual_sis_result.sis)
      np.testing.assert_array_equal(
          expected_sis_result.ordering_over_entire_backselect,
          actual_sis_result.ordering_over_entire_backselect)
      np.testing.assert_allclose(
          expected_sis_result.values_over_entire_backselect,
          actual_sis_result.values_over_entire_backselect)
      np.testing.assert_array_equal(expected_sis_result.mask,
                                    actual_sis_result.mask)
    except AssertionError:
      raise AssertionError('SISResults not equal. Expected %s, got %s.' %
                           (str(expected_sis_result), str(actual_sis_result)))

  def _assert_equal_sis_collection(self, expected_sis_collection,
                                   actual_sis_collection):
    if not expected_sis_collection:  # empty lists
      self.assertEqual(expected_sis_collection, actual_sis_collection)
      return

    try:
      # Must have same length
      self.assertLen(actual_sis_collection, len(expected_sis_collection))
      # Check that all SISResults are the same (in order)
      for expected_sis_result, actual_sis_result in zip(expected_sis_collection,
                                                        actual_sis_collection):
        self._assert_equal_sis_result(expected_sis_result, actual_sis_result)
    except AssertionError:
      raise AssertionError(
          'SIS collections not equal. Expected %s, got %s.' %
          (str(expected_sis_collection), str(actual_sis_collection)))

  @parameterized.named_parameters(
      dict(
          testcase_name='sis len 1',
          sis_result=sis.SISResult(
              sis=np.array([[0]]),
              ordering_over_entire_backselect=np.array([[2], [1], [3], [0]]),
              values_over_entire_backselect=np.array([10.0, 8.0, 5.0, 0.0]),
              mask=np.array([True, False, False, False]),
          ),
          expected_len=1),
      dict(
          testcase_name='sis, 2-dim idxs, len 3',
          sis_result=sis.SISResult(
              sis=np.array([[0, 1], [1, 2], [2, 3]]),
              ordering_over_entire_backselect=np.array([[2], [1], [3], [0]]),
              values_over_entire_backselect=np.array([10.0, 8.0, 5.0, 0.0]),
              mask=np.array([True, False, False, False]),
          ),
          expected_len=3),
  )
  def test_sisresult_len(self, sis_result, expected_len):
    actual_len = len(sis_result)
    self.assertEqual(expected_len, actual_len)

  @parameterized.named_parameters(
      dict(testcase_name='2-dim', shape=(4, 3)),
      dict(testcase_name='2-dim transposed', shape=(3, 4)),
      dict(testcase_name='1-dim', shape=(3,)),
      dict(testcase_name='3-dim', shape=(4, 3, 8)),
  )
  def test_make_empty_boolean_mask(self, shape):
    actual_mask = sis.make_empty_boolean_mask(shape)
    self.assertEqual(shape, actual_mask.shape)
    self.assertTrue(np.all(actual_mask))

  @parameterized.named_parameters(
      dict(
          testcase_name='2-dim mask over columns',
          shape=(2, 3),
          axis=0,
          expected_shape=(1, 3)),
      dict(
          testcase_name='2-dim mask over columns, as tuple',
          shape=(2, 3),
          axis=(0,),
          expected_shape=(1, 3)),
      dict(
          testcase_name='2-dim mask over rows',
          shape=(2, 3),
          axis=1,
          expected_shape=(2, 1)),
      dict(
          testcase_name='2-dim mask over all',
          shape=(2, 3),
          axis=(0, 1),
          expected_shape=(1, 1)),
      dict(
          testcase_name='3-dim mask over ax 1',
          shape=(4, 5, 6),
          axis=1,
          expected_shape=(4, 1, 6)),
      dict(
          testcase_name='3-dim mask over ax (1, 2)',
          shape=(4, 5, 6),
          axis=(1, 2),
          expected_shape=(4, 1, 1)),
  )
  def test_make_empty_boolean_mask_broadcast_over_axis(self, shape, axis,
                                                       expected_shape):
    actual_mask = sis.make_empty_boolean_mask_broadcast_over_axis(shape, axis)
    self.assertEqual(expected_shape, actual_mask.shape)
    self.assertTrue(np.all(actual_mask))

  @parameterized.named_parameters(
      dict(
          testcase_name='disjoint SIS-collection',
          collection=[
              sis.SISResult(
                  sis=np.array([[0], [1]]),
                  ordering_over_entire_backselect=np.array([[1], [0]]),
                  values_over_entire_backselect=np.array([10.0, 0.0]),
                  mask=np.array([True, False]),
              ),
              sis.SISResult(
                  sis=np.array([[2], [3]]),
                  ordering_over_entire_backselect=np.array([[1], [0]]),
                  values_over_entire_backselect=np.array([10.0, 0.0]),
                  mask=np.array([True, False]),
              ),
          ]),)
  def test_assert_sis_collection_disjoint(self, collection):
    sis._assert_sis_collection_disjoint(collection)

  @parameterized.named_parameters(
      dict(
          testcase_name='non-disjoint SIS-collection',
          collection=[
              sis.SISResult(
                  sis=np.array([[0], [1]]),
                  ordering_over_entire_backselect=np.array([[1], [0]]),
                  values_over_entire_backselect=np.array([10.0, 0.0]),
                  mask=np.array([True, False]),
              ),
              sis.SISResult(
                  sis=np.array([[1], [2]]),
                  ordering_over_entire_backselect=np.array([[1], [0]]),
                  values_over_entire_backselect=np.array([10.0, 0.0]),
                  mask=np.array([True, False]),
              ),
          ]),)
  def test_assert_sis_collection_disjoint_raises_error(self, collection):
    with self.assertRaises(AssertionError):
      sis._assert_sis_collection_disjoint(collection)

  @parameterized.named_parameters(
      dict(
          testcase_name='1-dim idxs, 1 idx',
          idx_array=np.array([[3]]),
          expected_tuple=(np.array([0]), np.array([3]))),
      dict(
          testcase_name='1-dim idxs, 2 idxs',
          idx_array=np.array([[1], [2]]),
          expected_tuple=(np.array([0, 1]), np.array([1, 2]))),
      dict(
          testcase_name='2-dim idxs, 2 idxs',
          idx_array=np.array([[0, 1], [1, 1]]),
          expected_tuple=(np.array([0, 1]), np.array([0, 1]), np.array([1,
                                                                        1]))),
      dict(
          testcase_name='3-dim idxs, 4 idxs',
          idx_array=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
          expected_tuple=(np.array([0, 1, 2, 3]), np.array([1, 4, 7, 10]),
                          np.array([2, 5, 8, 11]), np.array([3, 6, 9, 12]))),
  )
  def test_transform_next_masks_index_array_into_tuple(self, idx_array,
                                                       expected_tuple):
    actual_tuple = sis._transform_next_masks_index_array_into_tuple(idx_array)
    self.assertLen(actual_tuple, len(expected_tuple))
    for expected_column, actual_column in zip(expected_tuple, actual_tuple):
      np.testing.assert_array_equal(expected_column, actual_column)

  @parameterized.named_parameters(
      dict(testcase_name='1-dim idxs, 1 idx', idx_array=np.array([1])),
      dict(testcase_name='1-dim idxs, 2 idxs', idx_array=np.array([1, 2])),
      dict(
          testcase_name='3-dim idxs, 2 idxs',
          idx_array=np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])),
  )
  def test_transform_next_masks_index_array_into_tuple_raises_error(
      self, idx_array):
    with self.assertRaises(TypeError):
      _ = sis._transform_next_masks_index_array_into_tuple(idx_array)

  @parameterized.named_parameters(
      dict(
          testcase_name='no values masked',
          current_mask=np.array([True, True, True]),
          expected_next_masks=np.array([[False, True,
                                         True], [True, False, True],
                                        [True, True, False]]),
          expected_next_masks_idxs=np.array([[0], [1], [2]])),
      dict(
          testcase_name='partially masked',
          current_mask=np.array([True, False, True]),
          expected_next_masks=np.array([[False, False, True],
                                        [True, False, False]]),
          expected_next_masks_idxs=np.array([[0], [2]])),
      dict(
          testcase_name='partially masked 2',
          current_mask=np.array([False, False, True]),
          expected_next_masks=np.array([[False, False, False]]),
          expected_next_masks_idxs=np.array([[2]])),
      dict(
          testcase_name='partially masked larger',
          current_mask=np.array([True, True, False, True, True, False]),
          expected_next_masks=np.array([
              [False, True, False, True, True, False],
              [True, False, False, True, True, False],
              [True, True, False, False, True, False],
              [True, True, False, True, False, False],
          ]),
          expected_next_masks_idxs=np.array([[0], [1], [3], [4]])),
      dict(
          testcase_name='all values masked',
          current_mask=np.array([False, False, False]),
          expected_next_masks=np.array([]),
          expected_next_masks_idxs=np.array([])),
      dict(
          testcase_name='(3, 1) input',
          current_mask=np.array([[True], [True], [True]]),
          expected_next_masks=np.array([[[False], [True], [True]],
                                        [[True], [False], [True]],
                                        [[True], [True], [False]]]),
          expected_next_masks_idxs=np.array([[0, 0], [1, 0], [2, 0]])),
      dict(
          testcase_name='(1, 3) input',
          current_mask=np.array([[True, True, True]]),
          expected_next_masks=np.array([[[False, True, True]],
                                        [[True, False, True]],
                                        [[True, True, False]]]),
          expected_next_masks_idxs=np.array([[0, 0], [0, 1], [0, 2]])),
      dict(
          testcase_name='(1, 3) input, partially masked',
          current_mask=np.array([[True, False, True]]),
          expected_next_masks=np.array([[[False, False, True]],
                                        [[True, False, False]]]),
          expected_next_masks_idxs=np.array([[0, 0], [0, 2]])),
      dict(
          testcase_name='(1, 3) input, all masked',
          current_mask=np.array([[False, False, False]]),
          expected_next_masks=np.array([]),
          expected_next_masks_idxs=np.array([])),
      dict(
          testcase_name='(2, 2) input',
          current_mask=np.array([[True, True], [True, True]]),
          expected_next_masks=np.array([[[False, True], [True, True]],
                                        [[True, False], [True, True]],
                                        [[True, True], [False, True]],
                                        [[True, True], [True, False]]]),
          expected_next_masks_idxs=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
  )
  def test_produce_next_masks(self, current_mask, expected_next_masks,
                              expected_next_masks_idxs):
    actual_next_masks, actual_next_masks_idxs = sis._produce_next_masks(
        current_mask)
    np.testing.assert_array_equal(expected_next_masks, actual_next_masks)
    np.testing.assert_array_equal(expected_next_masks_idxs,
                                  actual_next_masks_idxs)

  @parameterized.named_parameters(
      dict(
          testcase_name='1-dim, single mask',
          input_to_mask=np.array([1, 2, 3, 4, 5]),
          fully_masked_input=np.array([0, 0, 0, 0, 0]),
          batch_of_masks=np.array([[False, True, False, True, True]]),
          expected_masked_inputs=np.array([[0, 2, 0, 4, 5]])),
      dict(
          testcase_name='1-dim, multiple masks',
          input_to_mask=np.array([1, 2, 3]),
          fully_masked_input=np.array([0, 0, 0]),
          batch_of_masks=np.array([[True, True, False], [True, True, True],
                                   [False, False, False], [False, True,
                                                           False]]),
          expected_masked_inputs=np.array([[1, 2, 0], [1, 2, 3], [0, 0, 0],
                                           [0, 2, 0]])),
      dict(
          testcase_name='2-dim, single mask',
          input_to_mask=np.array([[1, 2, 3], [4, 5, 6]]),
          fully_masked_input=np.array([[0, 0, 0], [0, 0, 0]]),
          batch_of_masks=np.array([[[True, False, False], [False, True,
                                                           True]]]),
          expected_masked_inputs=np.array([[[1, 0, 0], [0, 5, 6]]])),
      dict(
          testcase_name='2-dim, multiple masks',
          input_to_mask=np.array([[1, 2, 3], [4, 5, 6]]),
          fully_masked_input=np.array([[0, 0, 0], [0, 0, 0]]),
          batch_of_masks=np.array(
              [[[True, True, True], [True, True, True]],
               [[False, False, False], [False, False, False]],
               [[True, False, True], [False, True, False]]]),
          expected_masked_inputs=np.array([[[1, 2, 3], [4, 5, 6]],
                                           [[0, 0, 0], [0, 0, 0]],
                                           [[1, 0, 3], [0, 5, 0]]])),
      dict(
          testcase_name='1-dim, single mask, string inputs',
          input_to_mask=np.array(['A', 'B', 'C', 'D']),
          fully_masked_input=np.array(['-', '-', '-', '-']),
          batch_of_masks=np.array([[False, True, False, True]]),
          expected_masked_inputs=np.array([['-', 'B', '-', 'D']])),
  )
  def test_produce_masked_inputs(self, input_to_mask, fully_masked_input,
                                 batch_of_masks, expected_masked_inputs):
    actual_masked_inputs = sis.produce_masked_inputs(
        input_to_mask, fully_masked_input, batch_of_masks)
    np.testing.assert_array_equal(expected_masked_inputs, actual_masked_inputs)

  @parameterized.named_parameters(
      dict(
          testcase_name='1-dim, single mask, no batch dimension',
          input_to_mask=np.array([1, 2, 3]),
          fully_masked_input=np.array([0, 0, 0]),
          batch_of_masks=np.array([False, True, False])),)
  def test_produce_masked_inputs_raises_error(
      self, input_to_mask, fully_masked_input, batch_of_masks):
    with self.assertRaises(TypeError):
      _ = sis.produce_masked_inputs(input_to_mask, fully_masked_input,
                                    batch_of_masks)

  @parameterized.named_parameters(
      dict(
          testcase_name='L2 norm, 2-dim',
          f=_F_L2,
          current_input=np.array([1, 10]),
          current_mask=np.array([True, True]),
          fully_masked_input=np.array([0, 0]),
          expected_backselect_stack=[(np.array([0]), 10), (np.array([1]), 0)]),
      dict(
          testcase_name='L2 norm, 2-dim, all masked',
          f=_F_L2,
          current_input=np.array([1, 10]),
          current_mask=np.array([False, False]),
          fully_masked_input=np.array([0, 0]),
          expected_backselect_stack=[]),
      dict(
          testcase_name='L2 norm, 2-dim, reversed',
          f=_F_L2,
          current_input=np.array([10, 1]),
          current_mask=np.array([True, True]),
          fully_masked_input=np.array([0, 0]),
          expected_backselect_stack=[(np.array([1]), 10), (np.array([0]), 0)]),
      dict(
          testcase_name='L2 norm, 2-dim, partially masked',
          f=_F_L2,
          current_input=np.array([10, 1]),
          current_mask=np.array([False, True]),
          fully_masked_input=np.array([0, 0]),
          expected_backselect_stack=[(np.array([1]), 0)]),
      dict(
          testcase_name='L2 norm, 2-dim, partially masked, reversed',
          f=_F_L2,
          current_input=np.array([10, 1]),
          current_mask=np.array([True, False]),
          fully_masked_input=np.array([0, 0]),
          expected_backselect_stack=[(np.array([0]), 0)]),
      dict(
          testcase_name='L2 norm, 3-dim, same value',
          f=_F_L2,
          current_input=np.array([10, 10, 10]),
          current_mask=np.array([True, True, True]),
          fully_masked_input=np.array([0, 0, 0]),
          expected_backselect_stack=[(np.array([0]), np.sqrt(200)),
                                     (np.array([1]), 10), (np.array([2]), 0)]),
      dict(
          testcase_name='L2 norm, 4-dim, diff values',
          f=_F_L2,
          current_input=np.array([0.1, 10, 5, 1]),
          current_mask=np.array([True, True, True, True]),
          fully_masked_input=np.array([0, 0, 0, 0]),
          expected_backselect_stack=[(np.array([0]), np.sqrt(126)),
                                     (np.array([3]), np.sqrt(125)),
                                     (np.array([2]), 10), (np.array([1]), 0)]),
      dict(
          testcase_name='sum, 2x2 input, individual masking',
          f=_F_SUM,
          current_input=np.array([[10, 5], [2, 3]]),
          current_mask=np.array([[True, True], [True, True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_backselect_stack=[(np.array([1, 0]), 18),
                                     (np.array([1, 1]), 15),
                                     (np.array([0, 1]), 10),
                                     (np.array([0, 0]), 0)]),
      dict(
          testcase_name='sum, 2x2 input, mask broadcast over columns',
          f=_F_SUM,
          current_input=np.array([[10, 5], [2, 3]]),
          current_mask=np.array([[True, True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_backselect_stack=[(np.array([0, 1]), 12),
                                     (np.array([0, 0]), 0)]),
      dict(
          testcase_name='sum, 2x2 input, mask broadcast over rows',
          f=_F_SUM,
          current_input=np.array([[10, 5], [2, 3]]),
          current_mask=np.array([[True], [True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_backselect_stack=[(np.array([1, 0]), 15),
                                     (np.array([0, 0]), 0)]),
  )
  def test_backselect(self, f, current_input, current_mask, fully_masked_input,
                      expected_backselect_stack):
    actual_backselect_stack = sis._backselect(f, current_input, current_mask,
                                              fully_masked_input)
    self._assert_backselect_stack_equal(expected_backselect_stack,
                                        actual_backselect_stack)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty sis, threshold equals final value',
          backselect_stack=[(np.array([0]), 1.0), (np.array([2]), 0.9),
                            (np.array([1]), 0.7), (np.array([3]), 0.6)],
          threshold=0.6,
          expected_sis=[]),
      dict(
          testcase_name='empty sis, threshold less than final value',
          backselect_stack=[(np.array([0]), 1.0), (np.array([2]), 0.9),
                            (np.array([1]), 0.7), (np.array([3]), 0.6)],
          threshold=0.5,
          expected_sis=[]),
      dict(
          testcase_name='single element SIS, larger threshold',
          backselect_stack=[(np.array([0]), 1.0), (np.array([2]), 0.9),
                            (np.array([1]), 0.7), (np.array([3]), 0.6)],
          threshold=0.65,
          expected_sis=[np.array([3])]),
      dict(
          testcase_name='one element SIS, threshold equals value',
          backselect_stack=[(np.array([0]), 1.0), (np.array([2]), 0.9),
                            (np.array([1]), 0.7), (np.array([3]), 0.6)],
          threshold=0.7,
          expected_sis=[np.array([3])]),
      dict(
          testcase_name='two element SIS, threshold between values',
          backselect_stack=[(np.array([0]), 1.0), (np.array([2]), 0.9),
                            (np.array([1]), 0.7), (np.array([3]), 0.6)],
          threshold=0.8,
          expected_sis=[np.array([3]), np.array([1])]),
      dict(
          testcase_name='three element SIS',
          backselect_stack=[(np.array([0]), 1.0), (np.array([2]), 0.9),
                            (np.array([1]), 0.7), (np.array([3]), 0.6)],
          threshold=0.99,
          expected_sis=[np.array([3]),
                        np.array([1]),
                        np.array([2])]),
      dict(
          testcase_name='all element SIS',
          backselect_stack=[(np.array([0]), 1.0), (np.array([2]), 0.9),
                            (np.array([1]), 0.7), (np.array([3]), 0.6)],
          threshold=2.0,
          expected_sis=[
              np.array([3]),
              np.array([1]),
              np.array([2]),
              np.array([0])
          ]),
  )
  def test_find_sis_from_backselect(self, backselect_stack, threshold,
                                    expected_sis):
    actual_sis = sis._find_sis_from_backselect(backselect_stack, threshold)
    self.assertLen(actual_sis, len(expected_sis))
    for expected_idx, actual_idx in zip(expected_sis, actual_sis):
      np.testing.assert_array_equal(expected_idx, actual_idx)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty backselect_stack',
          backselect_stack=[],
          threshold=1.0),)
  def test_find_sis_from_backselect_raises_error(self, backselect_stack,
                                                 threshold):
    with self.assertRaises(ValueError):
      _ = sis._find_sis_from_backselect(backselect_stack, threshold)

  @parameterized.named_parameters(
      dict(
          testcase_name='L2 norm, 2-dim',
          f=_F_L2,
          threshold=1.0,
          current_input=np.array([.1, 10]),
          current_mask=np.array([True, True]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[1]]),
              ordering_over_entire_backselect=np.array([[0], [1]]),
              values_over_entire_backselect=np.array([10.0, 0.0]),
              mask=np.array([False, True]),
          )),
      dict(
          testcase_name='L2 norm, 2-dim, reversed',
          f=_F_L2,
          threshold=1.0,
          current_input=np.array([10, .1]),
          current_mask=np.array([True, True]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[0]]),
              ordering_over_entire_backselect=np.array([[1], [0]]),
              values_over_entire_backselect=np.array([10.0, 0.0]),
              mask=np.array([True, False]),
          )),
      dict(
          testcase_name='L2 norm, 3-dim',
          f=_F_L2,
          threshold=1.0,
          current_input=np.array([.1, 10, 5]),
          current_mask=np.array([True, True, True]),
          fully_masked_input=np.array([0, 0, 0]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[1]]),
              ordering_over_entire_backselect=np.array([[0], [2], [1]]),
              values_over_entire_backselect=np.array([np.sqrt(125), 10.0, 0.0]),
              mask=np.array([False, True, False]),
          )),
      dict(
          testcase_name='L2 norm, 3-dim, larger threshold',
          f=_F_L2,
          threshold=10.5,
          current_input=np.array([.1, 10, 5]),
          current_mask=np.array([True, True, True]),
          fully_masked_input=np.array([0, 0, 0]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[1], [2]]),
              ordering_over_entire_backselect=np.array([[0], [2], [1]]),
              values_over_entire_backselect=np.array([np.sqrt(125), 10.0, 0.0]),
              mask=np.array([False, True, True]),
          )),
      dict(
          testcase_name='L2 norm, 2-dim, all elms SIS',
          f=_F_L2,
          threshold=5.0,
          current_input=np.array([3, 4]),
          current_mask=np.array([True, True]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[1], [0]]),
              ordering_over_entire_backselect=np.array([[0], [1]]),
              values_over_entire_backselect=np.array([4.0, 0.0]),
              mask=np.array([True, True]),
          )),
      dict(
          testcase_name='L2 norm, 2-dim, no SIS',
          f=_F_L2,
          threshold=5.1,
          current_input=np.array([3, 4]),
          current_mask=np.array([True, True]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_result=None),
      dict(
          testcase_name='L2 norm, 3-dim, no SIS',
          f=_F_L2,
          threshold=1000,
          current_input=np.array([.1, 10, 5]),
          current_mask=np.array([True, True, True]),
          fully_masked_input=np.array([0, 0, 0]),
          expected_sis_result=None),
      dict(
          testcase_name='L2 norm, 3-dim, partially masked',
          f=_F_L2,
          threshold=1.0,
          current_input=np.array([.1, 10, 5]),
          current_mask=np.array([True, False, True]),
          fully_masked_input=np.array([0, 0, 0]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[2]]),
              ordering_over_entire_backselect=np.array([[0], [2]]),
              values_over_entire_backselect=np.array([5.0, 0.0]),
              mask=np.array([False, False, True]),
          )),
      dict(
          testcase_name='L2 norm, 2-dim, all masked',
          f=_F_L2,
          threshold=1.0,
          current_input=np.array([10, .1]),
          current_mask=np.array([False, False]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_result=None),
      dict(
          testcase_name='sum, (2, 2), individual masking, no initial masked',
          f=_F_SUM,
          threshold=4.0,
          current_input=np.array([[10, 5], [2, 3]]),
          current_mask=np.array([[True, True], [True, True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[0, 0]]),
              ordering_over_entire_backselect=np.array([[1, 0], [1, 1], [0, 1],
                                                        [0, 0]]),
              values_over_entire_backselect=np.array([18.0, 15.0, 10.0, 0.0]),
              mask=np.array([[True, False], [False, False]]),
          )),
      dict(
          testcase_name='sum, (2, 2), individual masking, broadcast over cols',
          f=_F_SUM,
          threshold=4.0,
          current_input=np.array([[10, 5], [2, 13]]),
          current_mask=np.array([[True, True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[0, 1]]),
              ordering_over_entire_backselect=np.array([[0, 0], [0, 1]]),
              values_over_entire_backselect=np.array([18.0, 0.0]),
              mask=np.array([[False, True]]),
          )),
      dict(
          testcase_name='sum, (2, 2), individual masking, broadcast over rows',
          f=_F_SUM,
          threshold=4.0,
          current_input=np.array([[10, 5], [2, 13]]),
          current_mask=np.array([[True], [True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_sis_result=sis.SISResult(
              sis=np.array([[1, 0]]),
              ordering_over_entire_backselect=np.array([[0, 0], [1, 0]]),
              values_over_entire_backselect=np.array([15.0, 0.0]),
              mask=np.array([[False], [True]]),
          )),
  )
  def test_find_sis(self, f, threshold, current_input, current_mask,
                    fully_masked_input, expected_sis_result):
    actual_sis_result = sis.find_sis(f, threshold, current_input, current_mask,
                                     fully_masked_input)
    self._assert_equal_sis_result(expected_sis_result, actual_sis_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='L2 norm, 2-dim, no SIS',
          f=_F_L2,
          threshold=1000,
          initial_input=np.array([.1, 10]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_collection=[]),
      dict(
          testcase_name='L2 norm, 2-dim, 1 SIS',
          f=_F_L2,
          threshold=1.0,
          initial_input=np.array([.1, 10]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[1]]),
                  ordering_over_entire_backselect=np.array([[0], [1]]),
                  values_over_entire_backselect=np.array([10.0, 0.0]),
                  mask=np.array([False, True]),
              ),
          ]),
      dict(
          testcase_name='L2 norm, 2-dim, 2 SIS',
          f=_F_L2,
          threshold=0.1,
          initial_input=np.array([.1, 10]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[1]]),
                  ordering_over_entire_backselect=np.array([[0], [1]]),
                  values_over_entire_backselect=np.array([10.0, 0.0]),
                  mask=np.array([False, True]),
              ),
              sis.SISResult(
                  sis=np.array([[0]]),
                  ordering_over_entire_backselect=np.array([[0]]),
                  values_over_entire_backselect=np.array([0.0]),
                  mask=np.array([True, False]),
              ),
          ]),
      dict(
          testcase_name='L2 norm, 2-dim, 2 SIS, reverse order',
          f=_F_L2,
          threshold=0.1,
          initial_input=np.array([10, .1]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[0]]),
                  ordering_over_entire_backselect=np.array([[1], [0]]),
                  values_over_entire_backselect=([10.0, 0.0]),
                  mask=np.array([True, False]),
              ),
              sis.SISResult(
                  sis=np.array([[1]]),
                  ordering_over_entire_backselect=np.array([[1]]),
                  values_over_entire_backselect=np.array([0.0]),
                  mask=np.array([False, True]),
              ),
          ]),
      dict(
          testcase_name='L2 norm, 2-dim, 1 SIS (both elms)',
          f=_F_L2,
          threshold=4.5,
          initial_input=np.array([3, 4]),
          fully_masked_input=np.array([0, 0]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[1], [0]]),
                  ordering_over_entire_backselect=np.array([[0], [1]]),
                  values_over_entire_backselect=np.array([4.0, 0.0]),
                  mask=np.array([True, True]),
              ),
          ]),
      dict(
          testcase_name='L2 norm, 3-dim, 2 SIS',
          f=_F_L2,
          threshold=1.0,
          initial_input=np.array([.1, 10, 5]),
          fully_masked_input=np.array([0, 0, 0]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[1]]),
                  ordering_over_entire_backselect=np.array([[0], [2], [1]]),
                  values_over_entire_backselect=np.array(
                      [np.sqrt(125), 10.0, 0.0]),
                  mask=np.array([False, True, False]),
              ),
              sis.SISResult(
                  sis=np.array([[2]]),
                  ordering_over_entire_backselect=np.array([[0], [2]]),
                  values_over_entire_backselect=np.array([5.0, 0.0]),
                  mask=np.array([False, False, True]),
              ),
          ]),
      dict(
          testcase_name='L2 norm, 3-dim, 3 SIS',
          f=_F_L2,
          threshold=1.0,
          initial_input=np.array([.9, .9, 10, 5]),
          fully_masked_input=np.array([0, 0, 0, 0]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[2]]),
                  ordering_over_entire_backselect=np.array([[0], [1], [3],
                                                            [2]]),
                  values_over_entire_backselect=np.array(
                      [np.sqrt(125.81),
                       np.sqrt(125), 10.0, 0.0]),
                  mask=np.array([False, False, True, False]),
              ),
              sis.SISResult(
                  sis=np.array([[3]]),
                  ordering_over_entire_backselect=np.array([[0], [1], [3]]),
                  values_over_entire_backselect=np.array(
                      [np.sqrt(25.81), 5.0, 0.0]),
                  mask=np.array([False, False, False, True]),
              ),
              sis.SISResult(
                  sis=np.array([[1], [0]]),
                  ordering_over_entire_backselect=np.array([[0], [1]]),
                  values_over_entire_backselect=np.array([0.9, 0.0]),
                  mask=np.array([True, True, False, False]),
              ),
          ]),
      dict(
          testcase_name='sum, (2, 2), individual masking, no initial mask',
          f=_F_SUM,
          threshold=4.0,
          initial_input=np.array([[10, 5], [2, 3]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[0, 0]]),
                  ordering_over_entire_backselect=np.array([[1, 0], [1, 1],
                                                            [0, 1], [0, 0]]),
                  values_over_entire_backselect=np.array(
                      [18.0, 15.0, 10.0, 0.0]),
                  mask=np.array([[True, False], [False, False]]),
              ),
              sis.SISResult(
                  sis=np.array([[0, 1]]),
                  ordering_over_entire_backselect=np.array([[1, 0], [1, 1],
                                                            [0, 1]]),
                  values_over_entire_backselect=np.array([8.0, 5.0, 0.0]),
                  mask=np.array([[False, True], [False, False]]),
              ),
              sis.SISResult(
                  sis=np.array([[1, 1], [1, 0]]),
                  ordering_over_entire_backselect=np.array([[1, 0], [1, 1]]),
                  values_over_entire_backselect=np.array([3.0, 0.0]),
                  mask=np.array([[False, False], [True, True]]),
              ),
          ]),
      dict(
          testcase_name='sum, (2, 2), individual masking, specify initial_mask',
          f=_F_SUM,
          threshold=4.0,
          initial_input=np.array([[10, 5], [2, 3]]),
          initial_mask=np.array([[True, True], [True, True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[0, 0]]),
                  ordering_over_entire_backselect=np.array([[1, 0], [1, 1],
                                                            [0, 1], [0, 0]]),
                  values_over_entire_backselect=np.array(
                      [18.0, 15.0, 10.0, 0.0]),
                  mask=np.array([[True, False], [False, False]]),
              ),
              sis.SISResult(
                  sis=np.array([[0, 1]]),
                  ordering_over_entire_backselect=np.array([[1, 0], [1, 1],
                                                            [0, 1]]),
                  values_over_entire_backselect=np.array([8.0, 5.0, 0.0]),
                  mask=np.array([[False, True], [False, False]]),
              ),
              sis.SISResult(
                  sis=np.array([[1, 1], [1, 0]]),
                  ordering_over_entire_backselect=np.array([[1, 0], [1, 1]]),
                  values_over_entire_backselect=np.array([3.0, 0.0]),
                  mask=np.array([[False, False], [True, True]]),
              ),
          ]),
      dict(
          testcase_name='sum, (2, 2), mask over cols',
          f=_F_SUM,
          threshold=10.0,
          initial_input=np.array([[10, 5], [2, 3]]),
          initial_mask=np.array([[True, True]]),
          fully_masked_input=np.array([[0, 0], [0, 0]]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[0, 0]]),
                  ordering_over_entire_backselect=np.array([[0, 1], [0, 0]]),
                  values_over_entire_backselect=np.array([12.0, 0.0]),
                  mask=np.array([[True, False]]),
              ),
          ]),
      dict(
          testcase_name='sum, (3, 2), mask over cols',
          f=_F_SUM,
          threshold=10.0,
          initial_input=np.array([[10, 5, 9], [2, 3, 1]]),
          initial_mask=np.array([[True, True, True]]),
          fully_masked_input=np.array([[0, 0, 0], [0, 0, 0]]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[0, 0]]),
                  ordering_over_entire_backselect=np.array([[0, 1], [0, 2],
                                                            [0, 0]]),
                  values_over_entire_backselect=np.array([22.0, 12.0, 0.0]),
                  mask=np.array([[True, False, False]]),
              ),
              sis.SISResult(
                  sis=np.array([[0, 2]]),
                  ordering_over_entire_backselect=np.array([[0, 1], [0, 2]]),
                  values_over_entire_backselect=np.array([10.0, 0.0]),
                  mask=np.array([[False, False, True]]),
              ),
          ]),
      dict(
          testcase_name='sum, (3, 2), mask over rows',
          f=_F_SUM,
          threshold=5.0,
          initial_input=np.array([[10, 5, 9], [2, 3, 1]]),
          initial_mask=np.array([[True], [True]]),
          fully_masked_input=np.array([[0, 0, 0], [0, 0, 0]]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[0, 0]]),
                  ordering_over_entire_backselect=np.array([[1, 0], [0, 0]]),
                  values_over_entire_backselect=np.array([24.0, 0.0]),
                  mask=np.array([[True], [False]]),
              ),
              sis.SISResult(
                  sis=np.array([[1, 0]]),
                  ordering_over_entire_backselect=np.array([[1, 0]]),
                  values_over_entire_backselect=np.array([0.0]),
                  mask=np.array([[False], [True]]),
              ),
          ]),
      dict(
          testcase_name='linregress, two SIS',
          f=_F_LINREGRESS,
          threshold=5.0,
          initial_input=np.array([5, 1, 6, 3]),
          initial_mask=np.array([True, True, True, True]),
          fully_masked_input=np.array([0, 0, 0, 0]),
          expected_sis_collection=[
              sis.SISResult(
                  sis=np.array([[0]]),
                  ordering_over_entire_backselect=np.array([[2], [1], [3],
                                                            [0]]),
                  values_over_entire_backselect=np.array([10.0, 8.0, 5.0, 0.0]),
                  mask=np.array([True, False, False, False]),
              ),
              sis.SISResult(
                  sis=np.array([[3], [1]]),
                  ordering_over_entire_backselect=np.array([[2], [1], [3]]),
                  values_over_entire_backselect=np.array([5.0, 3.0, 0.0]),
                  mask=np.array([False, True, False, True]),
              ),
          ]),
  )
  def test_sis_collection(self,
                          f,
                          threshold,
                          initial_input,
                          fully_masked_input,
                          expected_sis_collection,
                          initial_mask=None):
    actual_sis_collection = sis.sis_collection(
        f,
        threshold,
        initial_input,
        fully_masked_input,
        initial_mask=initial_mask)
    self._assert_equal_sis_collection(expected_sis_collection,
                                      actual_sis_collection)


if __name__ == '__main__':
  absltest.main()
