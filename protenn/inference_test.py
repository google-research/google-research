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

"""Tests for module inference.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from protenn import inference_lib
from protenn import test_util


class _InferrerFixture(object):
  """A mock inferrer object.

  See docstring for get_activations.
  """
  activation_type = 'serving_default'

  def __init__(self, activation_rank=1):
    """Constructs a mock inferrer with activation output of specified rank.

    Args:
      activation_rank: int. Use 1 for activations that have a single float per
        sequence, 2. for a vector per sequence, etc.
    """
    self._activation_rank = activation_rank

  def get_variable(self, x):
    if x == 'label_vocab:0':
      return np.array(['LABEL1'])
    else:
      raise ValueError(
          'Fixture does not have an implementation for this variable')

  def get_activations(self, input_seqs):
    """Returns a np.array with contents that are the length of each seq.

    The shape of the np.array is dictated by self._activation_rank - see
    docstring of __init__ for more information.

    Args:
      input_seqs: list of string.

    Returns:
      np.array of rank self._activation_rank, where the entries are the length
      of each input seq.
    """
    return np.reshape([len(s) for s in input_seqs],
                      [-1] + [1] * (self._activation_rank - 1))


class InferenceLibTest(parameterized.TestCase, tf.test.TestCase):

  def testInference(self):
    inferrer = inference_lib.Inferrer(
        test_util.savedmodel_path(),
        use_latest_savedmodel=True,
    )

    input_seq = 'AP'
    for total_size in range(1, 15, 3):
      full_list = [input_seq] * total_size
      activations = inferrer.get_activations(full_list)
      expected_inference_length = total_size
      actual_inference_length = len(activations)
      self.assertEqual(actual_inference_length, expected_inference_length)

      expected_num_amino_acids = len(input_seq)
      actual_num_amino_acids = activations[0].shape[0]
      self.assertEqual(actual_num_amino_acids, expected_num_amino_acids)

      expected_number_classes = 6
      actual_embedding_length = activations[0].shape[1]
      self.assertEqual(actual_embedding_length, expected_number_classes)

  def testStringInput(self):
    inferrer = inference_lib.Inferrer(
        test_util.savedmodel_path(),
        use_latest_savedmodel=True,
    )
    # Simulate failure to use a list.
    with self.assertRaisesRegex(
        ValueError, '`list_of_seqs` should be convertible to a '
        'numpy vector of strings. Got *'):
      inferrer.get_activations('QP')

  def testMemoizedInferrerLoading(self):
    inferrer = inference_lib.memoized_inferrer(
        test_util.savedmodel_path(),
        memoize_inference_results=True,
        use_latest_savedmodel=True,
    )
    memoized_inferrer = inference_lib.memoized_inferrer(
        test_util.savedmodel_path(),
        memoize_inference_results=True,
        use_latest_savedmodel=True,
    )

    self.assertIs(inferrer, memoized_inferrer)

  def testMemoizedInferenceResults(self):
    inferrer = inference_lib.Inferrer(
        test_util.savedmodel_path(),
        memoize_inference_results=True,
        use_latest_savedmodel=True,
    )
    activations = inferrer._get_activations_for_batch('ADE')
    memoized_activations = inferrer._get_activations_for_batch('ADE')

    self.assertIs(activations, memoized_activations)

  def testGetVariable(self):
    inferrer = inference_lib.Inferrer(
        test_util.savedmodel_path(),
        use_latest_savedmodel=True,
    )
    output = inferrer.get_variable('conv1d/bias:0')
    self.assertNotEmpty(output)

  def test_predictions_for_df(self):
    inferrer_fixture = _InferrerFixture()
    input_seqs = ['AAAA', 'DDD', 'EE', 'W']
    input_df = pd.DataFrame({
        'sequence_name': input_seqs,
        'sequence': input_seqs
    })
    actual_output_df = inference_lib.predictions_for_df(
        input_df, inferrer_fixture
    )

    self.assertEqual(actual_output_df['predictions'].values.tolist(),
                     [4, 3, 2, 1])

    self.assertEqual(actual_output_df.sequence_name.values.tolist(), input_seqs)

  def test_mean_sparse_acts(self):
    input_acts_1 = [(0, 1, 1), (1, 1, 1)]
    input_acts_2 = []
    input_sequence_length = 2
    num_output_classes = 3
    actual = inference_lib.mean_sparse_acts(
        input_sequence_length, [input_acts_1, input_acts_2], num_output_classes
    )
    expected = [(0, 1, 0.5), (1, 1, 0.5)]

    self.assertCountEqual(actual, expected)

  def test_mean_sparse_handles_things_that_round_to_zero(self):
    input_acts_1 = [(0, 1, 0.0000000001)]
    input_acts_2 = [(1, 1, 0.00000001)]
    input_sequence_length = 2
    num_output_classes = 3
    actual = inference_lib.mean_sparse_acts(
        input_sequence_length, [input_acts_1, input_acts_2], num_output_classes
    )
    # Expect no output, because rounding made things zero.
    expected = []

    self.assertCountEqual(actual, expected)

  def test_get_sparse_calls_by_inferrer(self):
    input_sequences = ['AAAA', 'DDD', 'EE']
    input_inferrer_list = [
        inference_lib.Inferrer(
            test_util.savedmodel_path(),
            use_latest_savedmodel=True,
        ),
    ] * 2
    actual = inference_lib.get_sparse_calls_by_inferrer(
        sequences=input_sequences, inferrer_list=input_inferrer_list
    )

    # Test to make sure each inferrer has an inference result.
    self.assertLen(actual, len(input_inferrer_list))

    # Test to make sure each sequence gets inference results from each inferrer.
    self.assertLen(actual[0], len(input_sequences))
    self.assertLen(actual[1], len(input_sequences))

    # Test to make sure the precision is truncated in the inference results.
    reasonable_number_of_decimal_places = 6
    # First dimension: inferrer. Second dimension: sequence.
    # Third dimension: coo_ijv_list. Fourth dimension: v (the value that should
    # have truncated precision.
    actual_number_with_decimal_places = actual[0][0][0][2]
    self.assertLessEqual(
        len(str(actual_number_with_decimal_places)),
        reasonable_number_of_decimal_places,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no sparse activations',
          input_sparse_act=[],
          input_sequence_length=2,
          input_family_to_clan={'PF00001': 'CL0192'},
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_min_domain_call_length=20,
          expected=[],
      ),
      dict(
          testcase_name='single family domain call gets included',
          input_sparse_act=[(i, 0, 1) for i in range(30)],
          input_family_to_clan={'PF00001': 'CL0192'},
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_min_domain_call_length=20,
          expected=[('PF00001', (1, 30))],
      ),
      dict(
          testcase_name='single clan domain call gets included',
          input_sparse_act=[(i, 1, 1) for i in range(30)],
          input_family_to_clan={'PF00001': 'CL0192'},
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_min_domain_call_length=20,
          expected=[('CL0192', (1, 30))],
      ),
      dict(
          testcase_name='single short domain call gets EXcluded',
          input_sparse_act=[(i, 0, 1) for i in range(3)],
          input_family_to_clan={'PF00001': 'CL0192'},
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_min_domain_call_length=20,
          expected=[],
      ),
      dict(
          testcase_name='two domain calls get included',
          input_sparse_act=(
              [(i, 0, 1) for i in range(5)] + [(i, 1, 1) for i in range(6, 10)]
          ),
          input_family_to_clan={'PF00001': 'CL0192'},
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_min_domain_call_length=1,
          expected=[('PF00001', (1, 5)), ('CL0192', (7, 10))],
      ),
      dict(
          testcase_name='normalization happens',
          input_sparse_act=(
              # These both are in the same clan, and that clan should get a
              # call.
              [(0, 0, 1)]
              + [(1, 1, 1)]
          ),
          input_family_to_clan={'PF00001': 'CL0192'},
          input_vocab=np.array(['PF00001', 'CL0192']),
          input_min_domain_call_length=2,
          expected=[('CL0192', (1, 2))],
      ),
      dict(
          testcase_name='competition happens',
          input_sparse_act=(
              # These should get competed, and only one call should be made.
              [(0, 0, 1), (1, 0, 1)]
              + [(1, 1, 1)]
          ),
          input_family_to_clan={'PF00001': 'PF00001', 'PF00002': 'PF00002'},
          input_vocab=np.array(['PF00001', 'PF00002']),
          input_min_domain_call_length=1,
          expected=[('PF00001', (1, 2))],
      ),
  )
  def test_get_competed_labels(
      self,
      input_sparse_act,
      input_family_to_clan,
      input_vocab,
      input_min_domain_call_length,
      expected,
      input_sequence_length=100,
      input_known_nested_domains=(),
      input_reporting_threshold=0.025,
  ):
    input_label_to_idx = {k: i for i, k in enumerate(input_vocab)}
    actual = inference_lib.get_competed_labels(
        sparse_act=input_sparse_act,
        sequence_length=input_sequence_length,
        family_to_clan=input_family_to_clan,
        label_to_idx=input_label_to_idx,
        vocab=input_vocab,
        known_nested_domains=input_known_nested_domains,
        reporting_threshold=input_reporting_threshold,
        min_domain_call_length=input_min_domain_call_length,
    )
    self.assertCountEqual(actual, expected)

  def test_get_preds_at_or_above_threshold(self):
    input_df = pd.DataFrame([
        {'sequence_name': 'seq2', 'sequence': 'AAA'},
        {'sequence_name': 'seq1', 'sequence': 'CC'},
    ])
    input_inferrer_list = [
        inference_lib.Inferrer(
            test_util.savedmodel_path(),
            use_latest_savedmodel=True,
        ),
    ] * 2
    input_model_cache_path = self.create_tempdir()
    with open(
        os.path.join(input_model_cache_path, 'vocab_pfam35.tsv'), 'w'
    ) as f:
      for class_num in range(1, 7):
        f.write(f'PF0000{class_num}\n')
    with open(
        os.path.join(input_model_cache_path, 'clans_pfam35.tsv'), 'w'
    ) as f:
      for class_num in range(6):
        f.write(f'PF0000{class_num}\t\t\t\t')
    with open(
        os.path.join(input_model_cache_path, 'nested_domains_pfam35.txt'),
        'w',
    ):
      pass
    input_reporting_threshold = 0.01
    input_min_domain_call_length = 1
    actual = inference_lib.get_preds_at_or_above_threshold(
        input_df=input_df,
        inferrer_list=input_inferrer_list,
        model_cache_path=input_model_cache_path,
        reporting_threshold=input_reporting_threshold,
        min_domain_call_length=input_min_domain_call_length,
    )

    # The model is untrained, and as such we expect every output to be fairly
    # close to .5. This causes all classes to be called, which causes
    # competition, and thus for PF00001 to win, because it's first
    # lexicographically.
    expected = [('PF00001', (1, 3))], [('PF00001', (1, 2))]
    self.assertSequenceEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
