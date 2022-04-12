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

"""Tests for negative_cache.retrieval_fns."""

from unittest import mock

import tensorflow.compat.v2 as tf
from negative_cache import retrieval_fns


class RetrievalFnsTest(tf.test.TestCase):

  def test_max_score_retrieval_fn(self):
    max_score_retrieval_fn = retrieval_fns.MaxScoreRetrievalFn()
    scores = tf.convert_to_tensor([[1.0, 2.0, 0.0], [-1.0, 0.0, 3.0],
                                   [4.0, -1.0, -1.0]])
    expected = tf.convert_to_tensor([[1], [2], [0]])
    actual = max_score_retrieval_fn(scores)
    self.assertAllEqual(expected, actual)

  def test_gumbel_max_retrieval_fn_with_mock_randomness(self):
    with mock.patch.object(retrieval_fns,
                           '_sample_gumbel') as mock_sample_gumbel:
      scores = tf.convert_to_tensor([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                                     [0.0, 0.0, -1.0]])
      mock_sample_gumbel.return_value = tf.convert_to_tensor([[0.0, 0.1, -0.1],
                                                              [1.0, -2.0, 3.3],
                                                              [0.1, -0.1, 2.0]])
      gumbel_max_retrieval_fn = retrieval_fns.GumbelMaxRetrievalFn()
      expected = tf.convert_to_tensor([[0], [0], [2]])
      actual = gumbel_max_retrieval_fn(scores)
    self.assertAllEqual(expected, actual)
    expected_shape = tf.convert_to_tensor([3, 3])
    mock_sample_gumbel.assert_called_once()
    mock_args, mock_kwargs = mock_sample_gumbel.call_args
    self.assertAllEqual(mock_args, (expected_shape,))
    self.assertEqual(mock_kwargs, {})

  def test_gumbel_max_retrieval_fn_with_temperature_and_mock_randomness(self):
    with mock.patch.object(retrieval_fns,
                           '_sample_gumbel') as mock_sample_gumbel:
      scores = tf.convert_to_tensor([[1.0, 0.0, 0.0]])
      mock_sample_gumbel.return_value = tf.convert_to_tensor([[0.0, 0.9, 0.0]])
      gumbel_max_retrieval_fn = retrieval_fns.GumbelMaxRetrievalFn(inv_temp=0.5)
      expected = tf.convert_to_tensor([[1]])
      actual = gumbel_max_retrieval_fn(scores)
    self.assertAllEqual(expected, actual)

  def test_gumbel_max_retrieval_fn_has_correct_output_shape(self):
    scores = tf.convert_to_tensor([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                                   [0.0, 0.0, -1.0]])
    gumbel_max_retrieval_fn = retrieval_fns.GumbelMaxRetrievalFn()
    output = gumbel_max_retrieval_fn(scores)
    self.assertAllEqual([3, 1], tf.shape(output))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
