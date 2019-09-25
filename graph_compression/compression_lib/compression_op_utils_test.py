# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for compression_op_utils."""

from absl.testing import absltest

from compression_lib import compression_op_utils


class CompressionOpUtilsTest(absltest.TestCase):

  def testCompressedRankComputationUsingCorrectDim(self):
    shape = (10, 20)
    rank_factor = 200
    expected_rank = 5  # half of the smaller dimension.
    rank = compression_op_utils.compute_compressed_rank_from_matrix_shape(
        shape, rank_factor)
    self.assertEqual(expected_rank, rank)

    shape = (20, 10)
    rank = compression_op_utils.compute_compressed_rank_from_matrix_shape(
        shape, rank_factor)
    self.assertEqual(expected_rank, rank)

  def testCompressedRankComputationLowerBound(self):
    shape = (10, 20)
    # Following rank_factor means 2x the current rank -- which is not
    # compression. Util method should return the current rank.
    rank_factor = 50
    rank = compression_op_utils.compute_compressed_rank_from_matrix_shape(
        shape, rank_factor)
    self.assertEqual(10, rank)

  def testCompressedRankComputationUnevenDivision(self):
    shape = (9, 20)
    rank_factor = 200
    rank = compression_op_utils.compute_compressed_rank_from_matrix_shape(
        shape, rank_factor)
    # Expected rank should be half of the original rank -- rounded up.
    self.assertEqual(5, rank)


if __name__ == '__main__':
  absltest.main()
