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

"""Tests for read solution file for Universal Embedding challenges."""

import os
from absl.testing import absltest

from universal_embedding_challenge import read_retrieval_solution


class LoadSolutionTest(absltest.TestCase):

  def testLoadSolutionWorks(self):
    # Define inputs.
    file_path = os.path.join(
        os.environ.get('TEST_TMPDIR', ''), 'retrieval_solution.csv')
    with open(file_path, 'w') as f:
      f.write('id,images,Usage\n')
      f.write('0223456789abcdef,fedcba9876543210 fedcba9876543200,Public\n')
      f.write('0323456789abcdef,fedcba9876543200,Private\n')
      f.write('0423456789abcdef,fedcba9876543220,Private\n')

    # Run tested function.
    (public_solution,
     private_solution) = read_retrieval_solution.LoadSolution(file_path)

    # Define expected results.
    expected_public_solution = {
        '0223456789abcdef': ['fedcba9876543210', 'fedcba9876543200'],
    }
    expected_private_solution = {
        '0323456789abcdef': ['fedcba9876543200'],
        '0423456789abcdef': ['fedcba9876543220'],
    }

    # Compare actual and expected results.
    self.assertDictEqual(public_solution, expected_public_solution)
    self.assertDictEqual(private_solution, expected_private_solution)


if __name__ == '__main__':
  absltest.main()
