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

"""Tests for setup stability score."""

import numpy as np
import pandas as pd
import tensorflow as tf

from stable_transfer.transferability.results_analysis import setup_stability_score


class SetupStabilityScoreTest(tf.test.TestCase):
  """Test the setup stability score."""

  def testEdgeCasesSsScore(self):
    """Test the edge cases of the SS score.

    Creation of random experiments with some overlapping components (a, b, c).
    Expected DataFrame structure:

    |Transf. Metric| CompA | CompB | CompC |Eval. Measure Score|
    ------------------------------------------------------------
    |     gbc      |   x   |   y   |   z   |        0.5        |
    |     leep     |   x   |   y   |   z   |        0.2        |
    |     logme    |   x   |   y   |   z   |        0.8        |
    |     gbc      |   x   |   y   |   k   |        0.5        |
    |     leep     |   x   |   y   |   k   |        0.8        |
    |     logme    |   x   |   y   |   k   |        0.2        |
    ...

    For example we can study the stability to the variation of the CompC z --> k
    For z the scores would be (0.5, 0.2, 0.8) while for k (0.5, 0.8, 0.2)
    The agreement for single pairs of experiments is measured with kendalltau
    """

    num_exp = 100
    same_comp_a = [0] * (num_exp * 3)
    same_comp_b = [1] * (num_exp * 3)
    diff_comp_c = np.repeat(np.arange(num_exp), 3)
    perfect_outcomes = [0, 1, 2] * num_exp
    np.random.seed(0)
    rnd_outcomes = np.random.uniform(0, 1, num_exp * 3)

    test_df = pd.DataFrame()
    test_df['comp_a'] = same_comp_a
    test_df['comp_b'] = same_comp_b
    test_df['comp_c'] = diff_comp_c
    test_df['Evaluation Measure Score'] = perfect_outcomes
    perfect_ss_score = setup_stability_score.get_setup_stability_score(
        test_df, varying='comp_c', fixing=['comp_a', 'comp_b'])
    # When the ranking is the same, the SS score should be 1.
    self.assertAlmostEqual(perfect_ss_score, 1.0)

    test_df['Evaluation Measure Score'] = rnd_outcomes
    rnd_ss_score = setup_stability_score.get_setup_stability_score(
        test_df, varying='comp_c', fixing=['comp_a', 'comp_b'])
    # When the ranking is random, the SS score should be close to 0.
    self.assertAlmostEqual(rnd_ss_score, 0.0, delta=0.01)

if __name__ == '__main__':
  tf.test.main()
