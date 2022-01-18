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

"""Tests for the Pfam library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import train_hmmer_model_for_paper

FLAGS = flags.FLAGS


class TestTrainHmmerModelForPaper(parameterized.TestCase):

  @parameterized.parameters(*train_hmmer_model_for_paper.FOLD_ENUM_VAULES)
  def testTrainAndTestPathsFromRandomSplit(self, fold):
    split = train_hmmer_model_for_paper.RANDOM_SPLIT
    train_path, test_path = (
        train_hmmer_model_for_paper.train_and_test_paths_from(
            split=split, fold=fold))

    self.assertIn('random', train_path)
    self.assertIn('random', test_path)
    self.assertIn(fold, test_path)

  @parameterized.parameters(*train_hmmer_model_for_paper.FOLD_ENUM_VAULES)
  def testTrainAndTestPathsFromClusteredSplit(self, fold):
    split = train_hmmer_model_for_paper.CLUSTERED_SPLIT
    train_path, test_path = (
        train_hmmer_model_for_paper.train_and_test_paths_from(
            split=split, fold=fold))

    self.assertIn('clustered', train_path)
    self.assertIn('clustered', test_path)
    self.assertIn(fold, test_path)

  def testTrainAndTestPathsFromRaisesErrorBadFold(self):
    with self.assertRaisesRegex(ValueError, 'fold'):
      train_hmmer_model_for_paper.train_and_test_paths_from(
          split=train_hmmer_model_for_paper.CLUSTERED_SPLIT,
          fold='THIS IS NOT A FOLD')

  def testTrainAndTestPathsFromRaisesErrorBadSplit(self):
    with self.assertRaisesRegex(ValueError, 'split'):
      train_hmmer_model_for_paper.train_and_test_paths_from(
          split='THIS IS NOT A SPLIT',
          fold=train_hmmer_model_for_paper.TRAIN_FOLD)


if __name__ == '__main__':
  absltest.main()
