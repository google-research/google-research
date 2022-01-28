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

"""Tests for experiment."""

from absl.testing import absltest
import numpy as np

from dp_topk import experiment


class ExperimentTest(absltest.TestCase):

  def test_linf_error_zero(self):
    true_top_k = np.array([99, 99, 38, 12, 11])
    est_top_k = np.array([99, 99, 38, 12, 11])
    linf_error = experiment.linf_error(true_top_k, est_top_k)
    expected_linf_error = 0
    self.assertAlmostEqual(linf_error, expected_linf_error, places=6)

  def test_linf_error_nonzero(self):
    true_top_k = np.array([99, 99, 38, 12, 11])
    est_top_k = np.array([99, 38, 12, 11, 0])
    linf_error = experiment.linf_error(true_top_k, est_top_k)
    expected_linf_error = 61
    self.assertAlmostEqual(linf_error, expected_linf_error, places=6)

  def test_l1_error_zero(self):
    true_top_k = np.array([99, 99, 38, 12, 11])
    est_top_k = np.array([99, 99, 38, 12, 11])
    l1_error = experiment.l1_error(true_top_k, est_top_k)
    expected_l1_error = 0
    self.assertAlmostEqual(l1_error, expected_l1_error, places=6)

  def test_l1_error_nonzero(self):
    true_top_k = np.array([99, 99, 38, 12, 11])
    est_top_k = np.array([99, 38, 12, 11, 0])
    l1_error = experiment.l1_error(true_top_k, est_top_k)
    expected_l1_error = 99
    self.assertAlmostEqual(l1_error, expected_l1_error, places=6)

  def test_k_relative_error_zero(self):
    true_top_k = np.array([99, 99, 38, 12, 11])
    est_top_k = np.array([99, 99, 38, 12, 11])
    k_relative_error = experiment.k_relative_error(true_top_k, est_top_k)
    expected_k_relative_error = 0
    self.assertAlmostEqual(
        k_relative_error, expected_k_relative_error, places=6)

  def test_k_relative_error_nonzero(self):
    true_top_k = np.array([99, 99, 38, 12, 11])
    est_top_k = np.array([99, 38, 12, 11, 0])
    k_relative_error = experiment.k_relative_error(true_top_k, est_top_k)
    expected_k_relative_error = 11
    self.assertAlmostEqual(
        k_relative_error, expected_k_relative_error, places=6)


if __name__ == '__main__':
  absltest.main()
