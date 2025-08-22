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

"""Tests for Cartan-Dynkin code."""

import unittest

from m_theory_lib import algebra
from m_theory_lib import cartan_dynkin
from m_theory_lib import m_util as mu
import numpy

# Uppercase/lowercase naming follows maths.
# pylint:disable=invalid-name


class CartanDynkinTests(unittest.TestCase):
  """Tests Cartan-Dynkin code."""

  def test_e7_56(self):
    """Tests the weightspace decomposition of the E7 56-irrep."""
    e7 = algebra.g.e7
    spin8 = algebra.g.spin8
    t_aMN = e7.t_a_ij_kl
    fano_mabs = numpy.stack([spin8.gamma_vvvvss[ijkl] for ijkl in e7.fano_ijkl],
                            axis=0)
    cartan7_op56_exact = mu.nsum('nsS,asS,aMN->nNM',
                                 fano_mabs,
                                 e7.v70_from_sc8x8[:, 0, :, :],
                                 t_aMN[:70, :, :]) / 8
    # Simulate numerically-noisy (with reproducible noise) Cartan generators.
    cartan7_op56 = cartan7_op56_exact + numpy.random.RandomState(0).normal(
        size=cartan7_op56_exact.shape, scale=1e-10)
    e7_56_ws = list(cartan_dynkin.get_weightspaces(cartan7_op56).items())
    self.assertEqual({w.shape for _, w in e7_56_ws}, {(1, 56)})
    weights_with_bad_eigenvalues = [
        weight for weight, _ in e7_56_ws
        if any(w not in (-0.5, 0, 0.5) for w in weight)]
    self.assertEqual(weights_with_bad_eigenvalues, [])
    weights_with_bad_allocation = [
        weight for weight, _ in e7_56_ws
        if sum(w == 0 for w in weight) != 4]
    self.assertEqual(weights_with_bad_allocation, [])

if __name__ == '__main__':
  unittest.main()
