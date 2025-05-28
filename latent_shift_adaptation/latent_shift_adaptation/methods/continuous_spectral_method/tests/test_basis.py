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

"""test the creat_basis.py file."""
import unittest
from latent_shift_adaptation.methods.continuous_spectral_method import create_basis
from latent_shift_adaptation.methods.continuous_spectral_method.utils import gram_schmidt
import numpy as np


class TestBasisFunctions(unittest.TestCase):
  """test the creat_basis.py file."""

  def test_basis_from_cluster(self):
    """test the basis from cluster method."""
    p = 1000
    data = np.zeros((p, 1))
    for i in range(p):
      s1 = 0.9*np.random.normal(-2, 1, size=(1, 1))
      s2 = 0.9*np.random.normal(0, 2, size=(1, 1))
      data[i] = s1 + s2

    out_list = create_basis.basis_from_cluster(data, 3, 0.425,
                                               select_pos=True)
    # test that all wieghts are positive,
    for f in out_list:
      assert (f.get_params()['weight'] >= 0.).all()
    # test that f is orthonormal
    ortho_list, r_mat = gram_schmidt(out_list)
    for f, g in zip(out_list, ortho_list):
      assert(f.get_params() == g.get_params())
    np.testing.assert_allclose(r_mat, np.eye(r_mat.shape[0]))

  def test_basis_from_centers(self):
    """test the basis from cluster method."""
    out_list = create_basis.basis_from_centers(np.array([-1, 2, 3]),
                                               0.425, select_pos=True)
    # test that all wieghts are positive,
    for f in out_list:
      assert((f.get_params()['weight'] >= 0.).all())
    # test that f is orthonormal
    ortho_list, r_mat = gram_schmidt(out_list)
    for f, g in zip(out_list, ortho_list):
      assert(f.get_params() == g.get_params())
    np.testing.assert_allclose(r_mat, np.eye(r_mat.shape[0]))


if __name__ == '__main__':
  unittest.main()
