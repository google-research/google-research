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

"""test the functions in utils.py file."""

import unittest
import cosde
from cosde import utils
from latent_shift_adaptation.methods.continuous_spectral_method.utils import gram_schmidt
import numpy as np
import sklearn


class TestUtils(unittest.TestCase):
  """test the functions in utils.py file."""

  def test_gram_schmidt(self):
    """test the Gram-Schmidt procedure."""
    eb_list = []
    kernel = sklearn.gaussian_process.kernels.RBF(0.45)
    for i in range(5):
      x = np.random.normal(2 * i, 1, size=(1, 1))
      eb = cosde.base.EigenBase(kernel, x, np.array([1.0]))
      eb_list.append(eb)
    out_list, r_mat = gram_schmidt(eb_list)
    for f in out_list:
      np.testing.assert_allclose(utils.l2_norm_base(f), 1.0)
    # test that all functions are orthonormal
    for f in out_list:
      for g in out_list:
        if f != g:
          np.testing.assert_allclose(
              utils.inner_product_base(f, g), 0.0, 1e-3, 1e-3
          )
    # test that the matrix is upper tirangular
    for i in range(5):
      for j in range(5):
        if j > i:
          np.testing.assert_allclose(r_mat[j, i], 0.0)


if __name__ == "main":
  unittest.main()
