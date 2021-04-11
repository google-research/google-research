# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for gmres."""
# pylint: skip-file

import jax.numpy as np
import numpy as onp

from . import gmres


class GmresTest(parameterized.TestCase):
  @parameterized.parameters(
    onp.arange(10))
  def test_gmres(self, rng):
    ''' asserts that gmres is a valid solver '''
    onp.random.seed(rng)
    b = onp.random.rand(10)
    Am = onp.random.rand(10,10)
    @Partial
    def A(x):
      return Am @ x

    x_opt, record = gmres.gmres(A, b, n=9, record=True)
    rec_diff = record[:-1] - record[1:]
    onp.testing.assert_array_equal(abs(rec_diff),rec_diff)
    onp.testing.assert_allclose(np.linalg.norm(A(x_opt)-b)/np.linalg.norm(b), record[-1], rtol=0,atol=5e-6)
    x_opt, record = gmres.gmres(A, b, n=10, record=True)
    onp.testing.assert_allclose(0, np.linalg.norm(A(x_opt)-b)/np.linalg.norm(b), rtol=0, atol=5e-5)

  def test_identity(self):
    x = onp.random.rand(5)
    y = gmres.identity(x)
    onp.testing.assert_allclose(x,y)

