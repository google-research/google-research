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

# pylint: skip-file
"""Tests for multigrid."""

from . import multigrid

from absl.testing import parameterized
import jax.numpy as np
import numpy as onp

class MultigridTest(parameterized.TestCase):
  @parameterized.parameters(
    onp.arange(10))
  def multigrid_convergence_test(self, rng):
    ''' asserts that multigrid is a valid iterative solver '''
    onp.random.seed(rng)
    mesh = multigrid.equations.Mesh(5)
    A = lambda x: mesh.matvec_helmholtz(0.0, 1.0, multigrid.equations.make_mask, multigrid.equations.make_mask_dual, x)
    b = onp.random.rand(5,5)
    x = onp.zeros((5,5))
    for j in range(10):
      x = multigrid._V_Cycle(x, b, 3, 'R')
    onp.testing.assert_allclose(np.linalg.norm(A(x)-b.ravel())/np.linalg.norm(b.ravel()), 0, rtol=0, atol=1e-5)
