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

"""Tests for the library of associated Legendre function."""
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
import numpy as np
import scipy.special as sp_special

from simulation_research.signal_processing.spherical import associated_legendre_function


def _apply_normalization(a):
  """Applies normalization to the associated Legendre functions."""
  num_l, num_m, _ = a.shape

  a_normalized = np.zeros_like(a)
  for l in range(num_l):
    for m in range(num_m):
      c0 = (2.0 * l + 1.0) * sp_special.factorial(l - m)
      c1 = (4.0 * math.pi) * sp_special.factorial(l + m)
      c2 = np.sqrt(c0 / c1)
      a_normalized[l, m] = c2 * a[l, m]

  return a_normalized


class AssociatedLegendreFunctionTest(parameterized.TestCase):

  # The maximum degree of the associated Legendre function.
  MAX_DEGREE = [3, 4, 6, 32, 64]

  # The size of the inputs.
  SIZE_INPUT = [2, 3, 4, 64, 128]

  @parameterized.parameters(*itertools.product(MAX_DEGREE, SIZE_INPUT))
  def testGenerateAssociatedLegendreFunctionWithNonnegativeOrder(self, l_max,
                                                                 num_x):

    # Generates dummy points on which the associated Legendre functions are
    # evaluated.
    x = np.linspace(-0.2, 0.9, num_x)

    actual_p = associated_legendre_function.gen_normalized_legendre(
        l_max, jnp.asarray(x))

    # The expected results are obtained from scipy.
    expected_p = np.zeros((l_max + 1, l_max + 1, num_x))
    for i in range(num_x):
      expected_p[:, :, i] = np.transpose(sp_special.lpmn(l_max, l_max, x[i])[0])

    # The results from scipy are not normalized and the comparison requires
    # normalizing the results.
    expected_p_normalized = _apply_normalization(expected_p)

    np.testing.assert_allclose(actual_p,
                               jnp.asarray(expected_p_normalized),
                               rtol=1e-06,
                               atol=3.2e-06)

if __name__ == '__main__':
  absltest.main()
