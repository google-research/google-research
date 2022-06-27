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

"""Tests for xc.utils."""

from absl.testing import absltest
from symbolic_functionals.syfes.xc import utils


def _add(a, b):
  return a + b


def _sub(a, b):
  return a - b


def _mul(a, b):
  return a * b


class UtilsTest(absltest.TestCase):

  def test_function_sum(self):
    function = utils.function_sum(_add, _sub, _mul)
    self.assertEqual(
        # test both positional and keyword arguments.
        function(2, b=1),
        # (2 + 1) + (2 - 1) + (2 * 1) = 6
        6)

  def test_get_hybrid_rsh_params(self):
    hybrid_coeff, rsh_params = utils.get_hybrid_rsh_params('wb97m_v')

    self.assertAlmostEqual(hybrid_coeff, 1.0)
    self.assertSequenceAlmostEqual(rsh_params, (0.3, 1.0, -0.85))


if __name__ == '__main__':
  absltest.main()
