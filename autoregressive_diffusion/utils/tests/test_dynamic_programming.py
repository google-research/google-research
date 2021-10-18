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

"""Tests dynamic programming code.
"""
from absl.testing import absltest

import numpy as np
from autoregressive_diffusion.utils import dynamic_programming


class DPTest(absltest.TestCase):
  """Tests categorical diffusion class."""

  def test_compute_fixed_budget(self):
    """Tests for a specific KL if the program computes cost correctly."""
    kl_per_t = np.array([5., 4., 3., 2.85, 2.80])
    budgets = [1, 2, 3, 4, 5]

    correct_outs = [25, 19, 18, 17.7, 17.65]

    _, costs = dynamic_programming.compute_fixed_budget(kl_per_t, budgets)

    self.assertTrue(np.allclose(costs, correct_outs))


if __name__ == '__main__':
  absltest.main()




