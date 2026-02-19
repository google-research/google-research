# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

from absl.testing import absltest
from dp_mm_domain import set_union_experiment


class ExperimentTest(absltest.TestCase):

  def test_compute_captured_mass_one(self):
    input_data = [
        [1, 2, 3],
        [1, 2, 4],
        [5],
        [1, 5, 6],
        [7, 8],
    ]
    output = [1, 2, 3, 4, 5, 6, 7, 8]
    captured_mass = set_union_experiment.compute_captured_mass(
        input_data, output
    )
    self.assertAlmostEqual(captured_mass, 1.0, places=6)

  def test_compute_captured_mass_zero(self):
    input_data = [
        [1, 2, 3],
        [1, 2, 4],
        [5],
        [1, 5, 6],
        [7, 8],
    ]
    output = []
    captured_mass = set_union_experiment.compute_captured_mass(
        input_data, output
    )
    self.assertAlmostEqual(captured_mass, 0.0, places=6)

  def test_compute_captured_mass_low(self):
    input_data = [
        [1, 2, 3],
        [1, 2, 4],
        [5, 10, 11],
        [1, 5, 6],
        [7, 8, 5],
    ]
    output = [10]
    captured_mass = set_union_experiment.compute_captured_mass(
        input_data, output
    )
    self.assertAlmostEqual(captured_mass, 1 - (14 / 15), places=6)

  def test_captured_captured_mass_high(self):
    input_data = [
        [1, 2, 3],
        [1, 2, 4],
        [5, 10, 11],
        [1, 5, 6],
        [7, 8, 5],
    ]
    output = [1, 2, 4, 5]
    captured_mass = set_union_experiment.compute_captured_mass(
        input_data, output
    )
    self.assertAlmostEqual(captured_mass, 1 - (6 / 15), places=6)


if __name__ == "__main__":
  absltest.main()
