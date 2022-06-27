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

"""Tests for stoichiometry."""

import tempfile
from absl.testing import absltest
from graph_sampler import stoichiometry


class StoichiometryTest(absltest.TestCase):

  def test_infer(self):
    expected_valences = {
        'C': 4,
        'N': 3,
        'N+': 4,
        'O': 2,
        'O-': 1,
        'F': 1,
        'H': 1,
    }
    inferred_valences = stoichiometry.get_valences(expected_valences.keys())
    self.assertEqual(expected_valences, inferred_valences)

    expected_charges = {
        'C': 0,
        'N': 0,
        'N+': 1,
        'O': 0,
        'O-': -1,
        'F': 0,
        'H': 0,
    }
    inferred_charges = stoichiometry.get_charges(expected_charges.keys())
    self.assertEqual(expected_charges, inferred_charges)

  def test_read_write(self):
    stoich = stoichiometry.Stoichiometry(dict(C=3, O=2, H=1))

    with tempfile.TemporaryFile(mode='w+') as f:
      stoich.write(f)
      f.seek(0)
      new_stoich = stoichiometry.read(f)
      self.assertEqual(new_stoich, stoich)

  def test_enumerate(self):
    stoichs = list(
        stoichiometry.enumerate_stoichiometries(
            3, heavy_elements=['C', 'N', 'N+', 'O', 'O-', 'F']))
    self.assertLen(stoichs, 58)

    stoichs = list(
        stoichiometry.enumerate_stoichiometries(3, heavy_elements=['C', 'F']))
    self.assertLen(stoichs, 9)


if __name__ == '__main__':
  absltest.main()
