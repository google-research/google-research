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

# Tester for topology_from_geometry

from typing import Tuple

from absl.testing import absltest

from google.protobuf import text_format
import numpy as np
import pandas as pd

from smu import dataset_pb2
from smu.geometry import bond_length_distribution
from smu.geometry import smu_molecule
from smu.geometry import topology_from_geom
from smu.parser import smu_utils_lib

# Only needed so we can alter the default bond matching

# For the empirical bond length distributions, the resolution used.
# Which is not necessarily the same as what is used in the production system.
RESOLUTION = 1000


def triangular_distribution(min_dist, dist_max_value,
                            max_dist):
  """Generate a triangular distribution.

  Args:
    min_dist: minimum X value
    dist_max_value: X value of the triangle peak
    max_dist: maximum X value

  Returns:
    Tuple of the X and Y coordinates that represent the distribution.
  """
  population = np.zeros(RESOLUTION, dtype=np.float32)
  x_extent = max_dist - min_dist
  peak_index = int(round((dist_max_value - min_dist) / x_extent * RESOLUTION))
  dy = 1.0 / peak_index
  for i in range(0, peak_index):
    population[i] = (i + 1) * dy

  dy = 1.0 / (RESOLUTION - peak_index)
  for i in range(peak_index, RESOLUTION):
    population[i] = 1.0 - (i - peak_index) * dy

  dx = x_extent / RESOLUTION
  distances = np.arange(min_dist, max_dist, dx, dtype=np.float32)

  return distances, population


@absltest.skip("This test was not updated when all Hs have to be explicit")
class TestTopoFromGeom(absltest.TestCase):

  def test_scores(self):
    carbon = dataset_pb2.BondTopology.ATOM_C
    single_bond = dataset_pb2.BondTopology.BondType.BOND_SINGLE
    double_bond = dataset_pb2.BondTopology.BondType.BOND_DOUBLE

    # For testing, turn off the need for complete matching.
    smu_molecule.default_must_match_all_bonds = False

    all_distributions = bond_length_distribution.AllAtomPairLengthDistributions(
    )
    x, y = triangular_distribution(1.0, 1.4, 2.0)
    df = pd.DataFrame({"length": x, "count": y})
    bldc1c = bond_length_distribution.EmpiricalLengthDistribution(df, 0.0)
    all_distributions.add(carbon, carbon, single_bond, bldc1c)

    x, y = triangular_distribution(1.0, 1.5, 2.0)
    df = pd.DataFrame({"length": x, "count": y})
    bldc2c = bond_length_distribution.EmpiricalLengthDistribution(df, 0.0)
    all_distributions.add(carbon, carbon, double_bond, bldc2c)

    bond_topology = text_format.Parse(
        """
atoms: ATOM_C
atoms: ATOM_C
bonds: {
  atom_a: 0
  atom_b: 1
  bond_type: BOND_SINGLE
}
""", dataset_pb2.BondTopology())

    geometry = text_format.Parse(
        """
atom_positions {
  x: 0.0
  y: 0.0
  z: 0.0
},
atom_positions {
  x: 0.0
  y: 0.0
  z: 0.0
}
""", dataset_pb2.Geometry())
    geometry.atom_positions[1].x = 1.4 / smu_utils_lib.BOHR_TO_ANGSTROMS

    matching_parameters = smu_molecule.MatchingParameters()
    matching_parameters.must_match_all_bonds = False
    fate = dataset_pb2.Conformer.FATE_SUCCESS
    conformer_id = 1001
    result = topology_from_geom.bond_topologies_from_geom(
        all_distributions, conformer_id, fate, bond_topology, geometry, matching_parameters)
    self.assertIsNotNone(result)
    self.assertEqual(len(result.bond_topology), 2)
    self.assertEqual(len(result.bond_topology[0].bonds), 1)
    self.assertEqual(len(result.bond_topology[1].bonds), 1)
    self.assertGreater(result.bond_topology[0].topology_score,
                       result.bond_topology[1].topology_score)
    self.assertEqual(result.bond_topology[0].bonds[0].bond_type, single_bond)
    self.assertEqual(result.bond_topology[1].bonds[0].bond_type, double_bond)


if __name__ == "__main__":
  absltest.main()
