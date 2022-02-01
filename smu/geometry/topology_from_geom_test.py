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

# Tester for topology_from_geometry

from absl.testing import absltest

import numpy as np
import pandas as pd

from google.protobuf import text_format
from smu import dataset_pb2
from smu.geometry import bond_length_distribution
from smu.geometry import smu_molecule
from smu.geometry import topology_from_geom
from smu.parser import smu_utils_lib

# Only needed so we can alter the default bond matching

# For the empirical bond length distributions, the resolution used.
# Which is not necessarily the same as what is used in the production system.
RESOLUTION = 1000


def triangular_distribution(min_dist, dist_max_value, max_dist):
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
        all_distributions, conformer_id, fate, bond_topology, geometry,
        matching_parameters)
    self.assertIsNotNone(result)
    self.assertLen(result.bond_topology, 2)
    self.assertLen(result.bond_topology[0].bonds, 1)
    self.assertLen(result.bond_topology[1].bonds, 1)
    self.assertEqual(result.bond_topology[0].bonds[0].bond_type, single_bond)
    self.assertEqual(result.bond_topology[1].bonds[0].bond_type, double_bond)
    self.assertGreater(result.bond_topology[0].topology_score,
                       result.bond_topology[1].topology_score)
    self.assertAlmostEqual(
        np.sum(np.exp([bt.topology_score for bt in result.bond_topology])), 1.0)
    self.assertAlmostEqual(result.bond_topology[0].geometry_score,
                           np.log(bldc1c.pdf(1.4)))
    self.assertAlmostEqual(result.bond_topology[1].geometry_score,
                           np.log(bldc2c.pdf(1.4)))

  def test_multi_topology_detection(self):
    """Tests that we can find multiple versions of the same topology."""
    single = dataset_pb2.BondTopology.BondType.BOND_SINGLE
    double = dataset_pb2.BondTopology.BondType.BOND_DOUBLE

    all_dist = bond_length_distribution.AllAtomPairLengthDistributions()
    for bond_type in [single, double]:
      all_dist.add(
          dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_N,
          bond_type,
          bond_length_distribution.FixedWindowLengthDistribution(
              1.0, 2.0, None))

    # This conformer is a flat aromatic square of nitrogens. The single and
    # double bonds can be rotated such that it's the same topology but
    # individual bonds have switched single/double.
    conformer = dataset_pb2.Conformer()

    conformer.bond_topologies.add(bond_topology_id=123, smiles="N1=NN=N1")
    conformer.bond_topologies[0].atoms.extend([
        dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_N,
    ])
    conformer.bond_topologies[0].bonds.extend([
        dataset_pb2.BondTopology.Bond(atom_a=0, atom_b=1, bond_type=single),
        dataset_pb2.BondTopology.Bond(atom_a=1, atom_b=2, bond_type=double),
        dataset_pb2.BondTopology.Bond(atom_a=2, atom_b=3, bond_type=single),
        dataset_pb2.BondTopology.Bond(atom_a=3, atom_b=0, bond_type=double),
    ])

    dist15a = 1.5 / smu_utils_lib.BOHR_TO_ANGSTROMS
    conformer.optimized_geometry.atom_positions.extend([
        dataset_pb2.Geometry.AtomPos(x=0, y=0, z=0),
        dataset_pb2.Geometry.AtomPos(x=0, y=dist15a, z=0),
        dataset_pb2.Geometry.AtomPos(x=dist15a, y=dist15a, z=0),
        dataset_pb2.Geometry.AtomPos(x=dist15a, y=0, z=0),
    ])

    matching_parameters = smu_molecule.MatchingParameters()
    result = topology_from_geom.bond_topologies_from_geom(
        bond_lengths=all_dist,
        conformer_id=123,
        fate=dataset_pb2.Conformer.FATE_SUCCESS,
        bond_topology=conformer.bond_topologies[0],
        geometry=conformer.optimized_geometry,
        matching_parameters=matching_parameters)

    self.assertLen(result.bond_topology, 2)

    # The returned order is arbitrary so we figure out which is is marked
    # as the starting topology.
    starting_idx = min([
        i for i, bt, in enumerate(result.bond_topology)
        if bt.is_starting_topology
    ])
    other_idx = (starting_idx + 1) % 2

    starting = result.bond_topology[starting_idx]
    self.assertTrue(starting.is_starting_topology)
    self.assertEqual(smu_utils_lib.get_bond_type(starting, 0, 1), single)
    self.assertEqual(smu_utils_lib.get_bond_type(starting, 1, 2), double)
    self.assertEqual(smu_utils_lib.get_bond_type(starting, 2, 3), single)
    self.assertEqual(smu_utils_lib.get_bond_type(starting, 3, 0), double)

    other = result.bond_topology[other_idx]
    self.assertFalse(other.is_starting_topology)
    self.assertEqual(smu_utils_lib.get_bond_type(other, 0, 1), double)
    self.assertEqual(smu_utils_lib.get_bond_type(other, 1, 2), single)
    self.assertEqual(smu_utils_lib.get_bond_type(other, 2, 3), double)
    self.assertEqual(smu_utils_lib.get_bond_type(other, 3, 0), single)


if __name__ == "__main__":
  absltest.main()
