# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
from smu.geometry import topology_from_geom
from smu.geometry import topology_molecule
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

  df = pd.DataFrame({'length': distances, 'count': population})
  return bond_length_distribution.Empirical(df, 0.0)


class TestTopoFromGeom(absltest.TestCase):

  def test_scores(self):
    carbon = dataset_pb2.BondTopology.ATOM_C
    single_bond = dataset_pb2.BondTopology.BondType.BOND_SINGLE
    double_bond = dataset_pb2.BondTopology.BondType.BOND_DOUBLE

    # For testing, turn off the need for complete matching.
    topology_molecule.default_must_match_all_bonds = False

    all_distributions = bond_length_distribution.AllAtomPairLengthDistributions(
    )
    bldc1c = triangular_distribution(1.0, 1.4, 2.0)
    all_distributions.add(carbon, carbon, single_bond, bldc1c)
    bldc2c = triangular_distribution(1.0, 1.5, 2.0)
    all_distributions.add(carbon, carbon, double_bond, bldc2c)

    molecule = dataset_pb2.Molecule()

    molecule.bond_topo.append(
        text_format.Parse(
            """
atom: ATOM_C
atom: ATOM_C
bond: {
  atom_a: 0
  atom_b: 1
  bond_type: BOND_SINGLE
}
""", dataset_pb2.BondTopology()))

    molecule.opt_geo.MergeFrom(
        text_format.Parse(
            """
atompos {
  x: 0.0
  y: 0.0
  z: 0.0
},
atompos {
  x: 0.0
  y: 0.0
  z: 0.0
}
""", dataset_pb2.Geometry()))
    molecule.opt_geo.atompos[1].x = (1.4 / smu_utils_lib.BOHR_TO_ANGSTROMS)

    matching_parameters = topology_molecule.MatchingParameters()
    matching_parameters.must_match_all_bonds = False
    molecule.prop.calc.fate = dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW
    molecule.mol_id = 1001
    result = topology_from_geom.bond_topologies_from_geom(
        molecule, all_distributions, matching_parameters)
    self.assertIsNotNone(result)
    self.assertLen(result.bond_topology, 2)
    self.assertLen(result.bond_topology[0].bond, 1)
    self.assertLen(result.bond_topology[1].bond, 1)
    self.assertEqual(result.bond_topology[0].bond[0].bond_type, single_bond)
    self.assertEqual(result.bond_topology[1].bond[0].bond_type, double_bond)
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
    all_dist.add(dataset_pb2.BondTopology.ATOM_N,
                 dataset_pb2.BondTopology.ATOM_N, single,
                 triangular_distribution(1.0, 1.5, 2.0))
    all_dist.add(dataset_pb2.BondTopology.ATOM_N,
                 dataset_pb2.BondTopology.ATOM_N, double,
                 triangular_distribution(1.0, 1.4, 2.0))

    # This molecule is a flat aromatic square of nitrogens. The single and
    # double bonds can be rotated such that it's the same topology but
    # individual bonds have switched single/double.
    # We set it so the bond lengths favor one of the two arrangements
    molecule = dataset_pb2.Molecule(mol_id=123)
    molecule.prop.calc.fate = dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW

    molecule.bond_topo.add(topo_id=123, smiles='N1=NN=N1')
    molecule.bond_topo[0].atom.extend([
        dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_N,
    ])
    molecule.bond_topo[0].bond.extend([
        dataset_pb2.BondTopology.Bond(atom_a=0, atom_b=1, bond_type=single),
        dataset_pb2.BondTopology.Bond(atom_a=1, atom_b=2, bond_type=double),
        dataset_pb2.BondTopology.Bond(atom_a=2, atom_b=3, bond_type=single),
        dataset_pb2.BondTopology.Bond(atom_a=3, atom_b=0, bond_type=double),
    ])

    dist15a = 1.5 / smu_utils_lib.BOHR_TO_ANGSTROMS
    dist14a = 1.4 / smu_utils_lib.BOHR_TO_ANGSTROMS
    molecule.opt_geo.atompos.extend([
        dataset_pb2.Geometry.AtomPos(x=0, y=0, z=0),
        dataset_pb2.Geometry.AtomPos(x=0, y=dist15a, z=0),
        dataset_pb2.Geometry.AtomPos(x=dist14a, y=dist15a, z=0),
        dataset_pb2.Geometry.AtomPos(x=dist14a, y=0, z=0),
    ])

    matching_parameters = topology_molecule.MatchingParameters()
    result = topology_from_geom.bond_topologies_from_geom(
        molecule, all_dist, matching_parameters)

    self.assertLen(result.bond_topology, 2)

    first = result.bond_topology[0]
    self.assertEqual(smu_utils_lib.get_bond_type(first, 0, 1), single)
    self.assertEqual(smu_utils_lib.get_bond_type(first, 1, 2), double)
    self.assertEqual(smu_utils_lib.get_bond_type(first, 2, 3), single)
    self.assertEqual(smu_utils_lib.get_bond_type(first, 3, 0), double)

    second = result.bond_topology[1]
    self.assertEqual(smu_utils_lib.get_bond_type(second, 0, 1), double)
    self.assertEqual(smu_utils_lib.get_bond_type(second, 1, 2), single)
    self.assertEqual(smu_utils_lib.get_bond_type(second, 2, 3), double)
    self.assertEqual(smu_utils_lib.get_bond_type(second, 3, 0), single)


class TestStandardTopologySensing(absltest.TestCase):
  """Tests standard_topology_sensing.

  Simple artifical case is a linear molecule
  OCNH which could have 1 or both bonding patterns
  O=C=N-H
  [O-]-C#[N+]-H

  We'll create some fake SMU bonding distances and use the real values
  from the covalent and Allen et al cases to explore a couple of cases.
  """

  def get_smu_dists(self):
    bld = bond_length_distribution.AllAtomPairLengthDistributions()
    # This is set up to make the O=C length of 1.25 a much better fit than
    # the [O-]-C bond
    bld.add(dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_C,
            dataset_pb2.BondTopology.BondType.BOND_SINGLE,
            triangular_distribution(1.2, 1.6, 1.8))
    bld.add(dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_C,
            dataset_pb2.BondTopology.BondType.BOND_DOUBLE,
            triangular_distribution(1.2, 1.25, 1.3))
    bld.add(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N,
            dataset_pb2.BondTopology.BondType.BOND_DOUBLE,
            bond_length_distribution.FixedWindow(1.1, 1.3, None))
    bld.add(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N,
            dataset_pb2.BondTopology.BondType.BOND_TRIPLE,
            bond_length_distribution.FixedWindow(1.2, 1.4, None))
    return bld

  def get_molecule(self, oc_dist, cn_dist):
    molecule = dataset_pb2.Molecule(mol_id=12345)
    molecule.bond_topo.append(dataset_pb2.BondTopology(smiles='N=C=O'))
    molecule.bond_topo[0].atom.extend([
        dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_C,
        dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_H
    ])
    molecule.bond_topo[0].bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=0,
            atom_b=1,
            bond_type=dataset_pb2.BondTopology.BondType.BOND_DOUBLE))
    molecule.bond_topo[0].bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=1,
            atom_b=2,
            bond_type=dataset_pb2.BondTopology.BondType.BOND_DOUBLE))
    molecule.bond_topo[0].bond.append(
        dataset_pb2.BondTopology.Bond(
            atom_a=2,
            atom_b=3,
            bond_type=dataset_pb2.BondTopology.BondType.BOND_SINGLE))

    molecule.opt_geo.atompos.append(dataset_pb2.Geometry.AtomPos(x=0, y=0, z=0))
    molecule.opt_geo.atompos.append(
        dataset_pb2.Geometry.AtomPos(
            x=0, y=0, z=oc_dist / smu_utils_lib.BOHR_TO_ANGSTROMS))
    molecule.opt_geo.atompos.append(
        dataset_pb2.Geometry.AtomPos(
            x=0, y=0, z=(oc_dist + cn_dist) / smu_utils_lib.BOHR_TO_ANGSTROMS))
    molecule.opt_geo.atompos.append(
        dataset_pb2.Geometry.AtomPos(
            x=0,
            y=0,
            z=(oc_dist + cn_dist + 1) / smu_utils_lib.BOHR_TO_ANGSTROMS))

    return molecule

  def get_smiles_id_dict(self):
    return {'N=C=O': 111, '[NH+]#C[O-]': 222}

  def test_without_smu(self):
    mol = self.get_molecule(1.25, 1.11)
    self.assertTrue(
        topology_from_geom.standard_topology_sensing(mol, self.get_smu_dists(),
                                                     self.get_smiles_id_dict()))

    self.assertLen(mol.bond_topo, 2)

    self.assertEqual(
        mol.bond_topo[0].info, dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_STARTING
        | dataset_pb2.BondTopology.SOURCE_MLCR)
    self.assertEqual(mol.bond_topo[0].smiles, 'N=C=O')
    self.assertEqual(mol.bond_topo[0].topo_id, 111)
    self.assertEqual(mol.bond_topo[0].topology_score, 0)
    self.assertNotEqual(mol.bond_topo[0].geometry_score, 0)

    self.assertEqual(
        mol.bond_topo[1].info, dataset_pb2.BondTopology.SOURCE_MLCR
        | dataset_pb2.BondTopology.SOURCE_CSD)
    self.assertEqual(mol.bond_topo[1].smiles, '[NH+]#C[O-]')
    self.assertEqual(mol.bond_topo[1].topo_id, 222)
    self.assertTrue(np.isnan(mol.bond_topo[1].topology_score))
    self.assertTrue(np.isnan(mol.bond_topo[1].geometry_score))

  def test_smu_and_covalent(self):
    mol = self.get_molecule(1.25, 1.25)
    self.assertTrue(
        topology_from_geom.standard_topology_sensing(mol, self.get_smu_dists(),
                                                     self.get_smiles_id_dict()))

    self.assertLen(mol.bond_topo, 2)

    self.assertEqual(
        mol.bond_topo[0].info, dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_STARTING
        | dataset_pb2.BondTopology.SOURCE_MLCR
        | dataset_pb2.BondTopology.SOURCE_CSD)
    self.assertEqual(mol.bond_topo[0].smiles, 'N=C=O')
    self.assertEqual(mol.bond_topo[0].topo_id, 111)
    self.assertLess(mol.bond_topo[0].topology_score, 0)
    self.assertNotEqual(mol.bond_topo[0].geometry_score, 0)

    self.assertEqual(
        mol.bond_topo[1].info, dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_MLCR)
    self.assertEqual(mol.bond_topo[1].smiles, '[NH+]#C[O-]')
    self.assertEqual(mol.bond_topo[1].topo_id, 222)
    self.assertLess(mol.bond_topo[1].topology_score, 0)
    self.assertNotEqual(mol.bond_topo[1].geometry_score, 0)


if __name__ == '__main__':
  absltest.main()
