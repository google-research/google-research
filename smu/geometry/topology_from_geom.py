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
"""Functions related to discerning the BondTopology from the geometry."""
import itertools

from typing import Dict, Tuple

import numpy as np
from rdkit import Chem

from smu import dataset_pb2
from smu.geometry import bond_length_distribution
from smu.geometry import topology_molecule
from smu.geometry import utilities
from smu.parser import smu_utils_lib

# The longest distance considered.
THRESHOLD = 2.0


def hydrogen_to_nearest_atom(bond_topology, distances, bond_lengths):
  """Generate a BondTopology with each Hydrogen atom to its nearest heavy atom.

  If bond_lengths is given, the distance of the hydrogen is checked to the
  nearest
  heavy is checked to be allowed under that distance

  Args:
    bond_topology: dataset.pb2.BondTopology
    distances: matrix of interatomic distances.
    bond_lengths: None or AllAtomPairLengthDistributions

  Returns:
    dataset_pb2.BondTopology
  """
  result = dataset_pb2.BondTopology()
  result.atom[:] = bond_topology.atom
  natoms = len(bond_topology.atom)
  for a1 in range(0, natoms):
    if bond_topology.atom[a1] != dataset_pb2.BondTopology.AtomType.ATOM_H:
      continue

    shortest_distance = 1.0e+30
    closest_heavy_atom = -1
    for a2 in range(0, natoms):
      if bond_topology.atom[a2] == dataset_pb2.BondTopology.AtomType.ATOM_H:
        continue

      if distances[a1, a2] >= THRESHOLD:
        continue

      if distances[a1, a2] < shortest_distance:
        shortest_distance = distances[a1, a2]
        closest_heavy_atom = a2

    if closest_heavy_atom < 0:
      return None

    if bond_lengths:
      if (bond_lengths[(bond_topology.atom[closest_heavy_atom],
                        dataset_pb2.BondTopology.ATOM_H)]
          [dataset_pb2.BondTopology.BOND_SINGLE].pdf(shortest_distance) == 0.0):
        return None

    bond = dataset_pb2.BondTopology.Bond(
        atom_a=a1,
        atom_b=closest_heavy_atom,
        bond_type=dataset_pb2.BondTopology.BondType.BOND_SINGLE)
    result.bond.append(bond)

  return result


def indices_of_heavy_atoms(bond_topology):
  """Return the indices of the heavy atoms in `bond_topology`.

  Args:
    bond_topology: Bond topology.

  Returns:
    Heavy atom indices.
  """
  return [
      i for i, t in enumerate(bond_topology.atom)
      if t != dataset_pb2.BondTopology.AtomType.ATOM_H
  ]


def bond_topologies_from_geom(molecule, bond_lengths, matching_parameters):
  """Return all BondTopology's that are plausible.

    Given a molecule described by `bond_topology` and `geometry`, return all
    possible
    BondTopology that are consistent with that.
    Note that `bond_topology` will be put in a canonical form.

  Args:
    molecule: dataset.pb2.Molecule
    bond_lengths: matrix of interatomic distances
    matching_parameters: MatchingParameters

  Returns:
    TopologyMatches
  """
  starting_topology = molecule.bond_topo[0]

  result = dataset_pb2.TopologyMatches()  # To be returned.
  result.starting_smiles = starting_topology.smiles
  result.mol_id = molecule.mol_id
  result.fate = molecule.prop.calc.fate

  natoms = len(starting_topology.atom)
  if natoms == 1:
    return result  # empty.

  if len(molecule.opt_geo.atompos) != natoms:
    return result  # empty
  distances = utilities.distances(molecule.opt_geo)

  # First join each Hydrogen to its nearest heavy atom, thereby
  # creating a minimal BondTopology from which all others can grow
  if matching_parameters.check_hydrogen_dists:
    minimal_bond_topology = hydrogen_to_nearest_atom(starting_topology,
                                                     distances, bond_lengths)
  else:
    minimal_bond_topology = hydrogen_to_nearest_atom(starting_topology,
                                                     distances, None)

  if minimal_bond_topology is None:
    return result

  heavy_atom_indices = [
      i for i, t in enumerate(starting_topology.atom)
      if t != dataset_pb2.BondTopology.AtomType.ATOM_H
  ]

  # For each atom pair, a list of possible bond types.
  # Key is a tuple of the two atom numbers, value is an np.array
  # with the score for each bond type.

  bonds_to_scores: Dict[Tuple[int, int], np.ndarray] = {}
  for (i, j) in itertools.combinations(heavy_atom_indices, 2):  # All pairs.
    dist = distances[i, j]
    if dist > THRESHOLD:
      continue
    try:
      possible_bonds = bond_lengths.probability_of_bond_types(
          starting_topology.atom[i], starting_topology.atom[j], dist)
    except KeyError:  # Happens when this bond type has no data
      continue
    if not possible_bonds:
      continue
    # Note that this relies on the fact that BOND_SINGLE==1 etc..
    btypes = np.zeros(4, np.float32)
    for key, value in possible_bonds.items():
      btypes[key] = value
    bonds_to_scores[(i, j)] = btypes

  if not bonds_to_scores:  # Seems unlikely.
    return result

  rdkit_mol = smu_utils_lib.bond_topology_to_rdkit_molecule(starting_topology)
  initial_ring_atom_count = utilities.ring_atom_count_mol(rdkit_mol)

  mol = topology_molecule.TopologyMolecule(minimal_bond_topology,
                                           bonds_to_scores, matching_parameters)

  search_space = mol.generate_search_state()
  for s in itertools.product(*search_space):
    bt = mol.place_bonds(list(s), matching_parameters)
    if not bt:
      continue

    rdkit_mol = smu_utils_lib.bond_topology_to_rdkit_molecule(bt)
    if matching_parameters.consider_not_bonded and len(
        Chem.GetMolFrags(rdkit_mol)) > 1:
      continue

    utilities.canonicalize_bond_topology(bt)

    if matching_parameters.ring_atom_count_cannot_decrease:
      ring_atoms = utilities.ring_atom_count_mol(rdkit_mol)
      if ring_atoms < initial_ring_atom_count:
        continue
      bt.ring_atom_count = ring_atoms

    bt.smiles = smu_utils_lib.compute_smiles_for_rdkit_molecule(
        rdkit_mol, include_hs=matching_parameters.smiles_with_h)

    bt.geometry_score = geometry_score(bt, distances, bond_lengths)
    result.bond_topology.append(bt)

  if len(result.bond_topology) > 1:
    result.bond_topology.sort(key=lambda bt: bt.score, reverse=True)

  score_sum = np.sum([bt.score for bt in result.bond_topology])
  for bt in result.bond_topology:
    bt.topology_score = np.log(bt.score / score_sum)
    bt.ClearField("score")

  return result


def geometry_score(bt, distances, bond_lengths):
  """Return summed P(geometry|topology)  for `bt` given `distances`.

  For each bond in `bt` compute the score associated with that kind of
  bond at the distance in `distances`, given the distribution in
  bond_lengths.
  Sum these for an overall score, which the caller will most likely
  place in bt.geometry_score.

  Args:
    bt: BondTopology
    distances: numpy array natoms*natoms bond_length_distribution
    bond_lengths: Pairwise bond lengths.

  Returns:
    floating point score.
  """

  result = 0.0
  for bond in bt.bond:
    a1 = bond.atom_a
    a2 = bond.atom_b
    atype1 = bt.atom[a1]
    atype2 = bt.atom[a2]
    if (atype1 == dataset_pb2.BondTopology.ATOM_H or
        atype2 == dataset_pb2.BondTopology.ATOM_H or
        bond.bond_type == dataset_pb2.BondTopology.BOND_UNDEFINED):
      continue
    dist = distances[a1][a2]
    result += np.log(
        bond_lengths.pdf_length_given_type(atype1, atype2, bond.bond_type,
                                           dist))

  return result


_CACHED_MLCR_DISTS = None
_CACHED_CSD_DISTS = None


def standard_topology_sensing(molecule, smu_bond_lengths, smiles_id_dict):
  """Modifies molecule with our standard set of topology sensing.

  Uses 3 sets of bond lengths to extract a set of bond topologies,
  replaces the bond topolgies in molecule, setting appropriate
  source fields.

  Special case: Some SMU1 and SMU2 will fail detection because they
  have no bonds or unique bonds (like F-F). In that case, we still
  set
  source = SOURCE_DDT | SOURCE_STARTING_TOPOLOGY
  and return False

  Args:
    molecule: dataset_pb2.Molecule
    smu_bond_lengths: AllAtomPairLengthDistributions, empirical distribution
      from SMU molecules
    smiles_id_dict: dictionary from smiles string to bond topology id

  Returns:
    Whether topology sensing was successful
  """
  global _CACHED_MLCR_DISTS
  global _CACHED_CSD_DISTS

  if not _CACHED_MLCR_DISTS:
    _CACHED_MLCR_DISTS = bond_length_distribution.make_mlcr_dists()
  if not _CACHED_CSD_DISTS:
    _CACHED_CSD_DISTS = bond_length_distribution.make_csd_dists()

  matching_parameters = topology_molecule.MatchingParameters()
  matching_parameters.must_match_all_bonds = True
  matching_parameters.smiles_with_h = False
  matching_parameters.smiles_with_labels = False
  matching_parameters.neutral_forms_during_bond_matching = True
  matching_parameters.consider_not_bonded = True
  matching_parameters.ring_atom_count_cannot_decrease = False

  smu_matches = bond_topologies_from_geom(
      molecule,
      bond_lengths=smu_bond_lengths,
      matching_parameters=matching_parameters)
  # print('SMU: ', [bt.smiles for bt in smu_matches.bond_topology])

  if not smu_matches.bond_topology:
    # This means the SMU matching failed. We're gong to set the first bond
    # topology as starting and notify the caller
    molecule.bond_topo[0].info = (
        dataset_pb2.BondTopology.SOURCE_DDT
        | dataset_pb2.BondTopology.SOURCE_STARTING)
    return False

  starting_topology = molecule.bond_topo[0]
  utilities.canonicalize_bond_topology(starting_topology)

  # in order to test for equivalent topologies, we jsut have to test
  # the bonds since atom order is fixed. We canonicalizaed the starting
  # and everything coming out of bond_topologies_from_geom has been
  # canonicalized.
  for bt in smu_matches.bond_topology:
    try:
      bt.topo_id = smiles_id_dict[bt.smiles]
    except KeyError:
      pass
    bt.info = dataset_pb2.BondTopology.SOURCE_DDT
    if bt.bond == starting_topology.bond:
      bt.info |= dataset_pb2.BondTopology.SOURCE_STARTING

  del molecule.bond_topo[:]
  molecule.bond_topo.extend(smu_matches.bond_topology)

  cov_matches = bond_topologies_from_geom(
      molecule,
      bond_lengths=_CACHED_MLCR_DISTS,
      matching_parameters=matching_parameters)
  # print('COV: ', [bt.smiles for bt in cov_matches.bond_topology])
  for bt in cov_matches.bond_topology:
    try:
      bt.topo_id = smiles_id_dict[bt.smiles]
    except KeyError:
      pass
    bt.info = dataset_pb2.BondTopology.SOURCE_MLCR
    bt.topology_score = np.nan
    bt.geometry_score = np.nan

  allen_matches = bond_topologies_from_geom(
      molecule,
      bond_lengths=_CACHED_CSD_DISTS,
      matching_parameters=matching_parameters)
  # print('ALLEN: ', [bt.smiles for bt in allen_matches.bond_topology])
  for bt in allen_matches.bond_topology:
    try:
      bt.topo_id = smiles_id_dict[bt.smiles]
    except KeyError:
      pass
    bt.info = dataset_pb2.BondTopology.SOURCE_CSD
    bt.topology_score = np.nan
    bt.geometry_score = np.nan

  for bt in itertools.chain(cov_matches.bond_topology,
                            allen_matches.bond_topology):
    found = False
    for query_bt in molecule.bond_topo:
      if query_bt.bond == bt.bond:
        query_bt.info |= bt.info
        found = True
        break
    if not found:
      molecule.bond_topo.append(bt)

  return True
