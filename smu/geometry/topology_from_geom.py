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

"""Functions related to discerning the BondTopology from the geometry."""
import itertools

from typing import Dict, List, Optional, Tuple

import apache_beam as beam
import numpy as np
import smu_molecule

import utilities

from smu import dataset_pb2
from smu.geometry import bond_length_distribution
from smu.parser import smu_utils_lib

# The longest distance considered.
THRESHOLD = 2.0


def hydrogen_to_nearest_atom(
    bond_topology,
    distances):
  """Generate a BondTopology that joins each Hydrogen atom to its nearest.

      heavy atom.
  Args:
    bond_topology:
    distances:

  Returns:
  """
  result = dataset_pb2.BondTopology()
  result.atoms[:] = bond_topology.atoms
  natoms = len(bond_topology.atoms)
  for a1 in range(0, natoms):
    if bond_topology.atoms[a1] != dataset_pb2.BondTopology.AtomType.ATOM_H:
      continue

    shortest_distance = 1.0e+30
    closest_heavy_atom = -1
    for a2 in range(0, natoms):
      if bond_topology.atoms[a2] == dataset_pb2.BondTopology.AtomType.ATOM_H:
        continue

      if distances[a1, a2] >= THRESHOLD:
        continue

      if distances[a1, a2] < shortest_distance:
        shortest_distance = distances[a1, a2]
        closest_heavy_atom = a2

    if closest_heavy_atom < 0:
      return None

    bond = dataset_pb2.BondTopology.Bond(
        atom_a=a1,
        atom_b=closest_heavy_atom,
        bond_type=dataset_pb2.BondTopology.BondType.BOND_SINGLE)
    result.bonds.append(bond)

  return result


def indices_of_heavy_atoms(
    bond_topology):
  """Return the indices of the heavy atoms in `bond_topology`.

  Args:
    bond_topology: Bond topology.

  Returns:
    Heavy atom indices.
  """
  return [
      i for i, t in enumerate(bond_topology.atoms)
      if t != dataset_pb2.BondTopology.AtomType.ATOM_H
  ]


def bond_topologies_from_geom(
    bond_lengths,
    bond_topology, geometry,
    matching_parameters
):
  """Return all BondTopology's that are plausible.

    Given a molecule described by `bond_topology` and `geometry`, return all
    possible
    BondTopology that are consistent with that.
    Note that `bond_topology` will be put in a canonical form.

  Args:
    bond_lengths:
    bond_topology:
    geometry:
    matching_parameters:

  Returns:
    TopologyMatches
  """
  result = dataset_pb2.TopologyMatches()  # To be returned.
  if len(bond_topology.atoms) == 1:
    return result  # empty.

  # Will be used when comparing perceived BondTopology's.
  # serialized_starting_form = bond_topology.SerializeToString()

  utilities.canonical_bond_topology(bond_topology)
  distances = utilities.distances(geometry)

  # First join each Hydrogen to its nearest heavy atom, thereby
  # creating a starting BondTopology from which all others can grow
  starting_bond_topology = hydrogen_to_nearest_atom(bond_topology, distances)
  if starting_bond_topology is None:
    return result

  heavy_atom_indices = [
      i for i, t in enumerate(bond_topology.atoms)
      if t != dataset_pb2.BondTopology.AtomType.ATOM_H
  ]

  # For each atom pair, a list of possible bond types.
  # Key is a tuple of the two atom numbers, value is an np.array
  # with the score for each bond type.

  bonds_to_scores: Dict[Tuple[int, int], np.array] = {}
  for c in itertools.combinations(heavy_atom_indices, 2):  # All pairs.
    i = c[0]
    j = c[1]
    dist = distances[i, j]
    if dist > THRESHOLD:
      continue
    btypes = np.zeros(4, np.float32)
    for btype in range(0, 4):
      try:
        btypes[btype] = bond_lengths.pdf_length_given_type(
          bond_topology.atoms[i], bond_topology.atoms[j], btype, dist)
      except KeyError:
        btypes[btype] = 0.0

    if np.count_nonzero(btypes) > 0:
      bonds_to_scores[(i, j)] = btypes

  if not bonds_to_scores:  # Seems unlikely.
    return result


# print(f"Mol with {len(bond_topology.atoms)} has {bonds_to_scores}")
  mol = smu_molecule.SmuMolecule(starting_bond_topology, bonds_to_scores,
                                 matching_parameters)

  search_space = mol.generate_search_state()
  for s in itertools.product(*search_space):
    bt = mol.place_bonds(list(s))
    if not bt:
      continue
    utilities.canonical_bond_topology(bt)
    if utilities.same_bond_topology(bond_topology, bt):
      bt.is_starting_topology = True
    bt.smiles = smu_utils_lib.compute_smiles_for_bond_topology(
        bt, include_hs=True, labeled_atoms=True)
    result.bond_topology.append(bt)

  if len(result.bond_topology) > 1:
    result.bond_topology.sort(key=lambda bt: bt.score, reverse=True)

  return result


class TopologyFromGeom(beam.DoFn):
  """Beam class for extracting BondTopology from Conformer protos."""

  def __init__(
      self,
      bond_lengths):
    super().__init__()
    self._bond_lengths = bond_lengths

  def process(self, conformer):
    """Called by Beam.

      Returns a TopologyMatches for the plausible BondTopology's in `conformer`.
    Args:
      conformer:

    Yields:
      dataset_pb2.TopologyMatches
    """
    matching_parameters = smu_molecule.MatchingParameters()
    yield bond_topologies_from_geom(self._bond_lengths,
                                    conformer.bond_topologies[0],
                                    conformer.optimized_geometry,
                                    matching_parameters)
