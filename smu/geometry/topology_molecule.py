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
"""Class that is responsible for building and assessing proposed.

   bonding patterns.
"""

import operator
from typing import List

import numpy as np

from smu import dataset_pb2
from smu.parser import smu_utils_lib


class MatchingParameters:
  """A class to specify optional matching parameters for TopologyMolecule.place_bonds."""

  def __init__(self):
    self.must_match_all_bonds: bool = True
    self.smiles_with_h: bool = False
    self.smiles_with_labels: bool = False

    # A variant on matching is to consider all N and O as neutral forms during
    # matching, and then as a post processing step, see whether a valid,
    # neutral, molecule can be formed.
    self.neutral_forms_during_bond_matching: bool = True

    # If not a bond is being considered during matching.
    self.consider_not_bonded = True

    # Avoid destroying rings if not bonded is enabled.
    # Note that only the ring atom count is considered.
    self.ring_atom_count_cannot_decrease = False

    # Should we verify that the hydrogens have an appropriate bond length?
    self.check_hydrogen_dists = False


def add_bond(a1, a2, btype, destination):
  """Add a new Bond to `destination`.

  Args:
    a1: atom
    a2: atom
    btype: bond type.
    destination: dataset_pb2.BondTopology
  """
  destination.bond.append(
      dataset_pb2.BondTopology.Bond(
          atom_a=a1,
          atom_b=a2,
          bond_type=smu_utils_lib.INTEGER_TO_BOND_TYPE[btype]))


class TopologyMolecule:
  """Holds information about partially built molecules."""

  def __init__(self, hydrogens_attached, bonds_to_scores, matching_parameters):
    """Class to perform bonding assessments.

    Args:
      hydrogens_attached: a BondTopology that has all atoms, and the bonds
        associated with the Hydrogen atoms.
      bonds_to_scores: A dict that maps tuples of pairs of atoms, to a numpy
        array of scores [0,3], for each possible bond type.
      matching_parameters: contains possible optional behaviour modifiers.

    Returns:
    """
    self._starting_bond_topology = hydrogens_attached
    self._natoms = len(hydrogens_attached.atom)
    self._heavy_atoms = sum(1 for atom in hydrogens_attached.atom
                            if atom != dataset_pb2.BondTopology.ATOM_H)

    self._contains_both_oxygen_and_nitrogen = False
    # If the molecule contains both N and O atoms, then we can
    # do more extensive atom type matching if requested.
    if matching_parameters.neutral_forms_during_bond_matching:
      self.set_contains_both_oxygen_and_nitrogen(hydrogens_attached)

    # For each atom, the maximum number of bonds that can be attached.
    self._max_bonds = np.zeros(self._natoms, dtype=np.int32)
    if (matching_parameters.neutral_forms_during_bond_matching and
        self._contains_both_oxygen_and_nitrogen):
      for i in range(0, self._natoms):
        self._max_bonds[i] = smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS_ANY_FORM[
            hydrogens_attached.atom[i]]
    else:
      for i in range(0, self._natoms):
        self._max_bonds[i] = smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS[
            hydrogens_attached.atom[i]]

    # With the Hydrogens attached, the number of bonds to each atom.
    self._bonds_with_hydrogens_attached = np.zeros((self._natoms),
                                                   dtype=np.int32)
    for bond in hydrogens_attached.bond:
      self._bonds_with_hydrogens_attached[bond.atom_a] += 1
      self._bonds_with_hydrogens_attached[bond.atom_b] += 1

    self._current_bonds_attached = np.zeros((self._natoms), dtype=np.int32)

    # We turn bonds_to_scores into two arrays. So they can be iterated
    # via itertools.

    self._bonds = list(bonds_to_scores.keys())
    self._scores = list(bonds_to_scores.values())

    # Initialize for probability type accumulation
    self._initial_score = 1.0
    self._accumulate_score = operator.mul

    # For testing, it can be convenient to allow for partial matches
    # For example this allows matching C-C and C=C without the need
    # to add explicit hydrogens
    self._must_match_all_bonds = matching_parameters.must_match_all_bonds

  def set_contains_both_oxygen_and_nitrogen(self, bt):
    """Examine `bt` and set self._contains_both_oxygen_and_nitrogen.

    Args:
      bt: BondTopology
    """
    self._contains_both_oxygen_and_nitrogen = False
    oxygen_count = 0
    nitrogen_count = 0
    for atom in bt.atom:
      if atom in [
          dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_NPOS
      ]:
        nitrogen_count += 1
      elif atom in [
          dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_ONEG
      ]:
        oxygen_count += 1

    if oxygen_count > 0 and nitrogen_count > 0:
      self._contains_both_oxygen_and_nitrogen = True

  def set_initial_score_and_incrementer(self, initial_score, op):
    """Update values used for computing scores."""
    self._initial_score = initial_score
    self._accumulate_score = op

  def _initialize(self):
    """Make the molecule ready for adding bonds between heavy atoms."""
    self._current_bonds_attached = np.copy(self._bonds_with_hydrogens_attached)

  def _place_bond(self, a1, a2, btype):
    """Possibly add a new bond to the current config.

    If the bond can be placed, updates self._current_bonds_attached for
    both `a`` and `a2`.
    Args:
      a1:
      a2:
      btype:

    Returns:
      Bool.
    """
    if self._current_bonds_attached[a1] + btype > self._max_bonds[a1]:
      return False
    if self._current_bonds_attached[a2] + btype > self._max_bonds[a2]:
      return False

    self._current_bonds_attached[a1] += btype
    self._current_bonds_attached[a2] += btype
    return True

  def generate_search_state(self):
    """For each pair of atoms, return a list of plausible bond types.

    This will be passed to itertools.product, which thereby enumerates all
    possible bonding combinations.
    Args:

    Returns:
      List of lists - one for each atom pair.
    """
    result: List[List[int]] = []
    for ndx in range(0, len(self._bonds)):
      # For each pair of atoms, the plausible bond types - non zero score.
      plausible_types: List[int] = []
      for i, score in enumerate(self._scores[ndx]):
        if score > 0.0:
          plausible_types.append(i)

      result.append(plausible_types)

    return result

  def place_bonds_inner(self, state):
    """Place bonds corresponding to `state`.

    No validity checking is done, the calling function is responsible
    for that.

    Args:
      state: for each pair of atoms, the kind of bond to be placed.

    Returns:
      If successful, a BondTopology.
    """
    self._current_bonds_attached = np.copy(self._bonds_with_hydrogens_attached)

    result = dataset_pb2.BondTopology()
    result.CopyFrom(self._starting_bond_topology)  # only Hydrogens attached.
    result.score = self._initial_score

    # Make sure each atoms gets at least one bond
    atom_got_bond = np.zeros(self._heavy_atoms)

    for i, btype in enumerate(state):
      if btype != dataset_pb2.BondTopology.BOND_UNDEFINED:
        a1 = self._bonds[i][0]
        a2 = self._bonds[i][1]
        if not self._place_bond(a1, a2, btype):
          return None
        add_bond(a1, a2, btype, result)
        atom_got_bond[a1] = 1
        atom_got_bond[a2] = 1

      result.score = self._accumulate_score(result.score,
                                            self._scores[i][btype])
    if not np.all(atom_got_bond):
      return None

    return result

  def place_bonds(self, state, matching_parameters):
    """Place bonds corresponding to `state`.

    Args:
      state: bonding pattern to be placed.
      matching_parameters: optional settings

    Returns:
      If successful, a BondTopology
    """
    bt = self.place_bonds_inner(state)
    if not bt:
      return None

    if (matching_parameters.neutral_forms_during_bond_matching and
        self._contains_both_oxygen_and_nitrogen):
      if not self.assign_charged_atoms(bt):
        return None
      # all bonds matched has already been checked.
      return bt

    # Optionally check whether all bonds have been matched
    if not self._must_match_all_bonds:
      return bt

    if not np.array_equal(self._current_bonds_attached, self._max_bonds):
      return None

    return bt

  def assign_charged_atoms(self, bt):
    """Assign (N, N+) and (O, O-) possibilities in `bt`.

    bt must contain both N and O atoms.
    Note that we assume _must_match_all_bonds, and return None if that cannot
    be achieved.

    Args:
      bt: BondTopology, bt.atom are updated in place

    Returns:
      True if successful, False otherwise
    """

    carbon = dataset_pb2.BondTopology.ATOM_C
    hydrogen = dataset_pb2.BondTopology.ATOM_H
    fluorine = dataset_pb2.BondTopology.ATOM_F
    nitrogen = dataset_pb2.BondTopology.ATOM_N
    npos = dataset_pb2.BondTopology.ATOM_NPOS
    oxygen = dataset_pb2.BondTopology.ATOM_O
    oneg = dataset_pb2.BondTopology.ATOM_ONEG
    net_charge = 0
    for i, atom in enumerate(bt.atom):
      if atom in [carbon, hydrogen, fluorine]:
        if self._max_bonds[i] != self._current_bonds_attached[i]:
          return False
      elif atom in [nitrogen, npos]:
        if self._current_bonds_attached[i] == 4:
          bt.atom[i] = npos
          net_charge += 1
        elif self._current_bonds_attached[i] == 3:
          bt.atom[i] = nitrogen
        else:
          return False
      elif atom in [oxygen, oneg]:
        if self._current_bonds_attached[i] == 2:
          bt.atom[i] = oxygen
        elif self._current_bonds_attached[i] == 1:
          bt.atom[i] = oneg
          net_charge -= 1
        else:  # not attached.
          return False

    if net_charge != 0:
      return False

    return True
