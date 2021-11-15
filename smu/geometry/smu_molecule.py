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

"""Class that is responsible for building and assessing proposed.

   bonding patterns.
"""

import operator

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from smu import dataset_pb2
from smu.parser import smu_utils_lib
from smu.geometry import utilities


class MatchingParameters:
  """A class to specify optional matching parameters for SmuMolecule.place_bonds."""

  def __init__(self):
    self._must_match_all_bonds: bool = True
    self._smiles_with_h: bool = False
    self._smiles_with_labels: bool = True
    # A variant on matching is to consider all N and O as neutral forms during
    # matching, and then as a post processing step, see whether a valid, 
    # neutral, molecule can be formed.
    self._neutral_forms_during_bond_matching: bool=False

  @property
  def must_match_all_bonds(self):
    return self._must_match_all_bonds

  @must_match_all_bonds.setter
  def must_match_all_bonds(self, value):
    self._must_match_all_bonds = value

  @property
  def smiles_with_h(self):
    return self._smiles_with_h

  @smiles_with_h.setter
  def smiles_with_h(self, value):
    self._smiles_with_h = value

  @property
  def smiles_with_labels(self):
    return self._smiles_with_labels

  @smiles_with_labels.setter
  def smiles_with_labels(self, value):
    self._smiles_with_labels = value

  @property
  def neutral_forms_during_bond_matching(self):
    return self._neutral_forms_during_bond_matching

  @neutral_forms_during_bond_matching.setter
  def neutral_forms_during_bond_matching(self, value):
    self._neutral_forms_during_bond_matching = value

def add_bond(a1, a2, btype,
             destination):
  """Add a new Bond to `destination`.

  Args:
    a1: atom
    a2: atom
    btype: bond type.
    destination:
  """
  destination.bonds.append(
      dataset_pb2.BondTopology.Bond(
          atom_a=a1,
          atom_b=a2,
          bond_type=smu_utils_lib.INTEGER_TO_BOND_TYPE[btype]))


class SmuMolecule:
  """Holds information about partially built molecules."""

  def __init__(self, hydrogens_attached,
               bonds_to_scores,
               matching_parameters):
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
    natoms = len(hydrogens_attached.atoms)

    self._contains_both_oxygen_and_nitrogen = False
    # If the molecule contains both N and O atoms, then we can
    # do more extensive atom type matching if requested.
    if matching_parameters.neutral_forms_during_bond_matching:
      self.set_contains_both_oxygen_and_nitrogen(hydrogens_attached)

    # For each atom, the maximum number of bonds that can be attached.
    self._max_bonds = np.zeros(natoms, dtype=np.int32)
    if matching_parameters.neutral_forms_during_bond_matching and self._contains_both_oxygen_and_nitrogen:
      for i in range(0, natoms):
        self._max_bonds[i] = utilities.max_bonds_any_form(hydrogens_attached.atoms[i])
    else:
      for i in range(0, natoms):
        self._max_bonds[i] = smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS[
            hydrogens_attached.atoms[i]]

    # With the Hydrogens attached, the number of bonds to each atom.
    self._bonds_with_hydrogens_attached = np.zeros((natoms), dtype=np.int32)
    for bond in hydrogens_attached.bonds:
      self._bonds_with_hydrogens_attached[bond.atom_a] += 1
      self._bonds_with_hydrogens_attached[bond.atom_b] += 1

    self._current_bonds_attached = np.zeros((natoms), dtype=np.int32)

    # We turn bonds_to_scores into two arrays. So they can be iterated
    # via itertools.

    self._bonds = list(bonds_to_scores.keys())
    self._scores = list(bonds_to_scores.values())

    self._initial_score = 0.0
    self._accumulate_score = operator.add

    # For testing, it can be convenient to allow for partial matches
    # For example this allows matching C-C and C=C without the need
    # to add explicit hydrogens
    self._must_match_all_bonds = matching_parameters.must_match_all_bonds


  def set_contains_both_oxygen_and_nitrogen(self, bt:dataset_pb2.BondTopology):
    """Examine `bt` and set self._contains_both_oxygen_and_nitrogen.
    Args:
      bt: BondTopology
    """
    self._contains_both_oxygen_and_nitrogen = False
    oxygen_count = 0
    nitrogen_count = 0
    for atom in bt.atoms:
      if atom in [dataset_pb2.BondTopology.ATOM_C,
                  dataset_pb2.BondTopology.ATOM_F,
                  dataset_pb2.BondTopology.ATOM_H]:
        continue
      if atom in [dataset_pb2.BondTopology.ATOM_O,
                  dataset_pb2.BondTopology.ATOM_ONEG]:
        oxygen_count += 1
      else:
        nitrogen_count += 1

    if oxygen_count > 0 and nitrogen_count > 0:
      self._contains_both_oxygen_and_nitrogen = True

  def set_initial_score_and_incrementer(self, initial_score,
                                        op):
    """Update values used for computing scores."""
    self._initial_score = initial_score
    self._accumulate_score = op

  def _initialize(self):
    """Make the molecule reading for adding bonds between heavy atoms."""
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
    print(f"_place_bond, currently", self._current_bonds_attached[a1], " max ", self._max_bonds[a1])
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

  def place_bonds_inner(self, state) -> Optional[dataset_pb2.BondTopology]:
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

    for i, btype in enumerate(state):
      a1 = self._bonds[i][0]
      a2 = self._bonds[i][1]
      print(f"Trying to place bond btw {a1} and {a2} btype {btype}")
      if not self._place_bond(a1, a2, btype):
        return None

      result.score = self._accumulate_score(result.score,
                                            self._scores[i][btype])
      if btype:
        add_bond(a1, a2, btype, result)

    return result

  def place_bonds(self, state, matching_parameters: MatchingParameters) -> Optional[dataset_pb2.BondTopology]:
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

    if matching_parameters.neutral_forms_during_bond_matching and self._contains_both_oxygen_and_nitrogen:
      if not self.assign_charged_atoms(bt):
        return None
      # all bonds matched has already been checked.
      return bt

    # Optionally check whether all bonds have been matched
    print("Bonds placed")
    if not self._must_match_all_bonds:
      return bt

    print(*zip(self._current_bonds_attached, self._max_bonds))
    if not np.array_equal(self._current_bonds_attached, self._max_bonds):
      return None

    return bt

  def assign_charged_atoms(self, bt:dataset_pb2.BondTopology) -> bool:
    """Assign (N, N+) and (O, O-) possibilities in `bt`.

    bt must contain both N and O atoms.
    Note that we assume _must_match_all_bonds, and return None if that cannot
    be achieved.

    Args:
      bt: BondTopology, bt.atoms are updated in place
    Returns: True if successful, False otherwise
    """

    carbon = dataset_pb2.BondTopology.ATOM_C
    hydrogen = dataset_pb2.BondTopology.ATOM_H
    fluorine = dataset_pb2.BondTopology.ATOM_F
    nitrogen = dataset_pb2.BondTopology.ATOM_N
    npos = dataset_pb2.BondTopology.ATOM_NPOS
    oxygen = dataset_pb2.BondTopology.ATOM_O
    oneg = dataset_pb2.BondTopology.ATOM_ONEG
    net_charge = 0
    for i, atom in enumerate(bt.atoms):
      if atom in [carbon, hydrogen, fluorine]:
        if self._max_bonds[i] != self._current_bonds_attached[i]:
          return False
      elif atom in [nitrogen, npos]:
        if self._current_bonds_attached[i] == 4:
          bt.atoms[i] = npos
          net_charge += 1
        elif self._current_bonds_attached[i] == 3:
          bt.atoms[i] = nitrogen
        else:
          return False
      elif atom in [oxygen, oneg]:
        if self._current_bonds_attached[i] == 2:
          bt.atoms[i] = oxygen
        elif self._current_bonds_attached[i] == 1:
          bt.atoms[i] = oneg
          net_charge -= 1
        else:    # Should never happen
          return False
        
    if net_charge != 0:
      return False

    return True
