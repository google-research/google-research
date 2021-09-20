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


class MatchingParameters:
  """A class to specify optional matching parameters for SmuMolecule.place_bonds."""

  def __init__(self):
    self._must_match_all_bonds: bool = True

  @property
  def must_match_all_bonds(self):
    return self._must_match_all_bonds

  @must_match_all_bonds.setter
  def must_match_all_bonds(self, value):
    self._must_match_all_bonds = value


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

    # For each atom, the maximum number of bonds that can be attached.
    self._max_bonds = np.zeros(natoms, dtype=np.int32)
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

  def place_bonds(self, state):
    """Place bonds corresponding to `state`.

    Args:
      state: for each pair of atoms, the kind of bond to be placed.

    Returns:
      If successful, the score.
    """
    self._current_bonds_attached = np.copy(self._bonds_with_hydrogens_attached)

    result = dataset_pb2.BondTopology()
    result.CopyFrom(self._starting_bond_topology)  # only Hydrogens attached.
    result.score = self._initial_score

    for i, btype in enumerate(state):
      a1 = self._bonds[i][0]
      a2 = self._bonds[i][1]
      if not self._place_bond(a1, a2, btype):
        return None

      result.score = self._accumulate_score(result.score,
                                            self._scores[i][btype])
      if btype:
        add_bond(a1, a2, btype, result)

    # Optionally check whether all bonds have been matched
    if not self._must_match_all_bonds:
      return result

    if not np.array_equal(self._current_bonds_attached, self._max_bonds):
      return None

    return result
