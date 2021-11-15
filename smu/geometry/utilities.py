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

"""Utility functions for SMU."""

import math

from typing import Any, List, Optional

import numpy as np

from rdkit import Chem

from smu import dataset_pb2
from smu.parser import smu_utils_lib

BOHR2ANSTROM = 0.529177

DISTANCE_BINS = 10000


def distance_between_atoms(geom, a1,
                           a2):
  """Return the distance between atoms `a1` and `a2` in `geom`.

  Args:
    geom:
    a1:
    a2:

  Returns:
    Distance in Angstroms.
  """
  return smu_utils_lib.bohr_to_angstroms(
      math.sqrt((geom.atom_positions[a1].x - geom.atom_positions[a2].x) *
                (geom.atom_positions[a1].x - geom.atom_positions[a2].x) +
                (geom.atom_positions[a1].y - geom.atom_positions[a2].y) *
                (geom.atom_positions[a1].y - geom.atom_positions[a2].y) +
                (geom.atom_positions[a1].z - geom.atom_positions[a2].z) *
                (geom.atom_positions[a1].z - geom.atom_positions[a2].z)))


def bonded(bond_topology):
  """Return an int array of the bonded atoms in `bond_topology`.

  Args:
    bond_topology:

  Returns:
    a numpy array of BondType's
  """
  natoms = len(bond_topology.atoms)
  connected = np.full((natoms, natoms), 0, dtype=np.int32)
  for bond in bond_topology.bonds:
    a1 = bond.atom_a
    a2 = bond.atom_b
    connected[a1, a2] = connected[a2, a1] = bond.bond_type
  return connected

def btype_to_nbonds(btype: dataset_pb2.BondTopology.BondType) -> int:
  """Convert `btype` to a number of bonds.

  Turns out that the enum is already set up so that simple
  integer conversion works.

  Args:
    btype:
  Returns:
    number of bonds
  """
  return int(btype)

def number_bonds(bt: dataset_pb2.BondTopology) -> np.array:
  """For each atom in `bt` return the number of bonds.
  single bonds count 1, double 2, triple 3.
  Args:
    bt: BondTopology
  Returns:
    Numpy array contains len(bt.atoms) numbers.
  """
  result = np.zeros(len(bt.atoms))
  for bond in bt.bonds:
    a1 = bond.atom_a
    a2 = bond.atom_b
    nb = btype_to_nbonds(bond.bond_type);
    result[a1] += nb
    result[a2] += nb
    
  return result


def distances(geometry):
  """Return a float array of the interatomic distances in `geometry`.

  Args:
    geometry:

  Returns:
    a numpy array of distances
  """
  natoms = len(geometry.atom_positions)
  result = np.full((natoms, natoms), 0.0, dtype=np.float32)
  for i in range(0, natoms):
    for j in range(i + 1, natoms):
      result[i, j] = result[j, i] = distance_between_atoms(geometry, i, j)
  return result


def rdkit_atom_to_atom_type(
    atom):
  """Atom to atom type.

  Args:
    atom:

  Returns:
    AtomType
  """
  if atom.GetAtomicNum() == 1:
    return dataset_pb2.BondTopology.ATOM_H
  if atom.GetAtomicNum() == 6:
    return dataset_pb2.BondTopology.ATOM_C
  if atom.GetAtomicNum() == 7:
    if atom.GetFormalCharge() == 0:
      return dataset_pb2.BondTopology.ATOM_N
    else:
      return dataset_pb2.BondTopology.ATOM_NPOS
  if atom.GetAtomicNum() == 8:
    if atom.GetFormalCharge() == 0:
      return dataset_pb2.BondTopology.ATOM_O
    else:
      return dataset_pb2.BondTopology.ATOM_ONEG
  if atom.GetAtomicNum() == 9:
    return dataset_pb2.BondTopology.ATOM_F

  raise ValueError(f"Unrecognized atom type {atom.GetAtomicNum()}")


def rdkit_bond_type_to_btype(
    bond_type):
  """Converts bond type.

  Args:
    bond_type:

  Returns:
  """
  if bond_type == Chem.rdchem.BondType.SINGLE:
    return dataset_pb2.BondTopology.BondType.BOND_SINGLE
  if bond_type == Chem.rdchem.BondType.DOUBLE:
    return dataset_pb2.BondTopology.BondType.BOND_DOUBLE
  if bond_type == Chem.rdchem.BondType.TRIPLE:
    return dataset_pb2.BondTopology.BondType.BOND_TRIPLE

  raise ValueError(f"Unrecognized bond type #{bond_type}")


def molecule_to_bond_topology(mol):
  """Molecule to bond topology.

  Args:
    mol:

  Returns:
    Bond topology.
  """
  bond_topology = dataset_pb2.BondTopology()
  for atom in mol.GetAtoms():
    bond_topology.atoms.append(rdkit_atom_to_atom_type(atom))

  for bond in mol.GetBonds():
    btype = rdkit_bond_type_to_btype(bond.GetBondType())
    bt_bond = dataset_pb2.BondTopology.Bond()
    bt_bond.atom_a = bond.GetBeginAtom().GetIdx()
    bt_bond.atom_b = bond.GetEndAtom().GetIdx()
    bt_bond.bond_type = btype
    bond_topology.bonds.append(bt_bond)

  return bond_topology


def canonical_bond_topology(bond_topology):
  """Transform the bonds attribute of `bond_topology` to a canonical form.

  Args:
    bond_topology:

  Returns:
    BondTopology
  """
  if len(bond_topology.bonds) < 2:
    return

  for bond in bond_topology.bonds:
    if bond.atom_a > bond.atom_b:
      bond.atom_a, bond.atom_b = bond.atom_b, bond.atom_a

  bond_topology.bonds.sort(key=lambda b: (b.atom_a, b.atom_b))


def same_bond_topology(bt1,
                       bt2):
  """Return True if bt1 == bt2.

  Note that there is no attempt to canonialise the protos.
  Args:
    bt1:
    bt2:

  Returns:
    Bool.
  """
  natoms = len(bt1.atoms)
  if len(bt2.atoms) != natoms:
    return False
  nbonds = len(bt1.bonds)

  if len(bt2.bonds) != nbonds:
    return False
  for i, t1 in enumerate(bt1.atoms):
    if t1 != bt2.atoms[i]:
      return False
  for i, b1 in enumerate(bt1.bonds):
    b2 = bt2.bonds[i]
    if b1.atom_a != b2.atom_a:
      return False
    if b1.atom_b != b2.atom_b:
      return False
    if b1.bond_type != b2.bond_type:
      return False
  return True


def visit(nbrs, atom, visited):
  """Recusrively visit nodes in the graph defined by `nbrs`.

  Args:
    nbrs:
    atom:
    visited:

  Returns:
    The number of nodes visited - including `atom`.
  """
  visited[atom] = 1
  result = 1  # To be returned.
  for nbr in nbrs[atom]:
    if visited[nbr] > 0:
      continue
    result += visit(nbrs, nbr, visited)

  return result


def is_single_fragment(bond_topology):
  """Return True if `bond_topology` is a single fragment.

  Args:
    bond_topology:

  Returns:
    True if `bond_topology` is a single fragment.
  """

  natoms = len(bond_topology.atoms)
  nbonds = len(bond_topology.bonds)
  # Some special cases are easy.
  if natoms == 1:
    return True
  if natoms == 2 and nbonds == 1:
    return True
  if natoms == 3 and nbonds == 2:
    return True
  if natoms == nbonds and natoms <= 4:
    return True

  connection_matrix = bonded(bond_topology)

  # Any row with all zero means a detached atom.
  if np.sum(connection_matrix.any(axis=1)) != natoms:
    return False

  # For each atom, the neighbours.
  attached: List[Any] = []
  for i in range(0, natoms):
    attached.append(np.ravel(np.argwhere(connection_matrix[i,])))


# neighbours = np.argwhere(connection_matrix > 0)

  visited = np.zeros(natoms, dtype=np.int32)
  # Mark anything with a single connection as visited.
  # Record the index of an atom that has multiple connections.
  a_multiply_connected_atom = -1
  for i in range(0, natoms):
    if bond_topology.atoms[i] == dataset_pb2.BondTopology.AtomType.ATOM_H:
      visited[i] = 1
      continue

    if len(attached[i]) > 1:
      a_multiply_connected_atom = i
      continue

    # A singly connected heavy atom. Mark visited if not of a two atom fragment.
    if len(attached[attached[i][0]]) > 1:
      visited[i] = 1

  if a_multiply_connected_atom < 0:  # Cannot happen
    return False

  number_visited = np.count_nonzero(visited) + visit(
      attached, a_multiply_connected_atom, visited)
  return number_visited == natoms

def geom_to_angstroms(geometry:dataset_pb2.Geometry) -> dataset_pb2.Geometry:
  """Convert all the coordinates in `geometry` to Angstroms
  Args:
    geometry: starting Geometry
  Returns
    New Geometry with adjusted coordinates.
  """
  result = dataset_pb2.Geometry()
  for atom in geometry.atom_positions:
    new_atom = dataset_pb2.Geometry.AtomPos()
    new_atom.x = smu_utils_lib.bohr_to_angstroms(atom.x)
    new_atom.y = smu_utils_lib.bohr_to_angstroms(atom.y)
    new_atom.z = smu_utils_lib.bohr_to_angstroms(atom.z)
    result.atom_positions.append(new_atom)

  return result

def max_bonds_any_form(atype: dataset_pb2.BondTopology.AtomType) -> int:
  """Return the max number of bonds for any form of `atype`.

    Args:
      atype: a dataset_pb2 atom type
    Returns:
      max number of bonds
  """
  if atype in [dataset_pb2.BondTopology.ATOM_C,
               dataset_pb2.BondTopology.ATOM_NPOS,
               dataset_pb2.BondTopology.ATOM_O,
               dataset_pb2.BondTopology.ATOM_F,
               dataset_pb2.BondTopology.ATOM_H]:
    return smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS[atype]

  if atype == dataset_pb2.BondTopology.ATOM_N:
    return 4

  if atype == dataset_pb2.BondTopology.ATOM_ONEG:
    return 2

def convert_to_neutral(bt: dataset_pb2.BondTopology) -> bool:
  """If possible, convert `bt` a neutral form.

  During matching, N+ and O- may have been converted to their neutral forms.
  During this post-processing, try to convert
    all 4 connected N atoms to N+
    all 1 connected O atoms to O-
  and see if the resulting molecule is neutral.

    Args:
      bt: BondTopology
    Returns:
      True if successful. `bt` may have been changed.
      Note that if we fail, `bt` will be in a changed, invalid state
  """


  charges = np.zeros(len(bt.atoms), dtype=np.int32)
  nbonds = number_bonds(bt)
  for i, atom in enumerate(bt.atoms):
    if atom in [dataset_pb2.BondTopology.ATOM_H,
                dataset_pb2.BondTopology.ATOM_C,
                dataset_pb2.BondTopology.ATOM_F]:
      continue

    if atom == dataset_pb2.BondTopology.ATOM_O:
      if nbonds[i] == 1:
        bt.atoms[i] = dataset_pb2.BondTopology.ATOM_ONEG
        charges[i] = -1
    elif atom == dataset_pb2.BondTopology.ATOM_ONEG:
      if nbonds[i] == 2:
        bt.atoms[i] = dataset_pb2.BondTopology.ATOM_O
    elif atom == dataset_pb2.BondTopology.ATOM_N:
      if nbonds[i] == 4:
        bt.atoms[i] = dataset_pb2.BondTopology.ATOM_NPOS
        charges[i] = 1
    elif atom == dataset_pb2.BondTopology.ATOM_NPOS:
      if nbonds[i] == 3:
        bt.atoms[i] = dataset_pb2.BondTopology.ATOM_N

  return positive_placed == negative_placed
