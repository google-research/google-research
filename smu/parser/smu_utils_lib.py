# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""This class provides shared utilities for parsing and writing SMU7 files."""

import collections
import csv
import enum
import itertools
import numpy as np
from rdkit import Chem
from rdkit import Geometry

from smu import dataset_pb2

# The stage1 files do not label their error fields explicitly. This is the list
# of fields in order.
STAGE1_ERROR_FIELDS = [
    'error_nstat1', 'error_nstatc', 'error_nstatt', 'error_frequencies'
]

ATOM_CHARS = 'cnofh'

ATOM_TYPE_TO_MAX_BONDS = {
    dataset_pb2.BondTopology.AtomType.ATOM_C: 4,
    dataset_pb2.BondTopology.AtomType.ATOM_N: 3,
    dataset_pb2.BondTopology.AtomType.ATOM_NPOS: 4,
    dataset_pb2.BondTopology.AtomType.ATOM_O: 2,
    dataset_pb2.BondTopology.AtomType.ATOM_ONEG: 1,
    dataset_pb2.BondTopology.AtomType.ATOM_F: 1,
    dataset_pb2.BondTopology.AtomType.ATOM_H: 1
}

# Assumes that charged and neutral forms can be switched
ATOM_TYPE_TO_MAX_BONDS_ANY_FORM = {
    dataset_pb2.BondTopology.AtomType.ATOM_C: 4,
    dataset_pb2.BondTopology.AtomType.ATOM_N: 4,
    dataset_pb2.BondTopology.AtomType.ATOM_NPOS: 4,
    dataset_pb2.BondTopology.AtomType.ATOM_O: 2,
    dataset_pb2.BondTopology.AtomType.ATOM_ONEG: 2,
    dataset_pb2.BondTopology.AtomType.ATOM_F: 1,
    dataset_pb2.BondTopology.AtomType.ATOM_H: 1
}
# The value is a pair of an atomic symbol and formal charge
ATOM_TYPE_TO_RDKIT = {
    dataset_pb2.BondTopology.AtomType.ATOM_C: ('C', 0),
    dataset_pb2.BondTopology.AtomType.ATOM_N: ('N', 0),
    dataset_pb2.BondTopology.AtomType.ATOM_NPOS: ('N', 1),
    dataset_pb2.BondTopology.AtomType.ATOM_O: ('O', 0),
    dataset_pb2.BondTopology.AtomType.ATOM_ONEG: ('O', -1),
    dataset_pb2.BondTopology.AtomType.ATOM_F: ('F', 0),
    dataset_pb2.BondTopology.AtomType.ATOM_H: ('H', 0),
}

ATOM_TYPE_TO_CHAR = {
    dataset_pb2.BondTopology.AtomType.ATOM_C: 'c',
    dataset_pb2.BondTopology.AtomType.ATOM_N: 'n',
    dataset_pb2.BondTopology.AtomType.ATOM_NPOS: 'n',
    dataset_pb2.BondTopology.AtomType.ATOM_O: 'o',
    dataset_pb2.BondTopology.AtomType.ATOM_ONEG: 'o',
    dataset_pb2.BondTopology.AtomType.ATOM_F: 'f',
    dataset_pb2.BondTopology.AtomType.ATOM_H: 'h'
}
ATOM_CHAR_TO_TYPE = {
    'c': dataset_pb2.BondTopology.AtomType.ATOM_C,
    'n': dataset_pb2.BondTopology.AtomType.ATOM_N,
    'o': dataset_pb2.BondTopology.AtomType.ATOM_O,
    'f': dataset_pb2.BondTopology.AtomType.ATOM_F,
    'h': dataset_pb2.BondTopology.AtomType.ATOM_H,
}
ATOM_CHAR_TO_ATOMIC_NUMBER = {
    'c': 6,
    'n': 7,
    'o': 8,
    'f': 9,
    'h': 1,
}

ATOM_TYPE_TO_ATOMIC_NUMBER = {
    dataset_pb2.BondTopology.AtomType.ATOM_C: 6,
    dataset_pb2.BondTopology.AtomType.ATOM_N: 7,
    dataset_pb2.BondTopology.AtomType.ATOM_NPOS: 7,
    dataset_pb2.BondTopology.AtomType.ATOM_O: 8,
    dataset_pb2.BondTopology.AtomType.ATOM_ONEG: 8,
    dataset_pb2.BondTopology.AtomType.ATOM_F: 9,
    dataset_pb2.BondTopology.AtomType.ATOM_H: 1
}

BOND_TYPE_TO_RDKIT = {
    dataset_pb2.BondTopology.BondType.BOND_SINGLE: Chem.rdchem.BondType.SINGLE,
    dataset_pb2.BondTopology.BondType.BOND_DOUBLE: Chem.rdchem.BondType.DOUBLE,
    dataset_pb2.BondTopology.BondType.BOND_TRIPLE: Chem.rdchem.BondType.TRIPLE,
}

RDKIT_TO_BOND_TYPE = {
    Chem.rdchem.BondType.SINGLE: dataset_pb2.BondTopology.BondType.BOND_SINGLE,
    Chem.rdchem.BondType.DOUBLE: dataset_pb2.BondTopology.BondType.BOND_DOUBLE,
    Chem.rdchem.BondType.TRIPLE: dataset_pb2.BondTopology.BondType.BOND_TRIPLE,
}

INTEGER_TO_BOND_TYPE = [
    dataset_pb2.BondTopology.BondType.BOND_UNDEFINED,
    dataset_pb2.BondTopology.BondType.BOND_SINGLE,
    dataset_pb2.BondTopology.BondType.BOND_DOUBLE,
    dataset_pb2.BondTopology.BondType.BOND_TRIPLE,
]

ERROR_CODES = collections.OrderedDict([
    # TODO(pfr): give the ones with just error codes better names
    ('nstat1', 'error_nstat1'),
    ('nstatc', 'error_nstatc'),
    ('nstatv', 'error_frequencies'),
    ('nstatt', 'error_nstatt'),
    ('nsvato', 'error_atomic_analysis'),
    ('nsvnsb', 'error_nmr_analysis_b3lyp_small_basis'),
    ('nsvnlb', 'error_nmr_analysis_b3lyp_large_basis'),
    ('nsvele', 'error_charge_analysis'),
    ('nsvego', 'error_nsvego'),
    ('nsveh3', 'error_energies_orbitals_pvtz'),
    ('nsveh4', 'error_energies_orbitals_pvqz'),
    ('nsvec3', 'error_energies_orbitals_pcvtz'),
    ('nsvexc', 'error_excitation_energies'),
    ('nsveca', 'error_single_point_energies'),
    ('nsvmr1', 'error_inconsistent_molecule_energy_turbomole_mrcc'),
    ('nsvmr2', 'error_inconsistent_cation_energy_turbomole_mrcc'),
    ('nsvvib', 'error_vib_mode'),
    ('nsvor1', 'error_inconsistent_molecule_energy_turbomole_orca'),
    ('nsvor2', 'error_inconsistent_cation_energy_turbomole_orca'),
    ('nsvrot', 'error_rotational_modes'),
    ('nsvnsp', 'error_nmr_analysis_pbe0_small_basis'),
    ('nsvnlp', 'error_nmr_analysis_pbe0_large_basis'),
    ('nsvg09', 'error_nsvg09'),
    ('nsvho1', 'error_nsvho1'),
    ('nsvho2', 'error_nsvho2'),
    ('nsvho3', 'error_nsvho3'),
    ('nsvneg', 'error_nsvneg'),
])

# SMU1 was not included in our original bond topology enumeration. So we stick
# these bond topologies at the very end of the list. We also have to special
# case them in the parsing / writing. The tuples are:
# id provided in dat files, bond topology id we use, atom, valence
SPECIAL_ID_CASES = [
    (999999, 899649, 'F', 1),
    (999998, 899650, 'O', 2),
    (999997, 899651, 'N', 3),
    (999996, 899652, 'C', 4),
]

# Conversion constant from Bohr to Angstroms
BOHR_TO_ANGSTROMS = 0.529177249


class StoichiometryError(Exception):

  def __init__(self, stoich_str):
    super().__init__()
    self.stoich_str = stoich_str

  def __str__(self):
    return f'Could not parse "{self.stoich_str}" as valid stoichiometry for SMU'


def special_case_bt_id_from_dat_id(dat_id, smiles):
  """Determines if dat_id is a special case.

  Special case handling for SMU1. We see special cases in two ways.
  * If dat_id is 0 (happens in stage1 files), we use the smiles string to
    determine the bt_id
  * If dat_id is listed in SPECIAL_ID_CASES (happens in stage2 files), we use
    mapping from there

  Args:
    dat_id: integer id from the ID line of the .dat file
    smiles: smiles string for this case

  Returns:
    None if this is not a special case, bond topology id otherwise

  Raises:
    ValueError: if ID is 0 and not a known special case
  """
  if dat_id == 0:
    # Note that the smiles string for these special SMU1 cases is just the atom
    matched_ids = [vals[1] for vals in SPECIAL_ID_CASES if vals[2] == smiles]
    if matched_ids:
      return matched_ids[0]
    else:
      raise ValueError(f'ID from .dat is 0, but {smiles} is not a special case')
  else:
    matched_ids = [vals[1] for vals in SPECIAL_ID_CASES if vals[0] == dat_id]
    if matched_ids:
      return matched_ids[0]
  return None


def special_case_dat_id_from_bt_id(bt_id):
  """Determines if bt_id is a special case.

  Special case handling for SMU1.

  Args:
    bt_id: integer bond topology id

  Returns:
    None if this is not a special case, id to use for .dat file otherwise
  """
  matched_ids = [vals[0] for vals in SPECIAL_ID_CASES if vals[1] == bt_id]
  if matched_ids:
    return matched_ids[0]
  return None


def bohr_to_angstroms(length):
  """Convert bohr units to angstroms.

  Args:
    length: float

  Returns:
    float
  """
  return length * BOHR_TO_ANGSTROMS


def get_composition(topology):
  """Returns the composition/stoichiometry of the molecule.

  The composition is returned as a 'x{num heavy atoms}_' followed by a single
  character per atom type followed by the atom count for the respective type.
  Atoms appear in order 'cnohf'. Types
  with 0-count are omitted and 1 counts are omitted.
  Example: x07_c4o2fh7

  Args:
    topology: A BondTopology protocol buffer message.

  Returns:
    composition(string)
  """
  counts = {char: 0 for char in ATOM_CHARS}
  heavy_atom_count = 0
  for atom in topology.atom:
    counts[ATOM_TYPE_TO_CHAR[atom]] += 1
    if atom != dataset_pb2.BondTopology.AtomType.ATOM_H:
      heavy_atom_count += 1
  components = []
  for c in ATOM_CHARS:
    if counts[c] == 0:
      continue
    elif counts[c] == 1:
      count_str = ''
    else:
      count_str = str(counts[c])
    components.append(c + count_str)
  return 'x{:02d}_{}'.format(heavy_atom_count, ''.join(components))


def get_original_label(molecule):
  """Returns the original id used to identify this molecule.

  We use an integer molecule id, but the original data used a form like
  x07_n6oh4.099599.008

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    string
  """
  bt_id = molecule.mol_id // 1000
  if special_case_dat_id_from_bt_id(bt_id):
    bt_id = 0
  return '{:s}.{:06d}.{:03d}'.format(
      get_composition(molecule.bond_topo[0]), bt_id, molecule.mol_id % 1000)


_STOICHIOMETRY_WITH_HYDROGENS_COMPONENTS = [
    'c', 'ch', 'ch2', 'ch3', 'ch4', 'n', 'nh', 'nh2', 'nh3', 'o', 'oh', 'oh2',
    'f', 'fh'
]


def _expanded_stoichiometry_from_h_counts(heavy_atoms, hydrogen_counts):
  """Returns expanded stoichiometry from H counts.

  Args:
    heavy_atoms: List of heavy atoms in a topology.
    hydrogen_counts: Look up for hydrogen count per atom.

  Returns:
    Stoichiometry with hydrogen count as part of atom type.
  """
  components = collections.defaultdict(int)
  for this_atom, h_count in zip(heavy_atoms, hydrogen_counts):
    this_component = ATOM_TYPE_TO_CHAR[this_atom]
    if h_count >= 1:
      this_component += 'h'
      if h_count > 1:
        this_component += str(h_count)
    components[this_component] += 1

  out = ''
  for got_component in _STOICHIOMETRY_WITH_HYDROGENS_COMPONENTS:
    if got_component not in components:
      continue
    out += f'({got_component})'
    if components[got_component] > 1:
      out += str(components[got_component])

  return out


def expanded_stoichiometry_from_topology(topology):
  """Get stoichiometry where hydrogen count is part of the atom type.

  Each heavy atom is typed by the number of hydrogens it's connected to, e.g.
  * c: carbon with no hydrogens
  * ch: carbon with one hydrogen
  * ch2: carbon with two hydrogens

  Each atom type is then included in the output with its count of how often it
  occurs (just like a normal stoichiometry).

  Atom types are in order 'cnof' then by number of hydrogens

  For example
  * benzene: (ch)6
  * water: (oh2)   (note that the 1 is implicit)
  * ethylene: (ch2)2
  * acrylic acid: (c)(ch)(ch2)(o)(oh)

  Args:
    topology: A BondTopology protocol buffer message.

  Returns:
    string
  """
  hydrogen_counts = compute_bonded_hydrogens(topology,
                                             compute_adjacency_matrix(topology))
  return _expanded_stoichiometry_from_h_counts(topology.atom, hydrogen_counts)


def _generate_hydrogen_assignments(heavy_atoms, total_h):
  """Recursively generate hydrogen atom assignments.

  Args:
    heavy_atoms: List of dataset_pb2.BondTopology.AtomType.
    total_h: int of total number of hydrogen atoms.

  Yields:
    List of hydrogen assignments.
  """
  # TODO(pfr): some error checking would be good
  if len(heavy_atoms) == 1:
    if total_h < ATOM_TYPE_TO_MAX_BONDS_ANY_FORM[heavy_atoms[0]]:
      yield [total_h]
    return
  if total_h == 0:
    yield [0] * len(heavy_atoms)
    return
  # It may look like this +1 is misplaced, but it's intentional.
  # e.g. Carbon has a max of 4 bonds, so we want to try range(3), one less then
  # the total, because, excpect for single atom special cases, one of the bonds
  # must be to a non-hydrogen.
  for num_for_first in range(
      min(ATOM_TYPE_TO_MAX_BONDS_ANY_FORM[heavy_atoms[0]], total_h + 1)):
    for other_assign in _generate_hydrogen_assignments(
        heavy_atoms[1:],
        total_h - num_for_first,
    ):
      yield [num_for_first] + other_assign


def expanded_stoichiometries_from_atom_list(heavy_atoms, total_h):
  """Get the list of possible expanded stoichiometries given atoms.

  See expanded_stoichiometry_from_topology for documentation on the expanded
  stoichiometry.

  Args:
    heavy_atoms: List of dataset_pb2.BondTopology.AtomType.
    total_h: int of total number of hydrogens.

  Returns:
    Set of expanded stoichiometry strings.
  """
  out = set()
  for assignment in _generate_hydrogen_assignments(heavy_atoms, total_h):
    out.add(_expanded_stoichiometry_from_h_counts(heavy_atoms, assignment))
  return out


_EXPAND_STOICHIOMETRY_SPECIAL_CASES = {
    'ch4': '(ch4)',
    'h4c': '(ch4)',
    'oh2': '(oh2)',
    'h2o': '(oh2)',
    'fh': '(fh)',
    'hf': '(fh)',
}


def expanded_stoichiometries_from_stoichiometry(stoich_str):
  """Generates a list possible expanded stoichiometries from a plain one.

  Note that some expanded stoichiometries will be returned even if an invalid
  number of hydrogens is given.
  For example "CN" generates (c)(n) and "CNH2" generates (ch2)(n)
  Further, even for some valid stoichiometries, expanded stoichiometries will be
  enerated that cannot correspond to a SMU molecule.
  For example "C2H2" will generate "(c)(ch2)"

  Args:
    stoich_str: string of a stoichiometry like "C6OH4" (case does not matter)

  Returns:
    set of strings

  Raises:
    StoichiometryError: If there are too many hydrogens or an unrecognized atom.
  """
  stoich_str = stoich_str.lower()
  if stoich_str in _EXPAND_STOICHIOMETRY_SPECIAL_CASES:
    out = set()
    out.add(_EXPAND_STOICHIOMETRY_SPECIAL_CASES[stoich_str])
    return out

  heavy_atoms = []
  total_h = 0

  parse_idx = 0
  while parse_idx < len(stoich_str):
    try:
      atom_type = ATOM_CHAR_TO_TYPE[stoich_str[parse_idx]]
    except KeyError as key_error:
      raise StoichiometryError(stoich_str) from key_error

    num_digits = 0
    while (parse_idx + num_digits + 1 < len(stoich_str) and
           stoich_str[parse_idx + num_digits + 1].isdigit()):
      num_digits += 1
    if num_digits == 0:
      atom_count = 1
    else:
      atom_count = int(stoich_str[parse_idx + 1:parse_idx + num_digits + 1])

    if atom_type == dataset_pb2.BondTopology.AtomType.ATOM_H:
      total_h = atom_count
    else:
      heavy_atoms.extend([atom_type] * atom_count)

    parse_idx += 1 + num_digits

  out = expanded_stoichiometries_from_atom_list(heavy_atoms, total_h)
  if not out:
    raise StoichiometryError(stoich_str)
  return out


def compute_adjacency_matrix(topology):
  """Helper function to determine the adjacency matrix between heavy atoms.

  Only the upper diagonal of the matrix is filled, all other entries are 0.
  All values are non-negative, with positive values giving the bond order.

  Args:
    topology: A BondTopology protocol buffer message.

  Returns:
    An NxN matrix, where N equals the number of heavy atoms in a molecule.
  """
  side_length = len([
      atom for atom in topology.atom
      if atom != dataset_pb2.BondTopology.AtomType.ATOM_H
  ])
  adjacency_matrix = [[0] * side_length for _ in range(side_length)]
  for bond in topology.bond:
    if topology.atom[bond.atom_a] == dataset_pb2.BondTopology.AtomType.ATOM_H:
      continue
    if topology.atom[bond.atom_b] == dataset_pb2.BondTopology.AtomType.ATOM_H:
      continue
    if bond.bond_type == dataset_pb2.BondTopology.BondType.BOND_SINGLE:
      adjacency_matrix[bond.atom_a][bond.atom_b] = 1
      adjacency_matrix[bond.atom_b][bond.atom_a] = 1
    elif bond.bond_type == dataset_pb2.BondTopology.BondType.BOND_DOUBLE:
      adjacency_matrix[bond.atom_a][bond.atom_b] = 2
      adjacency_matrix[bond.atom_b][bond.atom_a] = 2
    elif bond.bond_type == dataset_pb2.BondTopology.BondType.BOND_TRIPLE:
      adjacency_matrix[bond.atom_a][bond.atom_b] = 3
      adjacency_matrix[bond.atom_b][bond.atom_a] = 3
  return adjacency_matrix


def compact_adjacency_matrix_string(adjacency_matrix, join_str):
  """Gets adjacency matrix as one line string.

  This is the upper triangular part of the matrix.

  Args:
    adjacency_matrix: as returned by compute_adjacency_matrix
    join_str: characters to put between lines of the maxtrix

  Returns:
    string
  """
  out = []
  side_length = len(adjacency_matrix)
  for i in range(0, side_length - 1):
    out.append(''.join(
        str(adjacency_matrix[i][j]) for j in range(i + 1, side_length)))
  return join_str.join(out)


def compute_bonded_hydrogens(topology, adjacency_matrix):
  """Helper function to compute number of bonded hydrogens per heavy atom.

  Args:
    topology: A BondTopology protocol buffer message.
    adjacency_matrix: Matrix for all heavy atoms giving covalent bond orders.

  Returns:
    A list of integers (one per heavy atom) with bonded hydrogen counts.
  """
  side_length = len(adjacency_matrix)
  # Initialize with maximum number of hydrogens.
  # Only the first len(adjacency_matrix) atoms in the ordered topology are heavy
  # atoms capable of # binding protons.
  num_bonded_hydrogens = [
      ATOM_TYPE_TO_MAX_BONDS[atom] for atom in topology.atom[:side_length]
  ]
  # Subtract paired bonds (to other heavy atoms).
  for i in range(side_length):
    for j in range(i + 1, side_length):
      num_bonded_hydrogens[i] -= adjacency_matrix[i][j]
      num_bonded_hydrogens[j] -= adjacency_matrix[i][j]
  return num_bonded_hydrogens


def labeled_smiles(mol):
  """Return the smiles for `mol` with atom numbers labeled.

  For each atom in `mol` set the atom map number to the
  atom number.
  CCC -> C[C:1][C:2]
  because atom map 0 is never displayed.

  Args:
    mol: a rdkit_molecule.

  Returns:
    A labelled smiles string.
  """
  natoms = mol.GetNumAtoms()
  for i in range(1, natoms):
    mol.GetAtomWithIdx(i).SetAtomMapNum(i)

  to_be_returned: str = Chem.MolToSmiles(
      mol, kekuleSmiles=True, isomericSmiles=False)

  # Revert what we changed before returning.
  for i in range(1, natoms):
    mol.GetAtomWithIdx(i).SetAtomMapNum(0)

  return to_be_returned


def create_bond_topology(atoms, connectivity_matrix_string, hydrogens_string):
  """Creates a BondTopology from a compact string representation.

  Any hydrogens in the atoms string will be ignored. The appropriate number
  will be added based on what is in the hydrogens string.

  Args:
    atoms: a string like 'CCCCOON' (case insensitive) for the heavy atoms
    connectivity_matrix_string: a string for the uppertriangular connectivity
      matrix with bond orders, like '010210'
    hydrogens_string: a string for the number of hydrogens conencted to each
      heavy atom

  Returns:
    BondTopology

  Raises:
    ValueError: on unknown atom type
  """
  bond_topology = dataset_pb2.BondTopology()

  # Add the heavy atoms
  for atom_type in atoms.lower():
    if atom_type == 'h':
      continue
    try:
      bond_topology.atom.append(ATOM_CHAR_TO_TYPE[atom_type])
    except KeyError as key_error:
      raise ValueError('Unknown atom type: {}'.format(atom_type)) from key_error

  num_heavy_atoms = len(bond_topology.atom)

  # Now add the bonds between the heavy atoms
  if num_heavy_atoms > 1:
    for (i, j), bond_order in zip(
        np.nditer(np.triu_indices(num_heavy_atoms, k=1)),
        connectivity_matrix_string):
      if bond_order == '0':
        continue
      bond = bond_topology.bond.add()
      bond.atom_a = int(i)
      bond.atom_b = int(j)
      if bond_order == '1':
        bond.bond_type = dataset_pb2.BondTopology.BondType.BOND_SINGLE
      elif bond_order == '2':
        bond.bond_type = dataset_pb2.BondTopology.BondType.BOND_DOUBLE
      elif bond_order == '3':
        bond.bond_type = dataset_pb2.BondTopology.BondType.BOND_TRIPLE
      else:
        raise ValueError('Bad bond order {}'.format(bond_order))

  # Now add the hydrogens, and adjust charged atoms if the total bond counts
  # indicate that.
  expected_hydrogens = compute_bonded_hydrogens(
      bond_topology, compute_adjacency_matrix(bond_topology))
  for atom_idx, (actual_h, expected_h) in enumerate(
      zip(hydrogens_string, expected_hydrogens)):
    actual_h = int(actual_h)
    diff = expected_h - actual_h
    atom_type = bond_topology.atom[atom_idx]
    if diff == -1 and atom_type == dataset_pb2.BondTopology.AtomType.ATOM_N:
      bond_topology.atom[atom_idx] = dataset_pb2.BondTopology.AtomType.ATOM_NPOS
    elif diff == 1 and atom_type == dataset_pb2.BondTopology.AtomType.ATOM_O:
      bond_topology.atom[atom_idx] = dataset_pb2.BondTopology.AtomType.ATOM_ONEG
    elif diff:
      raise ValueError(
          f'Bad hydrogen count (actual={actual_h}, expected={expected_h} '
          f'for {atom_type}, index {atom_idx}')
    for _ in range(actual_h):
      bond_topology.atom.append(dataset_pb2.BondTopology.AtomType.ATOM_H)
      h_idx = len(bond_topology.atom) - 1
      bond = bond_topology.bond.add()
      bond.atom_a = atom_idx
      bond.atom_b = h_idx
      bond.bond_type = dataset_pb2.BondTopology.BondType.BOND_SINGLE

  return bond_topology


def parse_bond_topology_line(line):
  """Parses the a line from the enumeration of bond topologies.

  These files are generated by a fortran program that uses fixed width
  formatting that varies by topology size.

  Args:
    line: string

  Returns:
    num atoms (int),
    atoms str (like 'N+O O O-')
    connectivity matrix str (e.g. '010110')
    hydrogen count str (e.g. '3000')

  Raises:
    ValueError: on unexpected line format
  """
  line = line.rstrip()
  num_atoms = int(line[0:2])
  atoms_end = 4 + 2 * num_atoms
  connectivity_end = atoms_end + 2 + num_atoms * (num_atoms - 1) // 2
  if len(line) != connectivity_end + 2 + num_atoms:
    raise ValueError('Wrong line length: "{}"'.format(line))
  return (num_atoms, line[4:atoms_end], line[atoms_end + 2:connectivity_end],
          line[connectivity_end + 2:connectivity_end + 2 + num_atoms])


def generate_bond_topologies_from_csv(fileobj):
  """Generator for bond topologies stored in a csv.

  See merge_bond_topologies.py for the expected format.

  Args:
    fileobj: file like object

  Yields:
    BondTopology
  """
  reader = csv.reader(iter(fileobj))
  next(reader)  # skip the header line
  for row in reader:
    bt_id, _, atoms, connectivity, hydrogens, smiles = row
    # The atoms strings looks like 'C N N+O O-' where every atom has a space,
    # +, or - after it. create_bond_topology doesn't want the charge markings
    # (just a string like 'CNNOO') so the [::2] skips those.
    bond_topology = create_bond_topology(atoms[::2], connectivity, hydrogens)
    bond_topology.smiles = smiles
    bond_topology.topo_id = int(bt_id)
    yield bond_topology


def smiles_id_dict_from_csv(fileobj):
  """Generates a dict of smiles to id from bond_topology.csv.

  Args:
    fileobj: file like object

  Returns:
    dict of smiles to bodn topology id
  """
  smiles_id_dict = {}
  reader = csv.reader(iter(fileobj))
  next(reader)  # skip the header line
  for row in reader:
    bt_id, _, _, _, _, smiles = row
    smiles_id_dict[smiles] = int(bt_id)
  return smiles_id_dict


def bond_topology_to_rdkit_molecule(bond_topology):
  """Converts a bond topology proto to an RDKit molecule.

  Args:
    bond_topology: dataset_pb2.BondTopology

  Returns:
    rdkit.Chem.rdchem.RWMol
  """
  mol = Chem.rdchem.RWMol()
  for pb_atom_idx, pb_atom in enumerate(bond_topology.atom):
    symbol, charge = ATOM_TYPE_TO_RDKIT[pb_atom]
    atom = Chem.Atom(symbol)
    atom.SetFormalCharge(charge)
    atom_idx = mol.AddAtom(atom)
    assert atom_idx == pb_atom_idx

  for pb_bond in bond_topology.bond:
    mol.AddBond(pb_bond.atom_a, pb_bond.atom_b,
                BOND_TYPE_TO_RDKIT[pb_bond.bond_type])

  return mol


def get_bond_type(bond_topology, atom_idx0, atom_idx1):
  """Returns the type of bond in the topology.

  Args:
    bond_topology: datset_pb2.BondTopology
    atom_idx0: int, atom index
    atom_idx1: int atom index

  Returns:
    dataset_pb2.BondTopology.BondType
  """
  for bond in bond_topology.bond:
    if ((bond.atom_a == atom_idx0 and bond.atom_b == atom_idx1) or
        (bond.atom_a == atom_idx1 and bond.atom_b == atom_idx0)):
      return bond.bond_type
  return dataset_pb2.BondTopology.BondType.BOND_UNDEFINED


def bond_topology_sorting_key(bond_topology):
  """Returns tuple to use as sorting key.

  We eventually decided to sort bond topologies not be our estimated
  goodness of fit, but just from some simple parameters of the topology.
  This function shoudl be passed to a key parameter of a sorting function.

  Args:
    bond_topology: dataset_pb2.BondTopology

  Returns:
    tuple
  """
  return (bond_topology.topo_id,
          compact_adjacency_matrix_string(
              compute_adjacency_matrix(bond_topology), '.'))


# These are lower case so they can be used in a command line argument
class WhichTopologies(enum.Enum):
  """Enum of topology types."""
  # All topologies
  ALL = 1
  # The topology used during geometry finding
  STARTING = 3
  # All topologies matching the bond length ranges used in SMU
  DDT = 4
  # All topologies maatching a covalent bond length criteria from Meng and Lewis
  # (see dataset.proto for SourceType for details)
  MLCR = 5
  # All topologies maatching bond lengths from Cambridge Structural Database
  # (see dataset.proto for SourceType for details)
  CSD = 6


def iterate_bond_topologies(molecule, which):
  """Iterates over (possibly a subset of) bond topologies in a molecule.

  Args:
    molecule: dataset_pb2.Molecule
    which: WhichTopologies  Yields index of topology, dataset_pb2.BondTopology

  Yields:
    int (index in bond_topo), Bond topology.
  """
  if which == WhichTopologies.ALL:
    yield from enumerate(molecule.bond_topo)

  if which == WhichTopologies.STARTING:
    if (molecule.prop.calc.status >= 512 or molecule.duplicate_of > 0):
      yield 0, molecule.bond_topo[0]
    for bt_idx, bt in enumerate(molecule.bond_topo):
      if (bt.is_starting_topology or
          bt.info & dataset_pb2.BondTopology.SOURCE_STARTING):
        yield bt_idx, bt

  if which == WhichTopologies.DDT:
    for bt_idx, bt in enumerate(molecule.bond_topo):
      if not bt.info or bt.info & dataset_pb2.BondTopology.SOURCE_DDT:
        yield bt_idx, bt

  if which == WhichTopologies.MLCR:
    for bt_idx, bt in enumerate(molecule.bond_topo):
      if bt.info & dataset_pb2.BondTopology.SOURCE_MLCR:
        yield bt_idx, bt

  if which == WhichTopologies.CSD:
    for bt_idx, bt in enumerate(molecule.bond_topo):
      if bt.info & dataset_pb2.BondTopology.SOURCE_CSD:
        yield bt_idx, bt


def molecule_to_rdkit_molecules(molecule,
                                include_initial_geometries=True,
                                include_optimized_geometry=True,
                                which_topologies=WhichTopologies.ALL):
  """Converts a Molecule to RDKit molecules.

  Because a Molecule can include multiple bond topologies and geometries,
  multiple RDKit molecule objects can be produced

  The name of the molcule will be (all on one line)
  SMU <molid>
  bt=<bt_id>(<bt_idx>/<bt_count>)
  geom=[opt|init(<init_idx>/<init_count>)]
  where
    molid: mol_id
    bt_id: topo_id
    bt_idx: index in bond_topo
    bt_count: size of bond_topo
    init_idx: index in ini_geo
    init_count: size of ini_geo

  Args:
    molecule: dataset_pb2.Molecule
    include_initial_geometries: output molecule for each ini_geo
    include_optimized_geometry: output molecule for optimized_geometry
    which_topologies: WhichTopologies

  Yields:
    rdkit.Chem.rdchem.RWMol
  """
  bt_count = len(molecule.bond_topo)
  requested_bond_topo = [
      (bt, f'{bt.topo_id}({i+1}/{bt_count})')
      for i, bt in iterate_bond_topologies(molecule, which_topologies)
  ]

  # requested_geometries will be a list of tuples of
  # (goemetry, label)
  # where label is a string describing the geometry
  requested_geometries = []
  if include_initial_geometries:
    valid_init_geometries = [g for g in molecule.ini_geo if g.atompos]
    init_count = len(valid_init_geometries)
    requested_geometries.extend([
        (geom, f'init({i}/{init_count})')
        for i, geom in enumerate(valid_init_geometries, start=1)
    ])
  if include_optimized_geometry and molecule.opt_geo.atompos:
    requested_geometries.append((molecule.opt_geo, 'opt'))

  for bt, bt_label in requested_bond_topo:
    for geom, geom_label in requested_geometries:

      mol = bond_topology_to_rdkit_molecule(bt)
      mol.SetProp(
          '_Name',
          f'SMU {molecule.mol_id}, RDKIT {bt.smiles}, bt {bt_label}, geom {geom_label}'
      )

      # Add in the coordinates
      conf = Chem.Conformer(len(bt.atom))
      conf.Set3D(True)
      for atom_idx, pos in enumerate(geom.atompos):
        conf.SetAtomPosition(
            atom_idx,
            Geometry.Point3D(
                bohr_to_angstroms(pos.x), bohr_to_angstroms(pos.y),
                bohr_to_angstroms(pos.z)))
      mol.AddConformer(conf)

      # TODO(pfr): put the computed properties as properties of the molecule.

      yield mol


def compute_smiles_for_bond_topology(bond_topology,
                                     include_hs,
                                     labeled_atoms=False):
  """Calculate a canonical smiles for the given bond_topology.

  The bond topology may have the smiles field filled in but this method ignores
  that and calculates it directly from the atom and bond description.

  Args:
    bond_topology: dataset_pb2.BondTopology
    include_hs: whether to include hs in the smiles string
    labeled_atoms: whether or not to apply atom number labels.

  Returns:
    string
  """
  return compute_smiles_for_rdkit_molecule(
      bond_topology_to_rdkit_molecule(bond_topology),
      include_hs,
      labeled_atoms=labeled_atoms)


def compute_smiles_for_rdkit_molecule(mol, include_hs, labeled_atoms=False):
  """Calculate a canonical smiles for the given RDKit Molecule.

  Note that you probably should NOT have sanitized your RDKit molecule. The
  sanitization procedure can move bonds around in funny ways where here we are
  ignoring aromaticity and keeping bond orders fixed. Because we have such
  funny molecules here, RDKit sometimes does things we don't expect during
  sanitization.

  This is *almost* what RDKit does by default, but because we don't want to
  deal with aromaticity at all, we can't sanitize. That produces one case
  where RDKit will produce different SMILES for what is the same molecule.
  We catch that one case here.

  Note that it is the caller's responsibility to make sure that any Hs intended
  are in the mol. They will NOT be added by this function even when include_hs
  is given.

  If labeled_atoms is True, a smiles where every atom has as its atom map number
  the atom number within the molecule.

  Args:
    mol: rdkit Mol
    include_hs: whether to include hs in the smiles string
    labeled_atoms: whether or not to apply atom number labels.

  Returns:
    string
  """
  if not include_hs:
    mol = Chem.RemoveHs(mol, sanitize=False)
  if labeled_atoms:
    return labeled_smiles(mol)
  smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=False)
  # Yep, this is weird. Depending on the order of the atoms presented to RDKit
  # you can get either of these two smiles back. We arbitrarily picked one of
  # them to return. Note that this one case has no hydrogens so it doesn't
  # matter whether include_hs is True
  if smiles == 'C12=C3C4=C1C4=C23':
    return 'C12=C3C1=C1C2=C31'
  return smiles


def rdkit_atom_to_atom_type(atom):
  """Atom to atom type.

  Args:
    atom: RDKit atom

  Returns:
    dataset_pb2.AtomType

  Raises:
    ValueError: on unrecognized atom type
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

  raise ValueError(f'Unrecognized atom type {atom.GetAtomicNum()}')


def rdkit_molecule_to_bond_topology(mol):
  """Converts RDKit molecule to BondTopology.

  Args:
    mol: RDKit molecule

  Returns:
    dataset_pb2.BondTopology
  """
  bond_topology = dataset_pb2.BondTopology()
  for atom in mol.GetAtoms():
    bond_topology.atom.append(rdkit_atom_to_atom_type(atom))

  for bond in mol.GetBonds():
    bt_bond = dataset_pb2.BondTopology.Bond()
    bt_bond.atom_a = bond.GetBeginAtom().GetIdx()
    bt_bond.atom_b = bond.GetEndAtom().GetIdx()
    bt_bond.bond_type = RDKIT_TO_BOND_TYPE[bond.GetBondType()]
    bond_topology.bond.append(bt_bond)

  return bond_topology


def smiles_to_rdkit_molecule(smiles):
  """Converts a smiles string to a BondTopology.

  Uses RDKit, and because we avoid aromaticity, there's a little
  subtlety in how that is done.

  Args:
    smiles: string

  Returns:
    RDKit molecule
  """
  mol = Chem.MolFromSmiles(smiles, sanitize=False)
  Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS)
  mol = Chem.AddHs(mol)
  return mol


class SmilesCompareResult(enum.Enum):
  MISSING = 2
  MISMATCH = 3
  MATCH = 4

  def __str__(self):
    out = super(SmilesCompareResult, self).__str__()
    # remove the SmilesCompareResult. part
    return out[20:]


def bond_topology_smiles_comparison(bond_topology):
  """Compares the given smiles string to one generated by RDKit.

  The atom/bond structure defines a molecule that we can then turn into a
  SMILES string. There is also a smiles string in the bond topology that was
  generated by large computational pipeline and may have some issues.

  There are three output states. The first applicable one is returned.
  * MISSING: bond_topology does not have a smiles field
  * MISMATCH: bond_topology.smiles does not match the one generated by RDKit
  * MATCH: bond_topology.smiles matches one from RDKit

  Args:
    bond_topology: dataset_pb2.BondTopology

  Returns:
    SmilesCompareResult, SMILES(with H), SMILES(without H)
  """
  smiles_with_h = compute_smiles_for_bond_topology(
      bond_topology, include_hs=True)
  smiles_without_h = compute_smiles_for_bond_topology(
      bond_topology, include_hs=False)

  if not bond_topology.smiles:
    return SmilesCompareResult.MISSING, smiles_with_h, smiles_without_h

  if bond_topology.smiles == smiles_without_h:
    return SmilesCompareResult.MATCH, smiles_with_h, smiles_without_h
  else:
    return SmilesCompareResult.MISMATCH, smiles_with_h, smiles_without_h


class _MoleculeSource(enum.Enum):
  DUPLICATE = 0
  STAGE1 = 1
  STAGE2 = 2


def _molecule_source(mol):
  """Determines source of given molecule."""
  if not mol.HasField('prop'):
    if mol.duplicate_of == 0 and not mol.duplicate_found:
      raise ValueError(
          'Unknown molecule source, no properties or duplicates: ' + str(mol))
    return _MoleculeSource.DUPLICATE
  # We want this function to work for both internal pipeline and to be able to
  # call this with our final completed records. Unfortunatley, it's a little
  # hacky to tell the difference cleanly.

  # calculation_statistics is an internal only field, but it's it's present and
  # and has the right size (stage1 only has 2 computation steps),
  # it's got to be a stage 1 file.
  if len(mol.prop.calculation_statistics) == 2:
    return _MoleculeSource.STAGE1
  # If we are looking at a standard record from the end of the pipeline,
  # it won't have the errors field at all.
  if not mol.prop.HasField('calc'):
    return _MoleculeSource.STAGE2

  # If we have a complete record at the end of the pipeline, status will have a
  # value. Now you may note that we can't tell the difference between a missing
  # status value and a value of 0, but we lucked out that status 0 is always
  # stage 2.
  if not mol.prop.calculation_statistics and (mol.prop.calc.status < 0 or
                                              mol.prop.calc.status >= 512):
    return _MoleculeSource.STAGE1

  return _MoleculeSource.STAGE2


# A list of fields that will be returned by merge_molecule on a conflict.
# The fields for the STAGE1 molecule are first, then fields for the STAGE2
# molecule.
MERGE_CONFLICT_FIELDS = [
    'mol_id',
    'error_nstat1',
    'error_nstatc',
    'error_nstatv',
    'error_nstatt',
    'initial_geometry_energy_1',
    'initial_geometry_gradient_norm_1',
    'opt_geo_energy_1',
    'opt_geo_gradient_norm_1',
    'has_initial_geometry_1',
    'has_opt_geo_1',
    'initial_geometry_energy_2',
    'initial_geometry_gradient_norm_2',
    'opt_geo_energy_2',
    'opt_geo_gradient_norm_2',
    'has_initial_geometry_2',
    'has_opt_geo_2',
]


def merge_molecule(mol1, mol2):
  """Tries to merge information from two molecules.

  During the pipeline, we have partial information about molecules that we
  need to merge. This is the workhorse function for merging these.

  Only molecules with the same mol_id should be merged.

  The key concept is to identify a source of each molecule:
  * STAGE2: From end of pipeline, with mostly complete info
  * STAGE1: From after geometry optimization. Except for duplicate information
    which may have been merged, mostly contains duplicate information to
    STAGE2. However, in some cases it's expected that stage2 will differ
    because of reruns in STAGE2.
  * DUPLICATE: An almost bare molecule with just duplicate_of and/or
    duplicate_found fields

  May modify one of the inputs.

  Note that this is not the most general merge that the format suggests. In
  particular, it's expected that there is at most 1 ini_geo and
  1 bond_topo (and it's the same for all molecules). The final data won't
  be like this but it handles what's in the pipeline at this point we use this.

  While merging STAGE1 and STAGE2, conflicting values of some fields may be
  detected. If they are, then a list of all fields (from MERGE_CONFLICT_FIELDS)
  are returned in addition to doing the merge. If there is no conflict, None
  is returned as the second argument.

  ValueError is returned when a different error besides these expected
  differences is found.

  Args:
    mol1: dataset_pb2.Molecule
    mol2: dataset_pb2.Molecule

  Returns:
    dataset_pb2.Molecule, None or list of field values (see above)

  Raises:
    ValueError: if len(ini_geo) != 1, len(bond_topo) != 1,
      bond_topo differ, or incompatible duplicate_of fields
  """
  source1 = _molecule_source(mol1)
  source2 = _molecule_source(mol2)

  if source1 == source2:
    if source1 == _MoleculeSource.STAGE1 or source1 == _MoleculeSource.STAGE2:
      raise ValueError(
          'Can not merge two molecules of source {}'.format(source1))
    mol1.MergeFrom(mol2)
    return mol1, None

  if source2.value < source1.value:
    mol1, mol2 = mol2, mol1
    source1, source2 = source2, source1

  if len(mol1.ini_geo) > 1:
    raise ValueError('At most 1 ini_geo allowed, got {}'.format(
        len(mol1.ini_geo)))
  if len(mol2.ini_geo) > 1:
    raise ValueError('At most 1 ini_geo allowed, got {}'.format(
        len(mol2.ini_geo)))

  if len(mol1.bond_topo) > 1:
    raise ValueError('At most 1 bond_topo allowed, got {}'.format(
        len(mol1.ini_geo)))
  if len(mol2.bond_topo) > 1:
    raise ValueError('At most 1 bond_topo allowed, got {}'.format(
        len(mol2.ini_geo)))

  if mol1.bond_topo and mol2.bond_topo:
    if mol1.bond_topo[0] != mol2.bond_topo[0]:
      raise ValueError(
          'All bond topologies must be the same, got ids {} and {}'.format(
              mol1.bond_topo[0].topo_id, mol2.bond_topo[0].topo_id))

  # We set the conflict info here because we'll be messing around with fields
  # below. We may not need this if we don't actually find a conflict.
  conflict_info = [mol1.mol_id]
  conflict_info.append(mol1.prop.calc.error_nstat1)
  conflict_info.append(mol1.prop.calc.error_nstatc)
  conflict_info.append(mol1.prop.calc.error_frequencies)  # nstatv
  conflict_info.append(mol1.prop.calc.error_nstatt)
  for c in [mol1, mol2]:
    if c.ini_geo:
      conflict_info.append(c.ini_geo[0].energy.val)
      conflict_info.append(c.ini_geo[0].gnorm.val)
    else:
      conflict_info.extend([0.0, 0.0])
    conflict_info.append(c.opt_geo.energy.val)
    conflict_info.append(c.opt_geo.gnorm.val)
    conflict_info.append(bool(c.ini_geo) and bool(c.ini_geo[0].atompos))
    conflict_info.append(bool(len(c.opt_geo.atompos)))

  # The stage1 (in source1) and stage2 (in source2) is the only non-trivial
  # merge. We look for conflicts between them and then a few special cases.
  has_conflict = False
  if source1 == _MoleculeSource.STAGE1 and source2 == _MoleculeSource.STAGE2:
    if len(mol1.bond_topo) != 1 or len(mol2.bond_topo) != 1:
      has_conflict = True

    if len(mol1.ini_geo) != len(mol2.ini_geo):
      has_conflict = True
    elif len(mol1.ini_geo) == 1:
      if len(mol1.ini_geo[0].atompos) != len(mol2.ini_geo[0].atompos):
        has_conflict = True

    if mol1.HasField('opt_geo') != mol2.HasField('opt_geo'):
      has_conflict = True

    if len(mol1.opt_geo.atompos) != len(mol2.opt_geo.atompos):
      has_conflict = True

    for field in STAGE1_ERROR_FIELDS:
      # Only stage1 uses these old style error fields, so we just copy them
      # over
      setattr(mol2.prop.calc, field, getattr(mol1.prop.calc, field))

    for field_fn, atol in [
        (lambda c: c.ini_geo[0].energy, 2e-6),
        (lambda c: c.ini_geo[0].gnorm, 1e-6),
        (lambda c: c.opt_geo.energy, 2e-6),
        (lambda c: c.opt_geo.gnorm, 1e-6),
    ]:
      try:
        val1 = field_fn(mol1).val
      except IndexError:
        val1 = 0.0
      try:
        val2 = field_fn(mol2).val
      except IndexError:
        val2 = 0.0
      # In some cases, stage2 files have -1 for these fields where stage1
      # doesn't. At some point, stricter error checking was done such that
      # nonsense values were not put into the .dat. So if stage2 has a -1, we
      # just leave it.
      if val2 != -1.0:
        if not np.isclose(val1, val2, atol=atol, rtol=0):
          has_conflict = True

    # This isn't actually a conflict per-se, but we want to find anything that
    # is not an allowed set of combinations of error values.
    error_codes = (mol1.prop.calc.error_nstat1, mol1.prop.calc.error_nstatc,
                   mol1.prop.calc.error_frequencies,
                   mol1.prop.calc.error_nstatt)
    if mol1.prop.calc.error_frequencies == 101:
      # This happens for exactly one molecule. If anything else shows up
      # here we will mark it as a conflict so it comes out in that output
      if mol2.mol_id != 795795001:
        has_conflict = True
    elif error_codes not in [(1, 1, 1, 1), (3, 1, 1, 1), (2, 3, 2, 1),
                             (5, 1, 3, 1), (1, 1, 101, 1)]:
      has_conflict = True

    # After all of that, we always take the stage1 initial energy,
    # gradient norm, and positions.
    if mol2.ini_geo:
      mol2.ini_geo[0].CopyFrom(mol1.ini_geo[0])
    else:
      mol2.ini_geo.append(mol1.ini_geo[0])

    # The 800 and 700 are special cases where we want to take the stage1 data
    if (mol2.prop.calc.status == 800 or mol2.prop.calc.status == 700):
      # Flip back because we will base everything on the stage1 file
      mol1, mol2 = mol2, mol1
      source1, source2 = source2, source1

      mol2.prop.calc.status = (500 + mol1.prop.calc.status // 10)
      mol2.prop.calc.which_database = dataset_pb2.COMPLETE
      if np.any(np.asarray(mol2.prop.vib_freq.val) < -30):
        mol2.prop.calc.warn_vib_imag = 2
      elif np.any(np.asarray(mol2.prop.vib_freq.val) < 0):
        mol2.prop.calc.warn_vib_imag = 1

  # Move over all duplicate info.
  if (mol1.duplicate_of != 0 and mol2.duplicate_of != 0 and
      mol1.duplicate_of != mol2.duplicate_of):
    raise ValueError('Incompatible duplicate_of {} {}'.format(
        mol1.duplicate_of, mol2.duplicate_of))
  # max is just to get the non-zero one
  mol2.duplicate_of = max(mol1.duplicate_of, mol2.duplicate_of)
  mol2.duplicate_found.extend(mol1.duplicate_found)

  if not has_conflict:
    return mol2, None

  return mol2, conflict_info


def molecule_calculation_error_level(molecule):
  """Returns whether status codes indicate this molecule had serious errors.

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    integer, higher values are more srious errors
      8: serious problems
      7: major problems
      6: moderate problems
      5: cations problems, minor problems, serious warning
      4: cations problems, minor problems, vibrational analysis warning
      3: cations problems, otherwise sucess
      2: cations success, minor problems, serious warning
      1: cations success, minor problems, vibrational analysis warning
      0: cations success, no problems
  """
  source = _molecule_source(molecule)
  errors = molecule.prop.calc

  # The levels aren't very well defined for STAGE1.
  # We'll call all errors serious
  if source == _MoleculeSource.STAGE1:
    if errors.error_nstat1 != 1 and errors.error_nstat1 != 3:
      return 8

    if (errors.error_nstatc != 1 or errors.error_nstatt != 1 or
        errors.error_frequencies != 1):
      return 8

    return 0

  # Now logic for stage2 files.
  if errors.status >= 64:
    return 8
  elif errors.status >= 8:
    return 7
  elif errors.status >= 4:
    return 6

  # This is warning level 'C' from Bazel documentation.
  if (errors.warn_t1 > 1 or errors.warn_delta_t1 > 1 or
      errors.warn_bse_b6 > 1 or errors.warn_bse_eccsd > 1 or
      errors.warn_exc_ene > 1 or errors.warn_exc_osmin > 0 or
      errors.warn_exc_osmax > 0):
    warn_offset = 2
  # This is warning level 'B" from Bazel documentation.
  elif (errors.warn_vib_linear > 0 or errors.warn_vib_imag > 1):
    warn_offset = 1
  else:
    warn_offset = 0

  if errors.status == 0:
    return warn_offset
  else:
    return 3 + warn_offset


def filter_molecule_by_availability(molecule, allowed):
  """Filters fields of Molecule by availability annotations.

  *Modifies* the input molecule.

  Args:
    molecule: dataset_pb2.Molecule
    allowed: list of AvailabilityEnum
  """
  # A bit of a hack because original_molecule_index is the only field we
  # filter in the molecule not in the properties subfield.
  if dataset_pb2.INTERNAL_ONLY not in allowed:
    molecule.ClearField('original_molecule_index')
  for descriptor, _ in molecule.prop.ListFields():
    if (descriptor.GetOptions().Extensions[dataset_pb2.availability]
        not in allowed):
      molecule.prop.ClearField(descriptor.name)
  for geometry in itertools.chain([molecule.opt_geo], molecule.ini_geo):
    for descriptor, _ in geometry.ListFields():
      if descriptor.name == 'atompos':
        # We never filter atom positions and we can't call ClearField on it
        continue
      if (descriptor.GetOptions().Extensions[dataset_pb2.availability]
          not in allowed):
        geometry.ClearField(descriptor.name)


def should_include_in_standard(molecule):
  """Returns whether this molecule should be included in the Standard form.

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    boolean
  """
  if molecule.duplicate_of > 0:
    return False
  if molecule.prop.calc.which_database == dataset_pb2.COMPLETE:
    return False
  elif molecule.prop.calc.which_database == dataset_pb2.STANDARD:
    return True
  else:
    # This should only happen with stage1 only files.
    if molecule_calculation_error_level(molecule) > 0:
      return False
    else:
      return True


def molecule_to_standard(molecule):
  """Converts a Molecule from internal to 'Standard' form.

  The "Complete" dataset has all information that anyone could reasonably use.
  The "Standard" dataset is a reduced form with information that we trust and
  various simplifications.

  *Modifies* the input molecule.

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    dataset_pb2.Molecule or None (meaning that this molecule should be
      filtered)
  """
  if not should_include_in_standard(molecule):
    return None

  filter_molecule_by_availability(molecule, [dataset_pb2.STANDARD])

  return molecule


def clean_up_error_codes(molecule):
  """Cleans up error codes for the final dataset.

  Two major types of thigns need to be changed.
  * For stage1 only molecules, the new status code needs to be set
  * For stage2 molecules, the old style error codes need to be cleared.

  Modifies the input molecule

  Args:
    molecule: dataset_pb2.Molecule

  Raises:
    ValueError: on unexpected molecule format
  """
  source = _molecule_source(molecule)
  if source == _MoleculeSource.STAGE1:
    if molecule.prop.calc.status:
      # This is a special case where the stage1 molecule was already put
      # together as a final entry during the merging process. Everything
      # has already been set up.
      pass
    elif (molecule.prop.calc.error_nstat1 == 1 or
          molecule.prop.calc.error_nstat1 == 3):
      # This should be a duplciate. If we have no record of a dup, we'll
      # leaves is as stauts 0 and let it be caught by fate below
      if molecule.duplicate_of:
        molecule.prop.calc.status = -1
    elif molecule.prop.calc.error_nstat1 == 5:
      # optimization was successful, but optimized to different topology
      molecule.prop.calc.status = 590
    elif molecule.prop.calc.error_nstat1 == 2:
      # optimization failed. Clean up the error codes and remove some info
      molecule.prop.calc.status = 600
      molecule.ini_geo[0].ClearField('energy')
      molecule.ini_geo[0].ClearField('gnorm')
      molecule.ClearField('opt_geo')

    # If something isn't caught there, we'll let it go through with
    # status still unset. This will be categorized later in determine_fate
  elif source == _MoleculeSource.STAGE2:
    pass
  else:
    raise ValueError(
        f'Clean up can only handle Stage1 or 2 molecules, got {molecule}')

  for field in STAGE1_ERROR_FIELDS:
    molecule.prop.calc.ClearField(field)


def clean_up_sentinel_values(molecule):
  """Removes some snetinel values, relying on empty protobuf fields to indicate absence.

  Modifies the molecule

  Args:
    molecule: dataset_pb2.Molecule
  """
  for geometry in itertools.chain([molecule.opt_geo], molecule.ini_geo):
    for field in ['energy', 'gnorm']:
      if getattr(geometry, field).val == -1.0:
        geometry.ClearField(field)


_ZERO_FIELD_CHECK_SCALAR = [
    'spe_comp_b5',
    'spe_comp_b6',
    'spe_std_b3lyp_631ppgdp',
    'spe_std_b3lyp_augpcs1',
    'spe_std_cc2_tzvp',
    'spe_std_ccsd_2sd',
    'spe_std_ccsd_2sp',
    'spe_std_ccsd_3psd',
    'spe_std_ccsd_t_2sd',
    'spe_std_ccsd_t_2sp',
    'spe_comp_eccsd',
    'spe_std_hf_2sd',
    'spe_std_hf_2sp',
    'spe_std_hf_3',
    'spe_std_hf_34',
    'spe_std_hf_3psd',
    'spe_std_hf_4',
    'spe_std_hf_631gd',
    'spe_std_hf_cvtz',
    'spe_std_hf_tzvp',
    'spe_std_mp2_2sd',
    'spe_std_mp2_2sp',
    'spe_std_mp2_3',
    'spe_std_mp2_34',
    'spe_std_mp2_3psd',
    'spe_std_mp2_4',
    'spe_std_mp2_tzvp',
    'spe_std_mp2full_cvtz',
    'spe_check_pbe0_6311gd_tmol',
    'spe_stdcat_pbe0_6311gd_tmol',
    'spe_stdcat_pbe0_6311gd_mrcc',
    'spe_stdcat_pbe0_6311gd_orca',
    'spe_check_pbe0_6311gd_mrcc',
    'spe_check_pbe0_6311gd_orca',
    'spe_std_pbe0_631ppgdp',
    'spe_std_pbe0_augpc1',
    'spe_std_pbe0_augpcs1',
    'spe_std_pbe0d3_6311gd',
    'orb_ehomo_b3lyp_631ppgdp',
    'orb_ehomo_b3lyp_augpcs1',
    'orb_ehomo_hf_3',
    'orb_ehomo_hf_4',
    'orb_ehomo_hf_631gd',
    'orb_ehomo_hf_cvtz',
    'orb_ehomo_hf_tzvp',
    'orb_ehomo_pbe0_6311gd',
    'orb_ehomo_pbe0_631ppgdp',
    'orb_ehomo_pbe0_augpc1',
    'orb_ehomo_pbe0_augpcs1',
    'orb_elumo_b3lyp_631ppgdp',
    'orb_elumo_b3lyp_augpcs1',
    'orb_elumo_hf_3',
    'orb_elumo_hf_4',
    'orb_elumo_hf_631gd',
    'orb_elumo_hf_cvtz',
    'orb_elumo_hf_tzvp',
    'orb_elumo_pbe0_6311gd',
    'orb_elumo_pbe0_631ppgdp',
    'orb_elumo_pbe0_augpc1',
    'orb_elumo_pbe0_augpcs1',
    'at2_std_b5_eae',
    'at2_um_b5_eae',
    'at2_std_b6_eae',
    'at2_um_b6_eae',
    'at2_std_eccsd_eae',
    'at2_um_eccsd_eae',
    'at2_std_b5_ea0',
    'at2_um_b5_ea0',
    'at2_std_b6_ea0',
    'at2_um_b6_ea0',
    'at2_std_eccsd_ea0',
    'at2_um_eccsd_ea0',
    'at2_std_b5_hf0',
    'at2_um_b5_hf0',
    'at2_std_b6_hf0',
    'at2_um_b6_hf0',
    'at2_std_eccsd_hf0',
    'at2_um_eccsd_hf0',
    'at2_std_b5_hf298',
    'at2_um_b5_hf298',
    'at2_std_b6_hf298',
    'at2_um_b6_hf298',
    'at2_std_eccsd_hf298',
    'at2_um_eccsd_hf298',
]

_ZERO_FIELD_CHECK_ATOMIC = [
    'nmr_b3lyp_631ppgdp',
    'nmr_b3lyp_augpcs1',
    'nmr_pbe0_631ppgdp',
    'nmr_pbe0_augpcs1',
    'chg_esp_hf_631gd',
    'chg_esp_pbe0_augpc1',
    'chg_loe_hf_631gd',
    'chg_loe_pbe0_augpc1',
    'chg_mul_hf_631gd',
    'chg_mul_pbe0_augpc1',
    'chg_nat_hf_631gd',
    'chg_nat_pbe0_augpc1',
    'partial_charges_paboon_hf_6_31gd',
    'partial_charges_paboon_pbe0_aug_pc_1',
]


def find_zero_values(molecule):
  """Finds fields whose values are exactly 0.

  Fields that are exactly zero are likely to be problematic in some way so we
  look for
  a handful of these.

  Args:
    molecule: dataset_pb2.Molecule

  Yields:
    string of field name
  """
  properties = molecule.prop

  # excitation is different because it's a MultiScalar
  if properties.HasField('exc_ene_cc2_tzvp'):
    for value in properties.exc_ene_cc2_tzvp.val:
      if value == 0.0:
        yield 'exc_ene_cc2_tzvp'

  for field in _ZERO_FIELD_CHECK_SCALAR:
    if properties.HasField(field) and getattr(properties, field).val == 0.0:
      yield field

  for field in _ZERO_FIELD_CHECK_ATOMIC:
    if properties.HasField(field):
      for value in getattr(properties, field).val:
        if value == 0.0:
          yield field


def determine_fate(molecule):
  """Determines the cateogrical FateCategory for molecule.

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    dataset_pb2.Properties.FateCategory

  Raises:
    ValueError: on unrecognized error level or source
  """
  source = _molecule_source(molecule)
  if source == _MoleculeSource.DUPLICATE:
    # This shouldn't really happen in the real set so we'll just leave it as
    # undefined.
    return dataset_pb2.Properties.FATE_UNDEFINED

  elif source == _MoleculeSource.STAGE1:
    if molecule.duplicate_of > 0:
      this_btid = molecule.mol_id // 1000
      other_btid = molecule.duplicate_of // 1000
      if this_btid == other_btid:
        return dataset_pb2.Properties.FATE_DUPLICATE_SAME_TOPOLOGY
      else:
        return dataset_pb2.Properties.FATE_DUPLICATE_DIFFERENT_TOPOLOGY

    status = molecule.prop.calc.status
    if status == 600:
      return dataset_pb2.Properties.FATE_FAILURE_GEO_OPT
    elif status == 590:
      return dataset_pb2.Properties.FATE_FAILURE_TOPOLOGY_CHECK
    elif status == 570 or status == 580:
      return dataset_pb2.Properties.FATE_FAILURE_STAGE2
    else:
      # This means that we can find no reason this shouldn't have gone on to
      # stage2.
      return dataset_pb2.Properties.FATE_FAILURE_NO_RESULTS

  elif source == _MoleculeSource.STAGE2:
    error_level = molecule_calculation_error_level(molecule)
    if error_level == 8:
      return dataset_pb2.Properties.FATE_ERROR_SERIOUS
    elif error_level == 7:
      return dataset_pb2.Properties.FATE_ERROR_MAJOR
    elif error_level == 6:
      return dataset_pb2.Properties.FATE_ERROR_MODERATE
    elif error_level == 5:
      return dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_SERIOUS
    elif error_level == 4:
      return dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_MEDIUM_VIB
    elif error_level == 3:
      return dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_LOW
    elif error_level == 2:
      return dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_SERIOUS
    elif error_level == 1:
      return dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_MEDIUM_VIB
    elif error_level == 0:
      return dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW
    else:
      raise ValueError(f'Bad error_level {error_level}')

  else:
    raise ValueError(f'Got an unknown source {source}')


def get_starting_bond_topology_index(molecule):
  """Gets the index of the bond topology which generated this calculation.

  see iterate_bond_topologies for fun details

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    integer

  Raises:
    ValueError: if no starting topology can be found
  """
  try:
    bt_idx, _ = next(
        iterate_bond_topologies(molecule, WhichTopologies.STARTING))
    return bt_idx
  except StopIteration:
    raise ValueError(f'For molecule {molecule.mol_id}, no starting topology'
                    ) from StopIteration


def molecule_to_bond_topology_summaries(molecule):
  """Produces BondTopologySummary protos from Molecule.

  Since a molecule can be associated with many bond topologies, this can output
  potentially many summaries.

  Args:
    molecule: dataset_pb2.Molecule

  Yields:
    dataset_pb2.BondTopologySummary

  Raises:
    ValueError: on undefeined fate
  """
  summary = dataset_pb2.BondTopologySummary()
  try:
    starting_idx = get_starting_bond_topology_index(molecule)
    summary.bond_topology.CopyFrom(molecule.bond_topo[starting_idx])
    summary.count_attempted_molecules = 1
  except ValueError:
    starting_idx = None
    # In this case, we won't yield the summary at all so we don't set anything
    # about it.

  def filtered_topologies(source):
    observed_bt_id = set()
    # Special case DDT: We only want to filter the starting topology for the DDT
    # source.
    if (starting_idx is not None and
        source == dataset_pb2.BondTopology.SOURCE_DDT):
      observed_bt_id.add(molecule.bond_topo[starting_idx].topo_id)
    for bt in molecule.bond_topo:
      if not source & bt.info:
        continue
      if bt.topo_id in observed_bt_id:
        continue
      yield bt
      observed_bt_id.add(bt.topo_id)

  fate = molecule.prop.calc.fate

  if fate == dataset_pb2.Properties.FATE_UNDEFINED:
    raise ValueError(f'Molecule {molecule.mol_id} has undefined fate')

  elif fate == dataset_pb2.Properties.FATE_DUPLICATE_SAME_TOPOLOGY:
    summary.count_duplicates_same_topology = 1

  elif fate == dataset_pb2.Properties.FATE_DUPLICATE_DIFFERENT_TOPOLOGY:
    summary.count_duplicates_different_topology = 1

  elif (fate == dataset_pb2.Properties.FATE_FAILURE_GEO_OPT or
        fate == dataset_pb2.Properties.FATE_FAILURE_TOPOLOGY_CHECK or
        fate == dataset_pb2.Properties.FATE_FORCE_CONSTANT_FAILURE or
        fate == dataset_pb2.Properties.FATE_FAILURE_STAGE2):
    summary.count_failed_geometry_optimization = 1

  elif fate == dataset_pb2.Properties.FATE_FAILURE_NO_RESULTS:
    summary.count_kept_geometry = 1
    summary.count_missing_calculation = 1

  elif (fate == dataset_pb2.Properties.FATE_ERROR_SERIOUS or
        fate == dataset_pb2.Properties.FATE_ERROR_MAJOR or
        fate == dataset_pb2.Properties.FATE_ERROR_MODERATE):
    summary.count_kept_geometry = 1
    summary.count_calculation_with_error = 1
    for source, field in [(dataset_pb2.BondTopology.SOURCE_DDT,
                           'count_detected_match_itc_with_error'),
                          (dataset_pb2.BondTopology.SOURCE_MLCR,
                           'count_detected_match_mlcr_with_error'),
                          (dataset_pb2.BondTopology.SOURCE_CSD,
                           'count_detected_match_csd_with_error')]:
      for bt in filtered_topologies(source):
        other_summary = dataset_pb2.BondTopologySummary()
        other_summary.bond_topology.CopyFrom(bt)
        setattr(other_summary, field, 1)
        yield other_summary

  elif (
      fate == dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_SERIOUS or
      fate == dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_SERIOUS or
      fate == dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_MEDIUM_VIB or
      fate == dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_MEDIUM_VIB):
    summary.count_kept_geometry = 1
    summary.count_calculation_with_warning = 1
    for source, field in [(dataset_pb2.BondTopology.SOURCE_DDT,
                           'count_detected_match_itc_with_warning'),
                          (dataset_pb2.BondTopology.SOURCE_MLCR,
                           'count_detected_match_mlcr_with_warning'),
                          (dataset_pb2.BondTopology.SOURCE_CSD,
                           'count_detected_match_csd_with_warning')]:
      for bt in filtered_topologies(source):
        other_summary = dataset_pb2.BondTopologySummary()
        other_summary.bond_topology.CopyFrom(bt)
        setattr(other_summary, field, 1)
        yield other_summary

  elif (fate == dataset_pb2.Properties.FATE_SUCCESS_NEUTRAL_WARNING_LOW or
        fate == dataset_pb2.Properties.FATE_SUCCESS_ALL_WARNING_LOW):
    summary.count_kept_geometry = 1
    summary.count_calculation_success = 1
    for source, field in [(dataset_pb2.BondTopology.SOURCE_DDT,
                           'count_detected_match_itc_success'),
                          (dataset_pb2.BondTopology.SOURCE_MLCR,
                           'count_detected_match_mlcr_success'),
                          (dataset_pb2.BondTopology.SOURCE_CSD,
                           'count_detected_match_csd_success')]:
      for bt in filtered_topologies(source):
        other_summary = dataset_pb2.BondTopologySummary()
        other_summary.bond_topology.CopyFrom(bt)
        setattr(other_summary, field, 1)
        yield other_summary

  else:
    raise ValueError(f'Did not understand {fate}')

  if starting_idx is not None:
    yield summary

  # Now emit our multiple detection records
  observed_bt_id = set()
  yielded_multi_detect = set()
  for bt in molecule.bond_topo:
    if bt.topo_id not in observed_bt_id:
      observed_bt_id.add(bt.topo_id)
      continue
    if bt.topo_id not in yielded_multi_detect:
      other_summary = dataset_pb2.BondTopologySummary()
      other_summary.bond_topology.CopyFrom(bt)
      other_summary.count_multiple_detections = 1
      yield other_summary
      yielded_multi_detect.add(bt.topo_id)


def molecule_eligible_for_topology_detection(molecule):
  """Returns whether this molecule is worthy of topology detection.

  Simple duplicate marking or molecules with unreliable geometries are not
  generally useful to do topology detection.

  Args:
    molecule: dataset_pb2.Molecule

  Returns:
    bool
  """
  return (molecule.duplicate_of == 0 and molecule.prop.calc.status >= 0 and
          molecule.prop.calc.status < 512)
