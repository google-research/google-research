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

"""Utilities for creating and manipulating MoleculeGraph protos."""

import collections
import itertools
from typing import DefaultDict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from scipy.spatial import distance
import six
from six.moves import range

from gigamol.molecule_graph_proto import molecule_graph_pb2 as mgpb

# A curated set of rules from rdkit/Chem/Pharm3D/test_data/BaseFeatures.fdef
_HBOND_FEATURE_DEF = """
AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]
AtomType ChalcDonor [O,S;H1;+0]
DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor}]
  Family Donor
  Weights 1
EndFeature

AtomType NAcceptor [$([N&v3;H1,H2]-[!$(*=[O,N,P,S])])]
Atomtype NAcceptor [$([N;v3;H0])]
AtomType NAcceptor [$([n;+0])]
AtomType ChalcAcceptor [$([O,S;H1;v2]-[!$(*=[O,N,P,S])])]
AtomType ChalcAcceptor [O,S;H0;v2]
Atomtype ChalcAcceptor [O,S;-]
Atomtype ChalcAcceptor [o,s;+0]
AtomType HalogenAcceptor [F]
DefineFeature SingleAtomAcceptor [{NAcceptor},{ChalcAcceptor},{HalogenAcceptor}]
  Family Acceptor
  Weights 1
EndFeature

# this one is delightfully easy:
DefineFeature AcidicGroup [C,S](=[O,S,P])-[O;H1,H0&-1]
  Family NegIonizable
  Weights 1.0,1.0,1.0
EndFeature

AtomType CarbonOrArom_NonCarbonyl [$([C,a]);!$([C,a](=O))]
AtomType BasicNH2 [$([N;H2&+0][{CarbonOrArom_NonCarbonyl}])]
AtomType BasicNH1 [$([N;H1&+0]([{CarbonOrArom_NonCarbonyl}])[{CarbonOrArom_NonCarbonyl}])]
AtomType BasicNH0 [$([N;H0&+0]([{CarbonOrArom_NonCarbonyl}])([{CarbonOrArom_NonCarbonyl}])[{CarbonOrArom_NonCarbonyl}])]
AtomType BasicNakedN [N,n;X2;+0]
DefineFeature BasicGroup [{BasicNH2},{BasicNH1},{BasicNH0},{BasicNakedN}]
  Family PosIonizable
  Weights 1.0
EndFeature
"""
HydrogenBonding = collections.namedtuple('HydrogenBonding', ['acceptor', 'donor'
                                                            ])


class MoleculeGraphProtoError(Exception):
  """Error for various unexpected values in molecules.

  Keeps a tag attribute to allow easy classification of errors
  """

  def __init__(self, tag, message, *args):
    super(MoleculeGraphProtoError, self).__init__(message, *args)
    self.tag = tag


def element_symbols():
  """Gets mappings of element symbol strings to MoleculeGraph atom types.

  Returns:
    A dict mapping element symbol strings to MoleculeGraph atom types.

  Raises:
    AssertionError: If an element symbol is duplicated.
  """
  symbols = {
      'H': mgpb.MoleculeGraph.Atom.ATOM_H,
      'C': mgpb.MoleculeGraph.Atom.ATOM_C,
      'N': mgpb.MoleculeGraph.Atom.ATOM_N,
      'O': mgpb.MoleculeGraph.Atom.ATOM_O,
      'F': mgpb.MoleculeGraph.Atom.ATOM_F,
      'P': mgpb.MoleculeGraph.Atom.ATOM_P,
      'S': mgpb.MoleculeGraph.Atom.ATOM_S,
      'Cl': mgpb.MoleculeGraph.Atom.ATOM_CL,
      'Br': mgpb.MoleculeGraph.Atom.ATOM_BR,
      'I': mgpb.MoleculeGraph.Atom.ATOM_I,
      'Si': mgpb.MoleculeGraph.Atom.ATOM_SI,
      'Ge': mgpb.MoleculeGraph.Atom.ATOM_GE,
      'B': mgpb.MoleculeGraph.Atom.ATOM_B,
      'Se': mgpb.MoleculeGraph.Atom.ATOM_SE,
  }
  metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs',
            'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
            'Pb', 'Bi', 'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv']
  other = ['He', 'Ne', 'Ar', 'As', 'Kr', 'Sb', 'Te',
           'Xe', 'Po', 'At', 'Rn', 'Uus', 'Uuo', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
           'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac',
           'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
           'Md', 'No', 'Lr']
  for symbol in metals:
    assert symbol not in symbols
    symbols[symbol] = mgpb.MoleculeGraph.Atom.ATOM_METAL
  for symbol in other:
    assert symbol not in symbols
    symbols[symbol] = mgpb.MoleculeGraph.Atom.ATOM_OTHER
  return symbols


class MoleculeGraph(object):
  """Features extracted from the molecular graph.

  Attributes:
    mol: RDKit Mol.
    partial_charges: String partial charge type.
    smiles: String isomeric SMILES.
  """
  _symbols = element_symbols()

  def __init__(self,
               mol,
               partial_charges = None,
               compute_conformer = False):
    """Constructor.

    Args:
      mol: RDKit Mol.
      partial_charges: Partial charge type. If not None, one of 'gasteiger' or
        'mmff94' to compute partial charges, or 'from_property' to read the
        partial charge from the '_PartialCharge' property on each atom.
      compute_conformer: Boolean whether to clear existing conformers and
        generate 2D coordinates. At least one 2D or 3D conformer is required
        for computing spatial distances.
    """
    self.mol = mol
    self.partial_charges = partial_charges
    self.prepare_mol(compute_conformer)
    self.smiles = Chem.MolToSmiles(
        self.mol, canonical=True, isomericSmiles=True)

  def prepare_mol(self, compute_conformer):
    """Prepares molecule.

    Args:
      compute_conformer: If true, compute a default conformer for molecule.
    """
    if compute_conformer:
      self.mol.Compute2DCoords()  # Clears any existing conformers.
    Chem.AssignStereochemistry(self.mol)
    self.check_indices()

  def add_hydrogens(self, mol):
    """Adds hydrogens and verifies that original indices are unchanged.

    Args:
      mol: RDKit Mol.

    Returns:
      A new RDKit Mol with hydrogens added.

    Raises:
      AssertionError: If any of the atom equivalence checks fail.
    """
    mol_h = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
      atom_h = mol_h.GetAtomWithIdx(atom.GetIdx())
      assert atom.GetAtomicNum() == atom_h.GetAtomicNum()
      assert atom.GetHybridization() == atom_h.GetHybridization()
      assert atom.GetIsAromatic() == atom_h.GetIsAromatic()
    return mol_h

  def check_indices(self):
    """Checks that atom indices are continuous."""
    for expected_idx, atom in enumerate(self.mol.GetAtoms()):
      if atom.GetIdx() != expected_idx:
        raise MoleculeGraphProtoError(
            'bad-indices', 'Unexpected atom index %d not %d for molecule %s' %
            (atom.GetIdx(), expected_idx, atom.GetOwningMol().GetProp('_Name')))

  def get_ring_sizes(self):
    """For each atom, gets the sizes of all rings that include that atom.

    Returns:
      A dict mapping RDKit Atom indices to lists giving the sizes of all rings
      that include that atom. Atoms not in the dict are not part of any ring.
    """
    self.check_indices()
    sizes = collections.defaultdict(list)
    for ring in self.mol.GetRingInfo().AtomRings():
      size = len(ring)
      for atom_idx in ring:
        sizes[atom_idx].append(size)
    # Sorts ring sizes for clarity.
    sizes = {key: sorted(value) for key, value in six.iteritems(sizes)}
    return sizes

  def get_same_rings(self):
    """Gets indices for atoms in the same ring for rings of size 3-8.

    Returns:
      A dict mapping RDKit Atom indices to indices of atoms that are in the same
      ring. Atoms not in the dict are not part of any ring.
    """
    self.check_indices()
    rings = collections.defaultdict(set)
    for ring in self.mol.GetRingInfo().AtomRings():
      if len(ring) < 3 or len(ring) > 8:
        continue
      for atom_idx in ring:
        rings[atom_idx].update(ring)
    return rings

  def get_hybridization(
      self, atom):
    """Gets an atom's hybridization type.

    Args:
      atom: RDKit Atom.

    Returns:
      A mgpb.MoleculeGraph.Atom.HybridizationType value.

    Raises:
      MoleculeGraphProtoError: Unsupported hybridization type.
    """
    Chem.HybridizationType.S = 1
    tags = {
        Chem.HybridizationType.OTHER:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_OTHER,
        Chem.HybridizationType.S:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_S,
        Chem.HybridizationType.SP:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP,
        Chem.HybridizationType.SP2:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP2,
        Chem.HybridizationType.SP3:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP3,
        Chem.HybridizationType.SP3D:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP3D,
        Chem.HybridizationType.SP3D2:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_SP3D2,
        Chem.HybridizationType.UNSPECIFIED:
            mgpb.MoleculeGraph.Atom.HYBRIDIZATION_OTHER
    }
    try:
      return tags[atom.GetHybridization()]
    except KeyError:
      raise MoleculeGraphProtoError(
          'hybridization', 'Unsupported hybridization %s for molecule %s' %
          (atom.GetHybridization(),
           atom.GetOwningMol().GetProp('_Name'))) from None

  def get_hydrogen_bonding(self):
    """Gets hydrogen bonding character for all atoms.

    Returns:
      A dict mapping RDKit Atom indices to a HydrogenBonding object. Atom
      indices not in the dict are neither acceptors nor donors.

    Raises:
      TypeError: if more than one atom index is associated with the same
        acceptor or donor.
    """
    self.check_indices()
    factory = ChemicalFeatures.BuildFeatureFactoryFromString(_HBOND_FEATURE_DEF)
    features = factory.GetFeaturesForMol(self.mol)
    hb = collections.defaultdict(lambda: HydrogenBonding(False, False))
    for feat in features:
      family = feat.GetFamily().lower()
      if family in ['acceptor', 'donor']:
        if len(feat.GetAtomIds()) != 1:
          raise TypeError('More than one atom index for %s.' % family)
        idx = feat.GetAtomIds()[0]
        # pylint:disable=protected-access
        if family == 'acceptor':
          hb[idx] = hb[idx]._replace(acceptor=True)
        elif family == 'donor':
          hb[idx] = hb[idx]._replace(donor=True)
          # pylint:enable=protected-access
    return hb

  def get_atom_type(self, atom):
    """Gets atom type.

    Args:
      atom: RDKit Atom.

    Returns:
      A mgpb.MoleculeGraph.Atom.AtomType value.

    Raises:
      MoleculeGraphProtoError: Unsupported atom symbol.
    """
    try:
      atom_type = self._symbols[atom.GetSymbol()]
    except KeyError:
      raise MoleculeGraphProtoError(
          'atom-symbol', 'Unrecognized atom symbol for molecule {}: {}'.format(
              atom.GetSymbol(), self.smiles)) from None
    return atom_type

  def get_atom_chirality(self,
                         atom):
    """Gets atom chirality.

    Args:
      atom: RDKit Atom.

    Returns:
      A mgpb.MoleculeGraph.Atom.ChiralType value.

    Raises:
      MoleculeGraphProtoError: Unsupported chirality type.
    """
    if not atom.HasProp('_CIPCode'):
      chirality = mgpb.MoleculeGraph.Atom.CHIRAL_NONE
    elif atom.GetProp('_CIPCode') == 'R':
      chirality = mgpb.MoleculeGraph.Atom.CHIRAL_R
    elif atom.GetProp('_CIPCode') == 'S':
      chirality = mgpb.MoleculeGraph.Atom.CHIRAL_S
    else:
      raise MoleculeGraphProtoError(
          'chirality',
          'Unrecognized chirality: {}'.format(atom.GetProp('_CIPCode')))
    return chirality

  def get_bond_types(
      self):
    """Gets bond types for all bonds represented in the molecule graph.

    Returns:
      A dict mapping tuples of RDKit Atom indices (sorted by atom index) to
      mgpb.MoleculeGraph.AtomPair.BondType values. Atom tuples not found in the
      dict are not bonded.

    Raises:
      MoleculeGraphProtoError: Unsupported bond type.
    """
    tags = {
        Chem.BondType.SINGLE: mgpb.MoleculeGraph.AtomPair.BOND_SINGLE,
        Chem.BondType.DOUBLE: mgpb.MoleculeGraph.AtomPair.BOND_DOUBLE,
        Chem.BondType.TRIPLE: mgpb.MoleculeGraph.AtomPair.BOND_TRIPLE,
        Chem.BondType.AROMATIC: mgpb.MoleculeGraph.AtomPair.BOND_AROMATIC
    }
    bond_types = {}
    for bond in self.mol.GetBonds():
      key = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
      try:
        bond_types[key] = tags[bond.GetBondType()]
      except KeyError:
        raise MoleculeGraphProtoError(
            'bond-type',
            'Unrecognized BondType: {}'.format(bond.GetBondType())) from None
    return bond_types

  def get_partial_charges(self):
    """Calculates atomic partial charges.

    Returns:
      A dict mapping RDKit Atom indices to atomic partial charges.

    Raises:
      NotImplementedError: If self.partial_charges is not in
        ['gasteiger', 'mmff94', 'from_property'].
    """
    if not self.partial_charges:
      return None
    elif self.partial_charges == 'gasteiger':
      return self.get_gasteiger_partial_charges()
    elif self.partial_charges == 'mmff94':
      return self.get_mmff94_partial_charges()
    elif self.partial_charges == 'from_property':
      return self.get_partial_charges_from_property()
    else:
      raise NotImplementedError('Unrecognized partial charge model: %s' %
                                self.partial_charges)

  def get_gasteiger_partial_charges(self):
    """Calculates Gasteiger atomic partial charges.

    Returns:
      A dict mapping RDKit Atom indices to atomic partial charges.
    """
    AllChem.ComputeGasteigerCharges(self.mol)
    return {
        atom.GetIdx(): float(atom.GetProp('_GasteigerCharge'))
        for atom in self.mol.GetAtoms()
    }

  def get_mmff94_partial_charges(self):
    """Calculates MMFF94 atomic partial charges.

    Returns:
      A dict mapping RDKit Atom indices to atomic partial charges.
    """
    mol_with_h = self.add_hydrogens(self.mol)
    mol_with_h.Compute2DCoords()
    AllChem.MMFFOptimizeMolecule(mol_with_h)
    mp = AllChem.MMFFGetMoleculeProperties(mol_with_h)
    partial_charges = {}
    # Gets a list of the atom indices in the original molecule.
    # We don't want to include charge information for added hydrogens.
    original_indices = {a.GetIdx() for a in self.mol.GetAtoms()}
    for atom in mol_with_h.GetAtoms():
      idx = atom.GetIdx()
      if idx not in original_indices:
        continue
      partial_charges[idx] = mp.GetMMFFPartialCharge(idx)
    return partial_charges

  def get_partial_charges_from_property(self):
    """Gets partial charges from the '_PartialCharge' property on each atom.

    Note: does not compute charge -- assumes it has already been computed.

    Returns:
      A dict mapping RDKit Atom indices to atomic partial charges.
    """
    return {
        atom.GetIdx(): float(atom.GetProp('_PartialCharge'))
        for atom in self.mol.GetAtoms()
    }

  def to_proto(
      self,
      max_pair_distance = -1,
      calc_pair_spatial_distances = False,
      random_permute = None
  ):
    """Writes MoleculeGraph proto.

    Args:
      max_pair_distance: Integer. The max graph distance between a pair of atoms
        for it to generate an AtomPair proto
      calc_pair_spatial_distances: Boolean. If True, add a pairwise spatial
        distance feature based on the first conformer of the molecule.
      random_permute: Numpy RandomState or None. If not None, this numpy
        RandomState is used to take a random permutation of the atom ordering
        before generating the proto.

    Returns:
      A tuple containing a molecule identifier and a corresponding MoleculeGraph
      proto.
    """
    if random_permute is None:
      permutation = np.arange(self.mol.GetNumAtoms())
    else:
      permutation = random_permute.permutation(self.mol.GetNumAtoms())

    mol_pb = mgpb.MoleculeGraph()
    self._add_atom_pbs(mol_pb, permutation)
    self._add_atom_pair_pbs(mol_pb,
                            max_pair_distance, calc_pair_spatial_distances,
                            permutation)
    self._add_circular_fingerprint(mol_pb)
    mol_pb.smiles = self.smiles
    # Sums up atomic number, so error can be plotted against total electron
    # charge (which is the negation of total atomic number).
    tot_atomic_num = 0
    for atom in self.mol.GetAtoms():
      tot_atomic_num += atom.GetAtomicNum()
    mol_pb.tot_atomic_num = tot_atomic_num
    return mol_pb

  def _add_atom_pbs(self, mol_pb, permutation):
    """Writes Atom messages.

    Args:
      mol_pb: MoleculeGraph proto.
      permutation: permutation of the atom ordering (an ndarray of length
        mol.GetNumAtoms)
    """
    ring_sizes = self.get_ring_sizes()
    hydrogen_bonding = self.get_hydrogen_bonding()
    partial_charges = self.get_partial_charges()
    self.check_indices()
    for _ in range(self.mol.GetNumAtoms()):
      mol_pb.atoms.add()
    for i, atom in enumerate(self.mol.GetAtoms()):
      atom_pb = mol_pb.atoms[permutation[i]]
      atom_pb.atomic_num = atom.GetAtomicNum()
      atom_pb.element = atom.GetSymbol()
      atom_pb.type = self.get_atom_type(atom)
      atom_pb.aromatic = atom.GetIsAromatic()
      atom_pb.chirality = self.get_atom_chirality(atom)
      atom_pb.formal_charge = atom.GetFormalCharge()
      if partial_charges is not None:
        atom_pb.partial_charge = partial_charges[atom.GetIdx()]
      atom_pb.ring_sizes.extend(ring_sizes.get(atom.GetIdx(), []))
      atom_pb.hybridization = self.get_hybridization(atom)
      atom_pb.acceptor, atom_pb.donor = hydrogen_bonding.get(atom.GetIdx(),
                                                             (False, False))

  def _add_atom_pair_pbs(self, mol_pb,
                         max_pair_distance,
                         calc_pair_spatial_distances,
                         permutation):
    """Writes AtomPair messages.

    Only one possible ordering is written, where a_idx < b_idx.

    Args:
      mol_pb: MoleculeGraph proto.
      max_pair_distance: Integer. The max graph distance between a pair of atoms
        for it to generate an AtomPair proto
      calc_pair_spatial_distances: Boolean. If True, add a pairwise spatial
        distance feature based on the first conformer of the molecule.
      permutation: permutation of the atom ordering (an ndarray of length
        mol.GetNumAtoms)

    Raises:
      MoleculeGraphProtoError: if calc_pair_spatial_distances is true but mol
      has no
        conformers, or if the spatial distance between two atoms is too close to
        zero.
    """
    self.check_indices()
    bond_types = self.get_bond_types()
    graph_distances = Chem.GetDistanceMatrix(self.mol).astype(int)
    rings = self.get_same_rings()
    conformers = self.mol.GetConformers()
    conformer = None
    if conformers:
      conformer = conformers[0]
    elif calc_pair_spatial_distances:
      raise MoleculeGraphProtoError(
          'spatial-distances',
          'calc_pair_spatial_distances is true, but mol has no conformers')
    for atom_a in self.mol.GetAtoms():
      a_idx = atom_a.GetIdx()
      if conformer:
        a_pos = conformer.GetAtomPosition(a_idx)
      for atom_b in self.mol.GetAtoms():
        b_idx = atom_b.GetIdx()
        if a_idx >= b_idx:
          continue
        if (max_pair_distance >= 0 and
            graph_distances[a_idx, b_idx] > max_pair_distance):
          continue
        atom_pair_pb = mol_pb.atom_pairs.add()
        atom_pair_pb.a_idx = permutation[a_idx]
        atom_pair_pb.b_idx = permutation[b_idx]
        atom_pair_pb.bond_type = bond_types.get(
            (a_idx, b_idx), mgpb.MoleculeGraph.AtomPair.BOND_NONE)
        atom_pair_pb.graph_distance = int(graph_distances[a_idx, b_idx])
        if a_idx in rings:
          atom_pair_pb.same_ring = (b_idx in rings[a_idx])
        else:
          atom_pair_pb.same_ring = False
        if conformer:
          b_pos = conformer.GetAtomPosition(b_idx)
          if calc_pair_spatial_distances:
            atom_pair_pb.spatial_distance = distance.euclidean(
                (a_pos.x, a_pos.y, a_pos.z),
                (b_pos.x, b_pos.y, b_pos.z))
            if atom_pair_pb.spatial_distance < 1.0e-6:
              raise MoleculeGraphProtoError(
                  'spatial-distances',
                  'spatial distance between two atoms cannot be zero')

  def _add_circular_fingerprint(self, mol_pb):
    """Gets ECFP.

    Args:
      mol_pb: MoleculeGraph proto.

    Returns:
      Added a list of integers containing the ECFP to mol_pb binary features.
    """
    # Calculates isomeric ECFP4 hashed to 1024 bits.
    fp = AllChem.GetMorganFingerprintAsBitVect(
        self.mol, 2, nBits=1024, useChirality=True)
    array = np.zeros(1, dtype=int)
    DataStructs.ConvertToNumpyArray(fp, array)
    mol_pb.binary_features.extend(array)


def _values_to_distribution_summary(
    values
):
  pandas_summary = pd.Series(values).describe()
  return mgpb.SimpleMoleculeFeatures.DistributionSummary(
      min=pandas_summary.get('min', np.nan),
      max=pandas_summary.get('max', np.nan),
      mean=pandas_summary.get('mean', np.nan),
      median=pandas_summary.get('50%', np.nan),
      count=pandas_summary.get('count', np.nan),
      std=pandas_summary.get('std', np.nan),)


def proto_to_simple_features(mol_pb):
  """Converts a MoleculeGraph in proto form to a simple featurization.

  Args:
    mol_pb: a MoleculeGraph proto

  Returns:
    A SimpleMoleculeFeatures proto
  """
  out_pb = mgpb.SimpleMoleculeFeatures()

  out_pb.num_atoms = len(mol_pb.atoms)
  for element in (a.element for a in mol_pb.atoms):
    out_pb.element_type_counts[element] += 1
    if element != 'H':
      out_pb.num_heavy_atoms += 1

  out_pb.is_chiral = any(a.chirality != mgpb.MoleculeGraph.Atom.CHIRAL_NONE
                         for a in mol_pb.atoms)

  out_pb.num_hbond_acceptor = 0
  out_pb.num_hbond_donor = 0
  for a in mol_pb.atoms:
    if a.acceptor:
      out_pb.num_hbond_acceptor += 1
    if a.donor:
      out_pb.num_hbond_donor += 1

  out_pb.partial_charges_distribution.CopyFrom(_values_to_distribution_summary(
      a.partial_charge for a in mol_pb.atoms))
  out_pb.formal_charges_distribution.CopyFrom(_values_to_distribution_summary(
      a.formal_charge for a in mol_pb.atoms))

  out_pb.is_aromatic = any(a.aromatic for a in mol_pb.atoms)

  for bond_type in (p.bond_type for p in mol_pb.atom_pairs):
    if bond_type != mgpb.MoleculeGraph.AtomPair.BOND_NONE:
      out_pb.bond_type_counts[int(bond_type)] += 1

  all_ring_sizes = [a.ring_sizes for a in mol_pb.atoms]
  out_pb.ring_sizes.extend(list(set(itertools.chain(*all_ring_sizes))))

  out_pb.graph_distances_distribution.CopyFrom(
      _values_to_distribution_summary(
          p.graph_distance for p in mol_pb.atom_pairs))
  out_pb.spatial_distances_distribution.CopyFrom(
      _values_to_distribution_summary(
          p.spatial_distance for p in mol_pb.atom_pairs))

  return out_pb
