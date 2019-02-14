# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [
    'Roboto Condensed', 'Roboto Condensed Regular'
]

import seaborn as sns
import math
import rdkit
import itertools
from rdkit import Chem, DataStructs
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, Draw, Descriptors, QED
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import leastsq
from scipy import interpolate
import cairosvg as cs


def get_valid_actions(state, atom_types, allow_removal, allow_no_modification,
                      allowed_ring_sizes, allow_bonds_between_rings):
  """Computes the set of valid actions for a given state.

  Args:
    state: String SMILES; the current state. If None or the empty string, we
      assume an "empty" state with no atoms or bonds.
    atom_types: Set of string atom types, e.g. {'C', 'O'}.
    allow_removal: Boolean whether to allow actions that remove atoms and bonds.
    allow_no_modification: Boolean whether to include a "no-op" action.
    allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings.

  Returns:
    Set of RDKit Mol containing the valid actions (technically, the set of
    all states that are acceptable from the given state).

  Raises:
    ValueError: If state does not represent a valid molecule.
  """
  if not state:
    # Available actions are adding a node of each type.
    return copy.deepcopy(atom_types)
  mol = Chem.MolFromSmiles(state)
  if mol is None:
    raise ValueError('Received invalid state: %s' % state)
  # atom_valences = dict(
  # #zip(sorted(atom_types), molecules.atom_valences(sorted(atom_types))))
  # zip(sorted(atom_types), molecules_py.atom_valences(sorted(atom_types))))
  atom_valences = {'C': 4, 'H': 1, 'O': 2, 'N': 3}
  atoms_with_free_valence = {
      i: [
          atom.GetIdx()
          for atom in mol.GetAtoms()
          # Only atoms that allow us to replace at least one H with a new bond
          # are enumerated here.
          if atom.GetNumImplicitHs() >= i
      ] for i in range(1, max(atom_valences.values()))
  }
  valid_actions = set()
  valid_actions.update(
      _atom_addition(
          mol,
          atom_types=atom_types,
          atom_valences=atom_valences,
          atoms_with_free_valence=atoms_with_free_valence))
  valid_actions.update(
      _bond_addition(
          mol,
          atoms_with_free_valence=atoms_with_free_valence,
          allowed_ring_sizes=allowed_ring_sizes,
          allow_bonds_between_rings=allow_bonds_between_rings))
  if allow_removal:
    valid_actions.update(_bond_removal(mol))
  if allow_no_modification:
    #valid_actions.add(Chem.MolToSmiles(mol))
    valid_actions.add(Chem.Mol(mol))
  return valid_actions


def _atom_addition(state, atom_types, atom_valences, atoms_with_free_valence):
  """Computes valid actions that involve adding atoms to the graph.

  Actions:
    * Add atom (with a bond connecting it to the existing graph)

  Each added atom is connected to the graph by a bond. There is a separate
  action for connecting to (a) each existing atom with (b) each valence-allowed
  bond type. Note that the connecting bond is only of type single, double, or
  triple (no aromatic bonds are added).

  For example, if an existing carbon atom has two empty valence positions and
  the available atom types are {'C', 'O'}, this section will produce new states
  where the existing carbon is connected to (1) another carbon by a double bond,
  (2) another carbon by a single bond, (3) an oxygen by a double bond, and
  (4) an oxygen by a single bond.

  Args:
    state: RDKit Mol.
    atom_types: Set of string atoms.
    atom_valences: Dict mapping string atom types to integer valences.
    atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.

  Returns:
    Set of RDKit Mol; the available actions.
  """
  bond_order = {
      1: Chem.BondType.SINGLE,
      2: Chem.BondType.DOUBLE,
      3: Chem.BondType.TRIPLE,
  }
  atom_addition = set()
  for i in range(1, max(atom_valences.values())):
    if i not in bond_order:
      continue  # Skip valences that are too high.
    for atom in atoms_with_free_valence[i]:
      for element in atom_types:
        if atom_valences[element] >= i:
          new_state = Chem.RWMol(state)
          idx = new_state.AddAtom(Chem.Atom(element))
          new_state.AddBond(atom, idx, bond_order[i])
          sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
          if sanitization_result:
            continue  # Skip the molecule when sanitization fails.
          #atom_addition.add(Chem.MolToSmiles(new_state))
          atom_addition.add(new_state)
  return atom_addition


def _bond_addition(state, atoms_with_free_valence, allowed_ring_sizes,
                   allow_bonds_between_rings):
  """Computes valid actions that involve adding bonds to the graph.

  Actions (where allowed):
    * None->{single,double,triple}
    * single->{double,triple}
    * double->{triple}

  Note that aromatic bonds are not modified.

  Args:
    state: RDKit Mol.
    atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.
    allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings.

  Returns:
    Set of RDKit Mol; the available actions.
  """
  bond_orders = [
      None,
      Chem.BondType.SINGLE,
      Chem.BondType.DOUBLE,
      Chem.BondType.TRIPLE,
  ]
  bond_addition = set()
  for valence, atoms in atoms_with_free_valence.items():
    if valence > 3:
      continue  # Skip valences that are too high.
    for atom1, atom2 in itertools.combinations(atoms, 2):
      # Get the bond from a copy of the molecule so that SetBondType() doesn't
      # modify the original state.
      bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
      new_state = Chem.RWMol(state)
      # Kekulize the new state to avoid sanitization errors; note that bonds
      # that are aromatic in the original state are not modified (this is
      # enforced by getting the bond from the original state with
      # GetBondBetweenAtoms()).
      Chem.Kekulize(new_state, clearAromaticFlags=True)
      if bond is not None:
        if bond.GetBondType() not in bond_orders:
          continue  # Skip aromatic bonds.
        idx = bond.GetIdx()
        # Compute the new bond order as an offset from the current bond order.
        bond_order = bond_orders.index(bond.GetBondType())
        bond_order += valence
        if bond_order < len(bond_orders):
          idx = bond.GetIdx()
          bond.SetBondType(bond_orders[bond_order])
          new_state.ReplaceBond(idx, bond)
        else:
          continue
      # If do not allow new bonds between atoms already in rings.
      elif (not allow_bonds_between_rings and
            (state.GetAtomWithIdx(atom1).IsInRing() and
             state.GetAtomWithIdx(atom2).IsInRing())):
        continue
      # If the distance between the current two atoms is not in the
      # allowed ring sizes
      elif (allowed_ring_sizes is not None and
            len(Chem.rdmolops.GetShortestPath(
                state, atom1, atom2)) not in allowed_ring_sizes):
        continue
      else:
        new_state.AddBond(atom1, atom2, bond_orders[valence])
      sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
      if sanitization_result:
        continue  # Skip the molecule when sanitization fails.
      #bond_addition.add(Chem.MolToSmiles(new_state))
      bond_addition.add(new_state)
  return bond_addition


def _bond_removal(state):
  """Computes valid actions that involve removing bonds from the graph.

  Actions (where allowed):
    * triple->{double,single,None}
    * double->{single,None}
    * single->{None}

  Bonds are only removed (single->None) if the resulting graph has zero or one
  disconnected atom(s); the creation of multi-atom disconnected fragments is not
  allowed. Note that aromatic bonds are not modified.

  Args:
    state: RDKit Mol.

  Returns:
    Set of RDKit Mol; the available actions.
  """
  bond_orders = [
      None,
      Chem.BondType.SINGLE,
      Chem.BondType.DOUBLE,
      Chem.BondType.TRIPLE,
  ]
  bond_removal = set()
  for valence in [1, 2, 3]:
    for bond in state.GetBonds():
      # Get the bond from a copy of the molecule so that SetBondType() doesn't
      # modify the original state.
      bond = Chem.Mol(state).GetBondBetweenAtoms(bond.GetBeginAtomIdx(),
                                                 bond.GetEndAtomIdx())
      if bond.GetBondType() not in bond_orders:
        continue  # Skip aromatic bonds.
      new_state = Chem.RWMol(state)
      # Kekulize the new state to avoid sanitization errors; note that bonds
      # that are aromatic in the original state are not modified (this is
      # enforced by getting the bond from the original state with
      # GetBondBetweenAtoms()).
      Chem.Kekulize(new_state, clearAromaticFlags=True)
      # Compute the new bond order as an offset from the current bond order.
      bond_order = bond_orders.index(bond.GetBondType())
      bond_order -= valence
      if bond_order > 0:  # Downgrade this bond.
        idx = bond.GetIdx()
        bond.SetBondType(bond_orders[bond_order])
        new_state.ReplaceBond(idx, bond)
        sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
        if sanitization_result:
          continue  # Skip the molecule when sanitization fails.
        #bond_removal.add(Chem.MolToSmiles(new_state))
        bond_removal.add(new_state)
      elif bond_order == 0:  # Remove this bond entirely.
        atom1 = bond.GetBeginAtom().GetIdx()
        atom2 = bond.GetEndAtom().GetIdx()
        new_state.RemoveBond(atom1, atom2)
        sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
        if sanitization_result:
          continue  # Skip the molecule when sanitization fails.
        smiles = Chem.MolToSmiles(new_state)
        parts = sorted(smiles.split('.'), key=len)
        # We define the valid bond removing action set as the actions
        # that remove an existing bond, generating only one independent
        # molecule, or a molecule and an atom.
        if len(parts) == 1 or len(parts[0]) == 1:
          #bond_removal.add(parts[-1])
          bond_removal.add(Chem.MolFromSmiles(parts[-1]))
  return bond_removal


def highlights_diff(original_mol, next_mol):
  highlight_atoms = []
  original_num_atoms = len(original_mol.GetAtoms())
  next_num_atoms = len(next_mol.GetAtoms())
  for i in range(min(original_num_atoms, next_num_atoms)):
    if original_mol.GetAtoms()[i].GetSymbol() != next_mol.GetAtoms(
    )[i].GetSymbol():
      highlight_atoms.append(next_mol.GetAtoms()[i].GetIdx())
  if next_num_atoms > original_num_atoms:
    highlight_atoms.extend(range(original_num_atoms, next_num_atoms))

  highlight_bonds = []
  original_num_bonds = len(original_mol.GetBonds())
  next_num_bonds = len(next_mol.GetBonds())
  for i in range(min(original_num_bonds, next_num_bonds)):
    if original_mol.GetBonds()[i].GetBondType() != next_mol.GetBonds(
    )[i].GetBondType():
      highlight_bonds.append(next_mol.GetBonds()[i].GetIdx())
  if next_num_bonds > original_num_bonds:
    highlight_bonds.extend(range(original_num_bonds, next_num_bonds))
  return highlight_atoms, highlight_bonds


def tidy_smiles(smiles):
  new_smiles = {
      'weight_0': list(set(smiles['weight_0'][-30:])),
      'weight_1': list(set(smiles['weight_1'][-30:])),
      'weight_2': list(set(smiles['weight_2'][-150:])),
      'weight_3': list(set(smiles['weight_3'][-150:])),
      'weight_4': list(set(smiles['weight_4'][-150:])),
      'weight_5': list(set(smiles['weight_5'][-150:]))
  }
  return new_smiles


def get_properties(smiles, target_molecule='C1CCC2CCCCC2C1'):
  target_mol_fp = AllChem.GetMorganFingerprintAsBitVect(
      Chem.MolFromSmiles(target_molecule), radius=2, nBits=2048)
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return 0.0, 0.0
  fingerprint_structure = AllChem.GetMorganFingerprintAsBitVect(
      mol, radius=2, nBits=2048)
  sim = DataStructs.TanimotoSimilarity(target_mol_fp, fingerprint_structure)
  qed = QED.qed(mol)
  return sim, qed


def plot_multi_obj_opt(smiles, target_mol, idx=0):
  with open('all_molecules_with_id.json') as f:
    molid = json.load(f)
  colors = iter(cm.rainbow(np.linspace(0, 1, 6)))
  plt.figure()
  for i in range(6):
    ssl = smiles['weight_%i' % i]
    sim, qed = zip(
        *[get_properties(ss, target_molecule=target_mol) for ss in ssl])
    plt.scatter(sim, qed, label='w=%.1f' % (i * 0.2), color=next(colors))
  target_sim, target_qed = get_properties(target_mol, target_mol)
  plt.axvline(x=target_sim, ls='dashed', color='grey')
  plt.axhline(y=target_qed, ls='dashed', color='grey')
  leg = plt.legend()
  leg.get_frame().set_alpha(0.95)
  plt.ylim((-0.2, 1))
  plt.xlabel('Similarity')
  plt.ylabel('QED')
  plt.title(molid[target_mol])
  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.88)
  plt.savefig('batch/mult_obj_gen_{}.pdf'.format(idx))
  #plt.show()


def plot_multi_obj_gen_drug20():
  with open('multi_obj_opt_drug20.json') as f:
    data = json.load(f)
  plot_multi_obj_opt_multi_plot(result['smiles'], result['target_mol'], 2)


def plot_qed_improvements():
  with open('qed_imp_2.json') as f:
    improvements = json.load(f)

  def double_gaussian(x, params):
    (c1, mu1, sigma1, c2, mu2, sigma2) = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

  def double_gaussian_fit(params, y):
    fit = double_gaussian(x, params)
    return (fit - y)

  colors = list(iter(cm.rainbow(np.linspace(0, 1, 6))))
  colors = ['#eae471', '#c1e092', '#83b49d', '#448fad', '#3e60c3', '#5a26a6']
  plt.figure()
  start = -0.4
  end = 0.6
  for i in range(6):
    imp = np.array(improvements['weight_%i' % i])
    y, binEdges = np.histogram(imp, bins=40, range=(start, end))
    y = y.astype(np.float64)
    y /= y.sum()
    x = 0.5 * (binEdges[1:] + binEdges[:-1])
    if i == 0:
      fit = leastsq(lambda x: double_gaussian_fit(x, y),
                    [1, 0, 0.02, 1, 0.3, 0.1])
    elif i == 1:
      fit = leastsq(lambda x: double_gaussian_fit(x, y),
                    [1, 0, 0.02, 1, 0.1, 0.1])
    else:
      fit = leastsq(lambda x: double_gaussian_fit(x, y),
                    [1, 0, 0.02, 1, 0.1, 0.05])
    xx = np.linspace(start, end, 300)
    yy = double_gaussian(xx, fit[0])

    plt.plot(x, y, 'o', color=colors[i], alpha=0.3)
    plt.plot(
        xx,
        yy,
        color=colors[i],
        label='w=%.1f' % (i * 0.2),
    )
    plt.xlim(start, end)
    # plt.ylim(-0.02, 0.2)

  plt.legend()
  plt.xlabel('Improvements on QED')
  plt.ylabel('Normalized count')
  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.92)
  plt.savefig('qed_improvements.pdf')
  plt.show()


def plot_qed_relative_improvements():
  with open('qed_rel_imp_2.json') as f:
    improvements = json.load(f)

  def double_gaussian(x, params):
    (c1, mu1, sigma1, c2, mu2, sigma2) = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

  def double_gaussian_fit(params, y):
    fit = double_gaussian(x, params)
    return (fit - y)

  colors = list(iter(cm.rainbow(np.linspace(0, 1, 6))))
  colors = ['#eae471', '#c1e092', '#83b49d', '#448fad', '#3e60c3', '#5a26a6']
  plt.figure()
  start = -1
  end = 1
  for i in range(6):
    imp = np.array(improvements['weight_%i' % i])
    y, binEdges = np.histogram(imp, bins=40, range=(start, end))
    y = y.astype(np.float64)
    y /= y.sum()
    x = 0.5 * (binEdges[1:] + binEdges[:-1])
    if i == 0:
      fit = leastsq(lambda x: double_gaussian_fit(x, y),
                    [1, 0.5, 0.1, 1, 0.6, 0.1])
    elif i == 1:
      fit = leastsq(lambda x: double_gaussian_fit(x, y),
                    [1, 0.2, 0.05, 1, 0.5, 0.1])
    else:
      fit = leastsq(lambda x: double_gaussian_fit(x, y),
                    [1, 0, 0.1, 1, 0.4, 0.5])
    xx = np.linspace(start, end, 300)
    yy = double_gaussian(xx, fit[0])

    plt.plot(x, y, 'o', color=colors[i], alpha=0.3)
    plt.plot(
        xx,
        yy,
        color=colors[i],
        label='w=%.1f' % (i * 0.2),
    )
    plt.xlim(start, end)
    # plt.ylim(-0.02, 0.2)

  plt.legend()
  plt.xlabel('Relative improvements on QED')
  plt.ylabel('Normalized count')
  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.92)
  plt.savefig('qed_rel_improvements.pdf')
  plt.show()


def plot_drug20_smiles():
  with open('drug_20_smiles.json') as f:
    data = json.load(f)
  smiles = sum(data.values(), [])
  mols = [Chem.MolFromSmiles(ss) for ss in smiles]
  target_mol = 'CN1C(=O)C2(OCCO2)c3ccccc13'
  template1 = Chem.MolFromSmiles('N1C(=O)C2(OCCO2)c3ccccc13')
  AllChem.Compute2DCoords(template1, canonOrient=True)

  properties = [
      'SIM: %.3f\nQED: %.3f' % get_properties(mol, target_mol) for mol in smiles
  ]
  # img = Draw.MolsToGridImage(mols, molsPerRow=5,
  # subImgSize=(300, 150), useSVG=True)
  # imgsize = (280, 100)
  # drawer = rdMolDraw2D.MolDraw2DSVG(imgsize[0] * 5, imgsize[1] * 6 + 20,
  # imgsize[0], imgsize[1])
  # drawer.SetFontSize(0.8) # <- default is 0.5, so this makes the font half
  # drawer.drawOptions().legendFontSize = 18
  # drawer.DrawMolecules(mols, legends=properties)
  # drawer.FinishDrawing()
  # img = drawer.GetDrawingText()
  #cs.svg2pdf(bytestring=img.encode('utf-8'), write_to='drug20_smiles.pdf')
  for i in range(4, 5):
    smiles = data[f'weight_{i}']
    mols = [Chem.MolFromSmiles(ss) for ss in smiles]
    for mol in mols:
      try:
        AllChem.GenerateDepictionMatching2DStructure(mol, template1)
      except:
        pass
    properties = [
        'SIM: %.3f, QED: %.3f' % get_properties(mol, target_mol)
        for mol in smiles
    ]
    imgsize1 = [260, 340, 280, 280, 220, 220]
    imgsize = (240, imgsize1[i])
    drawer = rdMolDraw2D.MolDraw2DSVG(imgsize[0] * 5, imgsize[1] + 5,
                                      imgsize[0], imgsize[1])
    drawer.SetFontSize(0.8)  # <- default is 0.5, so this makes the font half
    drawer.drawOptions().legendFontSize = 18
    drawer.DrawMolecules(mols, legends=properties)
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()
    cs.svg2pdf(
        bytestring=img.encode('utf-8'), write_to=f'drug1_smiles_w{i}.pdf')


def plot_max_qed_mols_2():
  smiles = {
      'CCC1C(=O)C(C)(Cc2cnc3[nH]c4c(n23)CCO4)CC1C': 0.9480413389762415,
      'CCC(C)C12C3=CC(=O)C1Cc1cc4nc(OC)[nH]c4c(c12)C3': 0.9477126732214856,
      'CCCC1C(C)=NCC12COCC2n1cc2ocnc2c1O': 0.9469782135033733,
      'C=Cc1nc(OC)[nH]c1-c1c(C)cc2c3c1CC(=O)C(CC2)O3': 0.9465532716678036,
  }
  mols = [Chem.MolFromSmiles(k) for k, v in smiles.items()][:4]
  properties = ['QED: %.3f' % v for k, v in smiles.items()]
  # img = Draw.MolsToGridImage(mols, molsPerRow=2, legends=properties,
  # subImgSize=(300, 200), useSVG=True)
  drawer = rdMolDraw2D.MolDraw2DSVG(600, 400, 300, 200)
  drawer.SetFontSize(0.8)  # <- default is 0.5, so this makes the font half
  drawer.drawOptions().legendFontSize = 18
  drawer.DrawMolecules(mols, legends=properties)
  drawer.FinishDrawing()
  img = drawer.GetDrawingText()
  cs.svg2pdf(bytestring=img.encode('utf-8'), write_to='max_qed_mols_2.pdf')


def plot_max_logp_mols_2():
  logp = [11.7435, 11.7182, 11.7090, 11.7090]
  smiles = [
      'C=C(CCCCCCCC=C(CCCCCC)CCCCC(C)=CCCCC)C(C)(C)CCCCCCC(C)C',
      'C=C(CCCCC)CCCCCCC=C(CCCCCC=CCCC)CCCC(C)(C)CCCCCCC(C)C',
      'C=C(CCCCC(C)=CCCCCCC)CCCC(CCCCCC(C)C)=C(C)CCCCCCCC(C)(C)C',
      'C=C(CCCCC(C)=CCCCCCC)CCCC(CCCCCCCC(C)(C)C)=C(C)CCCCCC(C)C'
  ]
  mols = [Chem.MolFromSmiles(ss) for ss in smiles]
  properties = ['Penalized logP: %.2f' % v for v in logp]
  # img = Draw.MolsToGridImage(mols, molsPerRow=2, legends=properties,
  # subImgSize=(300, 200), useSVG=True)

  drawer = rdMolDraw2D.MolDraw2DSVG(600, 400, 300, 200)
  drawer.SetFontSize(0.8)  # <- default is 0.5, so this makes the font half
  drawer.drawOptions().legendFontSize = 18
  drawer.DrawMolecules(mols, legends=properties)
  drawer.FinishDrawing()
  img = drawer.GetDrawingText()

  cs.svg2pdf(bytestring=img.encode('utf-8'), write_to='max_logp_mols_2.pdf')


def plot_max_logp_mols():
  logp = [11.7069205, 11.63197045, 11.6280874, 11.62077996]
  smiles = [
      'CCCCCC=C(CCCC(CC)CC)CCC(C)(C)CCCCC(C)=CCCCCCCC(C)(C)CCCC',
      'C=C(CCCCCCCC=CCC)CCC(=CCCCCC(C)(C)CCCCCCC(CC)CC)CCCCC',
      'C=C(CCCCCC)CCCC(=CCCCCC(C)(C)CCCCCCC(CC)CC)CCCCC=CCCC',
      'C=C(CCCCC(C)(C)CCCC(C)(CC)CC)CCC(=CCCCCCC(CC)CC)CCCCCCCC'
  ]
  mols = [Chem.MolFromSmiles(ss) for ss in smiles]
  properties = ['Penalized logP: %.2f' % v for v in logp]
  # img = Draw.MolsToGridImage(mols, molsPerRow=2, legends=properties,
  # subImgSize=(300, 200), useSVG=True)

  drawer = rdMolDraw2D.MolDraw2DSVG(600, 400, 300, 200)
  drawer.SetFontSize(0.8)  # <- default is 0.5, so this makes the font half
  drawer.drawOptions().legendFontSize = 18
  drawer.DrawMolecules(mols, legends=properties)
  drawer.FinishDrawing()
  img = drawer.GetDrawingText()

  cs.svg2pdf(bytestring=img.encode('utf-8'), write_to='max_logp_mols.pdf')


def plot_noisy_qed_reward():
  colors = list(iter(cm.rainbow(np.linspace(0, 1, 4))))
  with open('noise.json') as f:
    all_qed = json.load(f)
  plt.figure()
  for i in range(4):
    qed = all_qed['robust_0.%i' % i]
    lq = len(qed)
    window = 200

    x = [j * 200 for j in range(lq // window - 1)]
    y = [
        np.mean(qed[window * j:window * (j + 1)])
        for j in range(lq // window - 1)
    ]
    fit = interpolate.UnivariateSpline(x, y, k=3)
    xx = np.linspace(0, 5000, 100)
    plt.plot(x, y, '-', alpha=0.2, color=colors[i])
    plt.plot(xx, fit(xx), label='robust, $\sigma$=0.%i' % i, color=colors[i])

    qed = all_qed['l2_0.%i' % i]
    lq = len(qed)
    window = 200

    x = [j * 200 for j in range(lq // window - 1)]
    y = [
        np.mean(qed[window * j:window * (j + 1)])
        for j in range(lq // window - 1)
    ]
    fit = interpolate.UnivariateSpline(x, y, k=3)
    xx = np.linspace(0, 5000, 100)
    plt.plot(x, y, ls='dashed', alpha=0.2, color=colors[i])
    plt.plot(
        xx,
        fit(xx),
        ls='dashed',
        label='l2, $\sigma$=0.%i' % i,
        color=colors[i])

  plt.xlim(0, 4600)
  plt.ylim(0.2, 1)
  plt.xlabel('Number of epochs')
  plt.ylabel('Reward')
  plt.legend(loc='upper left')
  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.92)
  plt.savefig('noisy_reward.pdf')
  plt.show()


def plot_final_vs_intermediate_reward():
  with open('final_vs_interm_reward.json') as f:
    all_qed = json.load(f)
  plt.figure()
  qed = all_qed['intermediate_reward']
  lq = len(qed)
  window = 200

  x = [j * window + 1 for j in range(lq // window - 1)]
  y = [
      np.mean(qed[window * j:window * (j + 1)]) for j in range(lq // window - 1)
  ]
  fit = interpolate.UnivariateSpline(
      x,
      y,
      k=3,
  )
  xx = np.linspace(0, 5000, 100)
  plt.plot(x, y, 'o', color='C0', alpha=0.2)
  plt.plot(xx, fit(xx), label='intermediate reward')

  qed = all_qed['final_reward']
  lq = len(qed)
  window = 200
  x = [j * window + 1 for j in range(lq // window - 1)]
  y = [
      np.mean(qed[window * j:window * (j + 1)]) for j in range(lq // window - 1)
  ]
  fit = interpolate.UnivariateSpline(
      x,
      y,
      k=3,
  )
  xx = np.linspace(0, 5000, 100)
  plt.plot(x, y, 'o', color='C1', alpha=0.2)
  plt.plot(xx, fit(xx), label='final reward')

  plt.xlim(0, 4600)
  plt.ylim(0.2, 0.8)
  plt.xlabel('Number of epochs')
  plt.ylabel('Reward')
  plt.legend(loc='upper left')
  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.92)
  plt.savefig('final_vs_intermediate_reward.pdf')
  plt.show()


def plot_qvals_with_change_20():
  highlightcolor = (0.98, 0.85, 0.37)

  with open('q_values_20.json') as f:
    qvals = json.load(f)

  original_smiles = 'CN1C(=O)C2(OCCO2)c2ccccc21'
  original_state = Chem.MolFromSmiles(original_smiles)
  original_state2 = Chem.MolFromSmiles(original_smiles)

  next_states = list(
      get_valid_actions(
          state=original_smiles,
          atom_types={'C', 'N', 'O'},
          allow_removal=False,
          allow_no_modification=True,
          allowed_ring_sizes={3, 5, 6},
          allow_bonds_between_rings=False))

  bond_removal_actions = [
      Chem.MolToSmiles(ss) for ss in _bond_removal(original_state2)
  ]
  # bond_removal_actions = {}
  stated = {Chem.MolToSmiles(s): s for s in next_states}
  mols = []
  ha = []
  hb = []
  hac = []
  hbc = []
  prop = []
  for k, v in sorted(qvals.items(), key=lambda x: -x[1]):
    if k in stated or k in bond_removal_actions:
      if k in stated:
        mol = stated[k]
        mols.append(mol)
        hla, hlb = highlights_diff(original_state, mol)
        ha.append(hla)
        hb.append(hlb)
        hac.append({a: highlightcolor for a in hla})
        hbc.append({b: highlightcolor for b in hlb})
      else:
        mols.append(Chem.MolFromSmiles(k))
        ha.append([])
        hb.append([])
        hac.append({})
        hbc.append({})
      prop.append('%.4f' % v)

  # img = Draw.MolsToGridImage(mols, molsPerRow=5,
  # subImgSize=(300, 150),
  # legends=prop,
  # highlightAtomLists=ha,
  # highlightBondLists=hb,
  # highlightAtomColors=hac,
  # highlightBondColors=hbc,
  # useSVG=True)

  nmols = len(mols)
  ncols = 5
  nrows = math.ceil(float(nmols) / ncols)
  drawer = rdMolDraw2D.MolDraw2DSVG(ncols * 220, nrows * 180 + 20, 220, 180)
  drawer.SetFontSize(0.75)  # <- default is 0.5, so this makes the font half
  drawer.drawOptions().legendFontSize = 20
  drawer.DrawMolecules(
      mols,
      legends=prop,
      highlightAtoms=ha,
      highlightBonds=hb,
      highlightAtomColors=hac,
      highlightBondColors=hbc)
  drawer.FinishDrawing()
  img = drawer.GetDrawingText()
  cs.svg2pdf(bytestring=img.encode('utf-8'), write_to='qval_mat_20.pdf')


def plot_opt_path_20():
  highlightcolor = (0.98, 0.85, 0.37)

  smiles = [
      'CN1C(=O)C2(OCCO2)c3ccccc13', 'C=C1COC2(O1)C(=O)N(C)c1ccccc12',
      'C=C1COC2(O1)C(=O)N(CO)c1ccccc12', 'C=C1COC2(O1)C(=O)N(CO)c1c(C)cccc12',
      'C=C1COC2(O1)C(=O)N(CO)c1c(CC)cccc12',
      'C=C1COC2(O1)C(=O)N(CO)c1c(CCC)cccc12',
      'C=C1COC2(O1)C(=O)N(CO)c1c(C(C)CC)cccc12',
      'C=C1COC2(O1)C(=O)N(CO)c1c(C(C)C(C)C)cccc12'
  ]

  template1 = Chem.MolFromSmiles(smiles[0])
  AllChem.Compute2DCoords(template1, canonOrient=True)

  mols = [Chem.MolFromSmiles(smiles[0])]
  ha = [[]]
  hb = [[]]
  hac = [{}]
  hbc = [{}]
  prop = ['Step: 0, QED: %.4f' % QED.qed(mols[0])]
  N = len(smiles)
  for i in range(1, N):
    original_smiles = smiles[i - 1]
    original_state = Chem.MolFromSmiles(original_smiles)

    next_states = list(
        get_valid_actions(
            state=original_smiles,
            atom_types={'C', 'N', 'O'},
            allow_removal=True,
            allow_no_modification=True,
            allowed_ring_sizes={3, 5, 6},
            allow_bonds_between_rings=False))

    stated = {Chem.MolToSmiles(s): s for s in next_states}
    current_smiles = smiles[i]

    mol = stated[current_smiles]
    mols.append(mol)
    hla, hlb = highlights_diff(original_state, mol)
    ha.append(hla)
    hb.append(hlb)
    hac.append({a: highlightcolor for a in hla})
    hbc.append({b: highlightcolor for b in hlb})
    prop.append('Step: %i, QED: %.4f' % (i, QED.qed(mol)))

  for i in range(8):
    AllChem.GenerateDepictionMatching2DStructure(mols[i], template1)

  # img = Draw.MolsToGridImage(mols, molsPerRow=3,
  # subImgSize=(300, 150),
  # legends=prop,
  # highlightAtomLists=ha,
  # highlightBondLists=hb,
  # highlightAtomColors=hac,
  # highlightBondColors=hbc,
  # useSVG=True)

  drawer = rdMolDraw2D.MolDraw2DSVG(220 * 4, 160 * 2 + 20, 220, 160)
  drawer.SetFontSize(0.8)  # <- default is 0.5, so this makes the font half
  drawer.drawOptions().legendFontSize = 18
  drawer.DrawMolecules(
      mols,
      legends=prop,
      highlightAtoms=ha,
      highlightBonds=hb,
      highlightAtomColors=hac,
      highlightBondColors=hbc)
  drawer.FinishDrawing()
  img = drawer.GetDrawingText()
  cs.svg2pdf(bytestring=img.encode('utf-8'), write_to='opt_path_20.pdf')


def plot_time_dependent_reward():
  with open('time_dependent.json') as f:
    all_qed = json.load(f)
  plt.figure()
  qed = all_qed['no_time']
  lq = len(qed)
  window = 200

  x = [j * window + 1 for j in range(lq // window - 1)]
  y = [
      np.mean(qed[window * j:window * (j + 1)]) for j in range(lq // window - 1)
  ]
  fit = interpolate.UnivariateSpline(
      x,
      y,
      k=3,
  )
  xx = np.linspace(0, 5000, 100)
  plt.plot(x, y, 'o', color='C0', alpha=0.2)
  plt.plot(xx, fit(xx), label='time-independent policy')

  qed = all_qed['with_time']
  lq = len(qed)
  window = 200
  x = [j * window + 1 for j in range(lq // window - 1)]
  y = [
      np.mean(qed[window * j:window * (j + 1)]) for j in range(lq // window - 1)
  ]
  fit = interpolate.UnivariateSpline(
      x,
      y,
      k=3,
  )
  xx = np.linspace(0, 5000, 100)
  plt.plot(x, y, 'o', color='C1', alpha=0.2)
  plt.plot(xx, fit(xx), label='time-dependent policy')

  plt.xlim(0, 4600)
  plt.ylim(0.2, 0.93)
  plt.xlabel('Number of epochs')
  plt.ylabel('Reward')
  plt.legend(loc='upper left')
  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.92)
  plt.savefig('time_heterogeneous.pdf')
  plt.show()


def plot_episode_length():
  with open('episode_length.json') as f:
    length_list = json.load(f)
  plt.figure()
  plt.hist(
      length_list,
      bins=[9, 10, 11, 12, 13, 14, 15, 16],
      edgecolor='black',
      linewidth=1.5)
  plt.xlabel('Number of steps before termination')
  plt.ylabel('Count')
  plt.title('Max Number of Steps: 20')

  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.90)
  plt.savefig('episode_length.pdf')
  plt.show()


def plot_episode_length_qed():
  with open('episode_length_qed.json') as f:
    length_list = json.load(f)
  plt.figure()
  plt.hist(length_list, bins=40, edgecolor='black', linewidth=1.5)
  plt.xlim((-1, 42))
  plt.xlabel('Number of steps before termination')
  plt.ylabel('Count')
  plt.title('Max Number of Steps: 40')

  plt.subplots_adjust(left=0.16, bottom=0.16, right=0.92, top=0.90)
  plt.savefig('episode_length_qed.pdf')
  plt.show()


def multi_obj_gen_stat():
  with open('multi_objective_generation.json') as f:
    data = json.load(f)
  objs = [(2.2, 0.84), (2.5, 0.27), (3.8, 0.84), (4.8, 0.27)]
  for i in range(1, 5):
    tarSAS = objs[i - 1][0]
    tarQED = objs[i - 1][1]
    prop = list(zip(*data[str(i)]))
    prop = [list(set(pp)) for pp in prop]
    print('targetSAS=%.3f, generatedSAS:mean=%.3f, var=%.3f,'
          'mean_absolute_difference=%.3f' %
          (tarSAS, np.mean(prop[0]), np.std(prop[0]),
           np.mean(np.abs(np.array(prop[0]) - tarSAS))))
    print('targetQED=%.3f, generatedQED:mean=%.3f, var=%.3f,'
          'mean_absolute_difference=%.3f' %
          (tarQED, np.mean(prop[1]), np.std(prop[1]),
           np.mean(np.abs(np.array(prop[1]) - tarQED))))


def plot_multi_obj_opt_multi_plot(smiles, target_mol, idx=0):
  with open('all_molecules_with_id.json') as f:
    molid = json.load(f)
  colors = iter(cm.rainbow(np.linspace(0, 1, 6)))
  colors = iter(cm.Set2(np.linspace(0, 1, 8)))
  colors = sns.color_palette('husl', 6)
  colors = ['#eae471', '#c1e092', '#83b49d', '#448fad', '#3e60c3', '#5a26a6']
  smiles = tidy_smiles(smiles)
  # plt.figure()
  all_sim = []
  all_qed = []
  target_sim, target_qed = get_properties(target_mol, target_mol)
  for i in range(6):
    ssl = smiles['weight_%i' % i]
    sim, qed = zip(
        *[get_properties(ss, target_molecule=target_mol) for ss in ssl])
    all_sim += list(sim)
    all_qed += list(qed)

  fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
  i = 0
  for row in ax:
    for col in row:
      ssl = smiles['weight_%i' % i]
      sim, qed = zip(
          *[get_properties(ss, target_molecule=target_mol) for ss in ssl])
      # col.scatter(all_sim, all_qed, color='#d4d4d4')
      col.scatter(sim, qed, label='w=%.1f' % (i * 0.2), color=colors[i])
      col.axvline(x=target_sim, ls='dashed', color='grey')
      col.axhline(y=target_qed, ls='dashed', color='grey')
      leg = col.legend(loc='lower left', handletextpad=0.0)
      leg.get_frame().set_alpha(0.75)
      col.set_ylim((-0.2, 1))
      col.set_xlim((-0.1, 1.1))
      i += 1
  fig.text(0.5, 0.02, 'Similarity', ha='center')
  fig.text(0.02, 0.5, 'QED', va='center', rotation='vertical')
  fig.text(0.5, 0.94, molid[target_mol], ha='center')
  plt.subplots_adjust(left=0.10, bottom=0.14, right=0.96, top=0.92, wspace=0.12)
  plt.savefig('batch/mult_obj_gen_{}.pdf'.format(idx))
  # plt.show()


# multi_obj_gen_stat()
# plot_opt_path_20()
# plot_qvals_with_change_20()
# plot_multi_obj_gen_drug20()
# plot_qed_relative_improvements()
plot_qed_improvements()
# plot_drug20_smiles()
# plot_max_qed_mols_2()
# plot_max_logp_mols_2()
# plot_noisy_qed_reward()
# plot_final_vs_intermediate_reward()
# plot_episode_length_qed()
# plot_episode_length()
