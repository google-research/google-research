// Implementation for SMU

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "Foundational/iwstring/iw_stl_hash_map.h"

#include "smu.h"

namespace smu {

int invalid_valence_rejected = 0;

// Return a vector containing the implicit_hydrogens value
// for each atom in `m`.
std::vector<int>
ImplicitHydrogens(Molecule& m) {
  const int matoms = m.natoms();
  std::vector<int> ih(matoms);
  for (int i = 0; i < matoms; ++i) {
    ih[i] = m.implicit_hydrogens(i);
  }

  return ih;
}

bool
OkCharges(const Molecule& m, const SMU_Params& params) {
  atom_number_t first_positive_charge = INVALID_ATOM_NUMBER;
  atom_number_t first_negative_charge = INVALID_ATOM_NUMBER;
  const int matoms = m.natoms();
  for (int i = 0; i < matoms; ++i) {
    const formal_charge_t fc = m.formal_charge(i);
    if (fc == 0) {
      continue;
    }

    if (fc > 0) {
      if (first_positive_charge >= 0) {
        return false;
      }
      first_positive_charge = i;
    } else {
      if (first_negative_charge >= 0) {
        return false;
      }
      first_negative_charge = i;
    }
  }

  // No charges encountered.
  if (first_negative_charge < 0 && first_positive_charge < 0) {
    return true;
  }

  // Molecule has net formal charge.
  if (first_negative_charge < 0 || first_positive_charge < 0) {
    return false;
  }

  // Charged atoms are only OK if adjacent.
  if (params.oneg_only_added_to_npos) {
    return m.are_bonded(first_negative_charge, first_positive_charge);
  }

  return true;
}

// Is atom `a` a negatively charged oxygen?
bool
IsOneg(const Atom& a) {
  return a.atomic_number() == 8 && a.formal_charge() == -1;
}

// Is atom `a` a pisitively charged nitrogen?
bool
IsNpos(const Atom& a) {
  return a.atomic_number() == 7 && a.formal_charge() == 1;
}

bool
OkFormalChargeAdjacency(const Atom& a1, const Atom & a2,
                        const SMU_Params& params) {
  const formal_charge_t fc1 = a1.formal_charge();
  if (fc1 == 0) {
    if (params.oneg_only_added_to_npos && IsOneg(a2) && ! IsNpos(a1)) {
      return false;
    }
    return true;
  }
  const formal_charge_t fc2 = a2.formal_charge();
  if (fc2 == 0) {
    return true;
  }

  // If non zero, cannot be the same.
  if (fc1 == fc2) {
    return false;
  }

  return true;
}

// Given two atoms that have `ih1` and `ih2` implicit hydrogens,
// return the maximum bond type (as an int) that can be made
// between them.
int MaxBtype(const int ih1, const int ih2) {
  int result = std::min(ih1, ih2);
  if (result > 3)
    result = 3;

  return result;
}

void
AddBond(Molecule & m, atom_number_t a1, atom_number_t a2, bond_type_t btype) {
  m.add_bond(a1, a2, btype);
}

// Returns true if `zatom` already has the maximum number of
// connections for that atom type.
bool
FullyConnected(Molecule& m, atom_number_t zatom) {
  static const std::array<int, 10> max_ncon{
    0,    // no element 0
    1,    // H 1
    0,    // He 2
    0,    // Li 3
    0,    // Be 4
    0,    // B  5
    4,    // C  6
    3,    // N  7
    2,    // O  8
    1,    // F  9
  };
  const Atom * a = m.atomi(zatom);

  return a->ncon() >= max_ncon[a->atomic_number()] + a->formal_charge();
}

// For all possible atom pairs in `m` generate a ring by joining
// the two atoms.
void 
AddRings(Molecule& m,
         const SMU_Params& params,
         IW_STL_Hash_Set& already_produced,
         SmuResults& smu_results) {

  const int matoms = m.natoms();
  if (matoms <= 2) {
    return;
  }

  static const std::array<int, 4> bonds{NOT_A_BOND, SINGLE_BOND, DOUBLE_BOND, TRIPLE_BOND};

  const std::vector<int> ih = ImplicitHydrogens(m);

  for (int i = 0; i < matoms; ++i) {
    if (ih[i] == 0 || FullyConnected(m, i)) {
      continue;
    }
    for (int j = i + 1; j < matoms; ++j) {
      if (ih[j] == 0 || FullyConnected(m, j)) {
        continue;
      }
      if (m.are_bonded(i, j)) {
        continue;
      }
      int max_bonds = MaxBtype(ih[i], ih[j]);
      for (int btype = 1; btype <= max_bonds; ++btype) {
        std::unique_ptr<Molecule> mcopy = std::make_unique<Molecule>(m);
//      cerr << "Adding ring bond btw " << i << " and " << j << "\n";
        AddBond(*mcopy, i, j, bonds[btype]);
        smu_results.Add(mcopy, already_produced, params.non_aromatic_unique_smiles);
      }
    }
  }
}

// Add an extra atom to `m`, based on `to_add`.
void
AddAtom(Molecule& m, const Atom& to_add) {
  m.add(to_add.element());
  if (to_add.formal_charge()) {
    m.set_formal_charge(m.natoms() - 1, to_add.formal_charge());
  }
}

// Build new molecules from `m` by adding new atoms to `zatom`.
void
ExpandFromAtom(Molecule & m,
               atom_number_t zatom,
               const SMU_Params& params,
               IW_STL_Hash_Set& already_produced,
               SmuResults& smu_results) {
  const int ih_zatom = m.implicit_hydrogens(zatom);
  if (ih_zatom == 0) {
    return;
  }

  static const std::array<int, 4> bonds{NOT_A_BOND, SINGLE_BOND, DOUBLE_BOND, TRIPLE_BOND};

  const Atom * a = m.atomi(zatom);

  int initial_natoms = m.natoms();

  for (Atom * to_add : params.to_add) {
    if (! OkFormalChargeAdjacency(*a, *to_add, params)) {
      continue;
    }
    const int max_btype = MaxBtype(ih_zatom, to_add->implicit_hydrogens());
    for (int btype = 1; btype <= max_btype; ++btype) {
      std::unique_ptr<Molecule> mcopy = std::make_unique<Molecule>(m);
      AddAtom(*mcopy, *to_add);
//    cerr << "To " << mcopy->smiles() << " addinb bond btw " << zatom << " and " << initial_natoms << " type " << bonds[btype] << "\n";
      AddBond(*mcopy, zatom, initial_natoms, bonds[btype]);
      // Because smu_results.Add may remove the pointer from mcopy on success, save it here.
      Molecule* msave = mcopy.get();
      if (smu_results.Add(mcopy, already_produced, params.non_aromatic_unique_smiles)) {
        AddRings(*msave, params, already_produced, smu_results);
      }
    }
  }
}

void ExpandMolecule(Molecule& m,
            const SMU_Params& params,
            IW_STL_Hash_Set& already_produced,
            SmuResults& smu_results) {
  const int matoms = m.natoms();
  for (int i = 0; i < matoms; ++i) {
    ExpandFromAtom(m, i, params, already_produced, smu_results);
  }
}


void
Expand(Molecule& starting_molecule,
       const SMU_Params& params,
       IW_STL_Hash_Set& already_produced,
       SmuResults& smu_results) {
  // Not really necessary.
  assert(starting_molecule.natoms() == 1);

  std::unique_ptr<Molecule> m = std::make_unique<Molecule>(starting_molecule);
  smu_results.Add(m, already_produced, params.non_aromatic_unique_smiles);

  // Range is max_atoms-1 since we have an initial molecule.
  for (int i = 0; i < params.max_atoms - 1; ++i) {
    const std::vector<IWString> current_set = smu_results.CurrentSmiles(i + 1);
//  cerr << " i = " << i << " current_set " << current_set.size() << "\n";
    for (const IWString & smi : current_set) {
      Molecule m;
      if (! m.build_from_smiles(smi)) {
        cerr << "Cannot build smiles " << smi << "\n";
        continue;
      }
      m.each_index_lambda([&m](int i) {m.unset_all_implicit_hydrogen_information(i);});
      ExpandMolecule(m, params, already_produced, smu_results);
    }
  }
}

}  // namespace smu
