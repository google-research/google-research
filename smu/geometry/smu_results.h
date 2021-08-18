#ifndef MOLECULE_TOOLS_SMU_RESULTS_H
#define MOLECULE_TOOLS_SMU_RESULTS_H

namespace smu {
// Results can be stored in a large or small memory form.
class SmuResults {
  public:
    SmuResults(bool keep_molecule_ptrs);

    int Add(std::unique_ptr<Molecule>& mol, bool non_aromatic_unique_smiles);

  private:
    bool _large_memory;
    IW_STL_Hash_Map<IWString, Molecule*> _smiles_to_mol;
    IW_STL_Hash_Map<IWString, int> _smiles_to_id;

    // The number of molecules pass in via the Add method.
    int _molecules_examined;

    // The number rejected for being already present.
    int _duplicates_discarded;

    // The number discarded for invalid valences.
    int _invalid_valence_rejected;
};

}  // namespace smu

#endif // MOLECULE_TOOLS_SMU_RESULTS_H
