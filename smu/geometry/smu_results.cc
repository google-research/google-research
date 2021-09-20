// Copyright 2021 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "geometry/smu.h"

namespace smu {

SmuResults::SmuResults() {
  _molecules_examined = 0;
  _duplicates_discarded = 0;
  _invalid_valence_rejected = 0;
  _already_produced_discarded = 0;
  _separator = ',';
}

SmuResults::~SmuResults() {}

size_t SmuResults::size() const { return _smiles_to_id.size(); }

void SmuResults::Report(std::ostream& output, bool single_line) const {
  const char line_sep = single_line ? ' ' : '\n';
  output << _molecules_examined << " examined" << line_sep;
  output << _duplicates_discarded << " duplicates" << line_sep;
  output << size() << " molecules" << line_sep;
  output << _invalid_valence_rejected << " invalid valence\n";
}

bool SmuResults::Add(std::unique_ptr<Molecule>& mol,
                     IW_STL_Hash_Set& already_produced,
                     bool non_aromatic_unique_smiles) {
  _molecules_examined++;
  if (_report_progress()) {
    Report(std::cerr, true);
  }

  if (!mol->valence_ok()) {
    std::cerr << " invalid valence " << mol->smiles() << "\n";
    _invalid_valence_rejected++;
    return 0;
  }

  const IWString& usmi = non_aromatic_unique_smiles
                             ? mol->non_aromatic_unique_smiles()
                             : mol->unique_smiles();

  if (already_produced.contains(usmi)) {
    _already_produced_discarded++;
    return false;
  }
  already_produced.emplace(usmi);

  if (_smiles_to_id.contains(usmi)) {
    _duplicates_discarded++;
    return false;
  }
  size_t id = _smiles_to_id.size() + 1;
  _smiles_to_id.emplace(usmi, id);

  return true;
}

// If the `m` passes quality tests, write to `output`.
// Returns true if written.
template <typename ID>
bool SmuResults::_maybe_write_molecule(Molecule& m, const IWString& smiles,
                                       const ID& id, const SMU_Params& params,
                                       std::ostream& output) const {
  if (!m.valence_ok()) {
    return false;
  }

  if (!OkCharges(m, params)) {
    return false;
  }

  output << smiles << _separator << id << '\n';
  return true;
}

int SmuResults::Write(const SMU_Params& params, std::ostream& output) const {
  int written = 0;
  for (const auto& [smiles, id] : _smiles_to_id) {
    Molecule m;
    if (!m.build_from_smiles(smiles)) {
      cerr << "Cannot build from " << smiles << "\n";
      continue;
    }
    if (_maybe_write_molecule(m, smiles, id, params, output)) {
      written++;
    }
  }

  return written;
}

int SmuResults::Write(int& next_id, const SMU_Params& params,
                      std::ostream& output) const {
  int written = 0;
  for (const auto& [smiles, id_not_used] : _smiles_to_id) {
    Molecule m;
    if (!m.build_from_smiles(smiles)) {
      cerr << "Cannot build from " << smiles << "\n";
      continue;
    }
    if (_maybe_write_molecule(m, smiles, next_id, params, output)) {
      next_id++;
      written++;
    }
  }

  std::cerr << "SmuResults::Write:scanned " << _smiles_to_id.size() << " wrote "
            << written << "\n";

  return written;
}

std::vector<IWString> SmuResults::CurrentSmiles(int natoms) const {
  std::vector<IWString> result;  // to be returned.
  result.reserve(_smiles_to_id.size());
  for (auto& [smiles, id_not_used] : _smiles_to_id) {
    if (smiles.length() < natoms) {
      continue;
    }
    if (count_atoms_in_smiles(smiles) != natoms) {
      continue;
    }
    result.emplace_back(smiles);
  }

  return result;
}

}  // namespace smu
