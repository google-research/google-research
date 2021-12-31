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

// Generate small molecule universe (SMU) exhaustively.

#include <algorithm>

#include "geometry/smu.h"
#include "third_party/lilly_mol/Foundational/cmdline/cmdline.h"
#include "third_party/lilly_mol/Molecule_Lib/molecule.h"

namespace smu {

using std::cerr;
using std::cout;

void usage(int rc) {
  cerr << "Generates SMU molecules\n";
  cerr << " -B <smi>          csv smiles of starting atoms (default "
          "C,N,[NH4+],O,[O-],F)\n";
  cerr << " -M <number>       max number of atoms in generated molecules\n";
  cerr << " -x                use non aromatic unique smiles\n";
  cerr << " -O                allow O- to be added to any atom (not just N+)\n";
  cerr << " -f <sep>          output separator\n";
  cerr << " -R <n>            report progress every <n> molecules examined\n";
  exit(0);
}

int initialise_starting_molecules(const std::vector<IWString>& smiles,
                                  std::vector<Molecule>& starting_molecules) {
  starting_molecules.reserve(smiles.size());
  for (const IWString& smi : smiles) {
    starting_molecules.emplace_back(Molecule());
    if (!starting_molecules.back().build_from_smiles(smi)) {
      cerr << "Huh, cannot build smiles " << smi << "\n";
      return 0;
    }
    starting_molecules.back().unset_all_implicit_hydrogen_information(0);
  }

  return starting_molecules.size();
}

int generate_smu(int argc, char** argv) {
  Command_Line cl(argc, argv, "vA:E:M:xOf:B:R:");

  if (cl.unrecognised_options_encountered()) {
    usage(1);
  }

  const int verbose = cl.option_count('v');

  if (!process_elements(cl, verbose)) {
    cerr << "Cannot parse element specifications\n";
    usage(2);
  }

  SMU_Params params;
  params.to_add.add(new Atom(6));
  params.to_add.add(new Atom(7));
  params.to_add.add(new Atom(7));
  params.to_add.last_item()->set_formal_charge(1);
  params.to_add.add(new Atom(8));
  params.to_add.add(new Atom(8));
  params.to_add.last_item()->set_formal_charge(-1);
  params.to_add.add(new Atom(9));

  int max_atoms = 7;
  if (cl.option_present('M')) {
    if (!cl.value('M', max_atoms) || max_atoms < 2) {
      cerr << "Invalid max atoms (-M)\n";
      usage(1);
    }
  }

  params.max_atoms = max_atoms;

  params.oneg_only_added_to_npos = true;
  if (cl.option_present('O')) {
    params.oneg_only_added_to_npos = false;
    if (verbose) cerr << "O- can be added to any existing atom\n";
  }

  params.non_aromatic_unique_smiles = false;
  if (cl.option_present('x')) {
    params.non_aromatic_unique_smiles = true;
    if (verbose) cerr << "Will use non aromatic unique smiles\n";
  }

  // The single atom molecules that start a build.
  // Note that we do not enforce them being single atom molecules.

  std::vector<IWString> smiles;
  if (cl.option_present('B')) {
    const_IWSubstring b;
    for (int i = 0; cl.value('B', b, i); ++i) {
      int j = 0;
      IWString smi;
      while (b.nextword(smi, j, ',')) {
        smiles.emplace_back(std::move(smi));
      }
    }
  } else {
    smiles = {"C", "N", "[NH4+]", "O", "[O-]", "F"};
  }

  std::vector<Molecule> starting_molecules;
  if (!initialise_starting_molecules(smiles, starting_molecules)) {
    cerr << "Cannot initialise starting molecules\n";
    return 1;
  }

  const int number_starting_molecules = starting_molecules.size();

  // One result for each starting molecule. Results get combined later.
  std::vector<SmuResults> smu_results(number_starting_molecules);

  if (cl.option_present('f')) {
    IWString f = cl.option_value('f');
    char_name_to_char(f);  // Easy specification of tab, space...
    std::for_each(smu_results.begin(), smu_results.end(),
                  [&f](SmuResults& res) { res.set_output_separator(f[0]); });
  }

  if (cl.option_present('R')) {
    std::for_each(smu_results.begin(), smu_results.end(),
                  [&cl, verbose](SmuResults& res) {
                    res.InitialiseProgressReporting(cl, 'R', verbose);
                  });
  }

  // For the reporting we do here, spread across multiple lines.
  constexpr bool single_line = false;

  // Save time by keeping track of what has been produced by other starting
  // molecules.
  IW_STL_Hash_Set already_produced;

  for (int i = 0; i < number_starting_molecules; ++i) {
    if (verbose > 1) {
      cerr << "Begin expansion from " << starting_molecules[i].smiles() << "\n";
    }
    Expand(starting_molecules[i], params, already_produced, smu_results[i]);
    if (verbose > 1) {
      cerr << "Smiles " << starting_molecules[i].smiles() << " generated "
           << smu_results[i].size() << " molecules\n";
      smu_results[i].Report(std::cerr, single_line);
    }
  }

  int molecules_generated = 0;
  int written = 0;
  int next_id = 1;
  for (int i = 0; i < number_starting_molecules; ++i) {
    molecules_generated += smu_results[i].size();
    written += smu_results[i].Write(next_id, params, std::cout);
  }

  if (verbose)
    cerr << "Across " << number_starting_molecules
         << " starting molecules generated " << molecules_generated << " wrote "
         << written << "\n";

  return 0;
}

}  // namespace smu

int main(int argc, char** argv) { return smu::generate_smu(argc, argv); }
