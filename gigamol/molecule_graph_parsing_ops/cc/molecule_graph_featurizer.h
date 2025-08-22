// Copyright 2025 The Google Research Authors.
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

#ifndef MOLECULE_GRAPH_FEATURIZER_H_
#define MOLECULE_GRAPH_FEATURIZER_H_

#include <map>
#include <vector>
#include "molecule_graph_proto/molecule_graph.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace graph_conv {

class MoleculeGraphFeaturizer {
 public:
  // Constructs a MoleculeGraphFeaturizer that computes MoleculeGraph pair and
  // atom features. Requires the following parameters to be set in the NodeDef:
  //
  // * max_atoms: The maximum number of atoms in a molecule.
  // * allow_overflow: If true, allow molecules with more than max_atoms atoms.
  //       Only the first max_atoms atoms will be used.
  // * max_pair_distance: The maximum graph distance between a pair of atoms for
  //       the pair to be featurized.
  explicit MoleculeGraphFeaturizer(const tensorflow::NodeDef& node_def);
  explicit MoleculeGraphFeaturizer(
      tensorflow::shape_inference::InferenceContext* c);

  // Gets the atom features for a MoleculeGraph::Atom.
  std::vector<float> GetDefaultAtomFeatures();
  std::vector<float> GetAtomFeatures(
      const research_gigamol::MoleculeGraph::Atom& atom);

  // Gets the atom pair features for a MoleculeGraph::AtomPair.
  std::vector<float> GetDefaultPairFeatures();
  std::vector<float> GetPairFeatures(
      const research_gigamol::MoleculeGraph::AtomPair& atom_pair);

  // The number of atom features.
  int GetNumAtomFeatures();

  // The number of atom types.
  int GetNumAtomTypes();

  // The number of bond types.
  int GetNumBondTypes();

  // The number of atom pair features.
  int GetNumPairFeatures();

  // The maximum number of atoms.
  int GetMaxAtoms();

  // The maximum graph distance for atom pair features.
  int GetMaxPairDistance();

  // Whether to silently ignore more than the first GetMaxAtoms() atoms.
  bool GetAllowOverflow();

  // The index into pair features of the first graph distance bin.
  int GetGraphDistanceSliceStart();

  // The size of the graph distances slice in the pair features.
  int GetGraphDistanceSliceSize();

 private:
  template <typename ContextT>
  explicit MoleculeGraphFeaturizer(const ContextT& c);

  // Maximum number of atoms
  int max_atoms_;

  // Maximum graph distance between pairs of atoms
  int max_pair_distance_;

  // If true, allows molecules with more than max_atoms atoms. Only the first
  // max_atoms atoms will be used.
  bool allow_overflow_;

  // atom features
  // AtomType
  // enum values not in this map will be ignored
  std::map<int, int> atom_type_map_;

  // pair features
  // BondType
  // enum values not in this map will be ignored
  std::map<int, int> bond_type_map_;

  // graph_distance bins
  std::vector<int> graph_distance_bins_;

  // ChiralType
  // enum values not in this map will be ignored
  static const std::map<int, int> chiral_type_map_;

  // maximum displacement from ring_size
  static const int ring_min_size_;
  static const int ring_max_size_;
  static const int ring_size_;

  // HybridizationType
  // enum values not in this map will be ignored
  static const std::map<int, int> hybridization_type_map_;
};  // end class MoleculeGraphFeaturizer

}  // end namespace graph_conv

#endif  // MOLECULE_GRAPH_FEATURIZER_H_
