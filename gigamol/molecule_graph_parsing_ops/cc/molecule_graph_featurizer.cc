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

#include "molecule_graph_parsing_ops/cc/molecule_graph_featurizer.h"
#include <map>
#include <vector>
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace graph_conv {

namespace {

// Adapter to expose NodeDefContext::GetAttr given a NodeDef, with the same
// signature as tensorflow::shape_inference::InferenceContext.
class NodeDefContext {
 public:
  explicit NodeDefContext(const tensorflow::NodeDef& node_def)
      : node_def_(node_def) {}

  template <class T>
  absl::Status GetAttr(absl::string_view attr_name, T* value) const {
    return tensorflow::GetNodeAttr(node_def_, attr_name, value);
  }

 private:
  const tensorflow::NodeDef& node_def_;
};

}  // namespace

// Constructs a MoleculeGraphFeaturizer that computes MoleculeGraph pair and
// atom features. Requires the following parameters to be set in the NodeDef:
//
// * max_atoms: The maximum number of atoms in a molecule.
// * allow_overflow: If true, allow molecules with more than max_atoms atoms.
//       Only the first max_atoms atoms will be used.
// * max_pair_distance: The maximum graph distance between a pair of atoms for
//       the pair to be featurized.
MoleculeGraphFeaturizer::MoleculeGraphFeaturizer(
    const tensorflow::NodeDef& node_def)
    : MoleculeGraphFeaturizer(NodeDefContext(node_def)) {}
MoleculeGraphFeaturizer::MoleculeGraphFeaturizer(
    tensorflow::shape_inference::InferenceContext* c)
    : MoleculeGraphFeaturizer(*c) {}
template <typename ContextT>
MoleculeGraphFeaturizer::MoleculeGraphFeaturizer(const ContextT& context) {
  TF_CHECK_OK(context.GetAttr("max_atoms", &max_atoms_));
  TF_CHECK_OK(context.GetAttr("allow_overflow", &allow_overflow_));
  TF_CHECK_OK(context.GetAttr("max_pair_distance", &max_pair_distance_));

  // atom features
  // AtomType
  // enum values not in this map will be ignored
  // Uses ATOM_NONE to indicate empty positions in the atom features tensor.
  atom_type_map_ = {
      {research_gigamol::MoleculeGraph_Atom::ATOM_H, 0},
      {research_gigamol::MoleculeGraph_Atom::ATOM_C, 1},
      {research_gigamol::MoleculeGraph_Atom::ATOM_N, 2},
      {research_gigamol::MoleculeGraph_Atom::ATOM_O, 3},
      {research_gigamol::MoleculeGraph_Atom::ATOM_F, 4},
      {research_gigamol::MoleculeGraph_Atom::ATOM_P, 5},
      {research_gigamol::MoleculeGraph_Atom::ATOM_S, 6},
      {research_gigamol::MoleculeGraph_Atom::ATOM_CL, 7},
      {research_gigamol::MoleculeGraph_Atom::ATOM_BR, 8},
      {research_gigamol::MoleculeGraph_Atom::ATOM_I, 9},
      {research_gigamol::MoleculeGraph_Atom::ATOM_METAL, 10},
      // Optionally: {research_gigamol::MoleculeGraph_Atom::ATOM_NONE, 11},
  };

  // pair features
  // BondType
  // enum values not in this map will be ignored
  // Uses BOND_NONE to indicate pairs that are not bonded.
  bond_type_map_ = {
    {research_gigamol::MoleculeGraph_AtomPair::BOND_SINGLE, 0},
    {research_gigamol::MoleculeGraph_AtomPair::BOND_DOUBLE, 1},
    {research_gigamol::MoleculeGraph_AtomPair::BOND_TRIPLE, 2},
    {research_gigamol::MoleculeGraph_AtomPair::BOND_AROMATIC, 3},
    // Optionally: {research_gigamol::MoleculeGraph_AtomPair::BOND_NONE, 4},
  };

  // graph_distance bins
  // Inserts zero to indicate distance for a pair with any ATOM_NONE atoms.
  graph_distance_bins_ = {1, 2, 3, 4, 5, 6, 7};
}

std::vector<float> MoleculeGraphFeaturizer::GetDefaultAtomFeatures() {
  std::vector<float> features(GetNumAtomFeatures(), 0.0);
  features[atom_type_map_.at(research_gigamol::MoleculeGraph_Atom::ATOM_NONE)] =
      1.0;
  return features;
}

std::vector<float> MoleculeGraphFeaturizer::GetAtomFeatures(
    const research_gigamol::MoleculeGraph::Atom& atom) {
  std::vector<float> features(GetNumAtomFeatures(), 0.0);
  int i = 0;

  // atom type (one-hot encoding of AtomType enum)
  CHECK(research_gigamol::MoleculeGraph_Atom::AtomType_IsValid(atom.type()));
  if (atom_type_map_.count(atom.type()) > 0) {
    features[i + atom_type_map_.at(atom.type())] = 1.0;
  }
  i += atom_type_map_.size();

  // chirality (one-hot encoding of ChiralType enum)
  CHECK(research_gigamol::MoleculeGraph_Atom::ChiralType_IsValid(
      atom.chirality()));
  if (chiral_type_map_.count(atom.chirality()) > 0)
    features[i + chiral_type_map_.at(atom.chirality())] = 1.0;
  i += chiral_type_map_.size();

  // formal charge
  features[i] = static_cast<float>(atom.formal_charge());
  ++i;

  // partial charge
  features[i] = atom.partial_charge();
  ++i;

  // ring size counts
  // out-of-bounds ring sizes are ignored
  for (int ring_size : atom.ring_sizes()) {
    if (ring_size >= ring_min_size_ && ring_size <= ring_max_size_) {
      ++features[i + ring_size - ring_min_size_];
    }
  }
  i += ring_size_;

  // hybridization
  CHECK(research_gigamol::MoleculeGraph_Atom::HybridizationType_IsValid(
      atom.hybridization()));
  if (hybridization_type_map_.count(atom.hybridization()) > 0)
    features[i + hybridization_type_map_.at(atom.hybridization())] = 1.0;
  i += hybridization_type_map_.size();

  // hydrogen bonding
  if (atom.acceptor()) {
    features[i] = 1.0;
  }
  ++i;
  if (atom.donor()) {
    features[i] = 1.0;
  }
  ++i;

  // aromaticity
  if (atom.aromatic()) {
    features[i] = 1.0;
  }
  ++i;

  CHECK_EQ(i, GetNumAtomFeatures());
  return features;
}

std::vector<float> MoleculeGraphFeaturizer::GetDefaultPairFeatures() {
  std::vector<float> features(GetNumPairFeatures(), 0.0);
  features[bond_type_map_.at(
      research_gigamol::MoleculeGraph_AtomPair::BOND_NONE)] = 1.0;
  features[bond_type_map_.size()] = 1.0;  // Zero graph distance.
  return features;
}

std::vector<float> MoleculeGraphFeaturizer::GetPairFeatures(
    const research_gigamol::MoleculeGraph::AtomPair& atom_pair) {
  std::vector<float> features(GetNumPairFeatures(), 0.0);
  int i = 0;

  // bond type (one-hot encoding of BondType enum)
  CHECK(research_gigamol::MoleculeGraph_AtomPair::BondType_IsValid(
      atom_pair.bond_type()));
  if (bond_type_map_.count(atom_pair.bond_type()) > 0)
    features[i + bond_type_map_.at(atom_pair.bond_type())] = 1.0;
  i += bond_type_map_.size();

  // graph distance
  // represented by a binning scheme that sets bits corresponding to
  // thresholds; e.g. if the bins are [1, <=2, <=3, <=5, <=8] and the
  // distance is 4, then the resulting portion of the feature vector is
  // [0.0, 0.0, 0.0, 1.0, 1.0].
  for (size_t bin_idx = 0; bin_idx < graph_distance_bins_.size();
    ++bin_idx) {
    if (atom_pair.graph_distance() <= graph_distance_bins_[bin_idx]) {
      features[i + bin_idx] = 1.0;
    }
  }

  // NOTE: if you change the order of pair features, be sure to update the
  // start of the distance slice in GetGraphDistanceSliceStart() and
  // GetSpatialDistanceIndex()
  CHECK_EQ(GetGraphDistanceSliceStart(), i);
  i += graph_distance_bins_.size();

  // same ring
  if (atom_pair.same_ring()) {
    features[i] = 1.0;
  }
  ++i;

  CHECK_EQ(i, GetNumPairFeatures());
  return features;
}

int MoleculeGraphFeaturizer::GetNumAtomFeatures() {
  return atom_type_map_.size() + chiral_type_map_.size() +
      hybridization_type_map_.size() + ring_size_ +
      5;  // formal charge, partial charge, aromaticity, H donor/acceptor
}

int MoleculeGraphFeaturizer::GetNumAtomTypes() {
  return atom_type_map_.size();
}

int MoleculeGraphFeaturizer::GetNumBondTypes() {
  return bond_type_map_.size();
}

int MoleculeGraphFeaturizer::GetNumPairFeatures() {
  // + 1 from same_ring feature, a bond based notion.
  return bond_type_map_.size() + graph_distance_bins_.size() + 1;
}

int MoleculeGraphFeaturizer::GetMaxAtoms() { return max_atoms_; }

int MoleculeGraphFeaturizer::GetMaxPairDistance() { return max_pair_distance_; }

bool MoleculeGraphFeaturizer::GetAllowOverflow() { return allow_overflow_; }

int MoleculeGraphFeaturizer::GetGraphDistanceSliceStart() {
  return bond_type_map_.size();
}

int MoleculeGraphFeaturizer::GetGraphDistanceSliceSize() {
  CHECK_GT(graph_distance_bins_.size(), 0);
  return graph_distance_bins_.size();
}

// ChiralType
// enum values not in this map will be ignored
const std::map<int, int> MoleculeGraphFeaturizer::chiral_type_map_ = {
    {research_gigamol::MoleculeGraph_Atom::CHIRAL_R, 0},
    {research_gigamol::MoleculeGraph_Atom::CHIRAL_S, 1},
};

// maximum displacement from ring_size
const int MoleculeGraphFeaturizer::ring_min_size_ = 3;
const int MoleculeGraphFeaturizer::ring_max_size_ = 8;
const int MoleculeGraphFeaturizer::ring_size_ =
    ring_max_size_ - ring_min_size_ + 1;

// HybridizationType
// enum values not in this map will be ignored
const std::map<int, int> MoleculeGraphFeaturizer::hybridization_type_map_ = {
    {research_gigamol::MoleculeGraph_Atom::HYBRIDIZATION_SP, 0},
    {research_gigamol::MoleculeGraph_Atom::HYBRIDIZATION_SP2, 1},
    {research_gigamol::MoleculeGraph_Atom::HYBRIDIZATION_SP3, 2},
};

}  // end namespace graph_conv
