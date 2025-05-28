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

#include "molecule_graph_proto/molecule_graph.pb.h"
#include "molecule_graph_parsing_ops/cc/molecule_graph_featurizer.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using tensorflow::tstring;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

class MoleculeGraphParserOp : public OpKernel {
 public:
  explicit MoleculeGraphParserOp(OpKernelConstruction* context)
      : OpKernel(context), featurizer_(def()) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& example_ids_in_tensor = context->input(0);
    OP_REQUIRES(context, example_ids_in_tensor.dims() == 1,
                errors::InvalidArgument("example_ids_in must be a vector"));
    const int32 batch_size_example_ids_in = example_ids_in_tensor.dim_size(0);

    const Tensor& in_molecule_tensor = context->input(1);
    OP_REQUIRES(context, in_molecule_tensor.dims() == 1,
                errors::InvalidArgument("molecule graphs must be a vector"));
    const int32 batch_size = in_molecule_tensor.dim_size(0);

    OP_REQUIRES(context, batch_size_example_ids_in == batch_size,
                errors::InvalidArgument("Input tensor size mismatch"));

    // Copies example_ids_in directly to example_ids
    context->set_output(0, example_ids_in_tensor);

    // Allocates tensors for other outputs
    Tensor* atom_features_;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       1, TensorShape({batch_size, featurizer_.GetMaxAtoms(),
                                       featurizer_.GetNumAtomFeatures()}),
                       &atom_features_));
    Tensor* pair_features_;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       2, TensorShape({batch_size, featurizer_.GetMaxAtoms(),
                                       featurizer_.GetMaxAtoms(),
                                       featurizer_.GetNumPairFeatures()}),
                       &pair_features_));
    Tensor* atom_mask_;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       3, TensorShape({batch_size, featurizer_.GetMaxAtoms()}),
                       &atom_mask_));
    Tensor* pair_mask_;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       4, TensorShape({batch_size, featurizer_.GetMaxAtoms(),
                                       featurizer_.GetMaxAtoms()}),
                       &pair_mask_));

    auto atom_features =
        atom_features_->shaped<float, 3>({batch_size, featurizer_.GetMaxAtoms(),
                                          featurizer_.GetNumAtomFeatures()});
    auto pair_features = pair_features_->shaped<float, 4>(
        {batch_size, featurizer_.GetMaxAtoms(), featurizer_.GetMaxAtoms(),
         featurizer_.GetNumPairFeatures()});
    auto atom_mask =
        atom_mask_->shaped<float, 2>({batch_size, featurizer_.GetMaxAtoms()});
    auto pair_mask = pair_mask_->shaped<float, 3>(
        {batch_size, featurizer_.GetMaxAtoms(), featurizer_.GetMaxAtoms()});

    atom_features.setZero();
    pair_features.setZero();
    atom_mask.setZero();
    pair_mask.setZero();

    // Gets atom and pair features for each molecule in a batch
    int32 max_pair_distance = featurizer_.GetMaxPairDistance();

    auto in_molecule = in_molecule_tensor.flat<tstring>();
    auto example_ids = example_ids_in_tensor.flat<tstring>();
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      research_gigamol::MoleculeGraph mol;

      // Sets mask to all 1s for empty string examples if they're allowed.
      // If not allowed, they'll trigger the "has no atoms" exception below.
      OP_REQUIRES(context, mol.ParseFromString(in_molecule(batch_idx)),
                  errors::InvalidArgument("Failed to parse MoleculeGraph for ",
                                          example_ids(batch_idx)));

      OP_REQUIRES(context, mol.atoms_size() > 0,
                  errors::InvalidArgument("Molecule ", example_ids(batch_idx),
                                          " has no atoms."));
      OP_REQUIRES(
          context, featurizer_.GetAllowOverflow() ||
                       mol.atoms_size() <= featurizer_.GetMaxAtoms(),
          errors::InvalidArgument("Molecule ", example_ids(batch_idx), " has ",
                                  mol.atoms_size(), " atoms, which is ",
                                  "more than the max allowed (",
                                  featurizer_.GetMaxAtoms(), ")"));
      if (mol.atoms_size() > featurizer_.GetMaxAtoms()) {
        LOG(WARNING) << "Molecule " << example_ids(batch_idx) << " has "
                     << mol.atoms_size() << " atoms, "
                     << "but only the first " << featurizer_.GetMaxAtoms()
                     << " atoms will be used";
      }
      // atom features
      for (int atom_idx = 0; atom_idx < mol.atoms_size(); ++atom_idx) {
        if (atom_idx >= featurizer_.GetMaxAtoms()) break;
        auto atom = mol.atoms(atom_idx);
        auto features = featurizer_.GetAtomFeatures(atom);
        for (Eigen::Index feat_idx = 0; feat_idx < features.size();
             ++feat_idx) {
          int prop = std::fpclassify(features[feat_idx]);
          if (prop == FP_NAN || prop == FP_INFINITE) {
            LOG(WARNING) << "Setting atom feature " << feat_idx
                         << " for molecule " << example_ids(batch_idx)
                         << " to 0 to avoid NaN/Inf";
            features[feat_idx] = 0.0;
          }
          atom_features(batch_idx, atom_idx, feat_idx) = features[feat_idx];
        }
        atom_mask(batch_idx, atom_idx) = true;
      }
      // pair features
      for (int i = 0; i < mol.atom_pairs_size(); ++i) {
        auto atom_pair = mol.atom_pairs(i);
        if (max_pair_distance >= 0 &&
            atom_pair.graph_distance() > max_pair_distance)
          continue;
        int a_idx = atom_pair.a_idx();
        int b_idx = atom_pair.b_idx();
        if (a_idx >= featurizer_.GetMaxAtoms() ||
            b_idx >= featurizer_.GetMaxAtoms())
          continue;
        auto features = featurizer_.GetPairFeatures(atom_pair);
        for (Eigen::Index feat_idx = 0; feat_idx < features.size();
             ++feat_idx) {
          int prop = std::fpclassify(features[feat_idx]);
          if (prop == FP_NAN || prop == FP_INFINITE) {
            LOG(WARNING) << "Setting pair feature " << feat_idx
                         << " for molecule " << example_ids(batch_idx)
                         << " to 0 to avoid NaN/Inf";
            features[feat_idx] = 0.0;
          }
          pair_features(batch_idx, a_idx, b_idx, feat_idx) = features[feat_idx];
          pair_features(batch_idx, b_idx, a_idx, feat_idx) = features[feat_idx];
        }
        pair_mask(batch_idx, a_idx, b_idx) = true;
        pair_mask(batch_idx, b_idx, a_idx) = true;
      }
    }
  }

 private:
  graph_conv::MoleculeGraphFeaturizer featurizer_;
};

REGISTER_OP("MoleculeGraphParser")
    .Input("example_ids_in: string")
    .Input("molecules: string")
    .Attr("max_atoms: int")
    .Attr("max_pair_distance: int = -1")
    .Attr("allow_overflow: bool = true")
    .Output("example_ids: string")
    .Output("atoms: float")
    .Output("pairs: float")
    .Output("atom_mask: float")
    .Output("pair_mask: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle example_ids_in;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &example_ids_in));
      DimensionHandle batch_size = c->Dim(example_ids_in, 0);

      graph_conv::MoleculeGraphFeaturizer featurizer(c);
      const auto max_atoms = featurizer.GetMaxAtoms();
      const auto num_atom_features = featurizer.GetNumAtomFeatures();
      const auto num_pair_features = featurizer.GetNumPairFeatures();

      c->set_output(0, c->Vector(batch_size));  // example_ids

      // atoms
      c->set_output(1,
                    c->MakeShape({batch_size, max_atoms, num_atom_features}));

      // pairs
      c->set_output(2, c->MakeShape({batch_size, max_atoms, max_atoms,
                                     num_pair_features}));
      c->set_output(3, c->Matrix(batch_size, max_atoms));  // atom_mask

      // pair_mask
      c->set_output(4, c->MakeShape({batch_size, max_atoms, max_atoms}));
      return absl::Status();
    })
    .Doc(R"doc(
Unpack atom and atom pair features from MoleculeGraph protos.

example_ids_in: A vector of length batch_size of the example id for each
  corresponding MoleculeGraph proto.
molecules: A vector of length batch_size of MoleculeGraph protos.
max_atoms: Maximum number of atoms in a molecule.
max_pair_distance: Maximum distance between atoms in pairs.
allow_overflow: If true, allow molecules with more than max_atoms atoms. Only
  the first max_atoms atoms will be used.
example_ids: A vector of length batch_size of the example id for each
  corresponding MoleculeGraph proto.
atoms: A tensor of shape batch_size x max_atoms x num_atom_features.
pairs: A tensor of shape batch_size x max_atoms x max_atoms x num_pair_features.
atom_mask: A tensor of shape batch_size x max_atoms indicating valid atoms. It
  is of type float to avoid casting from bool.
pair_mask: A tensor of shape batch_size x max_atoms x max_atoms indicating
  valid pairs. It is of type float to avoid casting from bool.
)doc");

REGISTER_KERNEL_BUILDER(Name("MoleculeGraphParser").Device(DEVICE_CPU),
                        MoleculeGraphParserOp);
}  // end namespace tensorflow
