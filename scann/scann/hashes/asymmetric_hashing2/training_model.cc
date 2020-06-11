// Copyright 2020 The Google Research Authors.
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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/hashes/asymmetric_hashing2/training_model.h"

#include "scann/data_format/datapoint.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

using QuantizationScheme = AsymmetricHasherConfig::QuantizationScheme;

template <typename T>
StatusOrPtr<Model<T>> Model<T>::FromCenters(
    vector<DenseDataset<FloatT>> centers,
    QuantizationScheme quantization_scheme) {
  if (centers.empty()) {
    return InvalidArgumentError("Cannot construct a Model from empty centers.");
  } else if (centers[0].size() < 1 || centers[0].size() > 256) {
    return InvalidArgumentError(absl::StrCat(
        "Each asymmetric hashing block must contain between 1 and 256 centers, "
        "not ",
        centers[0].size(), "."));
  }

  for (size_t i = 1; i < centers.size(); ++i) {
    if (centers[i].size() != centers[0].size()) {
      return InvalidArgumentError(absl::StrCat(
          "All asymmetric hashing blocks must have the same number of centers."
          "  (",
          centers[0].size(), " vs. ", centers[i].size(), "."));
    }
  }

  return unique_ptr<Model<T>>(
      new Model<T>(std::move(centers), quantization_scheme));
}

template <typename T>
StatusOr<unique_ptr<Model<T>>> Model<T>::FromProto(
    const CentersForAllSubspaces& proto) {
  const size_t num_blocks = proto.subspace_centers_size();
  if (num_blocks == 0) {
    return InvalidArgumentError(
        "Cannot build a Model from a serialized CentersForAllSubspaces with "
        "zero blocks.");
  }

  vector<DenseDataset<FloatT>> all_centers(num_blocks);
  Datapoint<FloatT> temp;
  for (size_t i = 0; i < num_blocks; ++i) {
    const size_t num_centers = proto.subspace_centers(i).center_size();
    for (size_t j = 0; j < num_centers; ++j) {
      temp.clear();
      SCANN_RETURN_IF_ERROR(temp.FromGfv(proto.subspace_centers(i).center(j)));
      all_centers[i].AppendOrDie(temp.ToPtr(), "");
    }

    all_centers[i].ShrinkToFit();
  }

  return FromCenters(std::move(all_centers), proto.quantization_scheme());
}

template <typename T>
CentersForAllSubspaces Model<T>::ToProto() const {
  CentersForAllSubspaces result;
  for (size_t i = 0; i < centers_.size(); ++i) {
    auto centers_serialized = result.add_subspace_centers();
    for (size_t j = 0; j < centers_[i].size(); ++j) {
      Datapoint<double> dp;
      centers_[i].GetDatapoint(j, &dp);
      *centers_serialized->add_center() = dp.ToGfv();
    }
  }

  result.set_quantization_scheme(quantization_scheme_);

  return result;
}

template <typename T>
Model<T>::Model(vector<DenseDataset<FloatT>> centers,
                QuantizationScheme quantization_scheme)
    : centers_(std::move(centers)),
      num_clusters_per_block_(centers_[0].size()),
      quantization_scheme_(quantization_scheme) {}

template <typename T>
bool Model<T>::CentersEqual(const Model& rhs) const {
  if (centers_.size() != rhs.centers_.size()) return false;
  for (size_t i : IndicesOf(centers_)) {
    if (centers_[i].dimensionality() != rhs.centers_[i].dimensionality() ||
        centers_[i].size() != rhs.centers_[i].size()) {
      return false;
    }
    auto this_span = centers_[i].data();
    auto rhs_span = rhs.centers_[i].data();
    if (!std::equal(this_span.begin(), this_span.end(), rhs_span.begin())) {
      return false;
    }
  }
  return true;
}

SCANN_INSTANTIATE_TYPED_CLASS(, Model);

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow
