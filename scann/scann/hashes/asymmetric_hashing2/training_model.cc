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

#include "scann/hashes/asymmetric_hashing2/training_model.h"

#include <algorithm>
#include <cstddef>
#include <utility>

#include "absl/strings/str_cat.h"
#include "hwy/highway.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/projection/chunking_projection.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing2 {

using QuantizationScheme = AsymmetricHasherConfig::QuantizationScheme;

template <typename T>
StatusOrPtr<Model<T>> Model<T>::FromCenters(
    vector<DenseDataset<FloatT>> centers,
    QuantizationScheme quantization_scheme) {
  if (centers.empty()) {
    return InvalidArgumentError("Cannot construct a Model from empty centers.");
  } else if (centers[0].empty() || centers[0].size() > 256) {
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
    const CentersForAllSubspaces& proto,
    std::optional<ProjectionConfig> projection_config) {
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
      SCANN_RETURN_IF_ERROR(all_centers[i].Append(temp.ToPtr(), ""));
    }

    all_centers[i].ShrinkToFit();
  }

  SCANN_ASSIGN_OR_RETURN(
      unique_ptr<Model<T>> result,
      FromCenters(std::move(all_centers), proto.quantization_scheme()));
  if (projection_config.has_value()) {
    SCANN_ASSIGN_OR_RETURN(
        auto projection,
        ChunkingProjectionFactory<T>(*projection_config,
                                     &proto.serialized_projection()));
    result->SetProjection(std::move(projection));
  }
  return result;
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
  if (projection_) {
    std::optional<SerializedProjection> serialized_projection =
        projection_->SerializeToProto();
    if (serialized_projection.has_value()) {
      *result.mutable_serialized_projection() =
          std::move(*serialized_projection);
    }
  }

  return result;
}

template <typename T>
StatusOr<shared_ptr<const ChunkingProjection<T>>> Model<T>::GetProjection(
    const ProjectionConfig& projection_config) const {
  if (projection_ == nullptr) {
    SCANN_ASSIGN_OR_RETURN(auto unique_projection,
                           ChunkingProjectionFactory<T>(projection_config));
    return {std::move(unique_projection)};
  }
  return projection_;
}

template <typename T>
void Model<T>::SetProjection(
    shared_ptr<const ChunkingProjection<T>> projection) {
  projection_ = std::move(projection);
}

template <typename T>
Model<T>::Model(vector<DenseDataset<FloatT>> centers,
                QuantizationScheme quantization_scheme)
    : centers_(std::move(centers)),
      num_clusters_per_block_(centers_[0].size()),
      quantization_scheme_(quantization_scheme) {
  if constexpr (!std::is_same_v<FloatT, float>) return;
  if (centers_.empty() || centers_[0].size() <= 16) {
    return;
  }
  const DatapointIndex centers_per_ah_block = centers_[0].size();
  const DatapointIndex total_centers = centers_.size() * centers_per_ah_block;
  const DatapointIndex simd_block_size =
      hwy::HWY_NAMESPACE::Lanes(hwy::HWY_NAMESPACE::ScalableTag<float>());
  if (centers_per_ah_block % simd_block_size != 0) return;
  VLOG(1) << "AH Model:  Transposing " << total_centers << " centers in "
          << simd_block_size << "-element blocks.";
  const DatapointIndex num_simd_blocks_per_ah_block =
      centers_per_ah_block / simd_block_size;
  size_t total_elements = 0;
  for (auto& ah_block_centers : centers_) {
    DCHECK_EQ(ah_block_centers.size(), centers_per_ah_block);
    total_elements += ah_block_centers.dimensionality() * centers_per_ah_block;
  }

  vector<FloatT> block_transposed_centers(total_elements);
  size_t dst_offset = 0;
  for (const DenseDataset<FloatT>& ah_block_centers : centers_) {
    const DimensionIndex dimensionality = ah_block_centers.dimensionality();

    auto get_val = [&](DatapointIndex dp_idx, DimensionIndex dim_idx) {
      DCHECK_LT(dp_idx, ah_block_centers.size());
      return ah_block_centers[dp_idx].values()[dim_idx];
    };

    for (DatapointIndex simd_block_idx : Seq(num_simd_blocks_per_ah_block)) {
      const DatapointIndex block_start = simd_block_idx * simd_block_size;
      const DatapointIndex block_end = block_start + simd_block_size;
      for (size_t dim_idx : Seq(dimensionality)) {
        for (DatapointIndex dp_idx : Seq(block_start, block_end)) {
          DCHECK_LT(dst_offset, block_transposed_centers.size());
          block_transposed_centers[dst_offset++] = get_val(dp_idx, dim_idx);
        }
      }
    }
  }
  DCHECK_EQ(dst_offset, block_transposed_centers.size());
  block_transposed_centers_ = std::move(block_transposed_centers);
}

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
}  // namespace research_scann
