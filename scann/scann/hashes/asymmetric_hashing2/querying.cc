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

#include "scann/hashes/asymmetric_hashing2/querying.h"

#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"

using std::shared_ptr;

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing2 {

PackedDataset CreatePackedDataset(
    const DenseDataset<uint8_t>& hashed_database) {
  PackedDataset result;
  result.bit_packed_data =
      asymmetric_hashing_internal::CreatePackedDataset(hashed_database);
  result.num_datapoints = hashed_database.size();
  result.num_blocks =
      (hashed_database.size() > 0) ? (hashed_database[0].nonzero_entries()) : 0;
  return result;
}

DenseDataset<uint8_t> UnpackDataset(const PackedDataset& packed) {
  const int num_dim = packed.num_blocks, num_dp = packed.num_datapoints;

  vector<uint8_t> unpacked(num_dim * num_dp);

  int idx = 0;
  for (int dp_block = 0; dp_block < num_dp / 32; dp_block++) {
    const int out_idx = 32 * dp_block;
    for (int dim = 0; dim < num_dim; dim++) {
      for (int offset = 0; offset < 16; offset++) {
        uint8_t data = packed.bit_packed_data[idx++];
        unpacked[(out_idx | offset) * num_dim + dim] = data & 15;
        unpacked[(out_idx | 16 | offset) * num_dim + dim] = data >> 4;
      }
    }
  }

  if (num_dp % 32 != 0) {
    const int out_idx = num_dp - (num_dp % 32);
    for (int dim = 0; dim < num_dim; dim++) {
      for (int offset = 0; offset < 16; offset++) {
        uint8_t data = packed.bit_packed_data[idx++];
        int idx1 = out_idx | offset, idx2 = out_idx | 16 | offset;
        if (idx1 < num_dp) unpacked[idx1 * num_dim + dim] = data & 15;
        if (idx2 < num_dp) unpacked[idx2 * num_dim + dim] = data >> 4;
      }
    }
  }
  return DenseDataset<uint8_t>(
      unpacked, make_unique<VariableLengthDocidCollection>(
                    VariableLengthDocidCollection::CreateWithEmptyDocids(
                        packed.num_datapoints)));
}

template <typename T>
AsymmetricQueryer<T>::AsymmetricQueryer(
    shared_ptr<const ChunkingProjection<T>> projector,
    shared_ptr<const DistanceMeasure> lookup_distance,
    shared_ptr<const Model<T>> model)
    : projector_(std::move(projector)),
      lookup_distance_(std::move(lookup_distance)),
      model_(std::move(model)) {}

template <typename T>
StatusOr<LookupTable> AsymmetricQueryer<T>::CreateLookupTable(
    const DatapointPtr<T>& query,
    AsymmetricHasherConfig::LookupType lookup_type,
    AsymmetricHasherConfig::FixedPointLUTConversionOptions
        float_int_conversion_options) const {
  switch (lookup_type) {
    case AsymmetricHasherConfig::FLOAT:
      return CreateLookupTable<float>(query, float_int_conversion_options);
    case AsymmetricHasherConfig::INT8:
    case AsymmetricHasherConfig::INT8_LUT16:
      return CreateLookupTable<int8_t>(query, float_int_conversion_options);
    case AsymmetricHasherConfig::INT16:
      return CreateLookupTable<int16_t>(query, float_int_conversion_options);
    default:
      return InvalidArgumentError("Unrecognized lookup type.");
  }
}

template <typename T>
SymmetricQueryer<T>::SymmetricQueryer(const DistanceMeasure& lookup_distance,
                                      const Model<T>& model, Option option)
    : num_clusters_per_block_(model.num_clusters_per_block()),
      num_blocks_(model.centers().size()),
      option_(option) {
  CHECK_NE(lookup_distance.specially_optimized_distance_tag(),
           DistanceMeasure::LIMITED_INNER_PRODUCT)
      << "Limited inner product distance not supported when using symmetric "
         "querying.";
  auto centers = model.centers();
  global_lookup_table_.reserve(std::pow(num_clusters_per_block_, 2) *
                               num_blocks_);

  for (size_t i = 0; i < num_blocks_; ++i) {
    const auto& cur_chunk_centers = centers[i];
    for (size_t j = 0; j < num_clusters_per_block_; ++j) {
      auto datapoint_j = cur_chunk_centers[j];
      for (size_t k = 0; k < num_clusters_per_block_; ++k) {
        global_lookup_table_.push_back(
            lookup_distance.GetDistance(datapoint_j, cur_chunk_centers[k]));
      }
    }
  }
}

template <typename T>
float SymmetricQueryer<T>::ComputeSingleApproximateDistance(
    const DatapointPtr<uint8_t>& hashed_dp1,
    const DatapointPtr<uint8_t>& hashed_dp2) const {
  const uint32_t num_clusters_sq = std::pow(num_clusters_per_block_, 2);
  double sum = 0.0;
  const float* matrix_ptr = global_lookup_table_.data();
  const auto* dp1 = hashed_dp1.values();
  const auto* dp2 = hashed_dp2.values();
  if (option_.uses_nibble_packing) {
    size_t i_end = hashed_dp1.nonzero_entries();

    if (num_blocks_ & 1) {
      --i_end;
      sum += matrix_ptr[(num_blocks_ - 1) * num_clusters_sq +
                        num_clusters_per_block_ * dp1[i_end] + dp2[i_end]];
    }
    for (size_t i = 0; i < i_end; ++i) {
      const uint8_t dp1_lo = dp1[i] & 0x0f;
      const uint8_t dp2_lo = dp2[i] & 0x0f;
      const uint8_t dp1_hi = dp1[i] >> 4;
      const uint8_t dp2_hi = dp2[i] >> 4;
      sum += matrix_ptr[num_clusters_per_block_ * dp1_lo + dp2_lo];
      sum += matrix_ptr[num_clusters_sq + num_clusters_per_block_ * dp1_hi +
                        dp2_hi];
      matrix_ptr += 2 * num_clusters_sq;
    }
  } else {
    for (size_t i = 0; i < hashed_dp1.nonzero_entries(); ++i) {
      sum += matrix_ptr[num_clusters_per_block_ * dp1[i] + dp2[i]];
      matrix_ptr += num_clusters_sq;
    }
  }

  return sum;
}

SCANN_INSTANTIATE_TYPED_CLASS(, AsymmetricQueryer);
SCANN_INSTANTIATE_TYPED_CLASS(, SymmetricQueryer);

}  // namespace asymmetric_hashing2
}  // namespace scann_ops
}  // namespace tensorflow
