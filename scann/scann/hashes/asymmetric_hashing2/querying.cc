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

#include "scann/hashes/asymmetric_hashing2/querying.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/hashes/internal/asymmetric_hashing_impl.h"
#include "scann/projection/chunking_projection.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

using std::shared_ptr;

namespace research_scann {
namespace asymmetric_hashing2 {

namespace {

enum LookupTableType : uint8_t {
  kNone = 0,
  kFloat = 1,
  kInt16 = 2,
  kInt8 = 3,
};

}

absl::StatusOr<std::vector<uint8_t>> LookupTable::ToBytes() const {
  std::vector<uint8_t> bytes;
  size_t extra;

  const int non_empty_lookup_tables = !float_lookup_table.empty() +
                                      !int16_lookup_table.empty() +
                                      !int8_lookup_table.empty();
  if (non_empty_lookup_tables != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "exactly one of float/int16/int8 lookup_table must be populated: ",
        non_empty_lookup_tables));
  }

  LookupTableType table_type = kNone;
  if (!float_lookup_table.empty()) {
    table_type = kFloat;
  } else if (!int16_lookup_table.empty()) {
    table_type = kInt16;
  } else if (!int8_lookup_table.empty()) {
    table_type = kInt8;
  }
  bytes.push_back(table_type);

  uint32_t table_size = 0;
  if (table_type == kFloat) {
    table_size = float_lookup_table.size();
  } else if (table_type == kInt16) {
    table_size = int16_lookup_table.size();
  } else if (table_type == kInt8) {
    table_size = int8_lookup_table.size();
  }
  extra = sizeof(table_size);
  bytes.resize(bytes.size() + extra);
  std::memcpy(bytes.data() + bytes.size() - extra, &table_size, extra);

  if (table_type == kFloat) {
    extra = float_lookup_table.size() * sizeof(float);
    bytes.resize(bytes.size() + extra);
    std::memcpy(bytes.data() + bytes.size() - extra, float_lookup_table.data(),
                extra);
  } else if (table_type == kInt16) {
    extra = int16_lookup_table.size() * sizeof(uint16_t);
    bytes.resize(bytes.size() + extra);
    std::memcpy(bytes.data() + bytes.size() - extra, int16_lookup_table.data(),
                extra);
  } else if (table_type == kInt8) {
    bytes.insert(bytes.end(), int8_lookup_table.begin(),
                 int8_lookup_table.end());
  }

  bool is_nan = std::isnan(fixed_point_multiplier);
  bytes.push_back(static_cast<uint8_t>(is_nan));

  if (!is_nan) {
    extra = sizeof(fixed_point_multiplier);
    bytes.resize(bytes.size() + extra);
    std::memcpy(bytes.data() + bytes.size() - extra, &fixed_point_multiplier,
                extra);
  }

  bytes.push_back(static_cast<uint8_t>(can_use_int16_accumulator));

  return bytes;
}

absl::StatusOr<LookupTable> LookupTable::FromBytes(
    absl::Span<const uint8_t> bytes) {
  LookupTable table;
  size_t offset = 0;
  size_t extra;

  LookupTableType table_type = static_cast<LookupTableType>(bytes[offset++]);
  if (!(table_type == kFloat || table_type == kInt16 || table_type == kInt8)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid table type encountered during deserialization: ", table_type));
  }

  uint32_t table_size;
  extra = sizeof(table_size);
  std::memcpy(&table_size, bytes.data() + offset, extra);
  offset += extra;
  if (table_size == 0) {
    return absl::InvalidArgumentError(
        "one of float/int16/int8 lookup_table must be populated");
  }

  if (table_type == kFloat) {
    table.float_lookup_table.resize(table_size);
    extra = table_size * sizeof(float);
    std::memcpy(table.float_lookup_table.data(), bytes.data() + offset, extra);
    offset += extra;
  } else if (table_type == kInt16) {
    table.int16_lookup_table.resize(table_size);
    extra = table_size * sizeof(uint16_t);
    std::memcpy(table.int16_lookup_table.data(), bytes.data() + offset, extra);
    offset += extra;
  } else if (table_type == kInt8) {
    table.int8_lookup_table.resize(table_size);
    std::memcpy(table.int8_lookup_table.data(), bytes.data() + offset,
                table_size);
    offset += table_size;
  }

  bool is_nan = static_cast<bool>(bytes[offset]);
  offset++;

  if (is_nan) {
    table.fixed_point_multiplier = std::numeric_limits<float>::quiet_NaN();
  } else {
    extra = sizeof(table.fixed_point_multiplier);
    std::memcpy(&table.fixed_point_multiplier, bytes.data() + offset, extra);
    offset += extra;
  }

  table.can_use_int16_accumulator = static_cast<bool>(bytes[offset]);

  return table;
}

PackedDataset CreatePackedDataset(
    const DenseDataset<uint8_t>& hashed_database) {
  PackedDataset result;
  result.bit_packed_data =
      asymmetric_hashing_internal::CreatePackedDataset(hashed_database);
  result.num_datapoints = hashed_database.size();
  result.num_blocks =
      (!hashed_database.empty()) ? (hashed_database[0].nonzero_entries()) : 0;
  return result;
}

DenseDataset<uint8_t> UnpackDataset(const PackedDatasetView& packed) {
  const size_t num_dim = packed.num_blocks, num_dp = packed.num_datapoints;

  vector<uint8_t> unpacked(num_dim * num_dp);

  int idx = 0;
  for (int dp_block = 0; dp_block < num_dp / kNumDatapointsPerBlock;
       dp_block++) {
    const int out_idx = kNumDatapointsPerBlock * dp_block;
    for (int dim = 0; dim < num_dim; dim++) {
      for (int offset = 0; offset < kPackedDatasetBlockSize; offset++) {
        uint8_t data = packed.bit_packed_data[idx++];
        unpacked[(out_idx | offset) * num_dim + dim] =
            data & (kPackedDatasetBlockSize - 1);
        unpacked[(out_idx | 16 | offset) * num_dim + dim] =
            data >> kPackedDataSetBlockSizeBits;
      }
    }
  }

  if (num_dp % kNumDatapointsPerBlock != 0) {
    const int out_idx = num_dp - (num_dp % kNumDatapointsPerBlock);
    for (int dim = 0; dim < num_dim; dim++) {
      for (int offset = 0; offset < kPackedDatasetBlockSize; offset++) {
        uint8_t data = packed.bit_packed_data[idx++];
        int idx1 = out_idx | offset,
            idx2 = out_idx | kPackedDatasetBlockSize | offset;
        if (idx1 < num_dp)
          unpacked[idx1 * num_dim + dim] = data & (kPackedDatasetBlockSize - 1);
        if (idx2 < num_dp)
          unpacked[idx2 * num_dim + dim] = data >> kPackedDataSetBlockSizeBits;
      }
    }
  }
  return DenseDataset<uint8_t>(std::move(unpacked), packed.num_datapoints);
}

PackedDatasetView CreatePackedDatasetView(const PackedDataset& packed_dataset) {
  PackedDatasetView result;
  result.bit_packed_data = absl::MakeConstSpan(packed_dataset.bit_packed_data);
  result.num_datapoints = packed_dataset.num_datapoints;
  result.num_blocks = packed_dataset.num_blocks;
  return result;
}

AsymmetricQueryerBase::AsymmetricQueryerBase(
    shared_ptr<const DistanceMeasure> lookup_distance)
    : lookup_distance_(std::move(lookup_distance)) {}

template <typename T>
AsymmetricQueryer<T>::AsymmetricQueryer(
    shared_ptr<const ChunkingProjection<T>> projector,
    shared_ptr<const DistanceMeasure> lookup_distance,
    shared_ptr<const Model<T>> model)
    : AsymmetricQueryerBase(std::move(lookup_distance)),
      projector_(std::move(projector)),
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

template <typename Dataset>
Status SetLUT16Hash(const DatapointPtr<uint8_t>& hashed, const size_t index,
                    Dataset* __restrict__ packed_struct) {
  MutableSpan<uint8_t> packed_dataset =
      MakeMutableSpan(packed_struct->bit_packed_data);

  const size_t hash_size = hashed.nonzero_entries();
  const size_t block_offset = index & 0x0f;

  const size_t offset = block_offset + (index & ~31) * hash_size / 2;
  SCANN_RET_CHECK_LE(offset + (hashed.nonzero_entries() - 1) * 16,
                     packed_dataset.size());
  SCANN_RET_CHECK_EQ(hashed.nonzero_entries(), packed_struct->num_blocks);

  if (index & 0x10) {
    for (int i = 0; i < hashed.nonzero_entries(); ++i) {
      packed_dataset[offset + i * 16] =
          (hashed.values()[i] << 4) | (packed_dataset[offset + i * 16] & 0x0f);
    }
  } else {
    for (int i = 0; i < hashed.nonzero_entries(); ++i) {
      packed_dataset[offset + i * 16] =
          hashed.values()[i] | (packed_dataset[offset + i * 16] & 0xf0);
    }
  }
  return OkStatus();
}

template <typename Dataset>
Datapoint<uint8_t> GetLUT16Hash(const size_t index,
                                const Dataset& packed_dataset) {
  const size_t hash_size = packed_dataset.num_blocks;
  const size_t block_offset = index & 0x0f;

  const size_t offset = block_offset + (index & ~31) * hash_size / 2;
  DCHECK_LE(offset + (hash_size - 1) * 16,
            packed_dataset.bit_packed_data.size());
  Datapoint<uint8_t> result;
  result.mutable_values()->reserve(hash_size);

  if (index & 0x10) {
    for (size_t i : Seq(hash_size)) {
      result.mutable_values()->push_back(
          (packed_dataset.bit_packed_data[offset + i * 16] >> 4));
    }
  } else {
    for (size_t i : Seq(hash_size)) {
      result.mutable_values()->push_back(
          (packed_dataset.bit_packed_data[offset + i * 16] & 0x0f));
    }
  }
  return result;
}

template Status SetLUT16Hash<PackedDataset>(
    const DatapointPtr<uint8_t>& hashed, size_t index,
    PackedDataset* __restrict__ packed_struct);
template Status SetLUT16Hash<PackedDatasetMutableView>(
    const DatapointPtr<uint8_t>& hashed, size_t index,
    PackedDatasetMutableView* __restrict__ packed_struct);
template Datapoint<uint8_t> GetLUT16Hash<PackedDataset>(
    size_t index, const PackedDataset& packed_dataset);
template Datapoint<uint8_t> GetLUT16Hash<PackedDatasetMutableView>(
    size_t index, const PackedDatasetMutableView& packed_dataset);
SCANN_INSTANTIATE_TYPED_CLASS(, AsymmetricQueryer);

}  // namespace asymmetric_hashing2
}  // namespace research_scann
