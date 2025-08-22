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

#include "scann/distance_measures/many_to_many/sfp8_transposed.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "absl/algorithm/container.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/scale_encoding_helpers.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace {

template <Int8TileSide kSide>
std::unique_ptr<Int8TileCodec> MakeCodec(size_t dims) {
#ifdef __x86_64__

  if (RuntimeSupportsAmx()) {
    return std::make_unique<amx::Int8TileCodecImpl<kSide>>(dims);
  } else if (RuntimeSupportsAvx512Vnni()) {
    return std::make_unique<avx512_vnni::Int8TileCodecImpl<kSide>>(dims);
  } else if (RuntimeSupportsAvx512()) {
    return std::make_unique<avx512::Int8TileCodecImpl<kSide>>(dims);
  } else if (RuntimeSupportsAvx2()) {
    return std::make_unique<avx2::Int8TileCodecImpl<kSide>>(dims);
  } else {
    return std::make_unique<sse4::Int8TileCodecImpl<kSide>>(dims);
  }
#endif

  return std::make_unique<fallback::Int8TileCodecImpl<kSide>>(dims);
}

std::unique_ptr<Int8TileCodec> MakeCodec(size_t dims, Int8TileSide side) {
  switch (side) {
    case Int8TileSide::kQuery:
      return MakeCodec<Int8TileSide::kQuery>(dims);
    case Int8TileSide::kDatabase:
      return MakeCodec<Int8TileSide::kDatabase>(dims);
  }
}

float SquaredL2Norm(ConstSpan<int8_t> dp, float scale) {
  return scale * scale * absl::c_inner_product(dp, dp, 0);
}

}  // namespace

absl::StatusOr<unique_ptr<SFP8SimdBlockTransposedDatabase>>
SFP8SimdBlockTransposedDatabase::Build(
    const DefaultDenseDatasetView<float>& float_dataset,
    double noise_shaping_threshold, Int8TileSide side) {
  const DimensionIndex dims = float_dataset.dimensionality();
  std::vector<float> scales(float_dataset.size());
  DenseDataset<int8_t> int_dataset;
  int_dataset.set_dimensionality(dims);
  int_dataset.Reserve(float_dataset.size());
  std::vector<float> multipliers(dims, 1.0f);
  std::string encoded;
  constexpr double kNoiseShapingThreshold = NAN;

  for (DatapointIndex i : IndicesOf(float_dataset)) {
    SCANN_RETURN_IF_ERROR(AppendQuantizeScaledFloatDatapointWithNoiseShaping(
        8, MakeDatapointPtr(float_dataset.GetDatapointSpan(i)), multipliers,
        ScaleEncoding::FLOAT32_SCALE_SUFFIX, kNoiseShapingThreshold, encoded));
    absl::string_view data;
    SCANN_RETURN_IF_ERROR(DecodeScaledDatapoint(
        8, ScaleEncoding::FLOAT32_SCALE_SUFFIX, encoded, scales[i], data));
    SCANN_RET_CHECK_EQ(data.size(), dims);
    int_dataset.AppendOrDie(
        MakeDatapointPtr(reinterpret_cast<const int8_t*>(data.data()), dims));
    encoded.clear();
  }
  return std::make_unique<SFP8SimdBlockTransposedDatabase>(int_dataset, scales,
                                                           side);
}

SFP8SimdBlockTransposedDatabase::SFP8SimdBlockTransposedDatabase(
    const DefaultDenseDatasetView<int8_t>& dataset, ConstSpan<float> scales,
    Int8TileSide side)
    : codec_(MakeCodec(dataset.dimensionality(), side)),
      size_(dataset.size()),

      padded_size_(NextMultipleOf(size_, 2 * codec_->block_datapoints())),
      payload_bytes_(padded_size_ * codec_->datapoint_bytes()),

      payload_(std::make_unique<int8_t[]>(payload_bytes_ +
                                          codec_->register_bytes())),
      scales_(std::make_unique<float[]>(padded_size_)),
      sums_(std::make_unique<int32_t[]>(padded_size_)),
      squared_l2_norms_(std::make_unique<float[]>(padded_size_)) {
  CHECK_EQ(scales.size(), size_);

  absl::c_copy(scales, scales_.get());
  std::fill(scales_.get() + size_, scales_.get() + padded_size_, 0.0f);
  std::fill(sums_.get() + size_, sums_.get() + padded_size_, 0);
  std::fill(squared_l2_norms_.get() + size_,
            squared_l2_norms_.get() + padded_size_, 0.0f);

  auto payload = MutableSpan<int8_t>(payload_.get(), payload_bytes_);
  absl::c_fill(payload, 0);
  for (DatapointIndex dp_idx : Seq(size_)) {
    const auto dp = dataset.GetDatapointSpan(dp_idx);
    sums_[dp_idx] = absl::c_accumulate(dp, 0);
    squared_l2_norms_[dp_idx] = SquaredL2Norm(dp, scales[dp_idx]);
    codec_->EncodeDatapoint(dp, dp_idx, payload);
  }
}

Datapoint<int8_t> SFP8SimdBlockTransposedDatabase::ReconstructDatapoint(
    DatapointIndex idx) const {
  CHECK(idx < size_);
  return codec_->ReconstructDatapoint(idx, payload());
}

}  // namespace research_scann
