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

#include "scann/utils/scale_encoding_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "absl/base/casts.h"
#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/one_to_many/one_to_many_helpers.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

namespace {

float CalculateDatapointScale(const DatapointPtr<float>& dp,
                              absl::Span<const float> fixed8_multipliers) {
  float max_abs = 0.0f;
  for (size_t i : Seq(dp.dimensionality())) {
    max_abs = std::max(max_abs,
                       std::abs(dp.values_span()[i] * fixed8_multipliers[i]));
  }
  return max_abs / numeric_limits<int8_t>::max();
}

template <typename T, typename Span>
absl::Span<T> CastCharSpan(Span span) {
  static_assert(sizeof(T) == sizeof(char));
  return MakeMutableSpan(reinterpret_cast<T*>(span.data()), span.size());
}

uint32_t DecodeBottomBitsData(int bits, absl::string_view encoded) {
  if (bits == 4) {
    return DecodeBottomBitsDataFromPackedInt4(
        CastCharSpan<const uint8_t>(encoded));
  } else {
    return DecodeBottomBitsDataFromInt8(CastCharSpan<const int8_t>(encoded));
  }
}

}  // namespace

absl::Status DecodeScaledDatapoint(int bits, ScaleEncoding scale_encoding,
                                   absl::string_view encoded, float& scale,
                                   absl::string_view& data) {
  SCANN_RET_CHECK(bits == 4 || bits == 8) << bits;
  data = encoded;
  switch (scale_encoding) {
    case UNSPECIFIED_SCALE_ENCODING:
      scale = 1.0f;
      return OkStatus();
    case FLOAT32_SCALE_SUFFIX:
      scale = absl::bit_cast<float>(
          absl::little_endian::Load32(encoded.end() - sizeof(float)));
      data.remove_suffix(sizeof(float));
      return OkStatus();
    case FLOAT32_SCALE_BOTTOM_BITS:
      scale = absl::bit_cast<float>(DecodeBottomBitsData(bits, encoded));
      return OkStatus();
  }
}

absl::StatusOr<size_t> ScaledDatapointEncodedBytes(int bits,
                                                   ScaleEncoding scale_encoding,
                                                   size_t dimension) {
  SCANN_RET_CHECK(bits == 8 || bits == 4) << bits;
  using one_to_many_low_level::DatapointBytes;
  if (bits == 4) {
    return DatapointBytes<uint8_t>(dimension, scale_encoding);
  } else {
    return DatapointBytes<int8_t>(dimension, scale_encoding);
  }
}

absl::Status AppendQuantizeScaledFloatDatapointWithNoiseShaping(
    int bits, DatapointPtr<float> dptr,
    absl::Span<const float> fixed8_multipliers, ScaleEncoding scale_encoding,
    double noise_shaping_threshold, std::string& out) {
  SCANN_ASSIGN_OR_RETURN(
      const size_t stride,
      ScaledDatapointEncodedBytes(bits, scale_encoding, dptr.dimensionality()));
  out.resize(out.size() + stride);
  auto quantized = MakeMutableSpan(out.data() + out.size() - stride, stride);
  return QuantizeScaledFloatDatapointWithNoiseShaping(
      bits, dptr, fixed8_multipliers, scale_encoding, noise_shaping_threshold,
      CastCharSpan<uint8_t>(quantized));
}

absl::Status QuantizeScaledFloatDatapointWithNoiseShaping(
    int bits, DatapointPtr<float> dptr,
    absl::Span<const float> fixed8_multipliers, ScaleEncoding scale_encoding,
    double noise_shaping_threshold, MutableSpan<uint8_t> encoded) {
  SCANN_RET_CHECK(bits == 8 || bits == 4) << bits;
  const size_t dims = dptr.dimensionality();
  SCANN_ASSIGN_OR_RETURN(const size_t stride, ScaledDatapointEncodedBytes(
                                                  bits, scale_encoding, dims));
  SCANN_RET_CHECK_EQ(encoded.size(), stride);
  std::unique_ptr<float[]> multipliers_storage;
  if (fixed8_multipliers.size() <= 1) {
    multipliers_storage = std::make_unique<float[]>(dims);
    const float fixed8_multiplier =
        fixed8_multipliers.size() == 1 ? fixed8_multipliers[0] : 1.0f;
    for (size_t i : Seq(dims)) {
      multipliers_storage[i] = fixed8_multiplier;
    }
    fixed8_multipliers = MakeConstSpan(multipliers_storage.get(), dims);
  }
  SCANN_RET_CHECK_EQ(fixed8_multipliers.size(), dims);

  float inv_scale = 1.0f;
  using one_to_many_low_level::DatapointBytes;
  if (bits == 4) {
    inv_scale *= (kFP4Max / kFP8Max);
  }
  std::optional<uint32_t> bottom_bits_data;
  if (scale_encoding != UNSPECIFIED_SCALE_ENCODING) {
    const float scale = CalculateDatapointScale(dptr, fixed8_multipliers);
    inv_scale /= scale;
    if (std::isinf(inv_scale)) {
      inv_scale = numeric_limits<float>::max();
    }
    uint32_t uint32_scale = absl::bit_cast<uint32_t>(scale);
    if (scale_encoding == FLOAT32_SCALE_BOTTOM_BITS) {
      bottom_bits_data = absl::little_endian::FromHost32(uint32_scale);
    } else {
      SCANN_RET_CHECK_EQ(scale_encoding, FLOAT32_SCALE_SUFFIX);
      absl::little_endian::Store32(encoded.end() - sizeof(float), uint32_scale);
      encoded.remove_suffix(sizeof(float));
    }
  }
  vector<float> scaled_storage;
  if (inv_scale != 1.0f) {
    scaled_storage.resize(dims);
    for (size_t i : Seq(dims)) {
      scaled_storage[i] = dptr.values_span()[i] * inv_scale;
    }
    dptr = MakeDatapointPtr(scaled_storage);
  }

  if (bits == 4) {
    return Int4QuantizePackFloatDatapointWithNoiseShaping(
        dptr, fixed8_multipliers, bottom_bits_data, noise_shaping_threshold,
        encoded);
  } else {
    return ScalarQuantizeFloatDatapointWithNoiseShaping(
               dptr, fixed8_multipliers, bottom_bits_data,
               noise_shaping_threshold, CastCharSpan<int8_t>(encoded))
        .status();
  }
}

static void ReconstructInt8Datapoint(
    ConstSpan<int8_t> data, ConstSpan<float> inverse_fixed8_multipliers,
    float scale, MutableSpan<float>& dp) {
  if (inverse_fixed8_multipliers.size() > 1) {
    for (size_t i : Seq(dp.size())) {
      dp[i] = scale * inverse_fixed8_multipliers[i] * data[i];
    }
  } else {
    for (size_t i : Seq(dp.size())) {
      dp[i] = scale * data[i];
    }
  }
}

static void ReconstructInt4Datapoint(
    ConstSpan<uint8_t> data, ConstSpan<float> inverse_fixed8_multipliers,
    float scale, MutableSpan<float>& dp) {
  std::vector<uint8_t> unpacked(dp.size());
  UnpackNibblesDatapoint(data, MakeMutableSpan(unpacked), dp.size());
  ReconstructInt8Datapoint(CastCharSpan<const int8_t>(unpacked),
                           inverse_fixed8_multipliers, scale, dp);
}

absl::Status ReconstructScaledDatapoint(
    int bits, ConstSpan<float> inverse_fixed8_multipliers,
    ScaleEncoding scale_encoding, absl::string_view encoded,
    MutableSpan<float>& dp) {
  SCANN_RET_CHECK(bits == 8 || bits == 4) << bits;
  absl::string_view data;
  float scale;
  SCANN_RETURN_IF_ERROR(
      DecodeScaledDatapoint(bits, scale_encoding, encoded, scale, data));
  if (inverse_fixed8_multipliers.size() == 1) {
    scale *= inverse_fixed8_multipliers[0];
  } else if (inverse_fixed8_multipliers.size() > 1) {
    SCANN_RET_CHECK_EQ(inverse_fixed8_multipliers.size(), dp.size());
  }
  if (bits == 4) {
    SCANN_RET_CHECK_EQ(DivRoundUp(dp.size(), 2), data.size());
    scale *= (kFP4Max / kFP8Max);
    SCANN_RET_CHECK_EQ(data.size(), DivRoundUp(dp.size(), 2));
    ReconstructInt4Datapoint(CastCharSpan<const uint8_t>(data),
                             inverse_fixed8_multipliers, scale, dp);
  } else {
    SCANN_RET_CHECK_EQ(data.size(), dp.size());
    ReconstructInt8Datapoint(CastCharSpan<const int8_t>(data),
                             inverse_fixed8_multipliers, scale, dp);
  }
  return OkStatus();
}

}  // namespace research_scann
