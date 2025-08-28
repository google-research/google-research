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

#ifndef SCANN_UTILS_SCALE_ENCODING_HELPERS_H_
#define SCANN_UTILS_SCALE_ENCODING_HELPERS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/one_to_many/scale_encoding.pb.h"
#include "scann/utils/common.h"

namespace research_scann {

absl::Status AppendQuantizeScaledFloatDatapointWithNoiseShaping(
    int bits, DatapointPtr<float> dptr, ConstSpan<float> fixed8_multipliers,
    ScaleEncoding scale_encoding, double noise_shaping_threshold,
    std::string& out);

absl::Status QuantizeScaledFloatDatapointWithNoiseShaping(
    int bits, DatapointPtr<float> dptr, ConstSpan<float> fixed8_multipliers,
    ScaleEncoding scale_encoding, double noise_shaping_threshold,
    MutableSpan<uint8_t> encoded);

absl::StatusOr<size_t> ScaledDatapointEncodedBytes(int bits,
                                                   ScaleEncoding scale_encoding,
                                                   size_t dimension);

absl::Status DecodeScaledDatapoint(int bits, ScaleEncoding scale_encoding,
                                   absl::string_view encoded, float& scale,
                                   absl::string_view& data);

absl::Status ReconstructScaledDatapoint(
    int bits, ConstSpan<float> inverse_fixed8_multipliers,
    ScaleEncoding scale_encoding, absl::string_view encoded,
    MutableSpan<float>& dp);

}  // namespace research_scann

#endif
