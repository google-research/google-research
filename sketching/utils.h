// Copyright 2021 The Google Research Authors.
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

#ifndef SKETCHING_UTILS_H_
#define SKETCHING_UTILS_H_

#include <utility>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"

namespace sketch {

typedef unsigned int uint;
typedef uint64 ULONG;
typedef std::pair<uint, float> IntFloatPair;

inline constexpr int HL = 31;
inline constexpr ULONG MOD = 2147483647;

inline uint Hash(ULONG a, ULONG b, ULONG x, ULONG size) {
  ULONG result = a * x + b;
  result = ((result >> HL) + result) & MOD;
  uint lresult = (uint)result;
  return lresult % size;
}

bool cmpByItem(const IntFloatPair& a, const IntFloatPair& b);

// A wrapper around absl's bit generator class. Could allow a switch to a
// deterministic generator for testing, not implemented.
class BitGenerator {
 public:
  BitGenerator();

  absl::BitGenRef * BitGen() {
    return &bit_gen_ref_;
  }

  absl::BitGen bit_gen_;
  absl::BitGenRef bit_gen_ref_;
};

std::vector<uint> FilterOutAboveThreshold(
    const std::vector<IntFloatPair>& candidates, float threshold);

std::pair<std::vector<IntFloatPair>, std::vector<float>> CreateStream(
    int stream_size, int lg_stream_range = 20, double zipf_param = 1.1);

}  // namespace sketch

#endif  // SKETCHING_UTILS_H_
