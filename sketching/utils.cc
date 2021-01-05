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

#include "utils.h"

#include <utility>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/random/zipf_distribution.h"

namespace sketch {

bool cmpByItem(const IntFloatPair& a, const IntFloatPair& b) {
  return a.first < b.first;
}

BitGenerator::BitGenerator() :
    bit_gen_ref_(bit_gen_)
{}

std::vector<uint> FilterOutAboveThreshold(
    const std::vector<IntFloatPair>& candidates, float threshold) {
  std::vector<uint> items;
  for (const auto& [key, value] : candidates) {
    if (value > threshold) {
      items.push_back(key);
    }
  }
  return items;
}

std::pair<std::vector<IntFloatPair>, std::vector<float>> CreateStream(
    int stream_size, int lg_stream_range, double zipf_param) {
  std::vector<IntFloatPair> data;
  uint stream_range = 1 << lg_stream_range;
  std::vector<float> counts(stream_range, 0.0);
  BitGenerator bit_gen;
  absl::BitGenRef& gen = *bit_gen.BitGen();
  absl::zipf_distribution<uint> zipf(stream_range, zipf_param, 1.0);
  ULONG a = absl::uniform_int_distribution<int>(0, stream_range - 1)(gen);
  ULONG b = absl::uniform_int_distribution<int>(0, stream_range - 1)(gen);
  for (int i = 0; i < stream_size; ++i) {
    uint k = Hash(a, b, zipf(gen), stream_range);
    counts.at(k) += 1.0;
    data.emplace_back(k, 1.0);
  }
  return {data, counts};
}

}  // namespace sketch
