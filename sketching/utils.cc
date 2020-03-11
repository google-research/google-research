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

#include "utils.h"

namespace sketch {

#define MOD 2147483647
#define HL  31

uint Hash(ULONG a, ULONG b, ULONG x, ULONG size) {
  ULONG result = a * x + b;
  result = ((result >> HL) + result) & MOD;
  uint lresult = (uint)result;
  return lresult % size;
}

bool cmpByItem(const IntFloatPair& a, const IntFloatPair& b) {
  return a.first < b.first;
}

bool cmpByValue(const IntFloatPair& a, const IntFloatPair& b) {
  return a.second > b.second;
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

}  // namespace sketch
