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

#ifndef SKETCHING_UTILS_H_
#define SKETCHING_UTILS_H_

#include <utility>
#include "glog/logging.h"
#include "absl/random/bit_gen_ref.h"

namespace sketch {

typedef unsigned int uint;
typedef unsigned long long ULONG;
typedef std::pair<uint, float> IntFloatPair;

unsigned int Hash(ULONG a, ULONG b, ULONG x, ULONG size);
bool cmpByItem(const IntFloatPair& a, const IntFloatPair& b);
bool cmpByValue(const IntFloatPair& a, const IntFloatPair& b);
uint log2int(uint val);

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

}  // namespace sketch

#endif  // SKETCHING_UTILS_H_
