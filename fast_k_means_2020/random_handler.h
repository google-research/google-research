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

// Simple utility function for handling random varialbes.

#ifndef FAST_K_MEANS_2020_RANDOM_HANDLER_H_
#define FAST_K_MEANS_2020_RANDOM_HANDLER_H_

#include <cstdint>
#include <random>

namespace fast_k_means {

class RandomHandler {
 public:
  // Get a random seed from the OS entropy device, or whatever
  static std::random_device rd;
  // Use the 64-bit Mersenne Twister 19937
  // generator and seed it with entropy.
  // In case of using changing random values, use std::mt19937_64 eng(rd);
  static std::mt19937_64 eng;
};

}  // namespace fast_k_means

#endif  // FAST_K_MEANS_2020_RANDOM_HANDLER_H_
