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

#ifndef EXPERIMENTAL_USERS_ASHKANNOROUZI_FAST_K_MEANS_2019_RANDOM_HANDLER_H_
#define EXPERIMENTAL_USERS_ASHKANNOROUZI_FAST_K_MEANS_2019_RANDOM_HANDLER_H_

#include <cstdint>
#include <random>

namespace fair_secretary {

class RandomHandler {
 public:
  // Get a random seed from the OS entropy device, or whatever
  // Use the 64-bit Mersenne Twister 19937
  // generator and seed it with entropy.
  // In case of using changing random values, use std::mt19937_64 eng_(rd_);
  static std::random_device rd_;

  // Define the distribution, by default it goes from 0 to MAX(unsigned long
  // long) or the type supported by your architecture.
  static std::mt19937_64 eng_;

  static std::default_random_engine generator_;
};

}  // namespace fair_secretary

#endif  // EXPERIMENTAL_USERS_ASHKANNOROUZI_FAST_K_MEANS_2019_RANDOM_HANDLER_H_
