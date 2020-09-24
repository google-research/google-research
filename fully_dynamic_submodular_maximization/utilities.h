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

//
//  Utilities
//

// Provides some utilites.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_UTILITIES_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_UTILITIES_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class RandomHandler {
 public:
  // Default-initialized, to get reproducible cross-platform randomness.
  static std::mt19937 generator_;

  // Shuffles a vector.
  // Own implementation to get cross-platform reproducibility.
  template <class T>
  static void Shuffle(std::vector<T>& input) {
    for (int i = static_cast<int>(input.size()) - 1; i > 0; --i) {
      std::swap(input[i], input[generator_() % (i + 1)]);
    }
  }

  // Checks if the randomness is as expected.
  static void CheckRandomNumberGenerator() {
    // "The 10000th consecutive invocation of a default-constructed
    //  std::mt19937 is required to produce the value 4123659995."
    for (int i = 0; i < 9999; ++i) generator_();
    if (generator_() != 4123659995) {
      std::cerr << "something is wrong with the generator: \n"
                << "the randomness might be different from what the original "
                   "implementation used\n\n";
    }
  }
};

// Returns a number in a way that is easier to read.
std::string PrettyNum(int64_t number);

// This functions is called if there is something wrong and the error string
// 'error' will be written in "cerr'.
void Fail(const std::string& error);

template <typename T, typename Fun>
void Filter(std::vector<T>& input, const Fun& predicate) {
  auto source = input.begin(), target = input.begin();
  for (; source != input.end(); ++source) {
    if (predicate(*source)) {
      *target = *source;
      ++target;
    }
  }
  input.erase(target, input.end());
}

std::vector<double> LogSpace(double small, double large, double base);

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_UTILITIES_H_
