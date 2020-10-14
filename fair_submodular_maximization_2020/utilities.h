// Copyright 2020 The Authors.
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
// Provides some utilites, like random shuffle, log space, pretty numbers.
//
#ifndef FAIR_SUBMODULAR_MAXIMIZATION_2020_UTILITIES_H_
#define FAIR_SUBMODULAR_MAXIMIZATION_2020_UTILITIES_H_

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
};

// Returns a number in a way that is easier to read.
// Formats numbers like 1078546 -> 1,078,546.
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

// Returns a sequence: small, ..., large.
// Where we roughly have a[i+1]/a[i] = base (we use base = 1+eps).
std::vector<double> LogSpace(double small, double large, double base);

#endif  // FAIR_SUBMODULAR_MAXIMIZATION_2020_UTILITIES_H_
