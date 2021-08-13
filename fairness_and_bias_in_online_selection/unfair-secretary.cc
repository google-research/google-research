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

#include "unfair-secretary.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace fair_secretary {

using std::vector;

SecretaryInstance UnfairSecretaryAlgorithm::ComputeSolution(
    const vector<SecretaryInstance>& elements) {
  double max_value = 0;
  int th = elements.size() * (1.0 / std::exp(1.0));
  // Compute max values.
  for (int i = 0; i < th; i++) {
    max_value = std::max(max_value, elements[i].value);
  }
  for (int i = th; i < elements.size(); i++) {
    if (elements[i].value >= max_value) {
      return elements[i];
    }
  }
  return SecretaryInstance(-1, -1);
}

SecretaryInstance UnfairSecretaryAlgorithm::ComputeSolutionSingleColor(
    const vector<SecretaryInstance>& elements, const vector<double>& prob) {
  double max_value = 0;
  int rand_color = 0;
  int rand = RandomHandler::eng_() % 1000000;
  double rand_balanced = static_cast<double>(rand) / 1000000;
  for (int i = 0; i < prob.size(); i++) {
    if (rand_balanced <= prob[i]) {
      rand_color = i;
      break;
    }
    rand_balanced -= prob[i];
  }
  int th = elements.size() * (1.0 / std::exp(1.0));
  // Compute max values.
  for (int i = 0; i < th; i++) {
    if (rand_color == elements[i].color) {
      max_value = std::max(max_value, elements[i].value);
    }
  }
  for (int i = th; i < elements.size(); i++) {
    if (elements[i].value >= max_value && elements[i].color == rand_color) {
      return elements[i];
    }
  }
  return SecretaryInstance(-1, -1);
}

}  // namespace fair_secretary
