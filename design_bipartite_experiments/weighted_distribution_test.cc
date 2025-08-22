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

#include "weighted_distribution.h"

#include <iostream>
#include <random>

int main() {
  exposure_design::WeightedDistribution<int *> distribution;
  int a{0}, b{0}, c{0};
  distribution.Add(&a, 2);
  distribution.Add(&b, 3);
  distribution.Add(&c, 1);
  std::mt19937 gen{std::random_device{}()};
  for (int i = 0; i < 6000; i++) *distribution(gen) += 1;
  std::cout << "a: " << a << " (expect about 2000)\n";
  std::cout << "b: " << b << " (expect about 3000)\n";
  std::cout << "c: " << c << " (expect about 1000)\n";
}
