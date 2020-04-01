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

#include "fec_hashing.h"
#include <cstddef>

#include "definitions.h"

namespace automl_zero {

using ::std::vector;

size_t WellMixedHash(
    const vector<double>& train_errors,
    const vector<double>& valid_errors,
    const size_t dataset_index,
    const IntegerT num_train_examples) {
  std::size_t seed = 42;
  for (const double error : train_errors) {
    HashCombine(seed, error);
  }
  for (const double error : valid_errors) {
    HashCombine(seed, error);
  }
  HashCombine(seed, dataset_index);
  HashCombine(seed, num_train_examples);
  return seed;
}

}  // namespace automl_zero
