// Copyright 2022 The Google Research Authors.
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

#ifndef AUTOML_ZERO_FEC_HASHING_H_
#define AUTOML_ZERO_FEC_HASHING_H_

#include <cstddef>

#include "definitions.h"
#include "executor.h"

namespace automl_zero {

namespace internal {

// Used to convert floats to ints for the purposes of hashing.
constexpr double kHashInversePrecision = 1e8;

inline size_t HashComponent(const double error) {
  const double hash_dbl = FlipAndSquash(error) * kHashInversePrecision;
  CHECK_GE(hash_dbl, static_cast<double>(std::numeric_limits<size_t>::min()));
  CHECK_LE(hash_dbl, static_cast<double>(std::numeric_limits<size_t>::max()));
  return static_cast<size_t>(hash_dbl);
}

}  // namespace internal

size_t WellMixedHash(const std::vector<double>& train_errors,
                     const std::vector<double>& valid_errors,
                     size_t dataset_index, IntegerT num_train_examples);

}  // namespace automl_zero

#endif  // AUTOML_ZERO_FEC_HASHING_H_
