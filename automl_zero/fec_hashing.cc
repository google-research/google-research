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

#include "util/hash/mix.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::std::vector;
using internal::HashComponent;

size_t WellMixedHash(
    const vector<double>& train_errors,
    const vector<double>& valid_errors,
    const size_t dataset_index,
    const IntegerT num_train_examples) {
  HashMix mix;
  for (const double error : train_errors) {
    mix.Mix(HashComponent(error));
  }
  for (const double error : valid_errors) {
    mix.Mix(HashComponent(error));
  }
  mix.Mix(dataset_index);
  CHECK(num_train_examples >= 0) << "num_train_examples must be >= 0.";
  mix.Mix(num_train_examples);
  return mix.get();
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
