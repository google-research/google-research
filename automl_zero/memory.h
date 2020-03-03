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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_MEMORY_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_MEMORY_H_

#include <array>

#include "glog/logging.h"
#include "definitions.h"
#include "absl/flags/flag.h"

namespace automl_zero {

// Instantiate only once per worker. Then use Wipe() to wipe clean. Unit tests
// may create multiple ones.
template<FeatureIndexT F>
class Memory {
 public:
  // Does NOT serve as a way to initialize the Memory.
  Memory();

  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;

  // Sets the Scalars, Vectors and Matrices to zero. Serves as a way to
  // initialize the memory.
  void Wipe();

  // Three typed-memory spaces.
  ::std::array<Scalar, kMaxScalarAddresses> scalar_;
  ::std::array<Vector<F>, kMaxVectorAddresses> vector_;
  ::std::array<Matrix<F>, kMaxMatrixAddresses> matrix_;
};

// Does NOT serve as a way to initialize the Memory.
// Simply ensures that the potentially dynamic Vector and Matrix objects have
// definite shape.
template<FeatureIndexT F>
Memory<F>::Memory() {
  for (Vector<F>& value : vector_) {
    value.resize(F, 1);
  }
  for (Matrix<F>& value : matrix_) {
    value.resize(F, F);
  }
}

template<FeatureIndexT F>
void Memory<F>::Wipe() {
  for (Scalar& value : scalar_) {
    value = 0.0;
  }
  for (Vector<F>& value : vector_) {
    value.setZero();
  }
  for (Matrix<F>& value : matrix_) {
    value.setZero();
  }
}

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_MEMORY_H_
