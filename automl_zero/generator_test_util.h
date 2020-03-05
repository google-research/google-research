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

#ifndef GENERATOR_TEST_UTIL_H_
#define GENERATOR_TEST_UTIL_H_

#include "algorithm.h"
#include "generator.h"
#include "testing/base/public/gmock.h"

namespace automl_zero {

// Returns a simple generator.
Generator SimpleGenerator();

// Returns a small fixed-size Algorithm with no-op instructions.
Algorithm SimpleNoOpAlgorithm();

// Returns a small fixed-size random Algorithm.
Algorithm SimpleRandomAlgorithm();

// Returns a Gz Algorithm with default parameters.
Algorithm SimpleGz();

// Returns a Gz Algorithm with default parameters.
Algorithm SimpleGrTildeGrWithBias();

// Returns a Algorithm where each component function has the data of
// instruction at position p set to the integer p.
Algorithm SimpleIncreasingDataAlgorithm();

}  // namespace automl_zero

#endif  // GENERATOR_TEST_UTIL_H_
