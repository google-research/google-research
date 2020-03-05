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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_ALGORITHM_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_ALGORITHM_H_

#include <array>
#include <cstdint>
#include <memory>
#include <ostream>

#include "algorithm.pb.h"
#include "instruction.h"
#include "absl/flags/flag.h"

namespace automl_zero {

class RandomGenerator;

// Denotes one of the three component functions in the Algorithm.
enum ComponentFunctionT : IntegerT {
  kSetupComponentFunction = 0,
  kPredictComponentFunction = 1,
  kLearnComponentFunction = 2
};

// The Algorithm describing an individual.
// NOTE: the default constructor does NOT serve as a way to initialize the
// Instruction.
class Algorithm {
 public:
  // A Algorithm without any instructions.
  Algorithm() {}

  explicit Algorithm(const SerializedAlgorithm& checkpoint_algorithm);

  Algorithm(const Algorithm& other);
  Algorithm& operator=(const Algorithm& other);
  Algorithm(Algorithm&& other);
  Algorithm& operator=(Algorithm&& other);

  bool operator ==(const Algorithm& other) const;
  bool operator !=(const Algorithm& other) const {
    return !(*this == other);
  }

  // Returns a human-readable representation.
  std::string ToReadable() const;

  // Serializes/deserializes a Algorithm to/from a amlz-specific proto.
  SerializedAlgorithm ToProto() const;
  void FromProto(const SerializedAlgorithm& checkpoint_algorithm);

  // Returns a reference to the given component function in the Algorithm.
  const std::vector<std::shared_ptr<const Instruction>>&
      ComponentFunction(ComponentFunctionT component_function_type) const;
  std::vector<std::shared_ptr<const Instruction>>* MutableComponentFunction(
      ComponentFunctionT component_function_type);

  // Setup, predict, and learn component functions.
  std::vector<std::shared_ptr<const Instruction>> setup_;
  std::vector<std::shared_ptr<const Instruction>> predict_;
  std::vector<std::shared_ptr<const Instruction>> learn_;
};

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_ALGORITHM_H_
