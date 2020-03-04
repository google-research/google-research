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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_RANDOMIZER_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_RANDOMIZER_H_

#include <memory>
#include <random>

#include "definitions.h"
#include "instruction.proto.h"

namespace automl_zero {

class RandomGenerator;
class Algorithm;

// A class to randomize Algorithms. Used for Algorithm mutation and
// Algorithm generation.
class Randomizer {
 public:
  Randomizer(
      // Ops that can be introduced into the setup component_function.
      // Empty means the component_function is not randomized.
      std::vector<Op> allowed_setup_ops,
      // Ops that can be introduced into the predict component_function.
      // Empty means the component_function is not randomized.
      std::vector<Op> allowed_predict_ops,
      // Ops that can be introduced into the learn component_function.
      // Empty means the component_function is not randomized.
      std::vector<Op> allowed_learn_ops,
      std::mt19937* bit_gen,
      RandomGenerator* rand_gen);

  // Randomizes the entire Algorithm (all three component_functions).
  // Does not change the component_function sizes.
  void Randomize(Algorithm* algorithm);

  // Randomizes all the instructions in the setup component_function.
  // Does not change the component_function size.
  void RandomizeSetup(Algorithm* algorithm);

  // Randomizes all the instructions in the predict component_function.
  // Does not change the component_function size.
  void RandomizePredict(Algorithm* algorithm);

  // Randomizes all the instructions in the learn component_function.
  // Does not change the component_function size.
  void RandomizeLearn(Algorithm* algorithm);

  // Return operations to introduce into the component_functions.
  Op SetupOp();
  Op PredictOp();
  Op LearnOp();

 private:
  const std::vector<Op> allowed_setup_ops_;
  const std::vector<Op> allowed_predict_ops_;
  const std::vector<Op> allowed_learn_ops_;
  std::mt19937* bit_gen_;
  RandomGenerator* rand_gen_;
};

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_RANDOMIZER_H_
