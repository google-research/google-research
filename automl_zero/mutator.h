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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_MUTATOR_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_MUTATOR_H_

#include <memory>
#include <random>

#include "algorithm.h"
#include "definitions.h"
#include "instruction.proto.h"
#include "mutator.proto.h"
#include "random_generator.h"
#include "randomizer.h"
#include "testing/production_stub/public/gunit_prod.h"

namespace automl_zero {

class Mutator {
 public:
  Mutator(
      // What mutations may be applied. See the MutationType enum.
      const MutationTypeList& allowed_actions,
      // The probability of mutating each time.
      double mutate_prob,
      // Ops that can be introduced into the setup component function. Empty
      // means the component function won't be mutated at all.
      const std::vector<Op>& allowed_setup_ops,
      // Ops that can be introduced into the predict component function. Empty
      // means the component function won't be mutated at all.
      const std::vector<Op>& allowed_predict_ops,
      // Ops that can be introduced into the learn component function. Empty
      // means the component function won't be mutated at all.
      const std::vector<Op>& allowed_learn_ops,
      // Minimum/maximum component function sizes.
      const IntegerT setup_size_min,
      const IntegerT setup_size_max,
      const IntegerT predict_size_min,
      const IntegerT predict_size_max,
      const IntegerT learn_size_min,
      const IntegerT learn_size_max,
      // The random bit generator.
      std::mt19937* bit_gen,
      // The random number generator.
      RandomGenerator* rand_gen);

  Mutator(const Mutator& other) = delete;
  Mutator& operator=(const Mutator& other) = delete;

  void Mutate(std::shared_ptr<const Algorithm>* algorithm);
  void Mutate(IntegerT num_mutations,
              std::shared_ptr<const Algorithm>* algorithm);

 private:
  friend Mutator SimpleMutator();
  FRIEND_TEST(MutatorTest, InstructionIndexTest);
  FRIEND_TEST(MutatorTest, SetupOpTest);
  FRIEND_TEST(MutatorTest, PredictOpTest);
  FRIEND_TEST(MutatorTest, LearnOpTest);
  FRIEND_TEST(MutatorTest, ComponentFunctionTest_SetupPredictLearn);
  FRIEND_TEST(MutatorTest, ComponentFunctionTest_Setup);
  FRIEND_TEST(MutatorTest, ComponentFunctionTest_Predict);
  FRIEND_TEST(MutatorTest, ComponentFunctionTest_Learn);
  FRIEND_TEST(MutatorTest, AlterParam);
  FRIEND_TEST(MutatorTest, RandomizeInstruction);
  FRIEND_TEST(MutatorTest, RandomizeComponentFunction);
  FRIEND_TEST(MutatorTest, RandomizeAlgorithm);

  // Used to create a simple instance for tests.
  Mutator();

  void MutateImpl(Algorithm* algorithm);

  // Randomizes a single parameter within one instruction. Keeps the same op.
  void AlterParam(Algorithm* algorithm);

  // Randomizes an instruction (all its parameters, including the op).
  void RandomizeInstruction(Algorithm* algorithm);

  // Randomizes all the instructions in one of the three component functions.
  // Does not change the component function size.
  void RandomizeComponentFunction(Algorithm* algorithm);

  // Inserts an instruction, making the component function longer. Has
  // no effect on a maximum-length component function.
  void InsertInstruction(Algorithm* algorithm);

  // Inserts an instruction, making the component function shorter. Has
  // no effect on a minimum-length component function.
  void RemoveInstruction(Algorithm* algorithm);

  // First removes an instruction, then inserts an instruction. Has
  // no effect on a zero-length component function.
  void TradeInstruction(Algorithm* algorithm);

  // Randomizes all the instructions in all of the component functions. Does not
  // change the component function sizes.
  void RandomizeAlgorithm(Algorithm* algorithm);

  void InsertInstructionUnconditionally(
      const Op op,
      std::vector<std::shared_ptr<const Instruction>>* component_function);

  void RemoveInstructionUnconditionally(
      std::vector<std::shared_ptr<const Instruction>>* component_function);

  // Return operations to introduce into the component functions.
  Op SetupOp();
  Op PredictOp();
  Op LearnOp();

  // Returns which instruction to mutate.
  InstructionIndexT InstructionIndex(InstructionIndexT component_function_size);

  // Returns which component function to mutate.
  ComponentFunctionT ComponentFunction();

  const MutationTypeList allowed_actions_;
  const double mutate_prob_;
  const std::vector<Op> allowed_setup_ops_;
  const std::vector<Op> allowed_predict_ops_;
  const std::vector<Op> allowed_learn_ops_;
  const bool mutate_setup_;
  const bool mutate_predict_;
  const bool mutate_learn_;
  const InstructionIndexT setup_size_min_;
  const InstructionIndexT setup_size_max_;
  const InstructionIndexT predict_size_min_;
  const InstructionIndexT predict_size_max_;
  const InstructionIndexT learn_size_min_;
  const InstructionIndexT learn_size_max_;
  std::unique_ptr<std::mt19937> bit_gen_owned_;
  std::mt19937* bit_gen_;
  std::unique_ptr<RandomGenerator> rand_gen_owned_;
  RandomGenerator* rand_gen_;
  Randomizer randomizer_;
};

}  // namespace automl_zero


#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_MUTATOR_H_
