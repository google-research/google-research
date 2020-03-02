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

#include "definitions.h"
#include "definitions.proto.h"
#include "algorithm.h"
#include "random_generator.h"
#include "randomizer.h"
#include "testing/production_stub/public/gunit_prod.h"
#include "util/random/mt_random.h"

namespace brain {
namespace evolution {
namespace amlz {

enum MutationAction : IntegerT {
  // Modifies a single parameter within one instruction. Does not change the op.
  kAlterParamMutationAction = 0,

  // Randomizes an instruction, including its op.
  kRandomizeInstructionMutationAction = 1,

  // Randomizes a whole component_function, preserving its size.
  kRandomizeComponentFunctionMutationAction = 2,

  // Does nothing. Useful for debugging.
  kIdentityMutationAction = 3,

  // Inserts an instruction in a component_function.
  kInsertInstructionMutationAction = 4,

  // Removes a mutation.
  kRemoveInstructionMutationAction = 5,

  // Removes a mutation and inserts another.
  kTradeInstructionMutationAction = 6,

  // Randomizes all component_functions.
  kRandomizeAlgorithmMutationAction = 7,
};

class Mutator {
 public:
  Mutator(
      // What mutations may be applied. See the MutationAction enum.
      const std::vector<MutationAction>& allowed_actions,
      // The probability of mutating each time.
      double mutate_prob,
      // Ops that can be introduced into the setup component_function. Empty
      // means the component_function won't be mutated at all.
      const std::vector<Op>& allowed_setup_ops,
      // Ops that can be introduced into the predict component_function. Empty
      // means the component_function won't be mutated at all.
      const std::vector<Op>& allowed_predict_ops,
      // Ops that can be introduced into the learn component_function. Empty
      // means the component_function won't be mutated at all.
      const std::vector<Op>& allowed_learn_ops,
      // Minimum/maximum component_function sizes.
      const IntegerT setup_size_min,
      const IntegerT setup_size_max,
      const IntegerT predict_size_min,
      const IntegerT predict_size_max,
      const IntegerT learn_size_min,
      const IntegerT learn_size_max,
      // The random bit generator.
      MTRandom* bit_gen,
      // The random number generator.
      RandomGenerator* rand_gen);

  // Similar to the first constructor, but provides a convenience transformation
  // from vector<IntegerT> to vector<MutationAction> that allows initializing
  // from a flag.
  Mutator(
      const std::vector<IntegerT>& allowed_actions,
      double mutate_prob,
      const std::vector<Op>& allowed_setup_ops,
      const std::vector<Op>& allowed_predict_ops,
      const std::vector<Op>& allowed_learn_ops,
      const IntegerT setup_size_min,
      const IntegerT setup_size_max,
      const IntegerT predict_size_min,
      const IntegerT predict_size_max,
      const IntegerT learn_size_min,
      const IntegerT learn_size_max,
      MTRandom* bit_gen,
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

  // Randomizes all the instructions in one of the three component_functions.
  // Does not change the component_function size.
  void RandomizeComponentFunction(Algorithm* algorithm);

  // Inserts an instruction, making the component_function longer. Has
  // no effect on a maximum-length component_function.
  void InsertInstruction(Algorithm* algorithm);

  // Inserts an instruction, making the component_function shorter. Has
  // no effect on a
  // minimum-length component_function.
  void RemoveInstruction(Algorithm* algorithm);

  // First removes an instruction, then inserts an instruction. Has
  // no effect on a zero-length component_function.
  void TradeInstruction(Algorithm* algorithm);

  // Randomizes all the instructions in all of the component_functions. Does not
  // change the component_function sizes.
  void RandomizeAlgorithm(Algorithm* algorithm);

  void InsertInstructionUnconditionally(
      const Op op,
      std::vector<std::shared_ptr<const Instruction>>* component_function);

  void RemoveInstructionUnconditionally(
      std::vector<std::shared_ptr<const Instruction>>* component_function);

  // Return operations to introduce into the component_functions.
  Op SetupOp();
  Op PredictOp();
  Op LearnOp();

  // Returns which instruction to mutate.
  InstructionIndexT InstructionIndex(InstructionIndexT component_function_size);

  // Returns which component_function to mutate.
  ComponentFunctionT ComponentFunction();

  const std::vector<MutationAction> allowed_actions_;
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
  std::unique_ptr<MTRandom> bit_gen_owned_;
  MTRandom* bit_gen_;
  std::unique_ptr<RandomGenerator> rand_gen_owned_;
  RandomGenerator* rand_gen_;
  Randomizer randomizer_;
};

}  // namespace amlz
}  // namespace evolution
}  // namespace brain


#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_MUTATOR_H_
