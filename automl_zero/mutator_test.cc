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

#include "mutator.h"

#include <memory>

#include "definitions.h"
#include "definitions.proto.h"
#include "algorithm.h"
#include "algorithm_test_util.h"
#include "generator.h"
#include "generator_test_util.h"
#include "mutator_test_util.h"
#include "random_generator.h"
#include "random_generator_test_util.h"
#include "test_util.h"
#include "gtest/gtest.h"
#include "util/random/mt_random.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::std::function;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::vector;  // NOLINT
using ::testing::Test;

TEST(MutatorTest, Runs) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<MutationAction>{  // allowed_actions
        kAlterParamMutationAction, kRandomizeInstructionMutationAction,
        kRandomizeComponentFunctionMutationAction,
        kInsertInstructionMutationAction, kTradeInstructionMutationAction,
        kRemoveInstructionMutationAction
      },
      0.5,  // mutate_prob
      {NO_OP, SCALAR_SUM_OP, VECTOR_SUM_OP},  // allowed_setup_ops
      {NO_OP, SCALAR_DIFF_OP, VECTOR_DIFF_OP},  // allowed_predict_ops
      {NO_OP, SCALAR_PRODUCT_OP, VECTOR_PRODUCT_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen,
      &rand_gen);
  Generator generator(kNoOpModel, 10, 10, 10, {}, {}, {}, nullptr,
                      nullptr);
  shared_ptr<const Algorithm> algorithm =
      make_shared<const Algorithm>(SimpleRandomAlgorithm());
  mutator.Mutate(10, &algorithm);
}

TEST(MutatorTest, CoversActions) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<MutationAction>{  // allowed_actions
        kInsertInstructionMutationAction, kTradeInstructionMutationAction,
        kRemoveInstructionMutationAction
      },
      1.0,  // mutate_prob
      {},  // allowed_setup_ops
      {NO_OP},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen,
      &rand_gen);
  const Algorithm algorithm = SimpleRandomAlgorithm();
  const IntegerT original_size = algorithm.predict_.size();
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
        mutator.Mutate(&mutated_algorithm);
        return static_cast<IntegerT>(mutated_algorithm->predict_.size());
      }),
      {original_size - 1, original_size, original_size + 1},
      {original_size - 1, original_size, original_size + 1}));
}

TEST(MutatorTest, RespectsMutateProb) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<MutationAction>{  // allowed_actions
        kInsertInstructionMutationAction},
      0.5,  // mutate_prob
      {},  // allowed_setup_ops
      {NO_OP},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen,
      &rand_gen);
  const Algorithm algorithm = SimpleRandomAlgorithm();
  const IntegerT original_size = algorithm.predict_.size();
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
        mutator.Mutate(&mutated_algorithm);
        return static_cast<IntegerT>(mutated_algorithm->predict_.size());
      }),
      {original_size, original_size + 1},
      {original_size, original_size + 1}));
}

TEST(MutatorTest, InstructionIndexTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0, {}, {}, {},
      0, 10000, 0, 10000, 0, 10000,
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<InstructionIndexT(void)>(
          [&](){return mutator.InstructionIndex(5);}),
      Range<InstructionIndexT>(0, 5),
      Range<InstructionIndexT>(0, 5)));
}

TEST(MutatorTest, SetupOpTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0,
      // allowed_setup_ops
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<Op(void)>([&](){
        return mutator.SetupOp();
      }),
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}));
}

TEST(MutatorTest, PredictOpTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0,
      {},  // allowed_setup_ops
      // allowed_predict_ops
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<Op(void)>([&](){
        return mutator.PredictOp();
      }),
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}));
}

TEST(MutatorTest, LearnOpTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0,
      {},  // allowed_setup_ops
      {},  // allowed_predict_ops
      // allowed_learn_ops
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<Op(void)>([&](){
        return mutator.LearnOp();
      }),
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}));
}

TEST(MutatorTest, ComponentFunctionTest_SetupPredictLearn) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0,
      {NO_OP, SCALAR_SUM_OP},  // allowed_setup_ops
      {NO_OP, SCALAR_SUM_OP},  // allowed_predict_ops
      {NO_OP, SCALAR_SUM_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<ComponentFunctionT(void)>([&](){
        return mutator.ComponentFunction();
      }),
      {kSetupComponentFunction, kPredictComponentFunction,
       kLearnComponentFunction},
      {kSetupComponentFunction, kPredictComponentFunction,
       kLearnComponentFunction}));
}

TEST(MutatorTest, ComponentFunctionTest_Setup) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0,
      {NO_OP, SCALAR_SUM_OP},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<ComponentFunctionT(void)>([&](){
        return mutator.ComponentFunction();
      }),
      {kSetupComponentFunction},
      {kSetupComponentFunction}));
}

TEST(MutatorTest, ComponentFunctionTest_Predict) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0,
      {},  // allowed_setup_ops
      {NO_OP, SCALAR_SUM_OP},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<ComponentFunctionT(void)>([&](){
        return mutator.ComponentFunction();
      }),
      {kPredictComponentFunction},
      {kPredictComponentFunction}));
}

TEST(MutatorTest, ComponentFunctionTest_Learn) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{}, 0.0,
      {},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {NO_OP, SCALAR_SUM_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
      function<ComponentFunctionT(void)>([&](){
        return mutator.ComponentFunction();
      }),
      {kLearnComponentFunction},
      {kLearnComponentFunction}));
}

TEST(MutatorTest, AlterParam) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  const Algorithm algorithm = SimpleRandomAlgorithm();
  Mutator mutator = SimpleMutator();
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.AlterParam(&mutated_algorithm);
        return CountDifferentInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.AlterParam(&mutated_algorithm);
        return CountDifferentSetupInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.AlterParam(&mutated_algorithm);
        return CountDifferentPredictInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.AlterParam(&mutated_algorithm);
        return CountDifferentLearnInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
}

TEST(MutatorTest, RandomizeInstruction) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  const Algorithm algorithm = SimpleRandomAlgorithm();
  Mutator mutator = SimpleMutator();
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeInstruction(&mutated_algorithm);
        return CountDifferentInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeInstruction(&mutated_algorithm);
        return CountDifferentSetupInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeInstruction(&mutated_algorithm);
        return CountDifferentPredictInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeInstruction(&mutated_algorithm);
        return CountDifferentLearnInstructions(mutated_algorithm, algorithm);
      }),
      {0, 1}, {1}));
}

TEST(MutatorTest, RandomizeComponentFunction) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  const Algorithm algorithm = SimpleRandomAlgorithm();
  Mutator mutator = SimpleMutator();
  const IntegerT setup_size = algorithm.setup_.size();
  const IntegerT predict_size = algorithm.predict_.size();
  const IntegerT learn_size = algorithm.learn_.size();
  vector<IntegerT> num_instr = {setup_size, predict_size, learn_size};
  const IntegerT max_instructions =
      *max_element(num_instr.begin(), num_instr.end());
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeComponentFunction(&mutated_algorithm);
        return CountDifferentInstructions(mutated_algorithm, algorithm);
      }),
      Range<IntegerT>(0, max_instructions + 1), {max_instructions}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeComponentFunction(&mutated_algorithm);
        return CountDifferentSetupInstructions(mutated_algorithm, algorithm);
      }),
      Range<IntegerT>(0, setup_size + 1),
      {setup_size}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeComponentFunction(&mutated_algorithm);
        return CountDifferentPredictInstructions(mutated_algorithm, algorithm);
      }),
      Range<IntegerT>(0, predict_size + 1),
      {predict_size}));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeComponentFunction(&mutated_algorithm);
        return CountDifferentLearnInstructions(mutated_algorithm, algorithm);
      }),
      Range<IntegerT>(0, learn_size + 1),
      {learn_size}));
}

TEST(MutatorTest, IdentityMutationAction_WorksCorrectly) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  const Algorithm algorithm = SimpleRandomAlgorithm();
  Mutator mutator(
      vector<IntegerT>{kIdentityMutationAction}, 1.0,
      {NO_OP, SCALAR_SUM_OP},  // allowed_setup_ops
      {NO_OP, SCALAR_SUM_OP},  // allowed_predict_ops
      {NO_OP, SCALAR_SUM_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
  mutator.Mutate(&mutated_algorithm);
  EXPECT_EQ(*mutated_algorithm, algorithm);
}

TEST(InsertInstructionMutationActionTest, CoversComponentFunctions) {
  const Algorithm no_op_algorithm = SimpleRandomAlgorithm();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {SCALAR_SUM_OP},  // allowed_setup_ops
      {SCALAR_SUM_OP},  // allowed_predict_ops
      {SCALAR_SUM_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, no_op_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
      mutator.Mutate(&mutated_algorithm);
      return DifferentComponentFunction(*mutated_algorithm, no_op_algorithm);
    }),
    {0, 1, 2}, {0, 1, 2}));
}

TEST(InsertInstructionMutationActionTest, CoversSetupPositions) {
  const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
  const IntegerT component_function_size = no_op_algorithm.setup_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {SCALAR_SUM_OP},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  ASSERT_GT(component_function_size, 0);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, no_op_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
      mutator.Mutate(&mutated_algorithm);
      return ScalarSumOpPosition(mutated_algorithm->setup_);
    }),
    Range<IntegerT>(0, component_function_size + 1),
    Range<IntegerT>(0, component_function_size + 1),
    3));
}

TEST(InsertInstructionMutationActionTest, CoversPredictPositions) {
  const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
  const IntegerT component_function_size = no_op_algorithm.predict_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {},  // allowed_setup_ops
      {SCALAR_SUM_OP},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  ASSERT_GT(component_function_size, 0);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, no_op_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
      mutator.Mutate(&mutated_algorithm);
      return ScalarSumOpPosition(mutated_algorithm->predict_);
    }),
    Range<IntegerT>(0, component_function_size + 1),
    Range<IntegerT>(0, component_function_size + 1),
    3));
}

TEST(InsertInstructionMutationActionTest, CoversLearnPositions) {
  const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
  const IntegerT component_function_size = no_op_algorithm.learn_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {SCALAR_SUM_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  ASSERT_GT(component_function_size, 0);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, no_op_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
      mutator.Mutate(&mutated_algorithm);
      return ScalarSumOpPosition(mutated_algorithm->learn_);
    }),
    Range<IntegerT>(0, component_function_size + 1),
    Range<IntegerT>(0, component_function_size + 1),
    3));
}

TEST(InsertInstructionMutationActionTest, InsertsWhenUnderMinSize) {
  const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
  const IntegerT setup_component_function_size = no_op_algorithm.setup_.size();
  const IntegerT predict_component_function_size =
      no_op_algorithm.predict_.size();
  const IntegerT learn_component_function_size = no_op_algorithm.learn_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {SCALAR_SUM_OP},  // allowed_setup_ops
      {SCALAR_SUM_OP},  // allowed_predict_ops
      {SCALAR_SUM_OP},  // allowed_learn_ops
      100, 10000, 100, 10000, 100, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
  mutator.Mutate(&mutated_algorithm);
  EXPECT_TRUE(mutated_algorithm->setup_.size() ==
                  setup_component_function_size + 1 ||
              mutated_algorithm->predict_.size() ==
                  predict_component_function_size + 1 ||
              mutated_algorithm->learn_.size() ==
                  learn_component_function_size + 1);
}

TEST(InsertInstructionMutationActionTest, DoesNotInsertWhenOverMaxSize) {
  const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {SCALAR_SUM_OP},  // allowed_setup_ops
      {SCALAR_SUM_OP},  // allowed_predict_ops
      {SCALAR_SUM_OP},  // allowed_learn_ops
      0, 1, 0, 1, 0, 1,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
  mutator.Mutate(&mutated_algorithm);
  EXPECT_EQ(*mutated_algorithm, no_op_algorithm);
}

TEST(InsertInstructionMutationActionTest, CoversSetupInstructions) {
  const Algorithm empty_algorithm;
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {SCALAR_SUM_OP, SCALAR_DIFF_OP},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, empty_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(empty_algorithm);
      mutator.Mutate(&mutated_algorithm);
      if (mutated_algorithm->setup_.size() == 1) {
        return static_cast<IntegerT>(mutated_algorithm->setup_[0]->op_);
      } else {
        return static_cast<IntegerT>(-1);
      }
    }),
    {SCALAR_SUM_OP, SCALAR_DIFF_OP}, {SCALAR_SUM_OP, SCALAR_DIFF_OP}, 3));
}

TEST(InsertInstructionMutationActionTest, CoversPredictInstructions) {
  const Algorithm empty_algorithm;
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {},  // allowed_setup_ops
      {SCALAR_SUM_OP, SCALAR_DIFF_OP},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, empty_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(empty_algorithm);
      mutator.Mutate(&mutated_algorithm);
      if (mutated_algorithm->predict_.size() == 1) {
        return static_cast<IntegerT>(mutated_algorithm->predict_[0]->op_);
      } else {
        return static_cast<IntegerT>(-1);
      }
    }),
    {SCALAR_SUM_OP, SCALAR_DIFF_OP}, {SCALAR_SUM_OP, SCALAR_DIFF_OP}, 3));
}

TEST(InsertInstructionMutationActionTest, CoversLearnInstructions) {
  const Algorithm empty_algorithm;
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kInsertInstructionMutationAction},
      1.0,  // mutate_prob
      {},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {SCALAR_SUM_OP, SCALAR_DIFF_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, empty_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(empty_algorithm);
      mutator.Mutate(&mutated_algorithm);
      if (mutated_algorithm->learn_.size() == 1) {
        return static_cast<IntegerT>(mutated_algorithm->learn_[0]->op_);
      } else {
        return static_cast<IntegerT>(-1);
      }
    }),
    {SCALAR_SUM_OP, SCALAR_DIFF_OP}, {SCALAR_SUM_OP, SCALAR_DIFF_OP}, 3));
}

TEST(RemoveInstructionMutationActionTest, CoversComponentFunctions) {
  const Algorithm random_algorithm = SimpleRandomAlgorithm();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kRemoveInstructionMutationAction},
      1.0, {NO_OP}, {NO_OP}, {NO_OP},
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, random_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(random_algorithm);
      mutator.Mutate(&mutated_algorithm);
      return DifferentComponentFunction(*mutated_algorithm, random_algorithm);
    }),
    {0, 1, 2}, {0, 1, 2}, 3));
}

TEST(RemoveInstructionMutationActionTest, CoversSetupPositions) {
  const Algorithm algorithm = SimpleIncreasingDataAlgorithm();
  const IntegerT component_function_size = algorithm.setup_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kRemoveInstructionMutationAction},
      1.0,  // mutate_prob
      {NO_OP},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  ASSERT_GT(component_function_size, 0);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
      mutator.Mutate(&mutated_algorithm);
      return MissingDataInComponentFunction(
          algorithm.setup_, mutated_algorithm->setup_);
    }),
    Range<IntegerT>(0, component_function_size),
    Range<IntegerT>(0, component_function_size),
    3));
}

TEST(RemoveInstructionMutationActionTest, CoversPredictPositions) {
  const Algorithm algorithm = SimpleIncreasingDataAlgorithm();
  const IntegerT component_function_size = algorithm.predict_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kRemoveInstructionMutationAction},
      1.0,  // mutate_prob
      {},  // allowed_setup_ops
      {NO_OP},  // allowed_predict_ops
      {},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  ASSERT_GT(component_function_size, 0);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
      mutator.Mutate(&mutated_algorithm);
      return MissingDataInComponentFunction(
          algorithm.predict_, mutated_algorithm->predict_);
    }),
    Range<IntegerT>(0, component_function_size),
    Range<IntegerT>(0, component_function_size),
    3));
}

TEST(RemoveInstructionMutationActionTest, CoversLearnPositions) {
  const Algorithm algorithm = SimpleIncreasingDataAlgorithm();
  const IntegerT component_function_size = algorithm.learn_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kRemoveInstructionMutationAction},
      1.0,  // mutate_prob
      {},  // allowed_setup_ops
      {},  // allowed_predict_ops
      {NO_OP},  // allowed_learn_ops
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  ASSERT_GT(component_function_size, 0);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(algorithm);
      mutator.Mutate(&mutated_algorithm);
      return MissingDataInComponentFunction(
          algorithm.learn_, mutated_algorithm->learn_);
    }),
    Range<IntegerT>(0, component_function_size),
    Range<IntegerT>(0, component_function_size),
    3));
}

TEST(RemoveInstructionMutationActionTest, DoesNotRemoveWhenUnderMinSize) {
  const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kRemoveInstructionMutationAction},
      1.0,  // mutate_prob
      {SCALAR_SUM_OP},  // allowed_setup_ops
      {SCALAR_SUM_OP},  // allowed_predict_ops
      {SCALAR_SUM_OP},  // allowed_learn_ops
      100, 10000, 100, 10000, 100, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
  mutator.Mutate(&mutated_algorithm);
  EXPECT_EQ(*mutated_algorithm, no_op_algorithm);
}

TEST(RemoveInstructionMutationActionTest, RemovesWhenOverMaxSize) {
  const Algorithm no_op_algorithm = SimpleNoOpAlgorithm();
  const IntegerT setup_component_function_size = no_op_algorithm.setup_.size();
  const IntegerT predict_component_function_size =
      no_op_algorithm.predict_.size();
  const IntegerT learn_component_function_size = no_op_algorithm.learn_.size();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kRemoveInstructionMutationAction},
      1.0,  // mutate_prob
      {SCALAR_SUM_OP},  // allowed_setup_ops
      {SCALAR_SUM_OP},  // allowed_predict_ops
      {SCALAR_SUM_OP},  // allowed_learn_ops
      0, 1, 0, 1, 0, 1,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  auto mutated_algorithm = make_shared<const Algorithm>(no_op_algorithm);
  mutator.Mutate(&mutated_algorithm);
  EXPECT_TRUE(mutated_algorithm->setup_.size() ==
                  setup_component_function_size - 1 ||
              mutated_algorithm->predict_.size() ==
                  predict_component_function_size - 1 ||
              mutated_algorithm->learn_.size() ==
                  learn_component_function_size - 1);
}

TEST(TradeInstructionMutationActionTest, CoversComponentFunctions) {
  const Algorithm random_algorithm = SimpleRandomAlgorithm();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kTradeInstructionMutationAction},
      1.0, {NO_OP}, {NO_OP}, {NO_OP},
      0, 10000, 0, 10000, 0, 10000,  // min/max component_function sizes
      &bit_gen, &rand_gen);
  EXPECT_TRUE(IsEventually(
    function<IntegerT(void)>([&mutator, random_algorithm](){
      auto mutated_algorithm = make_shared<const Algorithm>(random_algorithm);
      mutator.Mutate(&mutated_algorithm);
      return DifferentComponentFunction(*mutated_algorithm, random_algorithm);
    }),
    {-1, 0, 1, 2}, {0, 1, 2}, 3));
}

TEST(TradeInstructionMutationActionTest, PreservesSizes) {
  const Algorithm random_algorithm = SimpleRandomAlgorithm();
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Mutator mutator(
      vector<IntegerT>{kTradeInstructionMutationAction},
      1.0, {NO_OP}, {NO_OP}, {NO_OP},
      0, 10000, 0, 10000, 0, 10000,  // min/max prog. sizes
      &bit_gen, &rand_gen);
  auto mutated_algorithm = make_shared<const Algorithm>(random_algorithm);
  mutator.Mutate(&mutated_algorithm);
  EXPECT_EQ(mutated_algorithm->setup_.size(), random_algorithm.setup_.size());
  EXPECT_EQ(mutated_algorithm->predict_.size(),
            random_algorithm.predict_.size());
  EXPECT_EQ(mutated_algorithm->learn_.size(), random_algorithm.learn_.size());
}

TEST(MutatorTest, RandomizeAlgorithm) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  MTRandom bit_gen;
  const Algorithm algorithm = SimpleRandomAlgorithm();
  const IntegerT setup_size = algorithm.setup_.size();
  const IntegerT predict_size = algorithm.predict_.size();
  const IntegerT learn_size = algorithm.learn_.size();
  Mutator mutator(
      vector<IntegerT>{kRandomizeAlgorithmMutationAction},
      1.0,
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      setup_size, setup_size + 1,
      predict_size, predict_size + 1,
      learn_size, learn_size + 1,
      &bit_gen, &rand_gen);
  const IntegerT num_instr = setup_size + predict_size + learn_size;
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm mutated_algorithm = algorithm;
        mutator.RandomizeAlgorithm(&mutated_algorithm);
        return CountDifferentInstructions(mutated_algorithm, algorithm);
      }),
      Range<IntegerT>(0, num_instr + 1), {num_instr}));
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
