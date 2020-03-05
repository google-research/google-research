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

#include "algorithm.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

#include "task.h"
#include "task_util.h"
#include "definitions.h"
#include "executor.h"
#include "generator_test_util.h"
#include "memory.h"
#include "random_generator.h"
#include "test_util.h"
#include "gtest/gtest.h"

namespace automl_zero {
namespace {

using ::std::make_shared;  // NOLINT

TEST(AlgorithmTest, DefaultConstructionProducesCorrectComponentFunctionSizes) {
  Algorithm algorithm;
  EXPECT_EQ(algorithm.setup_.size(), 0);
  EXPECT_EQ(algorithm.predict_.size(), 0);
  EXPECT_EQ(algorithm.learn_.size(), 0);
}

TEST(AlgorithmTest, CopyConstructor) {
  Algorithm algorithm = SimpleRandomAlgorithm();
  Algorithm algorithm_copy = algorithm;
  EXPECT_TRUE(algorithm_copy == algorithm);
}

TEST(AlgorithmTest, CopyAssignmentOperator) {
  Algorithm algorithm = SimpleRandomAlgorithm();
  Algorithm algorithm_copy;
  algorithm_copy = algorithm;
  EXPECT_TRUE(algorithm_copy == algorithm);
}

TEST(AlgorithmTest, MoveConstructor) {
  Algorithm algorithm = SimpleRandomAlgorithm();
  Algorithm algorithm_copy = algorithm;
  Algorithm algorithm_move = std::move(algorithm);
  EXPECT_TRUE(algorithm_move == algorithm_copy);
}

TEST(AlgorithmTest, MoveAssignmentOperator) {
  Algorithm algorithm = SimpleRandomAlgorithm();
  Algorithm algorithm_copy = algorithm;
  Algorithm algorithm_move;
  algorithm_move = std::move(algorithm);
  EXPECT_TRUE(algorithm_move == algorithm_copy);
}

TEST(AlgorithmTest, CopyAssignmentOperator_SelfCopy) {
  Algorithm algorithm = SimpleRandomAlgorithm();
  Algorithm algorithm_copy = algorithm;
  algorithm_copy = algorithm_copy;
  EXPECT_TRUE(algorithm_copy == algorithm);
}

TEST(AlgorithmTest, EqualsOperator) {
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.predict_[1] =
      make_shared<const Instruction>(VECTOR_SUM_OP, 1, 2, 3);

  Algorithm algorithm_same = SimpleNoOpAlgorithm();
  algorithm_same.predict_[1] =
      make_shared<const Instruction>(VECTOR_SUM_OP, 1, 2, 3);

  Algorithm algorithm_different_instruction = SimpleNoOpAlgorithm();
  algorithm_different_instruction.predict_[1] =
      make_shared<const Instruction>(VECTOR_SUM_OP, 1, 1, 3);

  Algorithm algorithm_different_position = SimpleNoOpAlgorithm();
  algorithm_different_position.predict_[0] =
      make_shared<const Instruction>(VECTOR_SUM_OP, 1, 2, 3);

  Algorithm algorithm_different_component_function = SimpleNoOpAlgorithm();
  algorithm_different_component_function.learn_[0] =
      make_shared<const Instruction>(VECTOR_SUM_OP, 1, 2, 3);

  EXPECT_TRUE(algorithm == algorithm_same);
  EXPECT_FALSE(algorithm != algorithm_same);
  EXPECT_FALSE(algorithm == algorithm_different_instruction);
  EXPECT_TRUE(algorithm != algorithm_different_instruction);
  EXPECT_FALSE(algorithm == algorithm_different_position);
  EXPECT_TRUE(algorithm != algorithm_different_position);
  EXPECT_FALSE(algorithm == algorithm_different_component_function);
  EXPECT_TRUE(algorithm != algorithm_different_component_function);

  Algorithm random_algorithm = SimpleRandomAlgorithm();
  Algorithm same_random_algorithm = random_algorithm;
  EXPECT_TRUE(random_algorithm == same_random_algorithm);
  EXPECT_FALSE(random_algorithm != same_random_algorithm);
}

TEST(AlgorithmTest, ToFromProto) {
  Algorithm algorithm_src = SimpleRandomAlgorithm();
  Algorithm algorithm_dest;
  algorithm_dest = SimpleNoOpAlgorithm();
  algorithm_dest.FromProto(algorithm_src.ToProto());
  EXPECT_TRUE(algorithm_dest == algorithm_src);
}

TEST(AlgorithmTest, ToFromProtoIntoDifferentComponentFunctionSizes) {
  Algorithm algorithm_src = SimpleRandomAlgorithm();
  Algorithm algorithm_dest;
  algorithm_dest.FromProto(algorithm_src.ToProto());
  EXPECT_TRUE(algorithm_dest == algorithm_src);
}

}  // namespace
}  // namespace automl_zero
