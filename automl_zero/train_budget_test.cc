// Copyright 2021 The Google Research Authors.
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

#include "train_budget.h"

#include "algorithm.h"
#include "generator_test_util.h"
#include "gtest/gtest.h"

namespace automl_zero {

TEST(TrainBudgetTest, MatchesBaselineExactly) {
  Algorithm baseline_algorithm = SimpleGrTildeGrWithBias();
  TrainBudget train_budget(baseline_algorithm, 2.0);
  Algorithm algorithm = baseline_algorithm;
  EXPECT_EQ(train_budget.TrainExamples(algorithm, 100), 100);
  EXPECT_EQ(train_budget.TrainExamples(algorithm, 1000), 1000);
}

TEST(TrainBudgetTest, CheaperModelIsUnaffected) {
  Algorithm baseline_algorithm = SimpleGrTildeGrWithBias();
  TrainBudget train_budget(baseline_algorithm, 2.0);
  Algorithm algorithm = SimpleGz();
  EXPECT_EQ(train_budget.TrainExamples(algorithm, 100), 100);
  EXPECT_EQ(train_budget.TrainExamples(algorithm, 1000), 1000);
}

TEST(TrainBudgetTest, MoreExpensiveModelIsRejected) {
  Algorithm baseline_algorithm = SimpleGz();
  TrainBudget train_budget(baseline_algorithm, 2.0);
  Algorithm algorithm = SimpleGrTildeGrWithBias();
  EXPECT_EQ(train_budget.TrainExamples(algorithm, 100), 0);
  EXPECT_EQ(train_budget.TrainExamples(algorithm, 1000), 0);
}

}  // namespace automl_zero
