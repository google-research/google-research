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

#include "test_util.h"

#include <functional>
#include <iostream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/node_hash_set.h"
#include "definitions.h"

namespace automl_zero {

using ::std::function;  // NOLINT;
using ::std::pair;  // NOLINT;
// NOLINT;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

IntegerT cycle_up_to_five() {
  static IntegerT x = 0;
  return x++ % 5;
}

TEST(IsEventuallyTest, CorrectAnswer) {
  EXPECT_TRUE(IsEventually(
      function<IntegerT()>(cycle_up_to_five),
      {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}));
}

TEST(IsEventuallyTest, DisallowedNumber) {
  EXPECT_FALSE(IsEventually(
      function<IntegerT()>(cycle_up_to_five),
      {0, 1, 2, 4}, {0, 1, 2, 3, 4}));
}

TEST(IsEventuallyTest, MissingRequiredNumber) {
  EXPECT_FALSE(IsEventually(
      function<IntegerT()>(cycle_up_to_five),
      {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}));
}

TEST(IsEventuallyTest, NotRequiredNumber) {
  EXPECT_TRUE(IsEventually(
      function<IntegerT()>(cycle_up_to_five),
      {0, 1, 2, 3, 4}, {0, 1, 2, 4}));
}

TEST(IsEventuallyTest, MissingAllowedNumber) {
  EXPECT_TRUE(IsEventually(
      function<IntegerT()>(cycle_up_to_five),
      {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4}));
}

TEST(IsNeverTest, ExcludedValue) {
  EXPECT_FALSE(IsNever(
      function<IntegerT()>(cycle_up_to_five),
      {3}, 3.0));
}

TEST(IsNeverTest, NotExcludedValue) {
  EXPECT_TRUE(IsNever(
      function<IntegerT()>(cycle_up_to_five),
      {6}, 3.0));
}

TEST(RangeTest, WorksCorrectly) {
  EXPECT_THAT(Range<IntegerT>(0, 5), UnorderedElementsAre(0, 1, 2, 3, 4));
}

TEST(CartesianProductTest, WorksCorrectly) {
  EXPECT_THAT(CartesianProduct(Range<IntegerT>(0, 3),
                               Range<IntegerT>(3, 5)),
              UnorderedElementsAre(Pair(0, 3), Pair(0, 4),
                                   Pair(1, 3), Pair(1, 4),
                                   Pair(2, 3), Pair(2, 4)));
}

}  // namespace automl_zero
