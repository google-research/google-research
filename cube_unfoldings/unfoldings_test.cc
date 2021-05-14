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

#include "unfoldings.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace cube_unfoldings {
namespace {

TEST(UnfoldingsTest, TestNoMatchings) {
  EXPECT_EQ(NumberOfUniqueMatchings(":K`ACGO`ACG^"), 0);
}

TEST(UnfoldingsTest, TestSmall) {
  EXPECT_EQ(NumberOfUniqueMatchings(":GaYmLz"), 24);
}

TEST(UnfoldingsTest, TestMedium) {
  EXPECT_EQ(NumberOfUniqueMatchings(":K`EKIS`]{G^"), 52);
}

TEST(UnfoldingsTest, TestLarge) {
  EXPECT_EQ(NumberOfUniqueMatchings(":O`ESxqlISWoxuF"), 104512);
}

}  // namespace
}  // namespace cube_unfoldings
