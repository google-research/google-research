// Copyright 2022 The Google Research Authors.
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

#include "countmin.h"

#include <cstdio>
#include <vector>

#include "gtest/gtest.h"
#include "utils.h"

namespace sketch {
namespace {

class CountMinTest : public ::testing::Test {
 public:
  CountMinTest() {}
};

TEST_F(CountMinTest, TestAdd) {
  CountMin countmin(5, 8);
  countmin.Add(8, 3);
  EXPECT_LE(3, countmin.Estimate(8));
  countmin.Add(8, 2);
  EXPECT_LE(5, countmin.Estimate(8));
  countmin.Add(10, 2);
  EXPECT_LE(2, countmin.Estimate(10));
  countmin.Add(12, 3);
  EXPECT_LE(3, countmin.Estimate(12));
}

TEST_F(CountMinTest, TestMerge) {
  CountMin countmin(5, 8);
  CountMin countmin2(countmin);
  countmin.Add(8, 3);
  countmin.Add(8, 2);
  countmin.Add(10, 2);
  countmin.Add(12, 3);
  EXPECT_LE(3, countmin.Estimate(12));
  EXPECT_EQ(0, countmin2.Estimate(12));
  countmin2.Add(8, 3);
  countmin2.Add(8, 2);
  countmin2.Add(10, 2);
  countmin2.Add(12, 3);
  EXPECT_LE(5, countmin2.Estimate(8));
  countmin.Merge(countmin2);
  EXPECT_LE(10, countmin.Estimate(8));
  EXPECT_LE(4, countmin.Estimate(10));
  EXPECT_LE(6, countmin.Estimate(12));
  countmin.Reset();
  EXPECT_EQ(0, countmin.Estimate(8));
  EXPECT_EQ(0, countmin.Estimate(10));
  EXPECT_EQ(0, countmin.Estimate(12));
  countmin.Merge(countmin2);
  countmin.Merge(countmin2);
  EXPECT_LE(10, countmin.Estimate(8));
  EXPECT_LE(4, countmin.Estimate(10));
  EXPECT_LE(6, countmin.Estimate(12));
}


TEST_F(CountMinTest, TestSize) {
  // Fixed cost for 5 hashes = 248 bytes. Variable cost = 5 * 5 * hash_size.
  EXPECT_EQ(440, CountMin(5, 8).Size());
  EXPECT_EQ(41240, CountMin(5, 2048).Size());
}

class CountMinCUTest : public ::testing::Test {
 public:
  CountMinCUTest() {}
};

TEST_F(CountMinCUTest, TestAdd) {
  CountMinCU countmin(5, 8);
  countmin.Add(8, 3);
  EXPECT_LE(3, countmin.Estimate(8));

  CountMin cp(countmin);
  EXPECT_LE(3, cp.Estimate(8));
  cp.Add(9, 3);
  cp.Add(9, 3);
}

class CountMinHierarchicalTest : public ::testing::Test {
 public:
  CountMinHierarchicalTest() {}
};

TEST_F(CountMinHierarchicalTest, TestAdd) {
  CountMinHierarchical countmin(3, 4, 5, 1);
  countmin.Add(8, 3);
  EXPECT_LE(3, countmin.Estimate(8));
  countmin.Add(8, 2);
  EXPECT_LE(5, countmin.Estimate(8));
  countmin.Add(3, 2);
  EXPECT_LE(2, countmin.Estimate(3));
  countmin.Add(25, 3);
  EXPECT_LE(3, countmin.Estimate(25));

  std::vector<uint> hh = countmin.HeavyHitters(2.5);
  for (auto h : hh) {
    printf("HH %d\n", h);
  }
  CountMinHierarchical countmin2(countmin);
  EXPECT_LE(5, countmin2.Estimate(8));
  countmin.Merge(countmin2);
  EXPECT_LE(10, countmin.Estimate(8));
  EXPECT_LE(4, countmin.Estimate(3));
  EXPECT_LE(6, countmin.Estimate(25));
  hh.clear();
  hh = countmin.HeavyHitters(5);
  for (auto h : hh) {
    printf("HH %d\n", h);
  }
}

TEST_F(CountMinHierarchicalTest, TestAddCU) {
  CountMinHierarchicalCU countmin(3, 4, 5, 1);
  countmin.Add(8, 3);
  EXPECT_LE(3, countmin.Estimate(8));
  countmin.Add(8, 2);
  EXPECT_LE(5, countmin.Estimate(8));
  countmin.Add(3, 2);
  EXPECT_LE(2, countmin.Estimate(3));
  countmin.Add(25, 3);
  EXPECT_LE(3, countmin.Estimate(25));

  std::vector<uint> hh = countmin.HeavyHitters(2.5);
  for (auto h : hh) {
    printf("HH %d\n", h);
  }
  CountMinHierarchicalCU countmin2(countmin);
  EXPECT_LE(5, countmin2.Estimate(8));
  countmin.Merge(countmin2);
  EXPECT_LE(10, countmin.Estimate(8));
  EXPECT_LE(4, countmin.Estimate(3));
  EXPECT_LE(6, countmin.Estimate(25));
  hh.clear();
  hh = countmin.HeavyHitters(5);
  for (auto h : hh) {
    printf("HH %d\n", h);
  }
}

}  // namespace
}  // namespace sketch
