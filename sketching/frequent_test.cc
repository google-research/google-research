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

#include "frequent.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "utils.h"

namespace sketch {
namespace {

class FrequentTest : public ::testing::Test {
 public:
  FrequentTest() : freq_(new Frequent(100)) {}

 protected:
  std::unique_ptr<Frequent> freq_;
};

TEST_F(FrequentTest, TestBasic) {
  for (uint i = 0; i < 200; ++i) {
    freq_->Add(i, 4);
  }
  EXPECT_EQ(4.0, freq_->Estimate(10));
  freq_->Add(12, 3);
  EXPECT_EQ(4.0, freq_->Estimate(10));
  EXPECT_EQ(7.0, freq_->Estimate(12));
  EXPECT_THAT(freq_->HeavyHitters(6.9), testing::Contains(12));
  freq_->Add(10, 4);
  EXPECT_EQ(8.0, freq_->Estimate(10));
  EXPECT_EQ(7.0, freq_->Estimate(12));
  EXPECT_THAT(freq_->HeavyHitters(7.9), testing::Contains(10));
}

}  // namespace
}  // namespace sketch
