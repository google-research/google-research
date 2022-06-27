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

#include "lossy_weight.h"
#include "gtest/gtest.h"

namespace sketch {
namespace {

class LossyWeightTest : public ::testing::Test {
 public:
  LossyWeightTest() {
    lw_.reset(new LossyWeight(100, 2, 10));
  }

 protected:
  std::unique_ptr<LossyWeight> lw_;
};


TEST_F(LossyWeightTest, TestHashing) {
  for (uint i = 0; i < 200; ++i) {
    lw_->Add(i, 0.1);
  }
  lw_->Add(12, 3);
  lw_->Add(10, 4);
  lw_->ReadyToEstimate();
  std::vector<uint> items = lw_->HeavyHitters(3);
  EXPECT_EQ(2, items.size());
  EXPECT_FLOAT_EQ(4.1, lw_->Estimate(10));
  EXPECT_FLOAT_EQ(3.1, lw_->Estimate(12));

  LossyWeight lw2(*lw_);
  EXPECT_FLOAT_EQ(4.1, lw2.Estimate(10));
  EXPECT_FLOAT_EQ(3.1, lw2.Estimate(12));
  lw2.Merge(*lw_);
  EXPECT_FLOAT_EQ(8.2, lw2.Estimate(10));
  EXPECT_FLOAT_EQ(6.2, lw2.Estimate(12));
  EXPECT_FLOAT_EQ(0.2, lw2.Estimate(11));
}

}  // namespace
}  // namespace sketch
