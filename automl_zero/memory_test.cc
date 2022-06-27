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

#include "memory.h"

#include <cassert>
#include <iostream>

#include "definitions.h"
#include "util.h"
#include "gtest/gtest.h"

namespace automl_zero {

TEST(MemoryTest, NumAddressesAreWithinLimits) {
  const size_t max_addresses = Pow2(8 * sizeof(AddressT));
  EXPECT_LE(kMaxScalarAddresses, max_addresses);
  EXPECT_LE(kMaxVectorAddresses, max_addresses);
  EXPECT_LE(kMaxMatrixAddresses, max_addresses);
}

TEST(MemoryTest, WipeSetsValuesToZero) {
  const AddressT kSomeAddress = 1;
  const IntegerT kX = 2;
  const IntegerT kY = 3;

  Memory<4> memory;
  memory.scalar_[kSomeAddress] = 2.0;
  memory.vector_[kSomeAddress](kX, 0) = 4.0;
  memory.matrix_[kSomeAddress](kX, kY) = 0.5;
  EXPECT_EQ(memory.scalar_[kSomeAddress], 2.0);
  EXPECT_EQ(memory.vector_[kSomeAddress](kX, 0), 4.0);
  EXPECT_EQ(memory.matrix_[kSomeAddress](kX, kY), 0.5);

  memory.Wipe();
  EXPECT_EQ(memory.scalar_[kSomeAddress], 0.0);
  EXPECT_EQ(memory.vector_[kSomeAddress](kX, 0), 0.0);
  EXPECT_EQ(memory.matrix_[kSomeAddress](kX, kY), 0.0);
}

TEST(MemoryTest, RespectsFeaturesSize) {
  Memory<4> memory4;
  EXPECT_EQ(memory4.vector_[0].size(), 4);
  EXPECT_EQ(memory4.matrix_[0].rows(), 4);
  EXPECT_EQ(memory4.matrix_[0].cols(), 4);

  Memory<8> memory8;
  EXPECT_EQ(memory8.vector_[0].size(), 8);
  EXPECT_EQ(memory8.matrix_[0].rows(), 8);
  EXPECT_EQ(memory8.matrix_[0].cols(), 8);
}

}  // namespace automl_zero
