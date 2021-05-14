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

#include "nauty_wrapper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace cube_unfoldings {
namespace {
using testing::ElementsAreArray;
using testing::UnorderedElementsAreArray;

TEST(NautyWrapperTest, TestGraph) {
  auto data = TreeData::FromString(":GaYmLz");
  EXPECT_EQ(data.n, 8);
  std::vector<Edge> complement_edges = {
      {0, 2}, {0, 3}, {0, 4}, {0, 6}, {0, 7}, {1, 3}, {1, 4},
      {1, 5}, {1, 6}, {1, 7}, {2, 4}, {2, 5}, {2, 6}, {2, 7},
      {3, 5}, {3, 6}, {3, 7}, {4, 5}, {4, 6}, {4, 7}, {5, 7}};
  EXPECT_THAT(data.complement_edges, ElementsAreArray(complement_edges));
}

TEST(NautyWrapperTest, TestGenerators) {
  auto data = TreeData::FromString(":O`ESxrdBE\\X`AF");
  EXPECT_EQ(data.n, 16);
  // Check that applying automorphism group generators doesn't change the
  // (complement of the) tree.
  for (size_t i = 0; i < data.generators.size(); i++) {
    std::vector<Edge> new_edges;
    for (size_t j = 0; j < data.complement_edges.size(); j++) {
      int a = data.generators[i][data.complement_edges[j][0]];
      int b = data.generators[i][data.complement_edges[j][1]];
      if (a > b) std::swap(a, b);
      new_edges.push_back({a, b});
    }
    EXPECT_THAT(new_edges, UnorderedElementsAreArray(data.complement_edges));
  }
}

}  // namespace
}  // namespace cube_unfoldings
