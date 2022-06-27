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

#include "perfect_matchings.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace cube_unfoldings {
namespace {
using testing::UnorderedElementsAreArray;

void TestAllPerfectMatchings(const std::vector<Edge>& edges, size_t n,
                             const std::vector<Matching>& matchings) {
  std::vector<Matching> found;
  auto cb = [&found](const Matching& m) { found.push_back(m); };
  AllPerfectMatchings(edges, n, cb);
  EXPECT_THAT(found, UnorderedElementsAreArray(matchings));
}

TEST(PerfectMatchingsTest, AllPerfectMatchingsSimple) {
  std::vector<Edge> edges = {{0, 1}};
  std::vector<Matching> matchings = {{{0, 1}}};
  TestAllPerfectMatchings(edges, 2, matchings);
}

TEST(PerfectMatchingsTest, AllPerfectMatchingsK3) {
  std::vector<Edge> edges = {{0, 1}, {0, 2}, {1, 2}};
  std::vector<Matching> matchings = {};
  TestAllPerfectMatchings(edges, 3, matchings);
}

TEST(PerfectMatchingsTest, AllPerfectMatchingsK4) {
  std::vector<Edge> edges = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  std::vector<Matching> matchings = {
      {{0, 1}, {2, 3}}, {{0, 2}, {1, 3}}, {{0, 3}, {1, 2}}};
  TestAllPerfectMatchings(edges, 4, matchings);
}

}  // namespace
}  // namespace cube_unfoldings
