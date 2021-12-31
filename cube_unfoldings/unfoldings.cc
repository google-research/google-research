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

#include <assert.h>
#include <stdio.h>

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nauty_wrapper.h"
#include "perfect_matchings.h"

namespace cube_unfoldings {
namespace {
// Writes a matching in "canonical form": edges are sorted so that the first
// node is < the second one, and all edges are in ascending order.
void Canonicalize(Matching& pm) {
  for (size_t i = 0; i < pm.size(); i++) {
    int a = pm[i][0];
    int b = pm[i][1];
    if (a > b) std::swap(a, b);
    pm[i][0] = a;
    pm[i][1] = b;
  }
  std::sort(pm.begin(), pm.end());
}

// Find all the matchings that are equivalent to the given one under
// automorphism of the tree. All matchings are added to `all_found_matchings`.
void FindAllEquivalentMatchings(
    const Matching& pm, const std::vector<Generator>& tree_autom_gen,
    absl::flat_hash_set<Matching>* all_found_matchings) {
  absl::flat_hash_set<Matching> matchings;
  all_found_matchings->insert(pm);
  matchings.insert(std::move(pm));
  Matching new_m;
  std::vector<Matching> new_matchings;
  // Finds the closure of the current set of equivalent matchings under the
  // tree automorphism generators `tree_autom_gen`, by iteratively applying
  // the generators until no new matching is produced.
  while (true) {
    size_t n = matchings.size();
    for (size_t i = 0; i < tree_autom_gen.size(); i++) {
      new_matchings.clear();
      for (const Matching& m : matchings) {
        new_m = m;
        for (size_t j = 0; j < new_m.size(); j++) {
          new_m[j][0] = tree_autom_gen[i][new_m[j][0]];
          new_m[j][1] = tree_autom_gen[i][new_m[j][1]];
        }
        Canonicalize(new_m);
        if (!matchings.contains(new_m)) new_matchings.emplace_back(new_m);
      }
      for (Matching& m : new_matchings) {
        all_found_matchings->insert(m);
        matchings.insert(std::move(m));
      }
    }
    if (n == matchings.size()) break;
  }
}
}  // namespace

size_t NumberOfUniqueMatchings(const char* tree) {
  auto tree_data = TreeData::FromString(tree);
  absl::flat_hash_set<Matching> all_found_matchings;
  size_t num = 0;
  auto cb = [&](const Matching& matching) {
    if (all_found_matchings.contains(matching)) return;
    num++;
    FindAllEquivalentMatchings(matching, tree_data.generators,
                               &all_found_matchings);
  };
  AllPerfectMatchings(tree_data.complement_edges, tree_data.n, cb);

  return num;
}
}  // namespace cube_unfoldings
