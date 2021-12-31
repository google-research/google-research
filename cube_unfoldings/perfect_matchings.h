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

#ifndef CUBE_UNFOLDINGS_PERFECT_MATCHINGS_H_
#define CUBE_UNFOLDINGS_PERFECT_MATCHINGS_H_
#include <array>
#include <vector>

#include "absl/functional/function_ref.h"

namespace cube_unfoldings {
using Edge = std::array<int, 2>;
using Matching = std::vector<Edge>;

// Calls `cb` for all the perfect matchings in the graph with `n` nodes and
// edges defined by `edges`.
// Matchings are returned in "canonical form", i.e. with all edges in ascending
// order.
void AllPerfectMatchings(const std::vector<Edge>& edges, size_t n,
                         absl::FunctionRef<void(const Matching&)> cb);
}  // namespace cube_unfoldings

#endif  // CUBE_UNFOLDINGS_PERFECT_MATCHINGS_H_
