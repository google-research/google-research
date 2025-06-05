// Copyright 2025 The Google Research Authors.
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

#ifndef FORMULATION_H_
#define FORMULATION_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "graph.h"

namespace geo_algorithms {

struct Formulation {
  const Graph G;
  const absl::flat_hash_map<ArcIndex, int64_t> L;
  const absl::flat_hash_map<ArcIndex, int64_t> U;
  const absl::flat_hash_map<ArcIndex, int64_t> T;
  const absl::flat_hash_set<ArcIndex> P;
  const int path_src = -1;
  const int path_dst = -1;
};

// all flow and residual solution maps encoded sparsely.
// If edge is not present, value is 0.
struct ResidualSolution {
  absl::flat_hash_map<ArcIndex, int64_t> df;
  absl::flat_hash_map<ResidualIndex, int64_t> df_residual;
};

struct FlowSolution {
  absl::flat_hash_map<ArcIndex, int64_t> f;
  // exception: not present means that value is T(e)
  absl::flat_hash_map<ArcIndex, int64_t> a;
  absl::flat_hash_map<ArcIndex, int64_t> b;
};

// no default values for cut solution
struct CutSolution {
  absl::flat_hash_map<ArcIndex, int64_t> w;
  absl::flat_hash_map<int, int64_t> d;
};

}  // namespace geo_algorithms

#endif  // FORMULATION_H_
