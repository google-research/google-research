// Copyright 2020 The Google Research Authors.
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

// This file contains the definitions of parallel primitives and utilities
// which exploit multi-core parallelism.

#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_UTILS_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_UTILS_H_

#include <functional>

#include "external/gbbs/pbbslib/utilities.h"

namespace research_graph {
namespace parallel {

// Forks two sub-calls that do not return any data. Each sub-call is run in a
// separate fiber. The function is synchronous and will return only after both
// left() and right() complete. If do_parallel is false, it first evaluates
// left() and then right().
static inline void ParDo(const std::function<void()>& left,
                         const std::function<void()>& right,
                         bool do_parallel = true) {
  if (do_parallel) {
    pbbs::par_do(left, right);
  } else {
    left();
    right();
  }
}

// Runs f(start), ..., f(end) in parallel by recursively splitting the loop
// using par_do(). The function is synchronous and returns only after all loop
// iterations complete. The recursion stops and uses a sequential loop once the
// number of iterations in the subproblem becomes <= granularity.
static inline void ParallelForSplitting(size_t start, size_t end,
                                        size_t granularity,
                                        const std::function<void(size_t)>& f) {
  pbbs::parallel_for(start, end, f, granularity);
}

}  // namespace parallel
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_UTILS_H_
