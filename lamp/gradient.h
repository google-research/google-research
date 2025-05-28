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

#ifndef LAMP_GRADIENT_H
#define LAMP_GRADIENT_H

#include <vector>

#include "common.h"  // NOLINT(build/include)

// How to handle unit-simplex constraints
enum OptimMethod { PROJECT = 0, NORMALIZE = 1, GREEDY = 2, BALANCE = 3 };

struct OptimOptions {
  int max_outer_iter, max_weight_iter, max_transitions_iter;
  double tolerance;  // Stop optimization if no parameter changed by more than
                     // this amount.
  double learning_rate;
  OptimMethod method;
  int debug_level;
};

WeightVector OptimizeWeights(const TrailsSlice& trails,
                             const Model& input_model,
                             const OptimOptions& options, int outer_iter);

SparseMatrix OptimizeTransitions(const TrailsSlice& trails,
                                 const Model& input_model,
                                 const OptimOptions& options, int outer_iter);

Model Optimize(const TrailsSlice& trails, const Model& initial_model,
               const OptimOptions& options, const TrailsSlice& test_trails,
               const char* alg_name, EvalMap* eval_map);

// Occurences of 1 item in 1 trail.
struct TrailPositions {
  TrailPositions(const Trail* the_trail = nullptr) : trail(the_trail) {}

  const Trail* trail;
  std::vector<int> positions;
};
// All occcurences of 1 item.
typedef std::vector<TrailPositions> PostingList;
// Indexed by item.
typedef std::vector<PostingList> PositionIndex;

PositionIndex BuildIndex(const TrailsSlice& trails, int num_locations);

#endif
