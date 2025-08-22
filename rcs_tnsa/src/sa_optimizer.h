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

#ifndef SA_OPTIMIZER_H
#define SA_OPTIMIZER_H

#include <algorithm>
#include <random>
#include <tuple>

#include "tensor_network.h"  // NOLINT(build/include)

class SAOptimizer {
 public:
  SAOptimizer() = default;

  SAOptimizer(const TensorNetwork& tn);

  void SetSeed(unsigned int seed);

  double Optimize(double t0, double t1, size_t num_ts,
                  std::vector<size_t>& ordering);

  std::tuple<double, size_t> SlicedOptimizeGroupedSlicesSparseOutput(
      double t0, double t1, size_t num_ts, double max_width,
      size_t num_output_confs, std::vector<size_t>& ordering,
      std::vector<std::vector<size_t>>& slices_groups_ordering);

  std::tuple<double, size_t> SlicedOptimizeFullMemoryGroupedSlicesSparseOutput(
      double t0, double t1, size_t num_ts, double log2_max_memory,
      size_t num_output_confs, std::vector<size_t>& ordering,
      std::vector<std::vector<size_t>>& slices_groups_ordering);

 private:
  TensorNetwork tn_;
  size_t seed_;
  std::default_random_engine dre_;
};

void SwapMove(std::vector<size_t>& ordering);

template <typename T>
void SlicesMove(std::vector<T>& slices_ordering, size_t num_slices);

std::vector<double> LinInvSchedule(double t0, double t1, double num_ts);

#endif
