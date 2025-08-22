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

#pragma once

#include <memory>

#include "exposure_design.h"
#include "weighted_distribution.h"

namespace exposure_design {

using IndexedSparseVector = std::unordered_map<OutcomeIndex, double>;

struct Cluster {
  // Unweighted fraction of the edges accounted for by this cluster.
  double degree_fraction{0};
  // Squared L2 norm of the member `weights` below.
  double squared_l2_norm{0};
  // L1 norm of `weights`.
  double l1_norm{0};
  // Sum of the outcome weights of diversion units in this cluster.
  IndexedSparseVector weights;
  // Diversion units in this cluster. Not updated during the local search;
  // computed at the end.
  DiversionCluster members;
};

struct DiversionUnit;

struct OutcomeUnit {
  WeightedDistribution<DiversionUnit*> neighbor_dist;
};

struct DiversionUnit {
  DiversionId id;
  double degree_fraction{0};
  double squared_l2_norm{0};
  double l1_norm{0};
  const SparseVector* weights{nullptr};
  Cluster* cluster{nullptr};
  WeightedDistribution<OutcomeUnit*> neighbor_dist;
};

constexpr double Squared(double x) { return x * x; }

template <typename GenericSparseVector>
double SquaredL2Norm(const GenericSparseVector& x) {
  double sum{0};
  for (auto [_, x_i] : x) sum += Squared(x_i);
  return sum;
}

// Assumes x >= 0.
template <typename GenericSparseVector>
double L1Norm(const GenericSparseVector& x) {
  double sum{0};
  for (auto [i, x_i] : x) sum += x_i;
  return sum;
}

inline double DotProduct(const SparseVector& x, const IndexedSparseVector& y) {
  double sum{0};
  for (auto [i, x_i] : x)
    if (auto it{y.find(i)}; it != y.end()) sum += x_i * it->second;
  return sum;
}

inline void AddInPlace(IndexedSparseVector& x, const SparseVector& y) {
  for (auto [i, y_i] : y) x[i] += y_i;
}

inline void SubtractInPlace(IndexedSparseVector& x, const SparseVector& y) {
  for (auto [i, y_i] : y) x[i] -= y_i;
}

// The return value contains pointers into `instance`.
std::vector<std::unique_ptr<DiversionUnit>> MakeDiversionUnits(
    const Instance& instance);

void MoveUnitToCluster(DiversionUnit& unit, Cluster* new_cluster);

std::vector<std::unique_ptr<Cluster>> MakeSingletonClusters(
    const std::vector<std::unique_ptr<DiversionUnit>>& units);

inline double ObjectiveTerm(const Cluster& cluster,
                            const Parameters& parameters) {
  return (1 + parameters.phi) * cluster.squared_l2_norm -
         parameters.phi * Squared(cluster.l1_norm);
}

double Objective(const std::vector<std::unique_ptr<Cluster>>& clusters,
                 const Parameters& parameters);

// Returns true if it moves `unit`.
bool MoveIfNotWorse(DiversionUnit& unit, Cluster* new_cluster,
                    const Parameters& parameters);

DiversionClustering ExtractClustering(
    const std::vector<std::unique_ptr<DiversionUnit>>& units,
    const std::vector<std::unique_ptr<Cluster>>& clusters,
    const Parameters& parameters);

}  // namespace exposure_design
