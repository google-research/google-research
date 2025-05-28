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

#include "exposure_design.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exposure_design_internal.h"
#include "weighted_distribution.h"

namespace exposure_design {

DiversionClustering ComputeClustering(const Instance& instance,
                                      const Parameters& parameters) {
  std::vector<std::unique_ptr<DiversionUnit>> diversion_units{
      MakeDiversionUnits(instance)};
  std::vector<std::unique_ptr<Cluster>> clusters{
      MakeSingletonClusters(diversion_units)};
  // Construct neighbor distributions for wedge sampling.
  std::unordered_map<OutcomeIndex, OutcomeUnit> outcome_units;
  for (const auto& diversion_unit : diversion_units)
    for (auto [i, weights_i] : *diversion_unit->weights)
      outcome_units[i].neighbor_dist.Add(diversion_unit.get(), weights_i);
  for (const auto& diversion_unit : diversion_units)
    for (auto [i, weights_i] : *diversion_unit->weights)
      diversion_unit->neighbor_dist.Add(&outcome_units.at(i), weights_i);
  // Perform local search.
  std::mt19937 gen{std::random_device{}()};
  for (int t{0}; t < parameters.T; t++) {
    if (parameters.verbose)
      std::cerr << "iteration " << t << ", objective "
                << Objective(clusters, parameters) << "\n";
    std::shuffle(diversion_units.begin(), diversion_units.end(), gen);
    for (const auto& unit : diversion_units)
      MoveIfNotWorse(*unit,
                     unit->neighbor_dist(gen)->neighbor_dist(gen)->cluster,
                     parameters);
  }
  return ExtractClustering(diversion_units, clusters, parameters);
}

std::vector<std::unique_ptr<DiversionUnit>> MakeDiversionUnits(
    const Instance& instance) {
  std::size_t total_degree{0};
  for (const auto& [id, weights] : instance.diversion_units)
    total_degree += weights.size();
  std::vector<std::unique_ptr<DiversionUnit>> units;
  units.reserve(instance.diversion_units.size());
  for (const auto& [id, weights] : instance.diversion_units)
    units.push_back(std::make_unique<DiversionUnit>(DiversionUnit{
        .id{id},
        .degree_fraction{static_cast<double>(weights.size()) /
                         static_cast<double>(total_degree)},
        .squared_l2_norm{SquaredL2Norm(weights)},
        .l1_norm{L1Norm(weights)},
        .weights{&weights},
        .neighbor_dist{},
    }));
  return units;
}

void MoveUnitToCluster(DiversionUnit& unit, Cluster* new_cluster) {
  assert(new_cluster != nullptr);
  Cluster* old_cluster{unit.cluster};
  if (old_cluster != nullptr) {
    SubtractInPlace(old_cluster->weights, *unit.weights);
    old_cluster->l1_norm -= unit.l1_norm;
    old_cluster->squared_l2_norm -=
        unit.squared_l2_norm +
        2 * DotProduct(*unit.weights, old_cluster->weights);
    old_cluster->degree_fraction -= unit.degree_fraction;
  }
  unit.cluster = new_cluster;
  new_cluster->degree_fraction += unit.degree_fraction;
  new_cluster->squared_l2_norm +=
      unit.squared_l2_norm +
      2 * DotProduct(*unit.weights, new_cluster->weights);
  new_cluster->l1_norm += unit.l1_norm;
  AddInPlace(new_cluster->weights, *unit.weights);
}

std::vector<std::unique_ptr<Cluster>> MakeSingletonClusters(
    const std::vector<std::unique_ptr<DiversionUnit>>& units) {
  std::vector<std::unique_ptr<Cluster>> clusters;
  clusters.reserve(units.size());
  for (const auto& unit : units) {
    auto cluster{std::make_unique<Cluster>()};
    MoveUnitToCluster(*unit, cluster.get());
    clusters.push_back(std::move(cluster));
  }
  return clusters;
}

double Objective(const std::vector<std::unique_ptr<Cluster>>& clusters,
                 const Parameters& parameters) {
  double objective{0};
  for (const auto& cluster : clusters)
    objective += ObjectiveTerm(*cluster, parameters);
  return objective;
}

bool MoveIfNotWorse(DiversionUnit& unit, Cluster* new_cluster,
                    const Parameters& parameters) {
  assert(new_cluster != nullptr);
  Cluster* old_cluster{unit.cluster};
  assert(old_cluster != nullptr);
  if (new_cluster == old_cluster ||
      new_cluster->degree_fraction + unit.degree_fraction > 1 / parameters.k)
    return false;
  double current_objective{ObjectiveTerm(*old_cluster, parameters) +
                           ObjectiveTerm(*new_cluster, parameters)};
  MoveUnitToCluster(unit, new_cluster);
  double proposed_objective{ObjectiveTerm(*old_cluster, parameters) +
                            ObjectiveTerm(*new_cluster, parameters)};
  if (proposed_objective < current_objective) {
    MoveUnitToCluster(unit, old_cluster);
    return false;
  }
  return true;
}

DiversionClustering ExtractClustering(
    const std::vector<std::unique_ptr<DiversionUnit>>& units,
    const std::vector<std::unique_ptr<Cluster>>& clusters,
    const Parameters& parameters) {
  DiversionClustering clustering;
  clustering.objective = Objective(clusters, parameters);
  for (const auto& unit : units) unit->cluster->members.push_back(unit->id);
  for (const auto& cluster : clusters)
    if (!cluster->members.empty())
      clustering.clusters.push_back(std::move(cluster->members));
  return clustering;
}

}  // namespace exposure_design
