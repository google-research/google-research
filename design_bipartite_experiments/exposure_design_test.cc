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

#include "exposure_design_internal.h"

using namespace exposure_design;

int main() {
  assert(Squared(-2) == 4);

  {
    SparseVector x{{3, 1}, {4, 1}, {5, 9}};
    assert(SquaredL2Norm(x) == 83);
    assert(L1Norm(x) == 11);
  }

  {
    IndexedSparseVector x{{3, 1}, {4, 1}, {5, 9}};
    assert(SquaredL2Norm(x) == 83);
    assert(L1Norm(x) == 11);
  }

  assert(DotProduct({{2, 7}, {1, 8}, {4, 5}}, {{1, 8}, {2, 8}, {5, 3}}) == 120);

  {
    const Instance test_instance{{
        {"A", {{8, 1}}},
        {"B", {{8, 2}, {9, 3}}},
        {"C", {{9, 4}}},
    }};
    std::vector<std::unique_ptr<DiversionUnit>> units{
        MakeDiversionUnits(test_instance)};
    std::sort(
        units.begin(), units.end(),
        [](const std::unique_ptr<DiversionUnit>& a,
           const std::unique_ptr<DiversionUnit>& b) { return a->id < b->id; });
    assert(units[0]->id == "A");
    assert(units[0]->degree_fraction == 0.25);
    assert(units[0]->squared_l2_norm == 1);
    assert(units[0]->l1_norm == 1);
    assert(units[1]->id == "B");
    assert(units[1]->degree_fraction == 0.5);
    assert(units[1]->squared_l2_norm == 13);
    assert(units[1]->l1_norm == 5);
    assert(units[2]->id == "C");
    assert(units[2]->degree_fraction == 0.25);
    assert(units[2]->squared_l2_norm == 16);
    assert(units[2]->l1_norm == 4);

    std::vector<std::unique_ptr<Cluster>> clusters{
        MakeSingletonClusters(units)};
    assert(clusters[0]->degree_fraction == 0.25);
    assert(clusters[0]->squared_l2_norm == 1);
    assert(clusters[0]->l1_norm == 1);
    MoveUnitToCluster(*units[1], clusters[0].get());
    assert(clusters[0]->degree_fraction == 0.75);
    assert(clusters[0]->squared_l2_norm == 18);
    assert(clusters[0]->l1_norm == 6);
    MoveUnitToCluster(*units[2], clusters[0].get());
    assert(clusters[0]->degree_fraction == 1);
    assert(clusters[0]->squared_l2_norm == 58);
    assert(clusters[0]->l1_norm == 10);
    MoveUnitToCluster(*units[1], clusters[1].get());
    assert(clusters[0]->degree_fraction == 0.5);
    assert(clusters[0]->squared_l2_norm == 17);
    assert(clusters[0]->l1_norm == 5);
    assert(clusters[1]->degree_fraction == 0.5);
    assert(clusters[1]->squared_l2_norm == 13);
    assert(clusters[1]->l1_norm == 5);

    Parameters parameters{.phi{0.5}, .k{2}};
    assert(ObjectiveTerm(*clusters[0], parameters) ==
           1.5 * 17 - 0.5 * Squared(5));
    assert(ObjectiveTerm(*clusters[1], parameters) ==
           1.5 * 13 - 0.5 * Squared(5));

    // Rejected due to degree fraction.
    assert(!MoveIfNotWorse(*units[1], clusters[0].get(), parameters));
    // Same (singleton to singleton).
    assert(MoveIfNotWorse(*units[1], clusters[2].get(), parameters));
    // Better since A and C are unrelated.
    assert(MoveIfNotWorse(*units[2], clusters[1].get(), parameters));
    // Worse.
    assert(!MoveIfNotWorse(*units[2], clusters[0].get(), parameters));
    // Decrease k and B can join A...
    parameters.k = 1;
    assert(MoveIfNotWorse(*units[1], clusters[0].get(), parameters));
    // ...but not for good if phi is large enough.
    parameters.phi = 5;
    assert(MoveIfNotWorse(*units[1], clusters[2].get(), parameters));
    // Put it back.
    parameters.phi = 0.5;
    assert(MoveIfNotWorse(*units[1], clusters[0].get(), parameters));

    DiversionClustering clustering{
        ExtractClustering(units, clusters, parameters)};
    std::sort(
        clustering.clusters.begin(), clustering.clusters.end(),
        [](const std::vector<DiversionId>& a,
           const std::vector<DiversionId>& b) { return a.size() > b.size(); });
    for (DiversionCluster& cluster : clustering.clusters)
      std::sort(cluster.begin(), cluster.end());
    assert(clustering.objective ==
           (1.5 * (Squared(1 + 2) + Squared(3)) - 0.5 * Squared(1 + 5)) +
               (1.5 * Squared(4) - 0.5 * Squared(4)));
    assert(clustering.clusters.size() == 2);
    assert(clustering.clusters[0].size() == 2);
    assert(clustering.clusters[0][0] == "A");
    assert(clustering.clusters[0][1] == "B");
    assert(clustering.clusters[1].size() == 1);
    assert(clustering.clusters[1][0] == "C");
  }

  std::cout << "OK\n";
}
