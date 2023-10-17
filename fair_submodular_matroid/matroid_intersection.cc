// Copyright 2023 The Authors.
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

#include "matroid_intersection.h"

#include <cassert>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include "absl/container/btree_map.h"
#include "matroid.h"
#include "submodular_function.h"

void MaxIntersection(Matroid* matroid_a, Matroid* matroid_b,
                     const std::vector<int>& elements) {
  matroid_a->Reset();
  matroid_b->Reset();
  // Adjacency lists;
  std::map<int, std::vector<int>> exchange_graph;
  while (true) {
    // Greedily add elements to the solution;
    for (int element : elements) {
      if (matroid_a->InCurrent(element)) {
        continue;
      }
      if (matroid_a->CanAdd(element) && matroid_b->CanAdd(element)) {
        matroid_a->Add(element);
        matroid_b->Add(element);
      }
    }

    // Construct the exchange graph.
    exchange_graph.clear();
    for (int element : elements) {
      if (matroid_a->InCurrent(element)) {
        continue;
      }
      for (int a_swap : matroid_a->GetAllSwaps(element)) {
        exchange_graph[a_swap].push_back(element);
      }
      for (int b_swap : matroid_b->GetAllSwaps(element)) {
        exchange_graph[element].push_back(b_swap);
      }
    }

    // Find an augmenting path via BFS.
    std::map<int, int> bfs_parent;
    std::queue<int> queue;
    int aug_path_dest = -1;
    for (int element : elements) {
      if (matroid_a->InCurrent(element)) {
        continue;
      }
      if (matroid_a->CanAdd(element)) {
        bfs_parent[element] = -1;
        queue.push(element);
      }
    }
    while (!queue.empty()) {
      const int element = queue.front();
      queue.pop();
      if (!matroid_b->InCurrent(element) && matroid_b->CanAdd(element)) {
        aug_path_dest = element;
        break;
      }
      for (int neighbor : exchange_graph[element]) {
        if (!bfs_parent.count(neighbor)) {
          bfs_parent[neighbor] = element;
          queue.push(neighbor);
        }
      }
    }

    if (aug_path_dest == -1) {
      // No augmenting path found.
      break;
    }

    // Swap along the augmenting path.
    std::cerr << "we are applying an augmenting path" << std::endl;
    int out_element = aug_path_dest;
    int in_element = bfs_parent[aug_path_dest];
    while (in_element != -1) {
      matroid_a->Swap(out_element, in_element);
      matroid_b->Swap(out_element, in_element);
      out_element = bfs_parent[in_element];
      in_element = bfs_parent[out_element];
    }
    matroid_a->Add(out_element);
    matroid_b->Add(out_element);
  }

  assert(matroid_a->CurrentIsFeasible());
  assert(matroid_b->CurrentIsFeasible());
}

// Returns if an element is needed to be removed from `matroid_` to insert
// `element`. Returns "-1" if no element is needed to be remove and "-2" if
// the element cannot be swapped.
int MinWeightElementToRemove(Matroid* matroid,
                             absl::btree_map<int, double>& weight,
                             const std::set<int>& const_elements,
                             const int element) {
  if (matroid->CanAdd(element)) {
    return -1;
  }
  int best_element = -2;
  for (const int& swap : matroid->GetAllSwaps(element)) {
    if (const_elements.find(swap) != const_elements.end()) continue;
    if (best_element < 0 || weight[best_element] > weight[swap]) {
      best_element = swap;
    }
  }
  return best_element;
}

void SubMaxIntersection(Matroid* matroid_a, Matroid* matroid_b,
                        SubmodularFunction* sub_func_f,
                        const std::set<int>& const_elements,
                        const std::vector<int>& universe) {
  // DO NOT reset the matroids here.
  absl::btree_map<int, double> weight;
  for (const int& element : universe) {
    if (const_elements.count(element)) continue;  // don't add const_elements
    int first_swap =
        MinWeightElementToRemove(matroid_a, weight, const_elements, element);
    int second_swap =
        MinWeightElementToRemove(matroid_b, weight, const_elements, element);
    if (first_swap == -2 || second_swap == -2) continue;
    double total_decrease = weight[first_swap] + weight[second_swap];
    double cont_element = sub_func_f->DeltaAndIncreaseOracleCall(element);
    if (2 * total_decrease <= cont_element) {
      if (first_swap >= 0) {
        matroid_a->Remove(first_swap);
        matroid_b->Remove(first_swap);
        sub_func_f->Remove(first_swap);
      }
      if (second_swap >= 0 && first_swap != second_swap) {
        matroid_a->Remove(second_swap);
        matroid_b->Remove(second_swap);
        sub_func_f->Remove(second_swap);
      }
      matroid_a->Add(element);
      matroid_b->Add(element);
      sub_func_f->Add(element);
      weight[element] = cont_element;
    }
  }
}
