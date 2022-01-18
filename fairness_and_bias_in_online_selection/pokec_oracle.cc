// Copyright 2022 The Google Research Authors.
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

#include "pokec_oracle.h"

#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <string>

#include "absl/container/btree_map.h"
#include "utils.h"

namespace fair_secretary {

using std::map;
using std::string;
using std::vector;

vector<SecretaryInstance> PokecOracle::GetSecretaryInput() {
  string input_bmi = "";
  string input_edges = "";
  std::ifstream in_bmi(input_bmi);
  std::ifstream in_edges(input_edges);
  absl::btree_map<int, int> nodes;
  absl::btree_map<int, int> ids;
  int id, color;
  int counter = 0;
  while (in_bmi >> id >> color) {
    if (nodes.find(id) != nodes.end()) {
      std::cout << "Error: ID '" << id  << "' does not exists." << std::endl;
    }
    nodes[id] = color;
    ids[id] = counter++;
  }
  std::cout << counter << std::endl;
  vector<int> degrees(counter, 0);
  int v1, v2;
  while (in_edges >> v1 >> v2) {
    if (ids.find(v1) != ids.end() && ids.find(v2) != ids.end()) {
      degrees[ids[v1]]++;
      degrees[ids[v2]]++;
    }
  }
  vector<SecretaryInstance> instance;
  instance.reserve(counter);
  for (int i = 0; i < counter; i++) {
    instance.push_back(SecretaryInstance(degrees[i] + i * 0.0000001, nodes[i]));
  }
  num_colors = 0;
  for (int i = 0; i < instance.size(); i++) {
    num_colors = std::max(num_colors, instance[i].color);
  }
  num_colors++;
  return instance;
}

}  //  namespace fair_secretary
