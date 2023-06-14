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

#include <fstream>
#include <iostream>
#include <map>

#include "absl/container/btree_map.h"

int main() {
  std::ifstream fin_colors("color-BMI.txt");
  int processed = 0;
  int u, v, c;
  absl::btree_map<int, int> map_v_to_c;
  while (fin_colors >> v >> c) {
    map_v_to_c[v] = c;
    processed++;
  }

  fin_colors.close();
  std::cout << "processed = " << processed << std::endl;
  // processed = 582319

  std::ifstream fin_edges("soc-pokec-relationships.txt");
  std::ofstream fout_edges_bmi("BMI-soc-pokec-relationships.txt");
  processed = 0;
  while (fin_edges >> u >> v) {
    if (!map_v_to_c.count(u) || !map_v_to_c.count(v)) continue;
    fout_edges_bmi << u << " " << v << std::endl;
    processed++;
  }
  std::cout << "processed = " << processed << std::endl;
  // processed = 5834695
  fout_edges_bmi.close();
  fin_edges.close();
  return 0;
}
