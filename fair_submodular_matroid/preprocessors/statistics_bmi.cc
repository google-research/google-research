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
#include <string>
#include <utility>
#include <vector>

// this function splits string s by pattern pat
std::vector<std::string> split(std::string s, std::string pat) {
  std::vector<std::string> ret;
  ret.clear();
  int idx = 0, i = 0;
  while (idx < s.size()) {
    i = s.find(pat, idx);
    if (i == -1) break;
    if (i - idx) ret.push_back(s.substr(idx, i - idx));
    idx = i + pat.size();
  }
  if (idx < s.size()) ret.push_back(s.substr(idx));
  return ret;
}

// Extracts attributes that we care about.
// Description of attributes is given at
// https://snap.stanford.edu/data/soc-pokec-readme.txt
const int ID = 0;
const int HEIGHT_WEIGHT = 3;

int processed;

double computeBMI(const double height, const double weight) {
  return weight / (height * height / 10000.0);
}

int extract_next_number(const std::string &S, int idx) {
  int ret = 0;
  while (idx < S.size() && (S[idx] < '0' || S[idx] > '9')) idx++;

  while (idx < S.size() && (S[idx] >= '0' && S[idx] <= '9')) {
    ret = ret * 10 + S[idx] - '0';
    idx++;
  }

  return ret;
}

std::pair<int, int> extract_height_weight(const std::string &a) {
  if (a == "null") return {0, 0};
  int cm_index = a.find("cm");
  int kg_index = a.find("kg");
  if (cm_index == std::string::npos || kg_index == std::string::npos ||
      cm_index > kg_index)
    return {0, 0};

  int cm = extract_next_number(a, 0);
  int kg = extract_next_number(a, cm_index);
  if (cm == 0 || kg == 0 || cm > 222 || kg < 35 || cm < 120 || kg > 200)
    cm = kg = 0;
  return {cm, kg};
}

int BMIIndex(const int height, const int width) {
  double BMI = computeBMI(height, width);
  if (BMI < 18.5)
    return 1;
  else if (BMI <= 24.9)
    return 2;
  else if (BMI <= 29.9)
    return 3;
  else
    return 4;
}

int main() {
  // Update the paths as described in the README.md.
  std::ifstream fin("");
  std::ofstream fout("");

  processed = 0;
  std::string S;
  std::map<std::pair<int, int>, int> heigh_weight;

  heigh_weight.clear();
  while (std::getline(fin, S)) {
    std::vector<std::string> attr = split(S, "\t");
    processed++;
    std::pair<int, int> p = extract_height_weight(attr[HEIGHT_WEIGHT]);
    heigh_weight[p]++;
    if (p.first != 0)
      fout << attr[ID] << " " << BMIIndex(p.first, p.second) << std::endl;
  }
  fout.close();
  fin.close();

  std::cout << "processed = " << processed << std::endl;
  std::cout << 1.0 * heigh_weight[{0, 0}] / processed << std::endl;

  int cnt[5] = {0, 0, 0, 0, 0};
  for (auto &pp : heigh_weight) {
    if (pp.first.first == 0) {
      cnt[0] += pp.second;
      continue;
    }

    cnt[BMIIndex(pp.first.first, pp.first.second)] += pp.second;
  }

  for (int i = 0; i < 5; i++) std::cout << 1.0 * cnt[i] / processed << " ";
  std::cout << std::endl;

  std::cout << "Ignoring {0, 0}: " << std::endl;
  int offset = heigh_weight[{0, 0}];
  for (int i = 1; i < 5; i++)
    std::cout << 1.0 * cnt[i] / (processed - offset) << " ";
  std::cout << std::endl;

  return 0;
}
