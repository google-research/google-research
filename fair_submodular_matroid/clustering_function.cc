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

#include "clustering_function.h"

#include <assert.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "submodular_function.h"

using std::max;
using std::min;

ClusteringFunction::ClusteringFunction(
    const std::vector<std::vector<double>>& input) {
  input_ = input;
  for (int i = 0; i < input_.size(); i++) {
    double dist_orig = distance(i, -1);  // distance to origin
    max_value_ += dist_orig;
    min_dist_.push_back({dist_orig});
    universe_.push_back(i);
  }
}

void ClusteringFunction::Reset() {
  for (int i = 0; i < input_.size(); i++) {
    min_dist_[i] = {distance(i, -1)};
  }
}

const std::vector<int>& ClusteringFunction::GetUniverse() const {
  return universe_;
}

std::string ClusteringFunction::GetName() const { return "ClusteringFunction"; }

std::unique_ptr<SubmodularFunction> ClusteringFunction::Clone() const {
  return std::make_unique<ClusteringFunction>(*this);
}

double ClusteringFunction::GetMaxValue() { return max_value_; }

void ClusteringFunction::Add(int element) {
  // solution_.push_back(element);
  for (int i = 0; i < input_.size(); ++i) {
    min_dist_[i].insert(distance(element, i));
  }
}

double ClusteringFunction::Delta(int element) {
  double res = 0.0;
  for (int i = 0; i < input_.size(); ++i) {
    res += max(0.0, *min_dist_[i].begin() - distance(element, i));
  }
  return res;
}

void ClusteringFunction::Remove(int element) {
  for (int i = 0; i < input_.size(); ++i) {
    auto it = min_dist_[i].find(distance(element, i));
    assert(it != min_dist_[i].end());
    min_dist_[i].erase(it);
  }
}

double ClusteringFunction::RemovalDelta(int element) {
  double val = 0.0;
  for (int i = 0; i < input_.size(); ++i) {
    const double eval = distance(element, i);
    auto it = min_dist_[i].begin();
    if (*it == eval) {
      // element is the current minimum, so we look at the second-best
      ++it;
      val += *it - eval;
    }  // else: element is not the current minimum, so removing it won't change
       // things
  }
  return val;
}

// Not necessary, but overloaded for efficiency
double ClusteringFunction::RemoveAndIncreaseOracleCall(int element) {
  ++oracle_calls_;
  double val = 0.0;
  for (int i = 0; i < input_.size(); ++i) {
    const double before = *min_dist_[i].begin();
    auto it = min_dist_[i].find(distance(element, i));
    assert(it != min_dist_[i].end());
    min_dist_[i].erase(it);
    const double after = *min_dist_[i].begin();
    val += after - before;
  }
  return val;
}
double ClusteringFunction::Objective(const std::vector<int>& elements) const {
  if (elements.empty()) {
    return 0;
  }

  double res = max_value_;
  for (int i = 0; i < input_.size(); i++) {
    double min_dist = distance(i, -1);
    for (int element : elements) {
      min_dist = min(min_dist, distance(i, element));
    }
    res -= min_dist;
  }
  return res;
}

double ClusteringFunction::distance(int x, int y) const {
  double answer = 0;
  for (int i = 0; i < input_[x].size(); i++) {
    if (y == -1) {  // distance to origin
      answer += (input_[x][i]) * (input_[x][i]);
    } else {
      answer += (input_[x][i] - input_[y][i]) * (input_[x][i] - input_[y][i]);
    }
  }
  return answer;
}
