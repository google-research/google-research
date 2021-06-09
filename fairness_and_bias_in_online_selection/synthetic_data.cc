// Copyright 2021 The Google Research Authors.
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

#include "synthetic_data.h"

#include <iostream>

#include "random_handler.h"

namespace fair_secretary {

using std::vector;

vector<SecretaryInstance> SyntheticData::GetSecretaryInput(
    const vector<int>& sizes, const std::vector<double>& prob) {
  num_colors = sizes.size();
  vector<SecretaryInstance> instance;
  double rand = 2.0;
  if (!prob.empty()) {
    rand = static_cast<double>(RandomHandler::eng_()) /
           std::numeric_limits<uint64_t>::max();
  }
  for (int i = 0; i < sizes.size(); i++) {
    for (int j = 0; j < sizes[i]; j++) {
      instance.push_back(SecretaryInstance(
          static_cast<double>(RandomHandler::eng_()) / 10, i));
    }
    if (!prob.empty()) {
      if (prob[i] > rand && rand >= 0) {
        instance[instance.size() - 1].value =
            std::numeric_limits<uint64_t>::max();
      }
      rand -= prob[i];
    }
  }
  return instance;
}

vector<SecretaryInstance> SyntheticData::GetProphetInput(
    const int size,
    const vector<std::reference_wrapper<RandomDistribution>>& dist) {
  num_colors = size;
  vector<SecretaryInstance> instance;
  for (int i = 0; i < size; i++) {
    if (i < size / 2) {
      instance.push_back(SecretaryInstance(dist[0].get().Sample(), i, 0));

    } else {
      instance.push_back(SecretaryInstance(dist[1].get().Sample(), i, 1));
    }
  }
  return instance;
}

}  // namespace fair_secretary
