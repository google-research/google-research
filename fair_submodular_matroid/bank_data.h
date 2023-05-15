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

#ifndef FAIR_SUBMODULAR_MATROID_BANK_DATA_H_
#define FAIR_SUBMODULAR_MATROID_BANK_DATA_H_

#include <vector>

#include "absl/container/flat_hash_map.h"

class BankData {
 public:
  std::vector<std::vector<double>> input_;
  std::vector<int> age_grpcards_;
  std::vector<int> balance_grpcards_;
  absl::flat_hash_map<int, int> age_map_;
  absl::flat_hash_map<int, int> balance_map_;

  explicit BankData(const char input_path[]);
  ~BankData() = default;
};

#endif  // FAIR_SUBMODULAR_MATROID_BANK_DATA_H_
