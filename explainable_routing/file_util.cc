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

#include "file_util.h"

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "tsv_utils.h"

namespace geo_algorithms {

using utils_tsv::TsvReader;

std::vector<std::pair<int, int>> GetQueriesFromFile(
    const std::string& filename) {
  std::vector<std::pair<int, int>> queries;
  for (TsvReader reader(filename); !reader.AtEnd();) {
    const absl::flat_hash_map<std::string, std::string> row = reader.ReadRow();
    int source, target;
    CHECK(absl::SimpleAtoi(row.at("source_id"), &source));
    CHECK(absl::SimpleAtoi(row.at("target_id"), &target));
    queries.push_back(std::make_pair(source, target));
  }
  return queries;
}

}  // namespace geo_algorithms
