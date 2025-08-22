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

#ifndef FILE_UTIL_H_
#define FILE_UTIL_H_

#include "graph.h"
#include "tsv_utils.h"

namespace geo_algorithms {

using utils_tsv::TsvReader;

std::vector<std::pair<int, int>> GetQueriesFromFile(
    const std::string& filename);

}  // namespace geo_algorithms

#endif  // FILE_UTIL_H_
