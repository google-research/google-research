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

#ifndef PATH_STORE_H_
#define PATH_STORE_H_

#include <cstdlib>
#include <filesystem>

#include "formulation.h"
#include "graph.h"
#include "tsv_utils.h"

namespace geo_algorithms {

namespace fs = std::filesystem;

class PathStore {
 public:
  PathStore(const std::string& directory) : directory_(directory) {
    fs::create_directories(directory_);
  }

  void SavePathOrFailIfPresent(const std::string& filename,
                               const std::vector<AOrRIndex>& path);

  absl::StatusOr<std::vector<AOrRIndex>> ReadPathOrFailIfAbsent(
      const std::string& filename);

 private:
  const std::string directory_;
};

class CutStore {
 public:
  CutStore(const Formulation& problem, const std::string& directory)
      : problem_(problem), directory_(directory) {
    fs::create_directories(directory_);
  }

  void SaveCutOrFailIfPresent(const std::string& filename,
                              const CutSolution& cut);

  absl::StatusOr<CutSolution> ReadCutOrFailIfAbsent(
      const std::string& filename);

 private:
  const Formulation& problem_;
  const std::string directory_;
};

// deprecated because while encoding is more efficient,
// read time is not much faster than without caching
class SlowCutStore {
 public:
  SlowCutStore(const Formulation& problem, const std::string& directory)
      : problem_(problem), directory_(directory) {
    fs::create_directories(directory_);
  }

  void SaveCutOrFailIfPresent(const std::string& filename,
                              const CutSolution& cut);

  absl::StatusOr<CutSolution> ReadCutOrFailIfAbsent(
      const std::string& filename);

 private:
  const Formulation& problem_;
  const std::string directory_;
};

}  // namespace geo_algorithms

#endif  // PATH_STORE_H_
