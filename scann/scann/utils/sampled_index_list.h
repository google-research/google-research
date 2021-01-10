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



#ifndef SCANN_UTILS_SAMPLED_INDEX_LIST_H_
#define SCANN_UTILS_SAMPLED_INDEX_LIST_H_

#include "absl/types/variant.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/macros.h"

namespace research_scann {
namespace internal {

template <typename Index>
class SampledIndexList {
 public:
  SampledIndexList();

  SampledIndexList(Index start, Index stop);

  explicit SampledIndexList(vector<Index> sparse);

  bool GetNextIndex(Index* i);

  void RestartIndex();

 private:
  using DenseIndices = pair<Index, Index>;

  using SparseIndices = vector<Index>;

  const DenseIndices* dense() const {
    return absl::get_if<DenseIndices>(&indices_);
  }

  const SparseIndices* sparse() const {
    return absl::get_if<SparseIndices>(&indices_);
  }

  absl::variant<DenseIndices, SparseIndices> indices_;
  Index current_;
};

template <typename Index>
SampledIndexList<Index>::SampledIndexList()
    : indices_(DenseIndices(0, 0)), current_(0) {}

template <typename Index>
SampledIndexList<Index>::SampledIndexList(Index start, Index stop)
    : indices_(DenseIndices(start, stop)), current_(start) {}

template <typename Index>
SampledIndexList<Index>::SampledIndexList(vector<Index> sparse)
    : indices_(std::move(sparse)), current_(0) {}

template <typename Index>
bool SampledIndexList<Index>::GetNextIndex(Index* i) {
  if (dense()) {
    if (current_ == dense()->second) return false;
    *i = current_;
    ++current_;
  } else {
    if (current_ == sparse()->size()) return false;
    *i = (*sparse())[current_];
    ++current_;
  }
  return true;
}

template <typename Index>
void SampledIndexList<Index>::RestartIndex() {
  if (dense()) {
    current_ = dense()->first;
  } else {
    current_ = 0;
  }
}

}  // namespace internal
}  // namespace research_scann

#endif
