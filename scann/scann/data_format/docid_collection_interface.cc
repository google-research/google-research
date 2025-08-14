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

#include "scann/data_format/docid_collection_interface.h"

#include <cstddef>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "scann/data_format/docid_lookup.h"
#include "scann/utils/common.h"
#include "scann/utils/multi_stage_batch_pipeline.h"
#include "scann/utils/types.h"

namespace research_scann {

class HetergenousDocidLookupMap : public DocidLookupMap {
 public:
  explicit HetergenousDocidLookupMap(const DocidCollectionInterface* docids)
      : DocidLookupMap(docids),
        heterogeneous_index_set_({}, HeterogeneousHash{docids_},
                                 HeterogeneousEqual{docids_}) {}

  void Clear() final { heterogeneous_index_set_.clear(); }

  void Reserve(size_t size) final { heterogeneous_index_set_.reserve(size); }

  bool LookupDatapointIndex(string_view docid,
                            DatapointIndex* idx) const final {
    auto it = heterogeneous_index_set_.find(docid);
    if (it == heterogeneous_index_set_.end()) {
      return false;
    }
    *idx = *it;
    return true;
  }

  void LookupDatapointIndices(size_t num_docids, DocidGetter docid_getter,
                              LookupCallback callback) const final {
    auto prefetch_cb = [&](size_t idx, size_t) {
      heterogeneous_index_set_.prefetch(docid_getter(idx));
    };
    auto lookup_cb = [&](size_t idx, size_t) {
      auto it = heterogeneous_index_set_.find(docid_getter(idx));
      callback(idx, it == heterogeneous_index_set_.end()
                        ? kInvalidDatapointIndex
                        : *it);
    };

    constexpr size_t kBatchSize = 32;
    RunMultiStageBatchPipeline<kBatchSize, decltype(prefetch_cb),
                               decltype(lookup_cb)>(
        num_docids, {std::move(prefetch_cb), std::move(lookup_cb)});
  }

  Status RemoveDatapoint(string_view docid) final {
    auto it = heterogeneous_index_set_.find(docid);
    if (it == heterogeneous_index_set_.end()) {
      return NotFoundError(absl::StrCat("Docid not found: ", docid, "."));
    }
    heterogeneous_index_set_.erase(it);
    return OkStatus();
  }

  Status RemoveDatapoint(DatapointIndex dp_idx) final {
    if (dp_idx >= docids_->size()) {
      return OutOfRangeError(
          absl::StrCat("Datapoint index is out of range: ", dp_idx, "."));
    }
    auto it = heterogeneous_index_set_.find(dp_idx);
    if (it == heterogeneous_index_set_.end()) {
      return NotFoundError(
          absl::StrCat("Datapoint index not found: ", dp_idx, "."));
    }
    heterogeneous_index_set_.erase(it);
    return OkStatus();
  }

  Status AddDatapoint(string_view docid, DatapointIndex dp_idx) final {
    auto it = heterogeneous_index_set_.find(docid);
    if (it != heterogeneous_index_set_.end()) {
      return AlreadyExistsError(
          absl::StrCat("Docid already exists: ", docid, "."));
    }
    if (dp_idx >= docids_->size()) {
      return OutOfRangeError(
          absl::StrCat("Datapoint index is out of range: ", dp_idx, "."));
    }
    it = heterogeneous_index_set_.find(dp_idx);
    if (it != heterogeneous_index_set_.end()) {
      return AlreadyExistsError(
          absl::StrCat("Datapoint index already exists: ", dp_idx, "."));
    }
    if (docids_->Get(dp_idx) != docid) {
      return InvalidArgumentError(absl::StrCat(
          "Docid and datapoint index mismatch: ", docid, " vs. ", dp_idx, "."));
    }
    heterogeneous_index_set_.insert(dp_idx);
    return OkStatus();
  }

  string_view ImplName() const final { return "scann_heterogeneous"; }

 private:
  struct HeterogeneousHash {
    using is_transparent = void;

    size_t operator()(DatapointIndex dp_idx) const {
      return absl::Hash<string_view>{}(docids->Get(dp_idx));
    }
    size_t operator()(string_view docid) const {
      return absl::Hash<string_view>{}(docid);
    }

    const DocidCollectionInterface* docids = nullptr;
  };

  struct HeterogeneousEqual {
    using is_transparent = void;

    bool operator()(DatapointIndex dp_idx, string_view docid) const {
      return docids->Get(dp_idx) == docid;
    }
    bool operator()(string_view docid, DatapointIndex dp_idx) const {
      return docids->Get(dp_idx) == docid;
    }
    bool operator()(DatapointIndex dp_idx, DatapointIndex dp_idx2) const {
      return dp_idx == dp_idx2;
    }
    bool operator()(string_view docid, string_view docid2) const {
      return docid == docid2;
    }

    const DocidCollectionInterface* docids = nullptr;
  };

  absl::flat_hash_set<DatapointIndex, HeterogeneousHash, HeterogeneousEqual>
      heterogeneous_index_set_;
};

absl::StatusOr<std::unique_ptr<DocidLookupMap>> CreateDocidLookupMap(
    DocidCollectionInterface* docids) {
  std::unique_ptr<DocidLookupMap> map;

  map = std::make_unique<HetergenousDocidLookupMap>(docids);

  map->Reserve(docids->size());
  for (DatapointIndex i = 0; i < docids->size(); ++i) {
    string_view docid = docids->Get(i);
    if (!docid.empty()) {
      SCANN_RETURN_IF_ERROR(map->AddDatapoint(docid, i));
    }
  }

  return map;
}

}  // namespace research_scann
