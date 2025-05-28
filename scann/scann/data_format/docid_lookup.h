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

#ifndef SCANN_DATA_FORMAT_DOCID_LOOKUP_H_
#define SCANN_DATA_FORMAT_DOCID_LOOKUP_H_

#include "absl/functional/any_invocable.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class DocidLookup {
 public:
  virtual ~DocidLookup() = default;

  virtual bool LookupDatapointIndex(string_view docid,
                                    DatapointIndex* idx) const = 0;

  using LookupCallback =
      absl::AnyInvocable<void(size_t docids_idx, DatapointIndex dp_idx)>;

  virtual void LookupDatapointIndices(ConstSpan<string_view> docids,
                                      LookupCallback callback) const {
    for (size_t i = 0; i < docids.size(); ++i) {
      DatapointIndex dp_idx;
      callback(i, LookupDatapointIndex(docids[i], &dp_idx)
                      ? dp_idx
                      : kInvalidDatapointIndex);
    }
  }
};

}  // namespace research_scann

#endif
