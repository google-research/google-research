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

#ifndef SCANN_DATA_FORMAT_DOCID_COLLECTION_INTERFACE_H_
#define SCANN_DATA_FORMAT_DOCID_COLLECTION_INTERFACE_H_

#include <cstddef>
#include <optional>

#include "absl/base/nullability.h"
#include "scann/data_format/docid_lookup.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class DocidCollectionInterface {
 public:
  virtual ~DocidCollectionInterface() = default;

  virtual Status Append(string_view docid) = 0;

  virtual size_t size() const = 0;

  virtual bool empty() const = 0;

  virtual std::optional<size_t> fixed_len_size() const { return std::nullopt; }

  virtual string_view Get(size_t i) const = 0;

  virtual void MultiGet(size_t num_docids, DpIdxGetter docid_idx_getter,
                        StringSetter docid_setter) const = 0;

  virtual size_t capacity() const = 0;

  virtual size_t MemoryUsage() const = 0;

  virtual void Clear() = 0;

  virtual void Reserve(DatapointIndex n_elements) = 0;

  virtual void ShrinkToFit() = 0;

  virtual unique_ptr<DocidCollectionInterface> Copy() const = 0;

  class Mutator : public DocidLookup {
   public:
    ~Mutator() override = default;

    virtual Status AddDatapoint(string_view docid) = 0;

    virtual Status RemoveDatapoint(string_view docid) = 0;

    virtual void Reserve(size_t size) = 0;

    virtual Status RemoveDatapoint(DatapointIndex idx) = 0;

    string_view ImplName() const override = 0;
  };

  virtual StatusOr<DocidLookup*> GetDocidLookup() const {
    auto mutator = GetMutator();
    if (!mutator.ok()) {
      return mutator.status();
    }
    return mutator.value();
  }

  virtual StatusOr<Mutator*> GetMutator() const = 0;
};

class DocidLookupMap : public DocidLookup {
 public:
  ~DocidLookupMap() override = default;

  virtual void Clear() = 0;

  virtual void Reserve(size_t size) = 0;

  virtual Status AddDatapoint(string_view docid, DatapointIndex dp_idx) = 0;

  virtual Status RemoveDatapoint(string_view docid) = 0;

  virtual Status RemoveDatapoint(DatapointIndex dp_idx) = 0;

  string_view ImplName() const override = 0;

 protected:
  explicit DocidLookupMap(const DocidCollectionInterface* docids)
      : docids_(docids) {}

  const DocidCollectionInterface* docids_ = nullptr;
};

absl::StatusOr<std::unique_ptr<DocidLookupMap>> CreateDocidLookupMap(
    DocidCollectionInterface* docids);

}  // namespace research_scann

#endif
