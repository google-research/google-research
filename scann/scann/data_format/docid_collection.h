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



#ifndef SCANN_DATA_FORMAT_DOCID_COLLECTION_H_
#define SCANN_DATA_FORMAT_DOCID_COLLECTION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "scann/data_format/docid_collection_interface.h"
#include "scann/data_format/internal/string_view32.h"
#include "scann/oss_wrappers/scann_serialize.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class VariableLengthDocidCollection final : public DocidCollectionInterface {
 public:
  VariableLengthDocidCollection() = default;
  ~VariableLengthDocidCollection() final = default;

  VariableLengthDocidCollection(const VariableLengthDocidCollection& rhs);
  VariableLengthDocidCollection& operator=(
      const VariableLengthDocidCollection& rhs);

  static VariableLengthDocidCollection CreateWithEmptyDocids(
      DatapointIndex n_elements) {
    VariableLengthDocidCollection result;
    result.size_ = n_elements;
    return result;
  }

  VariableLengthDocidCollection(VariableLengthDocidCollection&& rhs) = default;
  VariableLengthDocidCollection& operator=(
      VariableLengthDocidCollection&& rhs) = default;

  Status Append(string_view docid) final;
  size_t size() const final { return size_; }
  bool empty() const final { return size_ == 0; }
  void ShrinkToFit() final;
  void Clear() final;

  unique_ptr<DocidCollectionInterface> Copy() const final {
    return unique_ptr<DocidCollectionInterface>(
        new VariableLengthDocidCollection(*this));
  }

  bool all_empty() const { return !impl_ && size_ > 0; }

  string_view Get(size_t i) const final {
    DCHECK_LT(i, size_);
    return (all_empty()) ? "" : impl_->Get(i);
  }

  void MultiGet(size_t num_docids, DpIdxGetter docid_idx_getter,
                StringSetter docid_setter) const final {
    if (!all_empty()) {
      impl_->MultiGet(num_docids, std::move(docid_idx_getter),
                      std::move(docid_setter));
    } else {
      for (size_t i = 0; i < num_docids; ++i) {
        DatapointIndex idx = docid_idx_getter(i);
        DCHECK_LT(idx, size_);
        docid_setter(idx, "");
      }
    }
  }

  size_t capacity() const final { return impl_ ? impl_->capacity() : 0; }

  size_t MemoryUsage() const final {
    return sizeof(this) + (impl_ ? (impl_->MemoryUsage()) : 0);
  }

  void Reserve(DatapointIndex n_elements) final;

  class Mutator : public DocidCollectionInterface::Mutator {
   public:
    static StatusOr<unique_ptr<Mutator>> Create(
        VariableLengthDocidCollection* docids);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;

    ~Mutator() final {}
    Status AddDatapoint(string_view docid) final;
    bool LookupDatapointIndex(string_view docid,
                              DatapointIndex* index) const final;
    Status RemoveDatapoint(string_view docid) final;
    Status RemoveDatapoint(DatapointIndex index) final;
    void Reserve(size_t size) final;

   private:
    explicit Mutator(VariableLengthDocidCollection* docids) : docids_(docids) {}
    VariableLengthDocidCollection* docids_ = nullptr;
    using string_view32 = data_format_internal::string_view32;
    absl::flat_hash_map<string_view32, DatapointIndex, string_view32::Hash>
        docid_lookup_;
  };

  StatusOr<DocidCollectionInterface::Mutator*> GetMutator() const final;

 private:
  Status AppendImpl(string_view docid);

  void InstantiateImpl();

  DatapointIndex size_ = 0;

  DatapointIndex expected_docid_size_ = 0;

  unique_ptr<DocidCollectionInterface> impl_ = nullptr;
  mutable unique_ptr<VariableLengthDocidCollection::Mutator> mutator_ = nullptr;
};

class FixedLengthDocidCollection final : public DocidCollectionInterface {
 public:
  FixedLengthDocidCollection(const FixedLengthDocidCollection& rhs);
  FixedLengthDocidCollection& operator=(const FixedLengthDocidCollection& rhs);

  FixedLengthDocidCollection(FixedLengthDocidCollection&& rhs) = default;
  FixedLengthDocidCollection& operator=(FixedLengthDocidCollection&& rhs) =
      default;

  explicit FixedLengthDocidCollection(size_t length) : docid_length_(length) {}
  ~FixedLengthDocidCollection() final = default;

  static StatusOr<FixedLengthDocidCollection> Iota(uint32_t length) {
    FixedLengthDocidCollection docids(sizeof(uint32_t));
    docids.Reserve(length);
    for (uint32_t i = 0; i < length; ++i) {
      std::string encoded;
      strings::KeyFromUint32(i, &encoded);
      SCANN_RETURN_IF_ERROR(docids.Append(encoded));
    }
    return docids;
  }

  Status Append(string_view docid) final;

  size_t size() const final { return size_; }
  bool empty() const final { return size_ == 0; }
  std::optional<size_t> fixed_len_size() const final { return docid_length_; }

  void ShrinkToFit() final { arr_.shrink_to_fit(); }

  unique_ptr<DocidCollectionInterface> Copy() const final {
    return unique_ptr<DocidCollectionInterface>(
        new FixedLengthDocidCollection(*this));
  }

  void Clear() final {
    size_ = 0;
    arr_.clear();
    mutator_ = nullptr;
  }

  string_view Get(size_t i) const final {
    DCHECK_LT(i, size_);
    return string_view(&arr_[i * docid_length_], docid_length_);
  }

  void MultiGet(size_t num_docids, DpIdxGetter docid_idx_getter,
                StringSetter docid_setter) const final;

  size_t capacity() const final { return arr_.capacity() / docid_length_; }

  size_t MemoryUsage() const final { return arr_.capacity() + sizeof(*this); }

  void Reserve(DatapointIndex n_elements) final;

  class Mutator : public DocidCollectionInterface::Mutator {
   public:
    static StatusOr<unique_ptr<Mutator>> Create(
        FixedLengthDocidCollection* docids);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;

    ~Mutator() final {}
    Status AddDatapoint(string_view docid) final;
    bool LookupDatapointIndex(string_view docid,
                              DatapointIndex* index) const final;
    Status RemoveDatapoint(string_view docid) final;
    Status RemoveDatapoint(DatapointIndex index) final;
    void Reserve(size_t size) final;

   private:
    static constexpr int kGrowthFactor = 2;
    explicit Mutator(FixedLengthDocidCollection* docids) : docids_(docids) {}

    FixedLengthDocidCollection* docids_ = nullptr;
    using string_view32 = data_format_internal::string_view32;
    absl::flat_hash_map<string_view32, DatapointIndex, string_view32::Hash>
        docid_lookup_;
  };

  StatusOr<DocidCollectionInterface::Mutator*> GetMutator() const final;

 private:
  Status AppendImpl(string_view docid);
  void ReserveImpl(DatapointIndex n_elements);

  std::vector<char> arr_ = {};

  size_t docid_length_;

  size_t size_ = 0;

  mutable unique_ptr<FixedLengthDocidCollection::Mutator> mutator_ = nullptr;
};

}  // namespace research_scann

#endif
