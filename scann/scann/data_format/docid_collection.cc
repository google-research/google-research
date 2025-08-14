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



#include "scann/data_format/docid_collection.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/container/inlined_vector.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "scann/data_format/docid_collection_interface.h"
#include "scann/data_format/docid_lookup.h"
#include "scann/data_format/internal/short_string_optimized_string.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/utils/common.h"
#include "scann/utils/memory_logging.h"
#include "scann/utils/multi_stage_batch_pipeline.h"
#include "scann/utils/types.h"

ABSL_FLAG(bool, use_memory_optimized_immutable_docid_collection, false,
          "Controls which implementation is used for immutable "
          "VariableLengthDocidCollection");

namespace research_scann {
namespace {

void AmortizedAppend(std::vector<char>& v, size_t to_add) {
  size_t new_size = v.size() + to_add;
  size_t capacity = v.capacity();

  if (new_size > capacity) v.reserve(std::max(new_size, 3 * capacity / 2));
  v.resize(new_size);
}

class MutableCollection;

class ImmutableCollection final : public DocidCollectionInterface {
 public:
  ImmutableCollection() = default;
  explicit ImmutableCollection(size_t size);
  ~ImmutableCollection() final = default;
  ImmutableCollection(const ImmutableCollection& rhs) = default;
  ImmutableCollection& operator=(const ImmutableCollection& rhs) = default;

  Status Append(string_view docid) final;
  size_t size() const final { return size_; }
  bool empty() const final { return size_ == 0; }
  void ShrinkToFit() final;
  void Clear() final;

  unique_ptr<DocidCollectionInterface> Copy() const final {
    return unique_ptr<DocidCollectionInterface>(new ImmutableCollection(*this));
  }

  string_view Get(size_t i) const final {
    DCHECK_LT(i, size_);
    return chunks_[c_idx(i)].Get(c_offset(i));
  }

  void MultiGet(size_t num_docids, DpIdxGetter docid_idx_getter,
                StringSetter docid_setter) const final {
    constexpr size_t kBatchSize = 24;
    absl::InlinedVector<DatapointIndex, kBatchSize> cached_indices(kBatchSize);
    auto stage1_cb = [&](size_t idx, size_t in_batch_idx) {
      cached_indices[in_batch_idx] = docid_idx_getter(idx);
      DatapointIndex i = cached_indices[in_batch_idx];
      DCHECK_LT(i, size_);
      absl::PrefetchToLocalCache(&chunks_[c_idx(i)]);
    };
    auto stage2_cb = [&](size_t, size_t in_batch_idx) {
      DatapointIndex i = cached_indices[in_batch_idx];
      absl::PrefetchToLocalCache(
          &chunks_[c_idx(i)].payload_offsets[c_offset(i)]);
    };
    auto stage3_cb = [&](size_t, size_t in_batch_idx) {
      DatapointIndex i = cached_indices[in_batch_idx];
      const size_t poffset = chunks_[c_idx(i)].payload_offsets[c_offset(i)];
      absl::PrefetchToLocalCache(&chunks_[c_idx(i)].payloads[poffset]);
    };
    auto stage4_cb = [&](size_t idx, size_t in_batch_idx) {
      DatapointIndex i = cached_indices[in_batch_idx];
      docid_setter(idx, chunks_[c_idx(i)].Get(c_offset(i)));
    };
    RunMultiStageBatchPipeline<kBatchSize, decltype(stage1_cb),
                               decltype(stage2_cb), decltype(stage3_cb),
                               decltype(stage4_cb)>(
        num_docids, {std::move(stage1_cb), std::move(stage2_cb),
                     std::move(stage3_cb), std::move(stage4_cb)});
  }

  size_t capacity() const final { return chunks_.size() * kChunkSize; }
  size_t MemoryUsage() const final;
  void Reserve(DatapointIndex n_elements) final;
  StatusOr<DocidCollectionInterface::Mutator*> GetMutator() const final;

 private:
  size_t c_idx(size_t i) const { return i / kChunkSize; }
  size_t c_offset(size_t i) const { return i % kChunkSize; }

  enum : size_t { kChunkSize = 8192 };
  struct Chunk {
    Chunk() { payload_offsets.reserve(kChunkSize); }

    string_view Get(size_t i) const {
      const size_t start = payload_offsets[i];
      const size_t size = (i + 1 == payload_offsets.size())
                              ? (payloads.size() - start)
                              : (payload_offsets[i + 1] - start);
      return {payloads.data() + start, size};
    }

    std::vector<char> payloads;

    std::vector<uint32_t> payload_offsets;
  };

  vector<Chunk> chunks_;

  size_t size_ = 0;

  friend class MutableCollection;
};

class MutableCollection final : public DocidCollectionInterface {
 public:
  MutableCollection() = default;
  explicit MutableCollection(size_t size);
  ~MutableCollection() final = default;

  MutableCollection(const MutableCollection& rhs);
  MutableCollection& operator=(const MutableCollection& rhs);

  static unique_ptr<MutableCollection> FromImmutableDestructive(
      ImmutableCollection* static_impl);

  Status Append(string_view docid) final;
  size_t size() const final { return size_; }
  bool empty() const final { return size_ == 0; }
  void ShrinkToFit() final;
  void Clear() final;

  unique_ptr<DocidCollectionInterface> Copy() const final {
    return unique_ptr<DocidCollectionInterface>(new MutableCollection(*this));
  }

  string_view Get(size_t i) const final {
    DCHECK_LT(i, size_);
    return chunks_[c_idx(i)].payload[c_offset(i)].ToStringPiece();
  }

  void MultiGet(size_t num_docids, DpIdxGetter docid_idx_getter,
                StringSetter docid_setter) const final {
    constexpr size_t kBatchSize = 24;
    absl::InlinedVector<DatapointIndex, kBatchSize> cached_indices(kBatchSize);
    auto stage1_cb = [&](size_t idx, size_t in_batch_idx) {
      cached_indices[in_batch_idx] = docid_idx_getter(idx);
      DatapointIndex i = cached_indices[in_batch_idx];
      absl::PrefetchToLocalCache(&chunks_[c_idx(i)]);
    };
    auto stage2_cb = [&](size_t, size_t in_batch_idx) {
      DatapointIndex i = cached_indices[in_batch_idx];
      absl::PrefetchToLocalCache(&chunks_[c_idx(i)].payload[c_offset(i)]);
    };
    auto stage3_cb = [&](size_t, size_t in_batch_idx) {
      DatapointIndex i = cached_indices[in_batch_idx];
      Fetch(i).prefetch();
    };
    auto stage4_cb = [&](size_t idx, size_t in_batch_idx) {
      DatapointIndex i = cached_indices[in_batch_idx];
      docid_setter(idx, Get(i));
    };
    RunMultiStageBatchPipeline<kBatchSize, decltype(stage1_cb),
                               decltype(stage2_cb), decltype(stage3_cb),
                               decltype(stage4_cb)>(
        num_docids, {std::move(stage1_cb), std::move(stage2_cb),
                     std::move(stage3_cb), std::move(stage4_cb)});
  }

  size_t capacity() const final { return chunks_.size() * kChunkSize; }
  size_t MemoryUsage() const final;
  void Reserve(DatapointIndex n_elements) final;

  StatusOr<DocidCollectionInterface::Mutator*> GetMutator() const final;

 private:
  size_t c_idx(size_t i) const { return i / kChunkSize; }
  size_t c_offset(size_t i) const { return i % kChunkSize; }

  ShortStringOptimizedString& Fetch(size_t i) {
    return chunks_[i / kChunkSize].payload[i % kChunkSize];
  }
  const ShortStringOptimizedString& Fetch(size_t i) const {
    return chunks_[i / kChunkSize].payload[i % kChunkSize];
  }

  enum : size_t { kChunkSize = 1024 };
  struct Chunk {
    Chunk() : payload(new ShortStringOptimizedString[kChunkSize]) {}
    Chunk(const Chunk& rhs) : Chunk() {
      std::copy(rhs.payload.get(), rhs.payload.get() + kChunkSize,
                payload.get());
    }

    Chunk& operator=(const Chunk& rhs) {
      std::copy(rhs.payload.get(), rhs.payload.get() + kChunkSize,
                payload.get());
      return *this;
    }

    Chunk(Chunk&& rhs) = default;
    Chunk& operator=(Chunk&&) = default;

    unique_ptr<ShortStringOptimizedString[]> payload;
  };

  vector<Chunk> chunks_;
  DatapointIndex size_ = 0;
  friend class VariableLengthDocidCollection::Mutator;
};

class ImmutableMemoryOptCollection : public DocidCollectionInterface {
 public:
  ImmutableMemoryOptCollection() = default;

  explicit ImmutableMemoryOptCollection(size_t size) {
    while (size >= kChunkSize) {
      chunks_.push_back(Chunk(kChunkSize, '\0'));
      size -= kChunkSize;
    }
    last_chunk_size_ = size;
    if (last_chunk_size_ != 0) {
      chunks_.push_back(Chunk(last_chunk_size_, '\0'));
    }
  }

  ImmutableMemoryOptCollection(const ImmutableMemoryOptCollection& rhs) =
      default;
  ImmutableMemoryOptCollection& operator=(
      const ImmutableMemoryOptCollection& rhs) = default;

  size_t size() const final {
    size_t num_chunks = chunks_.size();
    if (num_chunks == 0) return 0;
    return (num_chunks - 1) * kChunkSize + last_chunk_size_;
  }
  bool empty() const final { return size() == 0; }
  void ShrinkToFit() final {
    if (!chunks_.empty()) chunks_.back().shrink_to_fit();
    chunks_.shrink_to_fit();
  }
  void Clear() final {
    std::exchange(chunks_, {});
    last_chunk_size_ = 0;
  }

  Status Append(absl::string_view docid) final {
    if (chunks_.empty() || last_chunk_size_ == kChunkSize) {
      last_chunk_size_ = 0;
      chunks_.emplace_back();
    }

    StorePayload(docid, chunks_.back());

    if (++last_chunk_size_ == kChunkSize) {
      chunks_.back().shrink_to_fit();
    }
    return OkStatus();
  }

  std::unique_ptr<DocidCollectionInterface> Copy() const final {
    return std::make_unique<ImmutableMemoryOptCollection>(*this);
  }

  absl::string_view Get(size_t i) const final {
    DCHECK_LT(i, size());
    size_t chunk_num = i / kChunkSize;
    size_t chunk_index = i % kChunkSize;
    absl::string_view payload = LoadPayload(chunks_[chunk_num].data());
    for (size_t x = 0; x < chunk_index; ++x) {
      payload = LoadPayload(payload.data() + payload.size());
    }
    return payload;
  }

  void MultiGet(size_t num_docids, DpIdxGetter docid_idx_getter,
                StringSetter docid_setter) const final {
    for (size_t i = 0; i < num_docids; ++i) {
      DatapointIndex idx = docid_idx_getter(i);
      docid_setter(i, Get(idx));
    }
  }

  size_t capacity() const final { return chunks_.size() * kChunkSize; }

  size_t MemoryUsage() const final {
    size_t result = VectorStorage(chunks_);
    for (const auto& chunk : chunks_) {
      result += VectorStorage(chunk);
    }
    return result;
  }

  void Reserve(DatapointIndex n_elements) final {
    chunks_.reserve(DivRoundUp(n_elements, kChunkSize));
  }

  StatusOr<DocidCollectionInterface::Mutator*> GetMutator() const final {
    return UnimplementedError(
        "This should be handled by VariableLengthDocidCollection.");
  }

  std::unique_ptr<DocidCollectionInterface> ToMutable() && {
    auto result = std::make_unique<MutableCollection>();
    result->Reserve(size());
    for (auto& chunk : chunks_) {
      const char* ptr = chunk.data();
      while (ptr != chunk.data() + chunk.size()) {
        absl::string_view payload = LoadPayload(ptr);
        CHECK_OK(result->Append(payload));
        ptr = payload.data() + payload.size();
      }
      std::exchange(chunk, {});
    }
    Clear();
    return result;
  }

 private:
  using Chunk = std::vector<char>;

  static constexpr size_t kChunkSize = 64;

  absl::string_view LoadPayload(const char* data) const {
    uint32_t length = 0xFF & *data;
    if (ABSL_PREDICT_TRUE(length < 128)) {
      return absl::string_view(data + 1, length);
    }
    length = ~(((0xFF & data[0]) << 24) + ((0xFF & data[1]) << 16) +
               ((0xFF & data[2]) << 8) + ((0xFF & data[3]) << 0));
    return absl::string_view(data + 4, length);
  }

  void StorePayload(absl::string_view payload, Chunk& chunk) {
    assert((payload.size() < 0x80000000u) && "Payload is too large to store");
    uint32_t payload_size = payload.size();
    bool is_small = payload_size < 128;
    size_t current_size = chunk.size();
    AmortizedAppend(chunk, payload.size() + (is_small ? 1 : 4));
    char* data = chunk.data() + current_size;
    if (ABSL_PREDICT_TRUE(payload_size < 128)) {
      *data++ = static_cast<char>(payload_size);
    } else {
      uint32_t inverted_size = ~payload_size;

      *data++ = (inverted_size >> 24) & 0xFF;
      *data++ = (inverted_size >> 16) & 0xFF;
      *data++ = (inverted_size >> 8) & 0xFF;
      *data++ = (inverted_size >> 0) & 0xFF;
    }
    std::copy_n(payload.data(), payload_size, data);
  }

  size_t last_chunk_size_ = 0;

  std::vector<Chunk> chunks_;
};

}  // namespace

VariableLengthDocidCollection::VariableLengthDocidCollection(
    const VariableLengthDocidCollection& rhs) {
  size_ = rhs.size_;
  expected_docid_size_ = rhs.expected_docid_size_;
  if (rhs.impl_) impl_ = rhs.impl_->Copy();
}

VariableLengthDocidCollection& VariableLengthDocidCollection::operator=(
    const VariableLengthDocidCollection& rhs) {
  size_ = rhs.size_;
  expected_docid_size_ = rhs.expected_docid_size_;
  if (rhs.impl_) {
    impl_ = rhs.impl_->Copy();
  } else {
    impl_ = nullptr;
  }
  mutator_ = nullptr;
  return *this;
}

Status VariableLengthDocidCollection::AppendImpl(string_view docid) {
  if (!impl_) {
    if (docid.empty()) {
      ++size_;
      return OkStatus();
    }

    InstantiateImpl();
    impl_->Reserve(expected_docid_size_);
  }
  ++size_;
  return impl_->Append(docid);
}

Status VariableLengthDocidCollection::Append(string_view docid) {
  if (mutator_) {
    SCANN_RETURN_IF_ERROR(mutator_->AddDatapoint(docid));
    return OkStatus();
  }
  return AppendImpl(docid);
}

void VariableLengthDocidCollection::Clear() {
  mutator_ = nullptr;
  impl_ = nullptr;
  expected_docid_size_ = 0;
  size_ = 0;
}

void VariableLengthDocidCollection::ShrinkToFit() {
  if (impl_) impl_->ShrinkToFit();
}

void VariableLengthDocidCollection::InstantiateImpl() {
  if (mutator_) {
    impl_ = make_unique<MutableCollection>(size_);
  } else if (absl::GetFlag(
                 FLAGS_use_memory_optimized_immutable_docid_collection)) {
    impl_ = std::make_unique<ImmutableMemoryOptCollection>(size_);
  } else {
    impl_ = std::make_unique<ImmutableCollection>(size_);
  }
}

void VariableLengthDocidCollection::Reserve(DatapointIndex n_elements) {
  expected_docid_size_ = n_elements;
  if (impl_) impl_->Reserve(n_elements);
}

StatusOr<DocidCollectionInterface::Mutator*>
VariableLengthDocidCollection::GetMutator() const {
  if (!mutator_) {
    auto mutable_this = const_cast<VariableLengthDocidCollection*>(this);
    if (mutable_this->impl_) {
      if (auto* ptr = dynamic_cast<ImmutableCollection*>(impl_.get())) {
        mutable_this->impl_ = MutableCollection::FromImmutableDestructive(ptr);
      } else if (auto* ptr =
                     dynamic_cast<ImmutableMemoryOptCollection*>(impl_.get())) {
        mutable_this->impl_ = std::move(*ptr).ToMutable();
      }
    }
    SCANN_ASSIGN_OR_RETURN(
        mutator_, VariableLengthDocidCollection::Mutator::Create(mutable_this));
  }
  return static_cast<DocidCollectionInterface::Mutator*>(mutator_.get());
}

StatusOr<unique_ptr<VariableLengthDocidCollection::Mutator>>
VariableLengthDocidCollection::Mutator::Create(
    VariableLengthDocidCollection* docids) {
  if (!docids) {
    return InvalidArgumentError("Docids is nullptr");
  }

  SCANN_ASSIGN_OR_RETURN(auto docid_lookup, CreateDocidLookupMap(docids));
  return absl::WrapUnique<VariableLengthDocidCollection::Mutator>(
      new VariableLengthDocidCollection::Mutator(docids,
                                                 std::move(docid_lookup)));
}

bool VariableLengthDocidCollection::Mutator::LookupDatapointIndex(
    string_view docid, DatapointIndex* index) const {
  return docid_lookup_->LookupDatapointIndex(docid, index);
}

void VariableLengthDocidCollection::Mutator::LookupDatapointIndices(
    size_t num_docids, DocidGetter docid_getter,
    LookupCallback callback) const {
  docid_lookup_->LookupDatapointIndices(num_docids, std::move(docid_getter),
                                        std::move(callback));
}

void VariableLengthDocidCollection::Mutator::Reserve(size_t size) {
  docids_->Reserve(size);
  docid_lookup_->Reserve(size);
}

Status VariableLengthDocidCollection::Mutator::AddDatapoint(string_view docid) {
  if (!docid.empty()) {
    DatapointIndex index;
    if (LookupDatapointIndex(docid, &index)) {
      return AlreadyExistsError(
          absl::StrCat("Docid: ", docid, " is duplicated."));
    }
  }

  SCANN_RETURN_IF_ERROR(docids_->AppendImpl(docid));
  if (!docid.empty()) {
    return docid_lookup_->AddDatapoint(docids_->Get(docids_->size() - 1),
                                       docids_->size() - 1);
  }
  return OkStatus();
}

Status VariableLengthDocidCollection::Mutator::RemoveDatapoint(
    string_view docid) {
  DatapointIndex index;
  if (!LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("Docid: ", docid, " is not found."));
  }
  SCANN_RETURN_IF_ERROR(RemoveDatapoint(index));
  return OkStatus();
}

Status VariableLengthDocidCollection::Mutator::RemoveDatapoint(
    DatapointIndex index) {
  if (index >= docids_->size()) {
    return OutOfRangeError(
        absl::StrCat("Removing a datapoint out of bound: index = ", index,
                     ", but size() =  ", docids_->size(), "."));
  }
  if (docids_->all_empty()) {
    docids_->size_--;
    return OkStatus();
  }

  auto impl = down_cast<MutableCollection*>(docids_->impl_.get());
  DCHECK(impl);
  auto old_docid = impl->Get(impl->size() - 1);
  if (!old_docid.empty()) {
    SCANN_RETURN_IF_ERROR(docid_lookup_->RemoveDatapoint(old_docid));
  }

  if (index != impl->size() - 1) {
    auto new_docid = impl->Get(index);
    if (!new_docid.empty()) {
      SCANN_RETURN_IF_ERROR(docid_lookup_->RemoveDatapoint(new_docid));
    }

    impl->Fetch(index) = std::move(impl->Fetch(impl->size() - 1));

    new_docid = impl->Get(index);
    if (!new_docid.empty()) {
      SCANN_RETURN_IF_ERROR(docid_lookup_->AddDatapoint(new_docid, index));
    }
  } else {
    impl->Fetch(index) = ShortStringOptimizedString();
  }
  docids_->size_--;
  impl->size_--;

  return OkStatus();
}

namespace {

ImmutableCollection::ImmutableCollection(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    CHECK_OK(Append(""));
  }
}

Status ImmutableCollection::Append(string_view docid) {
  ++size_;
  if (chunks_.empty() || chunks_.back().payload_offsets.size() == kChunkSize) {
    chunks_.emplace_back();
  }
  auto& back_chunk = chunks_.back();
  const size_t start_offset = back_chunk.payloads.size();
  back_chunk.payload_offsets.push_back(start_offset);
  back_chunk.payloads.insert(back_chunk.payloads.end(), docid.begin(),
                             docid.end());

  if (back_chunk.payload_offsets.size() == kChunkSize) {
    back_chunk.payloads.shrink_to_fit();
  }
  return OkStatus();
}

void ImmutableCollection::ShrinkToFit() {
  if (!chunks_.empty()) {
    chunks_.back().payloads.shrink_to_fit();
    chunks_.back().payload_offsets.shrink_to_fit();
  }
  chunks_.shrink_to_fit();
}

void ImmutableCollection::Clear() {
  chunks_.clear();
  size_ = 0;
}

void ImmutableCollection::Reserve(DatapointIndex n_elements) {
  chunks_.reserve(DivRoundUp(n_elements, kChunkSize));
}

size_t ImmutableCollection::MemoryUsage() const {
  size_t result = VectorStorage(chunks_);
  for (const auto& chunk : chunks_) {
    result += VectorStorage(chunk.payloads);
    result += VectorStorage(chunk.payload_offsets);
  }
  return result;
}

StatusOr<DocidCollectionInterface::Mutator*> ImmutableCollection::GetMutator()
    const {
  return UnimplementedError(
      "This should be handled by VariableLengthDocidCollection.");
}

MutableCollection::MutableCollection(size_t size) {
  Reserve(size);
  for (size_t i = 0; i < size; ++i) {
    CHECK_OK(Append(""));
  }
}

MutableCollection::MutableCollection(const MutableCollection& rhs) {
  chunks_.emplace_back();
  Reserve(rhs.size_);
  for (size_t i = 0; i < rhs.size_; ++i) {
    Fetch(i) = rhs.Fetch(i);
  }
  size_ = rhs.size_;
}

MutableCollection& MutableCollection::operator=(const MutableCollection& rhs) {
  chunks_.clear();
  chunks_.emplace_back();
  Reserve(rhs.size_);
  for (size_t i = 0; i < rhs.size_; ++i) {
    Fetch(i) = rhs.Fetch(i);
  }
  size_ = rhs.size_;
  return *this;
}

unique_ptr<MutableCollection> MutableCollection::FromImmutableDestructive(
    ImmutableCollection* static_impl) {
  auto result = make_unique<MutableCollection>();
  result->Reserve(static_impl->size());
  for (auto& chunk : static_impl->chunks_) {
    for (size_t i : IndicesOf(chunk.payload_offsets)) {
      CHECK_OK(result->Append(chunk.Get(i)));
    }
    FreeBackingStorage(&chunk.payload_offsets);
    FreeBackingStorage(&chunk.payloads);
  }
  return result;
}

Status MutableCollection::Append(string_view docid) {
  ++size_;
  if (size_ > kChunkSize * chunks_.size()) {
    chunks_.emplace_back();
    DCHECK_LE(size_, kChunkSize * chunks_.size());
  }

  Fetch(size_ - 1) = ShortStringOptimizedString(docid);
  return OkStatus();
}

void MutableCollection::ShrinkToFit() {
  chunks_.resize((size_ + kChunkSize - 1) / kChunkSize);
  chunks_.shrink_to_fit();
}

size_t MutableCollection::MemoryUsage() const {
  size_t result =
      sizeof(*this) + sizeof(Chunk) * chunks_.capacity() +
      chunks_.size() * sizeof(ShortStringOptimizedString) * kChunkSize;

  for (size_t i = 0; i < size_; ++i) {
    result += Fetch(i).HeapStorageUsed();
  }
  return result;
}

void MutableCollection::Clear() {
  FreeBackingStorage(&chunks_);
  size_ = 0;
}

void MutableCollection::Reserve(DatapointIndex n_elements) {
  while (chunks_.size() * kChunkSize < n_elements) {
    chunks_.emplace_back();
  }
}

StatusOr<DocidCollectionInterface::Mutator*> MutableCollection::GetMutator()
    const {
  return UnimplementedError(
      "This should be handled by VariableLengthDocidCollection.");
}

}  // namespace

FixedLengthDocidCollection::FixedLengthDocidCollection(
    const FixedLengthDocidCollection& rhs) {
  docid_length_ = rhs.docid_length_;
  size_ = rhs.size_;
  arr_ = rhs.arr_;
}

FixedLengthDocidCollection& FixedLengthDocidCollection::operator=(
    const FixedLengthDocidCollection& rhs) {
  docid_length_ = rhs.docid_length_;
  size_ = rhs.size_;
  arr_ = rhs.arr_;
  return *this;
}

void FixedLengthDocidCollection::ReserveImpl(DatapointIndex n_elements) {
  arr_.reserve(n_elements * docid_length_);
}

void FixedLengthDocidCollection::Reserve(DatapointIndex n_elements) {
  if (mutator_) {
    mutator_->Reserve(n_elements);
    return;
  }
  ReserveImpl(n_elements);
}

Status FixedLengthDocidCollection::AppendImpl(string_view docid) {
  if (docid.size() != docid_length_) {
    return InvalidArgumentError(absl::StrCat(
        "Cannot append a docid of size ", docid.size(),
        " to a FixedLengthDocidCollection of length ", docid_length_, "."));
  }

  ++size_;
  arr_.insert(arr_.end(), docid.begin(), docid.end());
  DCHECK_EQ(size_ * docid_length_, arr_.size());
  return OkStatus();
}

Status FixedLengthDocidCollection::Append(string_view docid) {
  if (mutator_) {
    SCANN_RETURN_IF_ERROR(mutator_->AddDatapoint(docid));
    return OkStatus();
  }
  return AppendImpl(docid);
}

void FixedLengthDocidCollection::MultiGet(size_t num_docids,
                                          DpIdxGetter docid_idx_getter,
                                          StringSetter docid_setter) const {
  constexpr size_t kBatchSize = 24;
  absl::InlinedVector<DatapointIndex, kBatchSize> cached_indices(kBatchSize);

  auto stage1_cb = [&](size_t idx, size_t in_batch_idx) {
    cached_indices[in_batch_idx] = docid_idx_getter(idx);
    DatapointIndex i = cached_indices[in_batch_idx];
    absl::PrefetchToLocalCache(&arr_[i * docid_length_]);
  };
  auto stage2_cb = [&](size_t idx, size_t in_batch_idx) {
    DatapointIndex i = cached_indices[in_batch_idx];
    docid_setter(idx, Get(i));
  };

  RunMultiStageBatchPipeline<kBatchSize, decltype(stage1_cb),
                             decltype(stage2_cb)>(
      num_docids, {std::move(stage1_cb), std::move(stage2_cb)});
}

StatusOr<DocidCollectionInterface::Mutator*>
FixedLengthDocidCollection::GetMutator() const {
  if (!mutator_) {
    auto mutable_this = const_cast<FixedLengthDocidCollection*>(this);
    SCANN_ASSIGN_OR_RETURN(
        mutator_, FixedLengthDocidCollection::Mutator::Create(mutable_this));
  }
  return static_cast<DocidCollectionInterface::Mutator*>(mutator_.get());
}

StatusOr<unique_ptr<FixedLengthDocidCollection::Mutator>>
FixedLengthDocidCollection::Mutator::Create(
    FixedLengthDocidCollection* docids) {
  if (!docids) {
    return InvalidArgumentError("Docids is nullptr");
  }

  SCANN_ASSIGN_OR_RETURN(auto docid_lookup, CreateDocidLookupMap(docids));
  auto result = absl::WrapUnique<FixedLengthDocidCollection::Mutator>(
      new FixedLengthDocidCollection::Mutator(docids, std::move(docid_lookup)));
  return std::move(result);
}

bool FixedLengthDocidCollection::Mutator::LookupDatapointIndex(
    string_view docid, DatapointIndex* index) const {
  return docid_lookup_->LookupDatapointIndex(docid, index);
}

void FixedLengthDocidCollection::Mutator::LookupDatapointIndices(
    size_t num_docids, DocidGetter docid_getter,
    LookupCallback callback) const {
  docid_lookup_->LookupDatapointIndices(num_docids, std::move(docid_getter),
                                        std::move(callback));
}

void FixedLengthDocidCollection::Mutator::Reserve(size_t size) {
  docids_->ReserveImpl(size);
  docid_lookup_->Reserve(docids_->size());
}

Status FixedLengthDocidCollection::Mutator::AddDatapoint(string_view docid) {
  DatapointIndex index;
  if (LookupDatapointIndex(docid, &index)) {
    return AlreadyExistsError(
        absl::StrCat("Docid: ", docid, " is duplicated."));
  }

  if (docids_->capacity() == docids_->size()) {
    const size_t new_size = docids_->size() * kGrowthFactor + 1;
    Reserve(new_size);
  }
  SCANN_RETURN_IF_ERROR(docids_->AppendImpl(docid));
  return docid_lookup_->AddDatapoint(docids_->Get(docids_->size() - 1),
                                     docids_->size() - 1);
}

Status FixedLengthDocidCollection::Mutator::RemoveDatapoint(string_view docid) {
  DatapointIndex index;
  if (!LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("Docid: ", docid, " is not found."));
  }
  SCANN_RETURN_IF_ERROR(RemoveDatapoint(index));
  return OkStatus();
}

Status FixedLengthDocidCollection::Mutator::RemoveDatapoint(
    DatapointIndex index) {
  if (index >= docids_->size()) {
    return OutOfRangeError(
        absl::StrCat("Removing a datapoint out of bound: index = ", index,
                     ", but size() =  ", docids_->size(), "."));
  }

  SCANN_RETURN_IF_ERROR(
      docid_lookup_->RemoveDatapoint(docids_->Get(docids_->size() - 1)));
  if (index != docids_->size() - 1) {
    SCANN_RETURN_IF_ERROR(docid_lookup_->RemoveDatapoint(docids_->Get(index)));
    std::copy(
        docids_->arr_.begin() + (docids_->size() - 1) * docids_->docid_length_,
        docids_->arr_.begin() + docids_->size() * docids_->docid_length_,
        docids_->arr_.begin() + index * docids_->docid_length_);

    SCANN_RETURN_IF_ERROR(
        docid_lookup_->AddDatapoint(docids_->Get(index), index));
  }
  docids_->size_--;
  docids_->arr_.resize(docids_->size() * docids_->docid_length_);
  return OkStatus();
}

}  // namespace research_scann
