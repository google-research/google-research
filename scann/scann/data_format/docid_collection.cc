// Copyright 2022 The Google Research Authors.
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

#include <cstdint>

#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/utils/memory_logging.h"
#include "scann/utils/types.h"

namespace research_scann {

namespace {

class VariableLengthDocidCollectionImplMutable;

class VariableLengthDocidCollectionImplStatic final
    : public DocidCollectionInterface {
 public:
  VariableLengthDocidCollectionImplStatic() {}
  explicit VariableLengthDocidCollectionImplStatic(size_t size);
  ~VariableLengthDocidCollectionImplStatic() final {}
  VariableLengthDocidCollectionImplStatic(
      const VariableLengthDocidCollectionImplStatic& rhs) = default;
  VariableLengthDocidCollectionImplStatic& operator=(
      const VariableLengthDocidCollectionImplStatic& rhs) = default;

  Status Append(string_view docid) final;
  size_t size() const final { return size_; }
  bool empty() const final { return size_ == 0; }
  void ShrinkToFit() final;
  void Clear() final;

  unique_ptr<DocidCollectionInterface> Copy() const final {
    return unique_ptr<DocidCollectionInterface>(
        new VariableLengthDocidCollectionImplStatic(*this));
  }

  string_view Get(size_t i) const final {
    DCHECK_LT(i, size_);
    return chunks_[i / kChunkSize].Get(i % kChunkSize);
  }

  size_t capacity() const final { return chunks_.size() * kChunkSize; }
  size_t MemoryUsage() const final;
  void Reserve(DatapointIndex n_elements) final;
  StatusOr<DocidCollectionInterface::Mutator*> GetMutator() const final;

 private:
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

  friend class VariableLengthDocidCollectionImplMutable;
};

class VariableLengthDocidCollectionImplMutable final
    : public DocidCollectionInterface {
 public:
  VariableLengthDocidCollectionImplMutable() {}
  explicit VariableLengthDocidCollectionImplMutable(size_t size);
  ~VariableLengthDocidCollectionImplMutable() final {}

  VariableLengthDocidCollectionImplMutable(
      const VariableLengthDocidCollectionImplMutable& rhs);
  VariableLengthDocidCollectionImplMutable& operator=(
      const VariableLengthDocidCollectionImplMutable& rhs);

  static unique_ptr<VariableLengthDocidCollectionImplMutable>
  FromStaticImplDestructive(
      VariableLengthDocidCollectionImplStatic* static_impl);

  Status Append(string_view docid) final;
  size_t size() const final { return size_; }
  bool empty() const final { return size_ == 0; }
  void ShrinkToFit() final;
  void Clear() final;

  unique_ptr<DocidCollectionInterface> Copy() const final {
    return unique_ptr<DocidCollectionInterface>(
        new VariableLengthDocidCollectionImplMutable(*this));
  }

  string_view Get(size_t i) const final {
    DCHECK_LT(i, size_);
    return chunks_[i / kChunkSize].payload[i % kChunkSize].ToStringPiece();
  }

  size_t capacity() const final { return chunks_.size() * kChunkSize; }
  size_t MemoryUsage() const final;
  void Reserve(DatapointIndex n_elements) final;

  StatusOr<DocidCollectionInterface::Mutator*> GetMutator() const final;

 private:
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
    impl_ = make_unique<VariableLengthDocidCollectionImplMutable>(size_);
  } else {
    impl_ = make_unique<VariableLengthDocidCollectionImplStatic>(size_);
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
    if (mutable_this->impl_ &&
        typeid(*(mutable_this->impl_)) ==
            typeid(VariableLengthDocidCollectionImplStatic)) {
      mutable_this->impl_ =
          VariableLengthDocidCollectionImplMutable::FromStaticImplDestructive(
              down_cast<VariableLengthDocidCollectionImplStatic*>(impl_.get()));
    }
    TF_ASSIGN_OR_RETURN(
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
  auto result = absl::WrapUnique<VariableLengthDocidCollection::Mutator>(
      new VariableLengthDocidCollection::Mutator(docids));
  result->docid_lookup_.reserve(docids->size());
  for (DatapointIndex i = 0; i < docids->size(); ++i) {
    string_view docid = docids->Get(i);
    if (!docid.empty()) {
      auto emplace_result = result->docid_lookup_.emplace(docid, i);
      if (!emplace_result.second) {
        result->docid_lookup_.clear();
        return AlreadyExistsError(absl::StrCat(
            "Docids contain duplicates. First duplicated docid: ", docid, "."));
      }
    }
  }
  return std::move(result);
}

bool VariableLengthDocidCollection::Mutator::LookupDatapointIndex(
    string_view docid, DatapointIndex* index) const {
  auto it = docid_lookup_.find(docid);
  if (it == docid_lookup_.end()) {
    return false;
  }
  *index = it->second;
  return true;
}

void VariableLengthDocidCollection::Mutator::Reserve(size_t size) {
  docids_->Reserve(size);
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
    docid_lookup_[docids_->Get(docids_->size() - 1)] = docids_->size() - 1;
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

  auto impl = down_cast<VariableLengthDocidCollectionImplMutable*>(
      docids_->impl_.get());
  DCHECK(impl);
  string_view old_docid = impl->Get(impl->size() - 1);
  if (!old_docid.empty()) {
    docid_lookup_.erase(old_docid);
  }

  if (index != impl->size() - 1) {
    string_view new_docid = impl->Get(index);
    if (!new_docid.empty()) {
      docid_lookup_.erase(new_docid);
    }

    impl->Fetch(index) = std::move(impl->Fetch(impl->size() - 1));

    new_docid = impl->Get(index);
    if (!new_docid.empty()) {
      docid_lookup_[new_docid] = index;
    }
  } else {
    impl->Fetch(index) = ShortStringOptimizedString();
  }
  docids_->size_--;
  impl->size_--;

  return OkStatus();
}

namespace {

VariableLengthDocidCollectionImplStatic::
    VariableLengthDocidCollectionImplStatic(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    TF_CHECK_OK(Append(""));
  }
}

Status VariableLengthDocidCollectionImplStatic::Append(string_view docid) {
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

void VariableLengthDocidCollectionImplStatic::ShrinkToFit() {
  if (!chunks_.empty()) {
    chunks_.back().payloads.shrink_to_fit();
    chunks_.back().payload_offsets.shrink_to_fit();
  }
  chunks_.shrink_to_fit();
}

void VariableLengthDocidCollectionImplStatic::Clear() {
  chunks_.clear();
  size_ = 0;
}

void VariableLengthDocidCollectionImplStatic::Reserve(
    DatapointIndex n_elements) {
  chunks_.reserve(DivRoundUp(n_elements, kChunkSize));
}

size_t VariableLengthDocidCollectionImplStatic::MemoryUsage() const {
  size_t result = VectorStorage(chunks_);
  for (const auto& chunk : chunks_) {
    result += VectorStorage(chunk.payloads);
    result += VectorStorage(chunk.payload_offsets);
  }
  return result;
}

StatusOr<DocidCollectionInterface::Mutator*>
VariableLengthDocidCollectionImplStatic::GetMutator() const {
  return UnimplementedError(
      "This should be handled by VariableLengthDocidCollection.");
}

VariableLengthDocidCollectionImplMutable::
    VariableLengthDocidCollectionImplMutable(size_t size) {
  Reserve(size);
  for (size_t i = 0; i < size; ++i) {
    TF_CHECK_OK(Append(""));
  }
}

VariableLengthDocidCollectionImplMutable::
    VariableLengthDocidCollectionImplMutable(
        const VariableLengthDocidCollectionImplMutable& rhs) {
  chunks_.emplace_back();
  Reserve(rhs.size_);
  for (size_t i = 0; i < rhs.size_; ++i) {
    Fetch(i) = rhs.Fetch(i);
  }
  size_ = rhs.size_;
}

VariableLengthDocidCollectionImplMutable&
VariableLengthDocidCollectionImplMutable::operator=(
    const VariableLengthDocidCollectionImplMutable& rhs) {
  chunks_.clear();
  chunks_.emplace_back();
  Reserve(rhs.size_);
  for (size_t i = 0; i < rhs.size_; ++i) {
    Fetch(i) = rhs.Fetch(i);
  }
  size_ = rhs.size_;
  return *this;
}

unique_ptr<VariableLengthDocidCollectionImplMutable>
VariableLengthDocidCollectionImplMutable::FromStaticImplDestructive(
    VariableLengthDocidCollectionImplStatic* static_impl) {
  auto result = make_unique<VariableLengthDocidCollectionImplMutable>();
  result->Reserve(static_impl->size());
  for (auto& chunk : static_impl->chunks_) {
    for (size_t i : IndicesOf(chunk.payload_offsets)) {
      TF_CHECK_OK(result->Append(chunk.Get(i)));
    }
    FreeBackingStorage(&chunk.payload_offsets);
    FreeBackingStorage(&chunk.payloads);
  }
  return result;
}

Status VariableLengthDocidCollectionImplMutable::Append(string_view docid) {
  ++size_;
  if (size_ > kChunkSize * chunks_.size()) {
    chunks_.emplace_back();
    DCHECK_LE(size_, kChunkSize * chunks_.size());
  }

  Fetch(size_ - 1) = ShortStringOptimizedString(docid);
  return OkStatus();
}

void VariableLengthDocidCollectionImplMutable::ShrinkToFit() {
  chunks_.resize((size_ + kChunkSize - 1) / kChunkSize);
  chunks_.shrink_to_fit();
}

size_t VariableLengthDocidCollectionImplMutable::MemoryUsage() const {
  size_t result =
      sizeof(*this) + sizeof(Chunk) * chunks_.capacity() +
      chunks_.size() * sizeof(ShortStringOptimizedString) * kChunkSize;

  for (size_t i = 0; i < size_; ++i) {
    result += Fetch(i).HeapStorageUsed();
  }
  return result;
}

void VariableLengthDocidCollectionImplMutable::Clear() {
  FreeBackingStorage(&chunks_);
  size_ = 0;
}

void VariableLengthDocidCollectionImplMutable::Reserve(
    DatapointIndex n_elements) {
  while (chunks_.size() * kChunkSize < n_elements) {
    chunks_.emplace_back();
  }
}

StatusOr<DocidCollectionInterface::Mutator*>
VariableLengthDocidCollectionImplMutable::GetMutator() const {
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

StatusOr<DocidCollectionInterface::Mutator*>
FixedLengthDocidCollection::GetMutator() const {
  if (!mutator_) {
    auto mutable_this = const_cast<FixedLengthDocidCollection*>(this);
    auto statusor = FixedLengthDocidCollection::Mutator::Create(mutable_this);
    SCANN_RETURN_IF_ERROR(statusor.status());
    mutator_ = std::move(statusor).ValueOrDie();
  }
  return static_cast<DocidCollectionInterface::Mutator*>(mutator_.get());
}

StatusOr<unique_ptr<FixedLengthDocidCollection::Mutator>>
FixedLengthDocidCollection::Mutator::Create(
    FixedLengthDocidCollection* docids) {
  if (!docids) {
    return InvalidArgumentError("Docids is nullptr");
  }
  auto result = absl::WrapUnique<FixedLengthDocidCollection::Mutator>(
      new FixedLengthDocidCollection::Mutator(docids));
  result->docid_lookup_.reserve(docids->size());
  for (DatapointIndex i = 0; i < docids->size(); ++i) {
    string_view docid = docids->Get(i);
    if (!docid.empty()) {
      result->docid_lookup_[docid] = i;
    }
  }
  return std::move(result);
}

bool FixedLengthDocidCollection::Mutator::LookupDatapointIndex(
    string_view docid, DatapointIndex* index) const {
  auto it = docid_lookup_.find(docid);
  if (it == docid_lookup_.end()) {
    return false;
  }
  *index = it->second;
  return true;
}

void FixedLengthDocidCollection::Mutator::Reserve(size_t size) {
  docids_->ReserveImpl(size);
  docid_lookup_.clear();
  docid_lookup_.reserve(size);
  for (DatapointIndex i = 0; i < docids_->size(); ++i) {
    string_view docid = docids_->Get(i);
    if (!docid.empty()) {
      docid_lookup_[docid] = i;
    }
  }
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

  docid_lookup_[docids_->Get(docids_->size() - 1)] = docids_->size() - 1;
  return OkStatus();
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

  docid_lookup_.erase(docids_->Get(docids_->size() - 1));
  if (index != docids_->size() - 1) {
    docid_lookup_.erase(docids_->Get(index));
    std::copy(
        docids_->arr_.begin() + (docids_->size() - 1) * docids_->docid_length_,
        docids_->arr_.begin() + docids_->size() * docids_->docid_length_,
        docids_->arr_.begin() + index * docids_->docid_length_);

    docid_lookup_[docids_->Get(index)] = index;
  }
  docids_->size_--;
  docids_->arr_.resize(docids_->size() * docids_->docid_length_);
  return OkStatus();
}

}  // namespace research_scann
