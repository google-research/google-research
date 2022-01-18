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



#ifndef SCANN_DATA_FORMAT_INTERNAL_SHORT_STRING_OPTIMIZED_STRING_H_
#define SCANN_DATA_FORMAT_INTERNAL_SHORT_STRING_OPTIMIZED_STRING_H_

#include <cstdint>
#include <cstdlib>

#include "absl/types/optional.h"
#include "scann/oss_wrappers/scann_malloc_extension.h"
#include "scann/utils/types.h"

namespace research_scann {

class ShortStringOptimizedString {
 public:
  ShortStringOptimizedString() { memset(storage_, 0, kStorageSize); }

  explicit ShortStringOptimizedString(string_view orig) {
    ConstructFromStringPiece(orig);
  }

  ShortStringOptimizedString(const ShortStringOptimizedString& rhs)
      : ShortStringOptimizedString(rhs.ToStringPiece()) {}
  ShortStringOptimizedString& operator=(const ShortStringOptimizedString& rhs) {
    this->~ShortStringOptimizedString();
    ConstructFromStringPiece(rhs.ToStringPiece());
    return *this;
  }

  ShortStringOptimizedString& operator=(const string_view rhs) {
    this->~ShortStringOptimizedString();
    ConstructFromStringPiece(rhs);
    return *this;
  }

  ShortStringOptimizedString(ShortStringOptimizedString&& rhs) noexcept {
    memcpy(storage_, rhs.storage_, kStorageSize);
    rhs.ClearNoFree();
  }

  ShortStringOptimizedString& operator=(ShortStringOptimizedString&& rhs) {
    this->~ShortStringOptimizedString();
    memcpy(storage_, rhs.storage_, kStorageSize);
    rhs.ClearNoFree();
    return *this;
  }

  const char* data() const {
    return (size() <= kMaxInline) ? storage_ : heap_string();
  }

  uint32_t size() const {
    return *reinterpret_cast<const uint32_t*>(storage_ + kStorageSize -
                                              sizeof(uint32_t));
  }

  bool empty() const { return !size(); }

  string_view ToStringPiece() const { return string_view(data(), size()); }

  operator std::string() const { return std::string(ToStringPiece()); }

  size_t HeapStorageUsed() const {
    if (size() <= kMaxInline) return 0;
    absl::optional<size_t> true_size =
        MallocExtension::GetAllocatedSize(heap_string());
    return *true_size;
  }

  bool operator==(string_view s) const { return ToStringPiece() == s; }

  ~ShortStringOptimizedString() {
    if (size() > kMaxInline) {
      delete[] heap_string();
      memset(storage_, 0, kStorageSize);
    }
  }

 private:
  static_assert(sizeof(uint32_t) == 4, "The uint32 typedef is wrong.");

  static_assert(sizeof(char*) == 4 || sizeof(char*) == 8,
                "ScaNN only supports 32- and 64-bit flat memory models.");

  static constexpr size_t kStorageSize = (sizeof(char*) == 4) ? 8 : 16;

  static constexpr size_t kMaxInline = kStorageSize - sizeof(uint32_t);

  void ConstructFromStringPiece(string_view orig) {
    set_size(orig.size());
    if (orig.size() > kMaxInline) {
      char* heap_string = new char[orig.size()];
      memcpy(heap_string, orig.data(), orig.size());
      set_heap_string(heap_string);
    } else {
      memcpy(storage_, orig.data(), orig.size());
    }
  }

  void ClearNoFree() { memset(storage_, 0, kStorageSize); }

  const char* heap_string() const {
    return *reinterpret_cast<const char* const*>(storage_);
  }

  void set_heap_string(char* s) { *reinterpret_cast<char**>(storage_) = s; }

  void set_size(uint32_t s) {
    *(reinterpret_cast<uint32_t*>(storage_ + kStorageSize - sizeof(uint32_t))) =
        s;
  }

  union {
    char storage_[kStorageSize];

    pair<char*, uint32_t> for_alignment_only_;

    static_assert(sizeof(for_alignment_only_) == kStorageSize, "");
  };
};

static_assert(sizeof(ShortStringOptimizedString) == sizeof(string_view), "");

}  // namespace research_scann

#endif
