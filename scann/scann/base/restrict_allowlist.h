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



#ifndef SCANN_BASE_RESTRICT_ALLOWLIST_H_
#define SCANN_BASE_RESTRICT_ALLOWLIST_H_

#include <cstdint>
#include <limits>

#include "gtest/gtest_prod.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/bit_iterator.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/macros.h"

namespace research_scann {

class RestrictTokenMap;
class RestrictAllowlistConstView;

class RestrictAllowlist {
 public:
  RestrictAllowlist(DatapointIndex num_points, bool default_whitelisted);
  RestrictAllowlist() : RestrictAllowlist(0, false) {}
  ~RestrictAllowlist();

  RestrictAllowlist(std::vector<size_t>&& allowlist_array,
                    DatapointIndex num_points, bool default_whitelisted);

  RestrictAllowlist(const RestrictAllowlist& rhs);
  RestrictAllowlist(RestrictAllowlist&& rhs) noexcept = default;
  RestrictAllowlist& operator=(const RestrictAllowlist& rhs);
  RestrictAllowlist& operator=(RestrictAllowlist&& rhs) = default;

  explicit RestrictAllowlist(const RestrictAllowlistConstView& view);

  void Initialize(DatapointIndex num_points, bool default_whitelisted);

  RestrictAllowlist CopyWithCapacity(
      DatapointIndex capacity,
      vector<size_t>&& backing_storage = vector<size_t>()) const;

  RestrictAllowlist CopyWithSize(
      DatapointIndex size, bool default_whitelisted,
      vector<size_t>&& backing_storage = vector<size_t>()) const;

  void Append(bool is_whitelisted);

  void Resize(size_t num_points, bool default_whitelisted);

  bool CapacityAvailableForAppend(DatapointIndex dp_index) const;
  bool CapacityAvailableForAppend() const {
    return CapacityAvailableForAppend(num_points_);
  }

  bool IsWhitelisted(DatapointIndex dp_index) const {
    DCHECK_LT(dp_index, num_points_);
    return IsWhitelistedNoBoundsCheck(dp_index);
  }

  bool IsWhitelistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    if (dp_index >= num_points()) return default_value;
    return IsWhitelisted(dp_index);
  }

  void set_allowlist_recycling_fn(
      std::function<void(std::vector<size_t>&&)> f) {
    allowlist_recycling_fn_ = std::move(f);
  }

  DatapointIndex NumPointsWhitelisted() const;

  DatapointIndex num_points() const { return num_points_; }
  DatapointIndex size() const { return num_points_; }

  using Iterator = BitIterator<ConstSpan<size_t>, DatapointIndex>;

  Iterator WhitelistedPointIterator() const {
    return Iterator(allowlist_array_);
  }

  size_t GetWordContainingDatapoint(DatapointIndex dp_index) const {
    return allowlist_array_[dp_index / kBitsPerWord];
  }

  size_t* data() { return allowlist_array_.data(); }

  static uint8_t FindLSBSetNonZeroNative(size_t word) {
    static_assert(sizeof(word) == 8 || sizeof(word) == 4, "");
    return (sizeof(word) == 8) ? bits::FindLSBSetNonZero64(word)
                               : bits::FindLSBSetNonZero(word);
  }

  static constexpr size_t kBitsPerWord = sizeof(size_t) * CHAR_BIT;

  static constexpr size_t kOne = 1;

  static constexpr size_t kZero = 0;

  static constexpr size_t kAllOnes = ~kZero;

  static constexpr size_t kRoundDownMask = ~(kBitsPerWord - 1);

 private:
  bool IsWhitelistedNoBoundsCheck(DatapointIndex dp_index) const {
    return GetWordContainingDatapoint(dp_index) &
           (kOne << (dp_index % kBitsPerWord));
  }

  template <typename Lambda>
  void PointwiseLogic(const RestrictAllowlistConstView& rhs, Lambda lambda,
                      bool zero_trailing);

  std::vector<size_t> allowlist_array_;

  DatapointIndex num_points_;

  std::function<void(std::vector<size_t>&&)> allowlist_recycling_fn_;

  friend class RestrictTokenMap;

  friend class RestrictAllowlistConstView;

  FRIEND_TEST(RestrictAllowlist, ResizeUpwardDefaultWhitelisted);
  FRIEND_TEST(RestrictAllowlist, ResizeDownwardDefaultWhitelisted);
  FRIEND_TEST(RestrictAllowlist, ResizeDownwardDefaultNonWhitelisted);
  FRIEND_TEST(RestrictAllowlist, ResizeUpwardDefaultNonWhitelisted);
};

class DummyAllowlist {
 public:
  explicit DummyAllowlist(DatapointIndex num_points);

  bool IsAllowlisted(DatapointIndex dp_index) const { return true; }

  class Iterator {
   public:
    DatapointIndex value() const { return value_; }
    bool Done() const { return value_ >= num_points_; }
    void Next() { ++value_; }

   private:
    friend class DummyAllowlist;

    explicit Iterator(DatapointIndex num_points);

    DatapointIndex value_;

    DatapointIndex num_points_;
  };

  Iterator AllowlistedPointIterator() const { return Iterator(num_points_); }

 private:
  DatapointIndex num_points_;
};

class RestrictAllowlistConstView {
 public:
  RestrictAllowlistConstView() {}

  explicit RestrictAllowlistConstView(const RestrictAllowlist& whitelist)
      : allowlist_array_(whitelist.allowlist_array_.data()),
        num_points_(whitelist.num_points_) {}

  explicit RestrictAllowlistConstView(const RestrictAllowlist* whitelist)
      : allowlist_array_(whitelist ? whitelist->allowlist_array_.data()
                                   : nullptr),
        num_points_(whitelist ? whitelist->num_points_ : 0) {}

  RestrictAllowlistConstView(ConstSpan<size_t> storage,
                             DatapointIndex num_points)
      : allowlist_array_(storage.data()), num_points_(num_points) {
    DCHECK_EQ(storage.size(),
              DivRoundUp(num_points, RestrictAllowlist::kBitsPerWord));
  }

  bool IsWhitelisted(DatapointIndex dp_index) const {
    DCHECK_LT(dp_index, num_points_);
    return GetWordContainingDatapoint(dp_index) &
           (RestrictAllowlist::kOne
            << (dp_index % RestrictAllowlist::kBitsPerWord));
  }

  bool IsWhitelistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    if (dp_index >= num_points()) return default_value;
    return IsWhitelisted(dp_index);
  }

  size_t GetWordContainingDatapoint(DatapointIndex dp_index) const {
    return allowlist_array_[dp_index / RestrictAllowlist::kBitsPerWord];
  }

  const size_t* data() const { return allowlist_array_; }
  DatapointIndex num_points() const { return num_points_; }
  DatapointIndex size() const { return num_points_; }
  bool empty() const { return !allowlist_array_; }

  operator bool() const { return !empty(); }

 private:
  const size_t* allowlist_array_ = nullptr;
  DatapointIndex num_points_ = 0;
  friend class RestrictAllowlist;
};

}  // namespace research_scann

#endif
