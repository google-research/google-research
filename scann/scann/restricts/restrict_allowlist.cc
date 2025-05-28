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



#include "scann/restricts/restrict_allowlist.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "scann/utils/bits.h"
#include "scann/utils/common.h"

namespace research_scann {
namespace {

void ClearRemainderBits(MutableSpan<size_t> allowlist_array,
                        size_t num_points) {
  const uint8_t num_used_bits_in_last_word =
      num_points % RestrictAllowlist::kBitsPerWord;

  if (num_used_bits_in_last_word == 0) return;
  DCHECK(!allowlist_array.empty());
  allowlist_array[allowlist_array.size() - 1] = research_scann::GetLowBits(
      allowlist_array[allowlist_array.size() - 1], num_used_bits_in_last_word);
}

void SetRemainderBits(MutableSpan<size_t> allowlist_array, size_t num_points) {
  const uint8_t num_used_bits_in_last_word =
      num_points % RestrictAllowlist::kBitsPerWord;
  if (num_used_bits_in_last_word == 0) return;
  DCHECK(!allowlist_array.empty());
  allowlist_array[allowlist_array.size() - 1] |= RestrictAllowlist::kAllOnes
                                                 << num_used_bits_in_last_word;
}

}  // namespace

RestrictAllowlist::RestrictAllowlist(DatapointIndex num_points,
                                     bool default_allowlisted) {
  Initialize(num_points, default_allowlisted);
}

RestrictAllowlist::RestrictAllowlist(const RestrictAllowlistConstView& view)
    : allowlist_array_(
          view.allowlist_array_,
          view.allowlist_array_ + DivRoundUp(view.num_points_, kBitsPerWord)),
      num_points_(view.num_points_) {}

RestrictAllowlist::RestrictAllowlist(std::vector<size_t>&& allowlist_array,
                                     DatapointIndex num_points,
                                     bool default_allowlisted)
    : allowlist_array_(std::move(allowlist_array)), num_points_(num_points) {
  CHECK_EQ(allowlist_array_.size(), DivRoundUp(num_points, kBitsPerWord));

  VLOG(1) << "Using recycled allowlist_array_ at " << allowlist_array_.data();
  const size_t to_fill = default_allowlisted ? kAllOnes : 0;
  std::fill(allowlist_array_.begin(), allowlist_array_.end(), to_fill);
  if (default_allowlisted) {
    ClearRemainderBits(MakeMutableSpan(allowlist_array_), num_points);
  }
}

RestrictAllowlist::~RestrictAllowlist() {}

void RestrictAllowlist::Initialize(DatapointIndex num_points,
                                   bool default_allowlisted) {
  num_points_ = num_points;

  allowlist_array_.resize(0);
  allowlist_array_.resize(DivRoundUp(num_points, kBitsPerWord),
                          default_allowlisted ? kAllOnes : 0);
  if (default_allowlisted) {
    ClearRemainderBits(MakeMutableSpan(allowlist_array_), num_points);
  }
}

void RestrictAllowlist::Resize(size_t num_points, bool default_allowlisted) {
  if (default_allowlisted && num_points > num_points_) {
    SetRemainderBits(MakeMutableSpan(allowlist_array_), num_points_);
  }

  const size_t n_words =
      num_points / kBitsPerWord + (num_points % kBitsPerWord > 0);
  allowlist_array_.resize(n_words, (default_allowlisted ? kAllOnes : 0));
  num_points_ = num_points;
  ClearRemainderBits(MakeMutableSpan(allowlist_array_), num_points);
}

DatapointIndex RestrictAllowlist::NumPointsAllowlisted() const {
  DatapointIndex result = 0;
  for (size_t elem : allowlist_array_) {
    result += absl::popcount(elem);
  }
  return result;
}

RestrictAllowlist::RestrictAllowlist(const RestrictAllowlist& rhs) = default;
RestrictAllowlist& RestrictAllowlist::operator=(const RestrictAllowlist& rhs) =
    default;

DummyAllowlist::DummyAllowlist(DatapointIndex num_points)
    : num_points_(num_points) {}

DummyAllowlist::Iterator::Iterator(DatapointIndex num_points)
    : value_(0), num_points_(num_points) {}

RestrictAllowlist CreateAllowlist(RestrictAllowlistRecycler* recycler,
                                  DatapointIndex num_datapoints,
                                  bool default_allowed) {
  if (!recycler) {
    return RestrictAllowlist(num_datapoints, default_allowed);
  }
  vector<size_t> to_recycle = recycler->MaybeRemoveFromFreelist();
  VLOG(2) << "Creating allowlist with recycled at " << to_recycle.data();
  RestrictAllowlist result;
  if (to_recycle.empty()) {
    result = RestrictAllowlist(num_datapoints, default_allowed);
  } else {
    result = RestrictAllowlist(std::move(to_recycle), num_datapoints,
                               default_allowed);
  }
  result.set_allowlist_recycling_fn(recycler->AddToFreelistFunctor());
  return result;
}
}  // namespace research_scann
