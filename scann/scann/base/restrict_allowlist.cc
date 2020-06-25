// Copyright 2020 The Google Research Authors.
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



#include "scann/base/restrict_allowlist.h"

#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/common.h"

namespace tensorflow {
namespace scann_ops {
namespace {

void ClearRemainderBits(MutableSpan<size_t> whitelist_array,
                        size_t num_points) {
  const uint8_t num_leftover_bits =
      RestrictAllowlist::kBitsPerWord -
      num_points % RestrictAllowlist::kBitsPerWord;

  if (num_leftover_bits == RestrictAllowlist::kBitsPerWord) return;
  DCHECK(!whitelist_array.empty());
  whitelist_array[whitelist_array.size() - 1] &=
      RestrictAllowlist::kAllOnes >> num_leftover_bits;
}

void SetRemainderBits(MutableSpan<size_t> whitelist_array, size_t num_points) {
  const uint8_t num_used_bits_in_last_word =
      num_points % RestrictAllowlist::kBitsPerWord;
  if (num_used_bits_in_last_word == 0) return;
  DCHECK(!whitelist_array.empty());
  whitelist_array[whitelist_array.size() - 1] |= RestrictAllowlist::kAllOnes
                                                 << num_used_bits_in_last_word;
}

}  // namespace

RestrictAllowlist::RestrictAllowlist(DatapointIndex num_points,
                                     bool default_whitelisted) {
  Initialize(num_points, default_whitelisted);
}

RestrictAllowlist::RestrictAllowlist(const RestrictAllowlistConstView& view)
    : whitelist_array_(
          view.whitelist_array_,
          view.whitelist_array_ + DivRoundUp(view.num_points_, kBitsPerWord)),
      num_points_(view.num_points_) {}

void RestrictAllowlist::Initialize(DatapointIndex num_points,
                                   bool default_whitelisted) {
  num_points_ = num_points;
  whitelist_array_.clear();
  whitelist_array_.resize(DivRoundUp(num_points, kBitsPerWord),
                          default_whitelisted ? kAllOnes : 0);
  if (default_whitelisted) {
    ClearRemainderBits(MakeMutableSpan(whitelist_array_), num_points);
  }
}

void RestrictAllowlist::Resize(size_t num_points, bool default_whitelisted) {
  if (default_whitelisted && num_points > num_points_) {
    SetRemainderBits(MakeMutableSpan(whitelist_array_), num_points_);
  }

  const size_t n_words =
      num_points / kBitsPerWord + (num_points % kBitsPerWord > 0);
  whitelist_array_.resize(n_words, (default_whitelisted ? kAllOnes : 0));
  num_points_ = num_points;
  ClearRemainderBits(MakeMutableSpan(whitelist_array_), num_points);
}

DummyAllowlist::DummyAllowlist(DatapointIndex num_points)
    : num_points_(num_points) {}

DummyAllowlist::Iterator::Iterator(DatapointIndex num_points)
    : value_(0), num_points_(num_points) {}

}  // namespace scann_ops
}  // namespace tensorflow
