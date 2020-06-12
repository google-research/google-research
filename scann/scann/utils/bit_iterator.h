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



#ifndef SCANN__UTILS_BIT_ITERATOR_H_
#define SCANN__UTILS_BIT_ITERATOR_H_

#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename StorageT, typename PositionT = uint32_t>
class BitIterator {
 public:
  BitIterator() {}

  explicit inline BitIterator(StorageT whitelist_storage);

  PositionT value() const { return value_; }

  inline void Next();

  bool Done() const { return cur_whitelist_word_ == 0; }

 private:
  inline void NextNonzeroWord();

  StorageT whitelist_array_;

  size_t whitelist_array_position_ = 0;

  PositionT word_base_ = 0;

  PositionT value_ = 0;

  using WordType = decay_t<decltype(whitelist_array_[0])>;
  WordType cur_whitelist_word_ = 0;
};

namespace bits_internal {

template <typename Int>
inline int FindLSBSetNonZero(Int x) {
  DCHECK_NE(x, 0);
  if (sizeof(x) * CHAR_BIT > 32) {
    return bits::FindLSBSetNonZero64(x);
  } else {
    return bits::FindLSBSetNonZero(x);
  }
}

}  // namespace bits_internal

template <typename StorageT, typename PositionT>
BitIterator<StorageT, PositionT>::BitIterator(StorageT whitelist_storage)
    : whitelist_array_(std::move(whitelist_storage)), value_(0) {
  NextNonzeroWord();
  if (cur_whitelist_word_ == 0) return;
  value_ = word_base_ + bits_internal::FindLSBSetNonZero(cur_whitelist_word_);
}

template <typename StorageT, typename PositionT>
void BitIterator<StorageT, PositionT>::Next() {
  cur_whitelist_word_ &= cur_whitelist_word_ - 1;
  if (cur_whitelist_word_ == 0) {
    ++whitelist_array_position_;
    NextNonzeroWord();
    if (cur_whitelist_word_ == 0) return;
  }
  value_ = word_base_ + bits_internal::FindLSBSetNonZero(cur_whitelist_word_);
}

template <typename StorageT, typename PositionT>
void BitIterator<StorageT, PositionT>::NextNonzeroWord() {
  static constexpr size_t kBitsPerWord = sizeof(cur_whitelist_word_) * CHAR_BIT;
  for (; whitelist_array_position_ < whitelist_array_.size();
       ++whitelist_array_position_) {
    cur_whitelist_word_ = whitelist_array_[whitelist_array_position_];
    if (cur_whitelist_word_ != 0) {
      word_base_ = whitelist_array_position_ * kBitsPerWord;
      return;
    }
  }
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
