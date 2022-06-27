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

#ifndef SCANN_UTILS_ITERATORS_H_
#define SCANN_UTILS_ITERATORS_H_

#include <iterator>

#include "scann/utils/types.h"

namespace research_scann {

template <typename CollectionT>
class RandomAccessIterator {
 public:
  using Iter = RandomAccessIterator<CollectionT>;

  using difference_type = ptrdiff_t;
  using reference = decltype(std::declval<CollectionT>()[0]);
  using pointer = typename std::remove_reference<reference>::type*;
  using value_type = typename std::iterator_traits<pointer>::value_type;
  using iterator_category = std::random_access_iterator_tag;

  RandomAccessIterator(CollectionT* collection, size_t offset)
      : collection_(collection), offset_(offset) {}

  reference Get(size_t idx) const { return collection_->operator[](idx); }
  reference operator*() const { return Get(offset_); }
  reference operator[](difference_type d) const { return Get(offset_ + d); }
  pointer operator->() const {
    static_assert(std::is_lvalue_reference<reference>::value,
                  "The -> operator is not supported when "
                  "CollectionT::operator[] returns an immediate rvalue (eg "
                  "`int`) instead of an lvalue-reference (eg `const int&`).");
    return &Get(offset_);
  }

  Iter& operator++() {
    ++offset_;
    return *this;
  }
  Iter& operator--() {
    --offset_;
    return *this;
  }
  Iter operator++(int) {
    ++offset_;
    return *this;
  }
  Iter operator--(int) {
    --offset_;
    return *this;
  }
  Iter& operator+=(difference_type d) {
    offset_ += d;
    return *this;
  }
  Iter& operator-=(difference_type d) {
    offset_ -= d;
    return *this;
  }

  friend Iter operator+(const Iter& it, difference_type d) {
    return Iter(it.collection_, it.offset_ + d);
  }
  friend Iter operator+(difference_type d, const Iter& it) {
    return Iter(it.collection_, it.offset_ + d);
  }
  friend Iter operator-(const Iter& it, difference_type d) {
    return Iter(it.collection_, it.offset_ - d);
  }

  friend difference_type operator-(const Iter& a, const Iter& b) {
    DCHECK_EQ(a.collection_, b.collection_);
    return a.offset_ - b.offset_;
  }

  friend bool operator==(const Iter& a, const Iter& b) {
    DCHECK_EQ(a.collection_, b.collection_);
    return a.offset_ == b.offset_;
  }
  friend bool operator!=(const Iter& a, const Iter& b) {
    DCHECK_EQ(a.collection_, b.collection_);
    return a.offset_ != b.offset_;
  }
  friend bool operator<(const Iter& a, const Iter& b) {
    DCHECK_EQ(a.collection_, b.collection_);
    return a.offset_ < b.offset_;
  }
  friend bool operator>(const Iter& a, const Iter& b) {
    DCHECK_EQ(a.collection_, b.collection_);
    return a.offset_ > b.offset_;
  }
  friend bool operator<=(const Iter& a, const Iter& b) {
    DCHECK_EQ(a.collection_, b.collection_);
    return a.offset_ <= b.offset_;
  }
  friend bool operator>=(const Iter& a, const Iter& b) {
    DCHECK_EQ(a.collection_, b.collection_);
    return a.offset_ >= b.offset_;
  }

 private:
  CollectionT* collection_;
  size_t offset_ = 0;
};

}  // namespace research_scann

#endif
