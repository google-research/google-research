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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef SCANN__UTILS_INFINITE_ONE_ARRAY_H_
#define SCANN__UTILS_INFINITE_ONE_ARRAY_H_

#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
class InfiniteOneIterator {
 public:
  T operator*() const { return T(1); }
  T operator[](size_t i) const { return T(1); }
  void operator++() {}
  void operator++(int dummy) {}
  InfiniteOneIterator operator+=(size_t i) { return *this; }
  InfiniteOneIterator operator-=(size_t i) { return *this; }

  bool operator<(const InfiniteOneIterator<T>& x) const { return true; }
};

template <typename T>
class InfiniteOneArray {
 public:
  InfiniteOneArray() {}
  size_t size() const { return numeric_limits<size_t>::max(); }
  bool empty() const { return false; }
  T operator[](size_t i) const { return T(1); }
  T front() const { return T(1); }
  T back() const { return T(1); }
  InfiniteOneIterator<T> begin() { return InfiniteOneIterator<T>(); }
  InfiniteOneIterator<T> end() { return InfiniteOneIterator<T>(); }
  InfiniteOneArray data() const { return *this; }
};

template <typename T>
class InfiniteOneMatrix {
 public:
  InfiniteOneMatrix() {}
  size_t size() const { return numeric_limits<size_t>::max(); }
  bool empty() const { return false; }
  InfiniteOneArray<T> operator[](size_t i) const {
    return InfiniteOneArray<T>();
  }

  InfiniteOneArray<T> front() const { return InfiniteOneArray<T>(); }
  InfiniteOneArray<T> back() const { return InfiniteOneArray<T>(); }
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
