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



#ifndef SCANN_UTILS_ZIP_SORT_H_
#define SCANN_UTILS_ZIP_SORT_H_

#include "scann/utils/zip_sort_impl.h"

namespace research_scann {

template <typename Predicate, typename T, typename... U>
size_t ZipPartition(Predicate pred, T begin, T end, U... rest);

template <typename Comparator, typename T, typename... U>
void ZipNthElement(Comparator comp, size_t n, T begin, T end, U... rest) {
  zip_sort_internal::ZipNthElementImpl<Comparator, T, U...>(comp, n, begin, end,
                                                            rest...);
}

template <typename Comparator, typename T, typename... U>
void ZipNthElementBranchOptimized(Comparator comp, size_t n, T begin, T end,
                                  U... rest) {
  zip_sort_internal::ZipNthElementImplBranchOptimized<Comparator, T, U...>(
      comp, n, begin, end, rest...);
}

template <typename T, typename... U>
void ZipSort(T begin, T end, U... rest) {
  zip_sort_internal::DefaultComparator comp;
  zip_sort_internal::ZipSortImpl<zip_sort_internal::DefaultComparator, T, U...>(
      comp, 0, end - begin, zip_sort_internal::ComputeDepthLimit(end - begin),
      begin, rest...);
}

template <typename Comparator, typename T, typename... U>
void ZipSort(Comparator comp, T begin, T end, U... rest) {
  zip_sort_internal::ZipSortImpl<Comparator, T, U...>(
      comp, 0, end - begin, zip_sort_internal::ComputeDepthLimit(end - begin),
      begin, rest...);
}

template <typename Comparator, typename T, typename... U>
void ZipSortBranchOptimized(Comparator comp, T begin, T end, U... rest) {
  zip_sort_internal::ZipSortImplBranchOptimized<Comparator, T, U...>(
      comp, 0, end - begin, zip_sort_internal::ComputeDepthLimit(end - begin),
      begin, rest...);
}

template <typename T, typename... U>
void ZipSortBranchOptimized(T begin, T end, U... rest) {
  ZipSortBranchOptimized(zip_sort_internal::DefaultComparator(), begin, end,
                         rest...);
}

template <typename Iter, typename Compare>
void NthElementBranchOptimized(Iter begin, Iter nth, Iter end, Compare comp) {
  ZipNthElementBranchOptimized(comp, nth - begin, begin, end);
}

template <typename Iter>
void NthElementBranchOptimized(Iter begin, Iter nth, Iter end) {
  using T = typename std::iterator_traits<Iter>::value_type;
  ZipNthElementBranchOptimized(std::less<T>(), nth - begin, begin, end);
}

template <typename Iter, typename Comparator>
void SortBranchOptimized(Iter begin, Iter end, Comparator comp) {
  ZipSortBranchOptimized(comp, begin, end);
}

template <typename Iter>
void SortBranchOptimized(Iter begin, Iter end) {
  using T = typename std::iterator_traits<Iter>::value_type;
  ZipSortBranchOptimized(std::less<T>(), begin, end);
}

class NanAwareGreater {
 public:
  template <typename T>
  bool operator()(const T& a, const T& b) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(a)) return false;
      if (std::isnan(b)) return true;
    }
    return a > b;
  }
};

class NanAwareLess {
 public:
  template <typename T>
  bool operator()(const T& a, const T& b) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(a)) return false;
      if (std::isnan(b)) return true;
    }
    return a < b;
  }
};

}  // namespace research_scann

#endif
