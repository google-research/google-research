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



#ifndef SCANN__UTILS_ZIP_SORT_H_
#define SCANN__UTILS_ZIP_SORT_H_

#include "scann/utils/zip_sort_impl.h"

namespace tensorflow {
namespace scann_ops {

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

template <typename T, typename... U>
void ZipSortBranchOptimized(T begin, T end, U... rest) {
  zip_sort_internal::DefaultComparator comp;
  zip_sort_internal::ZipSortImplBranchOptimized<
      zip_sort_internal::DefaultComparator, T, U...>(
      comp, 0, end - begin, zip_sort_internal::ComputeDepthLimit(end - begin),
      begin, rest...);
}

template <typename Comparator, typename T, typename... U>
void ZipSortBranchOptimized(Comparator comp, T begin, T end, U... rest) {
  zip_sort_internal::ZipSortImplBranchOptimized<Comparator, T, U...>(
      comp, 0, end - begin, zip_sort_internal::ComputeDepthLimit(end - begin),
      begin, rest...);
}

template <typename Iter, typename Comparator>
void SortBranchOptimized(Iter begin, Iter end, Comparator comp) {
  zip_sort_internal::ZipSortImplBranchOptimized<Comparator, Iter>(
      comp, 0, end - begin, zip_sort_internal::ComputeDepthLimit(end - begin),
      begin);
}

template <typename Iter>
void SortBranchOptimized(Iter begin, Iter end) {
  SortBranchOptimized(
      begin, end, std::less<typename std::iterator_traits<Iter>::value_type>());
}

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

template <typename Predicate, typename T, typename... U>
size_t ZipPartition(Predicate pred, T begin, T end, U... rest);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
