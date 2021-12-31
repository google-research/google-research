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

#ifndef SCANN_UTILS_INDEX_SEQUENCE_H_
#define SCANN_UTILS_INDEX_SEQUENCE_H_

#include <utility>

namespace research_scann {

using ::std::index_sequence;
using ::std::make_index_sequence;

template <size_t... kInts>
inline constexpr size_t index_sequence_sum_v = (kInts + ...);

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#endif
template <size_t... kInts>
inline constexpr size_t index_sequence_last_v = (kInts, ...);
#ifdef __clang__
#pragma clang diagnostic pop
#endif

template <typename Seq0, typename Seq1>
struct index_sequence_all_but_last_impl;

template <size_t... kFirst, size_t kMiddle, size_t... kLast>
struct index_sequence_all_but_last_impl<
    std::index_sequence<kFirst...>, std::index_sequence<kMiddle, kLast...>> {
  using type = typename index_sequence_all_but_last_impl<
      std::index_sequence<kFirst..., kMiddle>,
      std::index_sequence<kLast...>>::type;
};

template <size_t... kFirst, size_t kLast>
struct index_sequence_all_but_last_impl<std::index_sequence<kFirst...>,
                                        std::index_sequence<kLast>> {
  using type = std::index_sequence<kFirst...>;
};

template <size_t... kInts>
using index_sequence_all_but_last_t = typename index_sequence_all_but_last_impl<
    std::index_sequence<>, std::index_sequence<kInts...>>::type;

static_assert(std::is_same_v<index_sequence_all_but_last_t<1, 2, 3>,
                             std::index_sequence<1, 2>>);

static_assert(std::is_same_v<index_sequence_all_but_last_t<3, 5, 4, 7>,
                             std::index_sequence<3, 5, 4>>);

}  // namespace research_scann

#endif
