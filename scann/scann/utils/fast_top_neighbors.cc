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

#include "scann/utils/fast_top_neighbors.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/strings/str_cat.h"
#include "scann/utils/bits.h"
#include "scann/utils/common.h"
#include "scann/utils/hwy-compact.h"
#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/zip_sort.h"

namespace research_scann {
namespace {

constexpr bool kShouldLog = false;

template <typename DistT, typename DatapointIndexT>
std::string FTN_DebugLogArrayContents(DatapointIndexT* indices, DistT* values,
                                      uint32_t* masks, size_t sz) {
  std::string txt;
  for (size_t j : Seq(DivRoundUp(sz, 32))) {
    absl::StrAppend(&txt, j ? " || " : "");
    uint32_t mask = masks ? masks[j] : 0;
    string_view sep = "";
    for (size_t offset : Seq(32)) {
      const size_t idx = 32 * j + offset;
      const bool idx_is_masked = mask & 1;
      mask >>= 1;
      if (idx == sz) {
        sep = ".  END: ";
      }
      if (idx_is_masked) {
        if constexpr (std::is_same_v<DatapointIndexT,
                                     std::pair<uint64_t, uint64_t>>) {
          StrAppendFormat(&txt, "%s%d:[%ld, %s]", sep, idx, indices[idx].first,
                          absl::StrCat(values[idx]));
        } else if constexpr (std::is_same_v<DatapointIndexT,
                                            std::shared_ptr<std::string>>) {
          StrAppendFormat(&txt, "%s%d:[%ld, %s]", sep, idx,
                          reinterpret_cast<uint64_t>(indices[idx].get()),
                          absl::StrCat(values[idx]));
        } else {
          StrAppendFormat(&txt, "%s%d:[%d, %s]", sep, idx, indices[idx],
                          absl::StrCat(values[idx]));
        }
      } else {
        if constexpr (std::is_same_v<DatapointIndexT,
                                     std::pair<uint64_t, uint64_t>>) {
          StrAppendFormat(&txt, "%s%d:[%ld, %s]", sep, idx, indices[idx].first,
                          absl::StrCat(values[idx]));
        } else if constexpr (std::is_same_v<DatapointIndexT,
                                            std::shared_ptr<std::string>>) {
          StrAppendFormat(&txt, "%s%d:[%ld, %s]", sep, idx,
                          reinterpret_cast<uint64_t>(indices[idx].get()),
                          absl::StrCat(values[idx]));
        } else {
          StrAppendFormat(&txt, "%s%d:{%d, %s}", sep, idx, indices[idx],
                          absl::StrCat(values[idx]));
        }
      }
      sep = ", ";
      if (idx == sz) {
        if (mask) {
          StrAppendFormat(&txt, ". EXTRA_MASK_BITS=%d", mask);
        }
        break;
      }
    }
  }
  return StrFormat("Array Contents: %s. sz=%d", txt, sz);
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE bool CompIV(DatapointIndexT idx_a, DatapointIndexT idx_b,
                         DistT value_a, DistT value_b) {
  const bool is_eq_or_nan =
      value_a == value_b || std::isunordered(value_a, value_b);
  if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
    return idx_a < idx_b;
  }
  return value_a < value_b;
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE void ZipSwap(size_t a, size_t b, DatapointIndexT* indices,
                          DistT* values) {
  std::swap(indices[a], indices[b]);
  std::swap(values[a], values[b]);
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE void CompOrSwap(size_t a, size_t b, DatapointIndexT* indices,
                             DistT* values) {
  if (!CompIV(indices[a], indices[b], values[a], values[b])) {
    ZipSwap(a, b, indices, values);
  }
}

template <typename DistT, typename DatapointIndexT>
SCANN_INLINE void SelectionSort(DatapointIndexT* indices, DistT* values,
                                size_t sz) {
  DCHECK_LE(sz, 3);
  switch (sz) {
    case 3:

      CompOrSwap(0, 1, indices, values);
      CompOrSwap(1, 2, indices, values);
      ABSL_FALLTHROUGH_INTENDED;
    case 2:

      CompOrSwap(0, 1, indices, values);
      ABSL_FALLTHROUGH_INTENDED;
    case 1:

      break;
  }
}

template <typename DistT>
SCANN_INLINE DistT FastMedianOf3(DistT v0, DistT v1, DistT v2) {
  DistT big = std::max(v0, v1);
  DistT sml = std::min(v0, v1);
  return std::max(sml, std::min(big, v2));
}

}  // namespace

#ifdef __x86_64__

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2

#define SCANN_TOPN_AVX2_ENABLED
#include "scann/utils/fast_top_neighbors_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
#undef SCANN_TOPN_AVX2_ENABLED
}  // namespace avx2

#endif

#if HWY_HAVE_CONSTEXPR_LANES
HWY_BEFORE_NAMESPACE();
namespace highway {
#define SCANN_SIMD_ATTRIBUTE

#include "scann/utils/fast_top_neighbors_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace highway
HWY_AFTER_NAMESPACE();
#else
namespace fallback {
#define SCANN_SIMD_ATTRIBUTE
#include "scann/utils/fast_top_neighbors_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace fallback
#endif

template <typename DistT, typename DatapointIndexT>
size_t ApproxNthElement(size_t keep_min, size_t keep_max, size_t sz,
                        DatapointIndexT* ii, DistT* dd, uint32_t* mm) {
  DCHECK_GT(keep_min, 0);
#ifdef __x86_64__
  if (RuntimeSupportsAvx2()) {
    return avx2::ApproxNthElementImpl(keep_min, keep_max, sz, ii, dd, mm);
  }
#endif

  return highway::ApproxNthElementImpl(keep_min, keep_max, sz, ii, dd, mm);
}

SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, int16_t, uint32_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, float, uint32_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, int16_t, uint64_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, float, uint64_t);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, int16_t, absl::uint128);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, float, absl::uint128);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, float, VectorDBDatapointIndexT);
SCANN_INSTANTIATE_FAST_TOP_NEIGHBORS(, float, std::shared_ptr<std::string>);

}  // namespace research_scann
