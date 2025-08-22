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

#include "scann/distance_measures/one_to_one/dot_product_highway.h"

#include <cstddef>
#include <cstdint>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "hwy/highway.h"
#include "scann/data_format/datapoint.h"
#include "scann/utils/intrinsics/attributes.h"

HWY_BEFORE_NAMESPACE();
namespace research_scann {
namespace dp_internal {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

SCANN_HIGHWAY_INLINE double DenseDotProductInt8FloatHighwayImpl(
    const int8_t* aptr, const float* bptr, size_t length) {
  const hn::ScalableTag<float> df;
  const hn::RebindToSigned<decltype(df)> d32;
  const hn::Rebind<int8_t, decltype(df)> to_d32;
  const hn::ScalableTag<int16_t> d16;
  const hn::Rebind<int8_t, decltype(d16)> to_d16;
  const hn::Half<decltype(to_d16)> to_d16_half;
  const size_t n = hn::Lanes(df);

  using VF = hn::Vec<decltype(df)>;
  const int8_t* aend = aptr + length;

  VF accumulator0 = hn::Zero(df);
  VF accumulator1 = hn::Zero(df);
  for (; aptr + 2 * n <= aend; aptr += 2 * n, bptr += 2 * n) {
    auto avals = hn::LoadU(to_d16, aptr);

    VF avals0 = hn::ConvertTo(
        df, hn::PromoteTo(d32, hn::LowerHalf(to_d16_half, avals)));
    VF bvals0 = hn::LoadU(df, bptr);
    accumulator0 = hn::MulAdd(avals0, bvals0, accumulator0);

    VF avals1 = hn::ConvertTo(
        df, hn::PromoteTo(d32, hn::UpperHalf(to_d16_half, avals)));
    VF bvals1 = hn::LoadU(df, bptr + n);
    accumulator1 = hn::MulAdd(avals1, bvals1, accumulator1);
  }

  if (aptr + (n) <= aend) {
    auto avals = hn::LoadU(to_d32, aptr);
    VF avals0 = hn::ConvertTo(df, hn::PromoteTo(d32, avals));
    VF bvals0 = hn::LoadU(df, bptr);
    accumulator0 = hn::MulAdd(avals0, bvals0, accumulator0);
    aptr += (n);
    bptr += (n);
  }

  float scalar_accumulator =
      hn::ReduceSum(df, hn::Add(accumulator0, accumulator1));

  DCHECK_LT(aend - aptr, n);
  for (; aptr < aend; ++aptr, ++bptr) {
    scalar_accumulator += static_cast<float>(*aptr) * *bptr;
  }

  return static_cast<double>(scalar_accumulator);
}

SCANN_HIGHWAY_INLINE double DenseDotProductInt16FloatHighwayImpl(
    const int16_t* aptr, const float* bptr, size_t length) {
  const hn::ScalableTag<float> df;
  const hn::RebindToSigned<decltype(df)> d32;
  const hn::Rebind<int16_t, decltype(df)> to_d32;
  const hn::ScalableTag<int16_t> d16;
  const hn::Half<decltype(d16)> d16_half;
  const size_t n = hn::Lanes(df);

  using VF = hn::Vec<decltype(df)>;
  const int16_t* aend = aptr + length;

  VF accumulator0 = hn::Zero(df);
  VF accumulator1 = hn::Zero(df);
  for (; aptr + 2 * n <= aend; aptr += 2 * n, bptr += 2 * n) {
    auto avals = hn::LoadU(d16, aptr);

    VF avals0 = hn::BitCast(df, hn::ShiftLeft<16>(hn::PromoteTo(
                                    d32, hn::LowerHalf(d16_half, avals))));
    VF bvals0 = hn::LoadU(df, bptr);
    accumulator0 = hn::MulAdd(avals0, bvals0, accumulator0);

    VF avals1 = hn::BitCast(df, hn::ShiftLeft<16>(hn::PromoteTo(
                                    d32, hn::UpperHalf(d16_half, avals))));
    VF bvals1 = hn::LoadU(df, bptr + n);
    accumulator1 = hn::MulAdd(avals1, bvals1, accumulator1);
  }

  if (aptr + n <= aend) {
    auto avals = hn::LoadU(to_d32, aptr);
    VF avals0 = hn::BitCast(df, hn::ShiftLeft<16>(hn::PromoteTo(d32, avals)));
    VF bvals0 = hn::LoadU(df, bptr);
    accumulator0 = hn::MulAdd(avals0, bvals0, accumulator0);
    aptr += n;
    bptr += n;
  }

  float scalar_accumulator =
      hn::ReduceSum(df, hn::Add(accumulator0, accumulator1));

  DCHECK_LT(aend - aptr, n);
  for (; aptr < aend; ++aptr, ++bptr) {
    scalar_accumulator +=
        absl::bit_cast<float>(static_cast<int32_t>(*aptr) << 16) * *bptr;
  }

  return static_cast<double>(scalar_accumulator);
}

}  // namespace HWY_NAMESPACE
}  // namespace dp_internal
}  // namespace research_scann
HWY_AFTER_NAMESPACE();

namespace research_scann {
namespace dp_internal {

SCANN_HIGHWAY_OUTLINE double DenseDotProductHighway(
    const DatapointPtr<int8_t>& a, const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  return HWY_NAMESPACE::DenseDotProductInt8FloatHighwayImpl(
      a.values(), b.values(), a.nonzero_entries());
}

SCANN_HIGHWAY_OUTLINE double DenseDotProductHighway(
    const DatapointPtr<int16_t>& a, const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  return HWY_NAMESPACE::DenseDotProductInt16FloatHighwayImpl(
      a.values(), b.values(), a.nonzero_entries());
}

}  // namespace dp_internal
}  // namespace research_scann
