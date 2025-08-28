#include <cstddef>
#include <type_traits>

#include "scann/utils/bfloat16_helpers.h"

#if SCANN_HIGHWAY_ONE_TO_MANY_DYNAMIC_DISPATCH
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "scann/distance_measures/one_to_many/one_to_many_impl_highway.inc"
#include "hwy/foreach_target.h"

#endif

#include "hwy/highway.h"

#if HWY_HAVE_CONSTEXPR_LANES
HWY_BEFORE_NAMESPACE();

namespace research_scann {
namespace highway {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <class DF, class D8>
SCANN_HIGHWAY_INLINE hn::Vec<DF> Int8ToFloat(const DF df, const D8 d8,
                                             hn::Vec<D8> int8_val) {
  const hn::RebindToSigned<DF> d32;
  return hn::ConvertTo(df, hn::PromoteTo(d32, int8_val));
}

template <class DF, class D16>
SCANN_HIGHWAY_INLINE hn::Vec<DF> Int16ToFloat(const DF df, const D16 d16,
                                              hn::Vec<D16> int16_val) {
  const hn::RebindToSigned<DF> d32;
  auto promoted = hn::PromoteTo(d32, int16_val);
  return hn::BitCast(df, hn::ShiftLeft<16>(promoted));
}

template <bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE hn::Vec<D> FusedMultiplyOpFixed8(const D d, hn::Vec<D> a,
                                                      hn::Vec<D> b,
                                                      hn::Vec<D> mult,
                                                      hn::Vec<D> accum) {
  using V = hn::Vec<D>;
  if constexpr (kIsSquaredL2) {
    V diff = hn::NegMulAdd(b, mult, a);
    return hn::MulAdd(diff, diff, accum);
  } else {
    return hn::NegMulAdd(a, b, accum);
  }
}

template <bool kIsSquaredL2>
SCANN_HIGHWAY_INLINE float FusedMultiplyOpFixed8Scalar(float a, float b,
                                                       float mult,
                                                       float accum) {
  if constexpr (kIsSquaredL2) {
    float diff = a - b * mult;
    return accum + diff * diff;
  } else {
    return accum - (a * b);
  }
}

template <bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE hn::Vec<D> FusedMultiplyOpBfloat16(const D d, hn::Vec<D> a,
                                                        hn::Vec<D> b,
                                                        hn::Vec<D> accum) {
  using V = hn::Vec<D>;
  if constexpr (kIsSquaredL2) {
    V diff = b - a;
    return hn::MulAdd(diff, diff, accum);
  } else {
    return hn::NegMulAdd(a, b, accum);
  }
}

template <bool kIsSquaredL2>
SCANN_HIGHWAY_INLINE float FusedMultiplyOpBfloat16Scalar(float a, int16_t b_int,
                                                         float accum) {
  const float b = Bfloat16Decompress(b_int);
  if constexpr (kIsSquaredL2) {
    float diff = a - b;
    return accum + diff * diff;
  } else {
    return accum - (a * b);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims1X(
    D d, const float* query, array<const int8_t*, 1>& ptrs,
    const float* inv_multipliers_for_squared_l2, size_t dim,
    hn::Vec<D>& accums) {
  const hn::Rebind<int8_t, D> d8;
  using VI8 = hn::Vec<decltype(d8)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI8 db_vals = hn::LoadU(d8, ptrs[0] + i);

    VF32 mult = hn::Zero(d);
    if constexpr (kIsSquaredL2) {
      mult = hn::LoadU(d, inv_multipliers_for_squared_l2 + i);
    }
    accums = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals), mult, accums);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims2X(
    D d, const float* query, array<const int8_t*, 2>& ptrs,
    const float* inv_multipliers_for_squared_l2, size_t dim,
    hn::Vec<D>& accums0, hn::Vec<D>& accums1) {
  const hn::Rebind<int8_t, D> d8;
  using VI8 = hn::Vec<decltype(d8)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI8 db_vals0 = hn::LoadU(d8, ptrs[0] + i);
    VI8 db_vals1 = hn::LoadU(d8, ptrs[1] + i);

    VF32 mult = hn::Zero(d);
    if constexpr (kIsSquaredL2) {
      mult = hn::LoadU(d, inv_multipliers_for_squared_l2 + i);
    }
    accums0 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals0), mult, accums0);
    accums1 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals1), mult, accums1);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims3X(
    D d, const float* query, array<const int8_t*, 3>& ptrs,
    const float* inv_multipliers_for_squared_l2, size_t dim,
    hn::Vec<D>& accums0, hn::Vec<D>& accums1, hn::Vec<D>& accums2) {
  const hn::Rebind<int8_t, D> d8;
  using VI8 = hn::Vec<decltype(d8)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI8 db_vals0 = hn::LoadU(d8, ptrs[0] + i);
    VI8 db_vals1 = hn::LoadU(d8, ptrs[1] + i);
    VI8 db_vals2 = hn::LoadU(d8, ptrs[2] + i);

    VF32 mult = hn::Zero(d);
    if constexpr (kIsSquaredL2) {
      mult = hn::LoadU(d, inv_multipliers_for_squared_l2 + i);
    }
    accums0 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals0), mult, accums0);
    accums1 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals1), mult, accums1);
    accums2 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals2), mult, accums2);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims4X(
    D d, const float* query, array<const int8_t*, 4>& ptrs,
    const float* inv_multipliers_for_squared_l2, size_t dim,
    hn::Vec<D>& accums0, hn::Vec<D>& accums1, hn::Vec<D>& accums2,
    hn::Vec<D>& accums3) {
  const hn::Rebind<int8_t, D> d8;
  using VI8 = hn::Vec<decltype(d8)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI8 db_vals0 = hn::LoadU(d8, ptrs[0] + i);
    VI8 db_vals1 = hn::LoadU(d8, ptrs[1] + i);
    VI8 db_vals2 = hn::LoadU(d8, ptrs[2] + i);
    VI8 db_vals3 = hn::LoadU(d8, ptrs[3] + i);

    VF32 mult = hn::Zero(d);
    if constexpr (kIsSquaredL2) {
      mult = hn::LoadU(d, inv_multipliers_for_squared_l2 + i);
    }
    accums0 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals0), mult, accums0);
    accums1 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals1), mult, accums1);
    accums2 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals2), mult, accums2);
    accums3 = FusedMultiplyOpFixed8<kIsSquaredL2>(
        d, qq_vals, Int8ToFloat(d, d8, db_vals3), mult, accums3);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims1X(D d, const float* query,
                                        array<const int16_t*, 1>& ptrs,
                                        size_t dim, hn::Vec<D>& accums) {
  const hn::Rebind<int16_t, D> d16;
  using VI16 = hn::Vec<decltype(d16)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI16 db_vals = hn::LoadU(d16, ptrs[0] + i);
    accums = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals), accums);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims2X(D d, const float* query,
                                        array<const int16_t*, 2>& ptrs,
                                        size_t dim, hn::Vec<D>& accums0,
                                        hn::Vec<D>& accums1) {
  const hn::Rebind<int16_t, D> d16;
  using VI16 = hn::Vec<decltype(d16)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI16 db_vals0 = hn::LoadU(d16, ptrs[0] + i);
    VI16 db_vals1 = hn::LoadU(d16, ptrs[1] + i);
    accums0 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals0), accums0);
    accums1 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals1), accums1);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims3X(D d, const float* query,
                                        array<const int16_t*, 3>& ptrs,
                                        size_t dim, hn::Vec<D>& accums0,
                                        hn::Vec<D>& accums1,
                                        hn::Vec<D>& accums2) {
  const hn::Rebind<int16_t, D> d16;
  using VI16 = hn::Vec<decltype(d16)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI16 db_vals0 = hn::LoadU(d16, ptrs[0] + i);
    VI16 db_vals1 = hn::LoadU(d16, ptrs[1] + i);
    VI16 db_vals2 = hn::LoadU(d16, ptrs[2] + i);
    accums0 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals0), accums0);
    accums1 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals1), accums1);
    accums2 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals2), accums2);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, class D>
SCANN_HIGHWAY_INLINE void HandleXDims4X(D d, const float* query,
                                        array<const int16_t*, 4>& ptrs,
                                        size_t dim, hn::Vec<D>& accums0,
                                        hn::Vec<D>& accums1,
                                        hn::Vec<D>& accums2,
                                        hn::Vec<D>& accums3) {
  const hn::Rebind<int16_t, D> d16;
  using VI16 = hn::Vec<decltype(d16)>;
  using VF32 = hn::Vec<D>;
  const size_t n = hn::Lanes(d);
  for (size_t i = dim; i < (dim + kNumDims); i += n) {
    VF32 qq_vals = hn::LoadU(d, query + i);

    VI16 db_vals0 = hn::LoadU(d16, ptrs[0] + i);
    VI16 db_vals1 = hn::LoadU(d16, ptrs[1] + i);
    VI16 db_vals2 = hn::LoadU(d16, ptrs[2] + i);
    VI16 db_vals3 = hn::LoadU(d16, ptrs[3] + i);
    accums0 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals0), accums0);
    accums1 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals1), accums1);
    accums2 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals2), accums2);
    accums3 = FusedMultiplyOpBfloat16<kIsSquaredL2>(
        d, qq_vals, Int16ToFloat(d, d16, db_vals3), accums3);
  }
}

template <class D>
SCANN_HIGHWAY_INLINE void HorizontalSum1X(D d, hn::Vec<D>& accums,
                                          float* results) {
  *results = hn::ReduceSum(d, accums);
}

template <class D>
SCANN_HIGHWAY_INLINE void HorizontalSum2X(D d, hn::Vec<D>& accums0,
                                          hn::Vec<D>& accums1, float* results0,
                                          float* results1) {
  *results0 = hn::ReduceSum(d, accums0);
  *results1 = hn::ReduceSum(d, accums1);
}

template <class D>
SCANN_HIGHWAY_INLINE void HorizontalSum3X(D d, hn::Vec<D>& accums0,
                                          hn::Vec<D>& accums1,
                                          hn::Vec<D>& accums2, float* results0,
                                          float* results1, float* results2) {
  *results0 = hn::ReduceSum(d, accums0);
  *results1 = hn::ReduceSum(d, accums1);
  *results2 = hn::ReduceSum(d, accums2);
}

template <class D>
SCANN_HIGHWAY_INLINE void HorizontalSum4X(D d, hn::Vec<D>& accums0,
                                          hn::Vec<D>& accums1,
                                          hn::Vec<D>& accums2,
                                          hn::Vec<D>& accums3, float* results0,
                                          float* results1, float* results2,
                                          float* results3) {
  *results0 = hn::ReduceSum(d, accums0);
  *results1 = hn::ReduceSum(d, accums1);
  *results2 = hn::ReduceSum(d, accums2);
  *results3 = hn::ReduceSum(d, accums3);
}

SCANN_HIGHWAY_INLINE double StaticallyInvokeOneToOneDenseDotProduct(
    const DatapointPtr<float>& qq, const DatapointPtr<int8_t>& db) {
  return ::research_scann::dp_internal::DenseDotProductHighway(db, qq);
}

SCANN_HIGHWAY_INLINE double StaticallyInvokeOneToOneDenseDotProduct(
    const DatapointPtr<float>& qq, const DatapointPtr<int16_t>& db) {
  return ::research_scann::dp_internal::DenseDotProductHighway(db, qq);
}

template <bool kIsSquaredL2>
SCANN_HIGHWAY_INLINE float ComputeOneToOneScoreImpl(
    const float* __restrict__ query, const int8_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  if constexpr (kIsSquaredL2) {
    array<const int8_t*, 1> ptrs = {ptr};
    V accums = hn::Zero(d);
    size_t dim = 0;
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims1X<16, kIsSquaredL2>(
          d, query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }

    float dist;
    HorizontalSum1X(d, accums, &dist);

    for (; dim < dimensionality; dim++) {
      float mult = 0.0;
      if constexpr (kIsSquaredL2) {
        mult = inv_multipliers_for_squared_l2[dim];
      }
      dist = FusedMultiplyOpFixed8Scalar<kIsSquaredL2>(
          query[dim], static_cast<float>(ptrs[0][dim]), mult, dist);
    }
    return dist;
  } else {
    DatapointPtr<float> qq_dptr(nullptr, query, dimensionality, dimensionality);
    DatapointPtr<int8_t> db_dptr(nullptr, ptr, dimensionality, dimensionality);
    return -StaticallyInvokeOneToOneDenseDotProduct(qq_dptr, db_dptr);
  }
}

template <size_t kUnrollBy, bool kIsSquaredL2>
SCANN_HIGHWAY_INLINE void ComputeOneToManyScoreImpl(
    const float* __restrict__ query, array<const int8_t*, kUnrollBy>& ptrs,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const size_t dimensionality, array<float, kUnrollBy>& results) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;
  V accums0 = hn::Zero(d);
  V accums1 = hn::Zero(d);
  V accums2 = hn::Zero(d);
  V accums3 = hn::Zero(d);
  size_t dim = 0;

  if constexpr (kUnrollBy == 4) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims4X<16, kIsSquaredL2>(d, query, ptrs,
                                      inv_multipliers_for_squared_l2, dim,
                                      accums0, accums1, accums2, accums3);
    }
    HorizontalSum4X(d, accums0, accums1, accums2, accums3, &results[0],
                    &results[1], &results[2], &results[3]);
  } else if constexpr (kUnrollBy == 3) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims3X<16, kIsSquaredL2>(d, query, ptrs,
                                      inv_multipliers_for_squared_l2, dim,
                                      accums0, accums1, accums2);
    }
    HorizontalSum3X(d, accums0, accums1, accums2, &results[0], &results[1],
                    &results[2]);
  } else if constexpr (kUnrollBy == 2) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims2X<16, kIsSquaredL2>(d, query, ptrs,
                                      inv_multipliers_for_squared_l2, dim,
                                      accums0, accums1);
    }
    HorizontalSum2X(d, accums0, accums1, &results[0], &results[1]);
  } else {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims1X<16, kIsSquaredL2>(
          d, query, ptrs, inv_multipliers_for_squared_l2, dim, accums0);
    }
    HorizontalSum1X(d, accums0, &results[0]);
  }

  for (; dim < dimensionality; ++dim) {
    for (size_t jj : Seq(kUnrollBy)) {
      float mult = 0.0;
      if constexpr (kIsSquaredL2) {
        mult = inv_multipliers_for_squared_l2[dim];
      } else {
        mult = 0.0;
      }
      results[jj] = FusedMultiplyOpFixed8Scalar<kIsSquaredL2>(
          query[dim], static_cast<float>(ptrs[jj][dim]), mult, results[jj]);
    }
  }
}

template <bool kIsSquaredL2>
SCANN_HIGHWAY_INLINE float ComputeOneToOneScoreImpl(
    const float* __restrict__ query, const int16_t* ptr,
    size_t dimensionality) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;

  if constexpr (kIsSquaredL2) {
    array<const int16_t*, 1> ptrs = {ptr};
    V accums = hn::Zero(d);
    size_t dim = 0;
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims1X<16, kIsSquaredL2>(d, query, ptrs, dim, accums);
    }

    float dist;
    HorizontalSum1X(d, accums, &dist);

    for (; dim < dimensionality; dim++) {
      dist = FusedMultiplyOpBfloat16Scalar<kIsSquaredL2>(query[dim],
                                                         ptrs[0][dim], dist);
    }
    return dist;
  } else {
    DatapointPtr<float> qq_dptr(nullptr, query, dimensionality, dimensionality);
    DatapointPtr<int16_t> db_dptr(nullptr, ptr, dimensionality, dimensionality);
    return -StaticallyInvokeOneToOneDenseDotProduct(qq_dptr, db_dptr);
  }
}

template <size_t kUnrollBy, bool kIsSquaredL2>
SCANN_HIGHWAY_INLINE void ComputeOneToManyScoreImpl(
    const float* __restrict__ query, array<const int16_t*, kUnrollBy>& ptrs,
    const size_t dimensionality, array<float, kUnrollBy>& results) {
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;
  V accums0 = hn::Zero(d);
  V accums1 = hn::Zero(d);
  V accums2 = hn::Zero(d);
  V accums3 = hn::Zero(d);
  size_t dim = 0;

  if constexpr (kUnrollBy == 4) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims4X<16, kIsSquaredL2>(d, query, ptrs, dim, accums0, accums1,
                                      accums2, accums3);
    }
    HorizontalSum4X(d, accums0, accums1, accums2, accums3, &results[0],
                    &results[1], &results[2], &results[3]);
  } else if constexpr (kUnrollBy == 3) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims3X<16, kIsSquaredL2>(d, query, ptrs, dim, accums0, accums1,
                                      accums2);
    }
    HorizontalSum3X(d, accums0, accums1, accums2, &results[0], &results[1],
                    &results[2]);
  } else if constexpr (kUnrollBy == 2) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims2X<16, kIsSquaredL2>(d, query, ptrs, dim, accums0, accums1);
    }
    HorizontalSum2X(d, accums0, accums1, &results[0], &results[1]);
  } else {
    for (; dim + 16 <= dimensionality; dim += 16) {
      HandleXDims1X<16, kIsSquaredL2>(d, query, ptrs, dim, accums0);
    }
    HorizontalSum1X(d, accums0, &results[0]);
  }

  for (; dim < dimensionality; ++dim) {
    for (size_t jj : Seq(kUnrollBy)) {
      results[jj] = FusedMultiplyOpBfloat16Scalar<kIsSquaredL2>(
          query[dim], ptrs[jj][dim], results[jj]);
    }
  }
}

SCANN_HIGHWAY_INLINE float ComputeOneToOneScore(
    const float* __restrict__ query, const int8_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  return ComputeOneToOneScoreImpl<false>(
      query, ptr, inv_multipliers_for_squared_l2, dimensionality);
}

SCANN_HIGHWAY_INLINE float ComputeOneToOneScoreSquaredL2(
    const float* __restrict__ query, const int8_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  return ComputeOneToOneScoreImpl<true>(
      query, ptr, inv_multipliers_for_squared_l2, dimensionality);
}

SCANN_HIGHWAY_INLINE float ComputeOneToOneScore(
    const float* __restrict__ query, const int16_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  return ComputeOneToOneScoreImpl<false>(query, ptr, dimensionality);
}

SCANN_HIGHWAY_INLINE float ComputeOneToOneScoreSquaredL2(
    const float* __restrict__ query, const int16_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  return ComputeOneToOneScoreImpl<true>(query, ptr, dimensionality);
}

#define COMPUTE_ONE_TO_MANY_SCORE(unroll_by)                                   \
  SCANN_HIGHWAY_INLINE void ComputeOneToManyScore##unroll_by(                  \
      const float* __restrict__ query, array<const int8_t*, unroll_by>& ptrs,  \
      const float* __restrict__ inv_multipliers_for_squared_l2,                \
      const size_t dimensionality, array<float, unroll_by>& results) {         \
    ComputeOneToManyScoreImpl<unroll_by, false>(                               \
        query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results); \
  }                                                                            \
  SCANN_HIGHWAY_INLINE void ComputeOneToManyScore##unroll_by(                  \
      const float* __restrict__ query, array<const int16_t*, unroll_by>& ptrs, \
      const float* __restrict__ inv_multipliers_for_squared_l2,                \
      const size_t dimensionality, array<float, unroll_by>& results) {         \
    ComputeOneToManyScoreImpl<unroll_by, false>(query, ptrs, dimensionality,   \
                                                results);                      \
  }

#define COMPUTE_ONE_TO_MANY_SCORE_SQUAREDL2(unroll_by)                         \
  SCANN_HIGHWAY_INLINE void ComputeOneToManyScoreSquaredL2##unroll_by(         \
      const float* __restrict__ query, array<const int8_t*, unroll_by>& ptrs,  \
      const float* __restrict__ inv_multipliers_for_squared_l2,                \
      const size_t dimensionality, array<float, unroll_by>& results) {         \
    ComputeOneToManyScoreImpl<unroll_by, true>(                                \
        query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results); \
  }                                                                            \
  SCANN_HIGHWAY_INLINE void ComputeOneToManyScoreSquaredL2##unroll_by(         \
      const float* __restrict__ query, array<const int16_t*, unroll_by>& ptrs, \
      const float* __restrict__ inv_multipliers_for_squared_l2,                \
      const size_t dimensionality, array<float, unroll_by>& results) {         \
    ComputeOneToManyScoreImpl<unroll_by, true>(query, ptrs, dimensionality,    \
                                               results);                       \
  }

COMPUTE_ONE_TO_MANY_SCORE(1);
COMPUTE_ONE_TO_MANY_SCORE_SQUAREDL2(1);
COMPUTE_ONE_TO_MANY_SCORE(2);
COMPUTE_ONE_TO_MANY_SCORE_SQUAREDL2(2);
COMPUTE_ONE_TO_MANY_SCORE(3);
COMPUTE_ONE_TO_MANY_SCORE_SQUAREDL2(3);
COMPUTE_ONE_TO_MANY_SCORE(4);
COMPUTE_ONE_TO_MANY_SCORE_SQUAREDL2(4);

#undef COMPUTE_ONE_TO_MANY_SCORE
#undef COMPUTE_ONE_TO_MANY_SCORE_SQUAREDL2

template <bool kIsSquaredL2, size_t kUnrollBy>
SCANN_HIGHWAY_INLINE void ComputeOneToManyScores(
    const int8_t* __restrict__ query, array<const uint8_t*, kUnrollBy> ptrs,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality, size_t num_outer_iters,
    array<int32_t, kUnrollBy>& results) {
  static_assert(!kIsSquaredL2, "Uint4Int8 doesn't support squared l2.");
  static_assert(kUnrollBy == 1, "Uint4Int8 doesn't support unrolling.");
  const uint8_t* ptr = ptrs[0];
  const size_t kWidth = 128;
  const hn::FixedTag<int8_t, kWidth / 8> d8;
  const hn::FixedTag<uint8_t, kWidth / 8> du8;
  const hn::FixedTag<int32_t, kWidth / 32> d32;
  const auto kMask = hn::Set(du8, 0x0F);
  auto acc = hn::Zero(d32);
  for (size_t i = 0; i * kWidth < dimensionality * 4; ++i) {
    auto packed_data = hn::LoadU(du8, ptr + i * kWidth / 8);
    auto data_low = hn::And(packed_data, kMask);
    auto data_high = hn::ShiftRight<4>(packed_data);
    auto query_low = hn::LoadU(d8, query + 2 * i * kWidth / 8);
    auto query_high = hn::LoadU(d8, query + 2 * i * kWidth / 8 + kWidth / 8);
    acc = hn::SumOfMulQuadAccumulate(d32, hn::BitCast(d8, data_low), query_low,
                                     acc);
    acc = hn::SumOfMulQuadAccumulate(d32, hn::BitCast(d8, data_high),
                                     query_high, acc);
  }
  results[0] = -hn::ReduceSum(d32, acc);
}

SCANN_HIGHWAY_INLINE void PermuteQueryForUint4Int8(const int8_t* query,
                                                   int8_t* permuted_query,
                                                   size_t dimensionality) {
  DCHECK_EQ(dimensionality % 64, 0);

  const size_t kWidth = 128;
  const hn::FixedTag<int8_t, kWidth / 8> d;
  for (size_t i = 0; i < dimensionality; i += 2 * kWidth / 8) {
    auto q0 = hn::LoadU(d, query + i);
    auto q1 = hn::LoadU(d, query + i + kWidth / 8);

    auto q_even = hn::ConcatEven(d, q1, q0);
    auto q_odd = hn::ConcatOdd(d, q1, q0);
    hn::StoreU(q_even, d, permuted_query + i);
    hn::StoreU(q_odd, d, permuted_query + i + kWidth / 8);
  }
}

template <typename CallbackT, typename ResultT, typename DataT>
SCANN_HIGHWAY_INLINE void InvokeCallbackHighway(const CallbackT& callback,
                                                size_t result_idx, ResultT val,
                                                size_t datapoint_bytes,
                                                const DataT* ptr) {
  if constexpr (one_to_many_low_level::NeedsBottomBitsSideData<
                    CallbackT>::value) {
    static_assert(std::is_same_v<DataT, int8_t> ||
                  std::is_same_v<DataT, uint8_t>);
    if (datapoint_bytes % 32 == 0) {
      const hn::FixedTag<DataT, 16> d;
      const hn::FixedTag<uint8_t, 16> du8;
      const hn::FixedTag<uint16_t, 8> du16;
      auto data_low = hn::Zero(d);
      auto data_high = hn::Zero(d);
      for (size_t i = 0; i < datapoint_bytes; i += 32) {
        data_low = hn::Xor(data_low, hn::LoadU(d, ptr + i));
        data_high = hn::Xor(data_high, hn::LoadU(d, ptr + i + 16));
      }
      auto data_low_u16 = hn::BitCast(du16, data_low);
      auto data_high_u16 = hn::BitCast(du16, data_high);
      if constexpr (std::is_same_v<DataT, uint8_t>) {
        data_low_u16 = hn::Xor(data_low_u16, hn::ShiftRight<4>(data_low_u16));
        data_high_u16 =
            hn::Xor(data_high_u16, hn::ShiftRight<4>(data_high_u16));
      }
      data_low_u16 = hn::ShiftLeft<7>(data_low_u16);
      data_high_u16 = hn::ShiftLeft<7>(data_high_u16);
      auto data_low_u8 = hn::BitCast(du8, data_low_u16);
      auto data_high_u8 = hn::BitCast(du8, data_high_u16);
      uint32_t bottom_bits =
          hn::BitsFromMask(du8, hn::Gt(data_low_u8, hn::Set(du8, 0x7F))) |
          (hn::BitsFromMask(du8, hn::Gt(data_high_u8, hn::Set(du8, 0x7F)))
           << 16);

      if constexpr (std::is_same_v<DataT, int8_t>) {
        DCHECK_EQ(bottom_bits, DecodeBottomBitsDataFromInt8(
                                   MakeConstSpan(ptr, datapoint_bytes)));
      } else {
        static_assert(std::is_same_v<DataT, uint8_t>);
        DCHECK_EQ(bottom_bits, DecodeBottomBitsDataFromPackedInt4(
                                   MakeConstSpan(ptr, datapoint_bytes)));
      }
      callback.invoke(result_idx, val, bottom_bits);
      return;
    }
  }
  InvokeCallback(callback, result_idx, val, datapoint_bytes, ptr);
}
}  // namespace HWY_NAMESPACE
}  // namespace highway
}  // namespace research_scann
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace research_scann {
namespace highway {

#if SCANN_HIGHWAY_ONE_TO_MANY_DYNAMIC_DISPATCH

HWY_EXPORT(ComputeOneToOneScore);
HWY_EXPORT(ComputeOneToOneScoreSquaredL2);
HWY_EXPORT(ComputeOneToManyScore1);
HWY_EXPORT(ComputeOneToManyScoreSquaredL21);
HWY_EXPORT(ComputeOneToManyScore2);
HWY_EXPORT(ComputeOneToManyScoreSquaredL22);
HWY_EXPORT(ComputeOneToManyScore3);
HWY_EXPORT(ComputeOneToManyScoreSquaredL23);
HWY_EXPORT(ComputeOneToManyScore4);
HWY_EXPORT(ComputeOneToManyScoreSquaredL24);

#define HWY_DISPATCH(func) HWY_DYNAMIC_DISPATCH(func)

#else

#define HWY_DISPATCH(func) HWY_NAMESPACE::func

#endif

namespace {

template <typename ValueT>
inline size_t GetDatapointIndex(ValueT* result, size_t index) {
  return index;
}
template <typename ValueT>
inline size_t GetDatapointIndex(MutableSpan<ValueT> result, size_t index) {
  return index;
}

template <typename IndexT, typename ValueT>
inline size_t GetDatapointIndex(pair<IndexT, ValueT>* result, size_t index) {
  return result[index].first;
}
template <typename IndexT, typename ValueT>
inline size_t GetDatapointIndex(MutableSpan<pair<IndexT, ValueT>> result,
                                size_t index) {
  return result[index].first;
}

}  // namespace

template <int kDimensionality, size_t kUnrollBy, bool kHasIndices,
          bool kIsSquaredL2, bool kShouldPrefetch, typename DataT,
          typename DatasetViewT, typename IndexT, typename ResultElemT,
          typename CallbackT, typename QueryT>
SCANN_HIGHWAY_INLINE void OneToManyIntFloatTemplate(
    const QueryT* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dimensionality =
      kDimensionality > 0 ? kDimensionality : dataset_view.dimensionality();
  using IntT = std::remove_reference_t<decltype(dataset_view.GetPtr(0)[0])>;

  const size_t num_datapoints = result.size();
  if (num_datapoints == 0 || dimensionality == 0) return;

  constexpr size_t kMinPrefetchAheadBytes = 2304;

  constexpr size_t kCacheLine = 64;
  const size_t datapoint_bytes = std::is_same_v<DataT, uint8_t>
                                     ? DivRoundUp(dimensionality, 2)
                                     : sizeof(DataT) * dimensionality;
  const size_t cache_lines_per_datapoint =
      DivRoundUp(datapoint_bytes, kCacheLine);
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(
        1, kMinPrefetchAheadBytes /
               (kUnrollBy * cache_lines_per_datapoint * kCacheLine));
  }

  auto get_db_ptr = [indices, &dataset_view, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const IntT* {
    const size_t idx = kHasIndices ? indices[i] : GetDatapointIndex(result, i);
    callback.prefetch(idx);
    return dataset_view.GetPtr(idx);
  };

  const size_t num_outer_iters = num_datapoints / kUnrollBy;

  if constexpr (kShouldPrefetch) {
    for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
         j++) {
      const IntT* prefetch_ptr = get_db_ptr(j);
      for (size_t n : Seq(cache_lines_per_datapoint)) {
        absl::PrefetchToLocalCacheNta(prefetch_ptr + n * kCacheLine);
      }
    }

    for (size_t j : Seq(std::min(num_prefetch_datapoints, num_outer_iters))) {
      array<const IntT*, kUnrollBy> prefetch_ptrs;
      for (size_t jj : Seq(kUnrollBy)) {
        prefetch_ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
      }

      for (size_t n : Seq(cache_lines_per_datapoint)) {
        for (size_t jj : Seq(kUnrollBy)) {
          absl::PrefetchToLocalCacheNta(prefetch_ptrs[jj] + n * kCacheLine);
        }
      }
    }
  }

  std::array<QueryT, kDimensionality> query_stack_storage;
  std::unique_ptr<QueryT[]> permuted_query_heap_storage;
  if constexpr (std::is_same_v<QueryT, int8_t> &&
                std::is_same_v<DataT, uint8_t>) {
    QueryT* permuted_query;

    const size_t round_up_dims = NextMultipleOf(dimensionality, 64);
    if constexpr (kDimensionality > 0) {
      DCHECK_EQ(round_up_dims, kDimensionality);
      permuted_query = query_stack_storage.data();
    } else {
      permuted_query_heap_storage = std::make_unique<QueryT[]>(round_up_dims);
      permuted_query = permuted_query_heap_storage.get();
    }
    std::unique_ptr<QueryT[]> query_heap_storage;
    if (dimensionality != round_up_dims) {
      query_heap_storage = std::make_unique<QueryT[]>(round_up_dims);
      std::copy(query, query + dimensionality, query_heap_storage.get());
      std::fill(query_heap_storage.get() + dimensionality,
                query_heap_storage.get() + round_up_dims, 0);
      query = query_heap_storage.get();
    }

    HWY_DISPATCH(PermuteQueryForUint4Int8)
    (query, permuted_query, round_up_dims);
    query = permuted_query;
  } else {
    if constexpr (kDimensionality > 0) {
      DCHECK_EQ(dimensionality, kDimensionality);

      std::copy(query, query + kDimensionality, query_stack_storage.data());
      query = query_stack_storage.data();
    }
  }

  if constexpr (kUnrollBy > 1) {
    for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
         j++) {
      const IntT* ptr = get_db_ptr(j);
      float result;
      if constexpr (kIsSquaredL2) {
        result = HWY_DISPATCH(ComputeOneToOneScoreSquaredL2)(
            query, ptr, inv_multipliers_for_squared_l2, dimensionality);
      } else {
        result = HWY_DISPATCH(ComputeOneToOneScore)(
            query, ptr, inv_multipliers_for_squared_l2, dimensionality);
      }

      InvokeCallback(callback, j, result, datapoint_bytes, ptr);
    }
  }

  for (size_t j : Seq(num_outer_iters)) {
    if constexpr (kShouldPrefetch) {
      if (j + num_prefetch_datapoints < num_outer_iters) {
        const size_t prefetch_j = j + num_prefetch_datapoints;

        array<const IntT*, kUnrollBy> prefetch_ptrs;
        for (size_t jj : Seq(kUnrollBy)) {
          prefetch_ptrs[jj] = get_db_ptr(prefetch_j + jj * num_outer_iters);
        }

        for (size_t n : Seq(cache_lines_per_datapoint)) {
          for (size_t jj : Seq(kUnrollBy)) {
            absl::PrefetchToLocalCacheNta(prefetch_ptrs[jj] + n * kCacheLine);
          }
        }
      }
    }

    array<const IntT*, kUnrollBy> ptrs;
    for (size_t jj : Seq(kUnrollBy)) {
      ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
    }

    using ResultT = std::conditional_t<std::is_same_v<QueryT, int8_t> &&
                                           std::is_same_v<DataT, uint8_t>,
                                       int32_t, float>;
    array<ResultT, kUnrollBy> results;
    if constexpr (std::is_same_v<QueryT, int8_t> &&
                  std::is_same_v<DataT, uint8_t>) {
      HWY_DISPATCH(ComputeOneToManyScores)<kIsSquaredL2, kUnrollBy>(
          query, ptrs, inv_multipliers_for_squared_l2, dimensionality, j,
          results);
    } else if constexpr (kUnrollBy == 4) {
      if constexpr (kIsSquaredL2) {
        HWY_DISPATCH(ComputeOneToManyScoreSquaredL24)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      } else {
        HWY_DISPATCH(ComputeOneToManyScore4)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      }
    } else if constexpr (kUnrollBy == 3) {
      if constexpr (kIsSquaredL2) {
        HWY_DISPATCH(ComputeOneToManyScoreSquaredL23)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      } else {
        HWY_DISPATCH(ComputeOneToManyScore3)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      }
    } else if constexpr (kUnrollBy == 2) {
      if constexpr (kIsSquaredL2) {
        HWY_DISPATCH(ComputeOneToManyScoreSquaredL22)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      } else {
        HWY_DISPATCH(ComputeOneToManyScore2)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      }
    } else {
      if constexpr (kIsSquaredL2) {
        HWY_DISPATCH(ComputeOneToManyScoreSquaredL21)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      } else {
        HWY_DISPATCH(ComputeOneToManyScore1)
        (query, ptrs, inv_multipliers_for_squared_l2, dimensionality, results);
      }
    }

    for (size_t jj : Seq(kUnrollBy)) {
      HWY_DISPATCH(InvokeCallbackHighway)
      (callback, j + jj * num_outer_iters, results[jj], datapoint_bytes,
       ptrs[jj]);
    }
  }
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_HIGHWAY_OUTLINE void OneToManyInt8FloatImpl(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dims = dataset_view.dimensionality();
  if (dims == 128) {
    OneToManyIntFloatTemplate<128, 3, kHasIndices, kIsSquaredL2, true, int8_t>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  } else if (dims == 64) {
    OneToManyIntFloatTemplate<64, 3, kHasIndices, kIsSquaredL2, true, int8_t>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  } else {
    OneToManyIntFloatTemplate<0, 3, kHasIndices, kIsSquaredL2, true, int8_t>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  }
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_HIGHWAY_OUTLINE void OneToManyBf16FloatImpl(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dims = dataset_view.dimensionality();
  if (dims == 128) {
    OneToManyIntFloatTemplate<128, 3, kHasIndices, kIsSquaredL2, true, int16_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  } else if (dims == 64) {
    OneToManyIntFloatTemplate<64, 3, kHasIndices, kIsSquaredL2, true, int16_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  } else {
    OneToManyIntFloatTemplate<0, 3, kHasIndices, kIsSquaredL2, true, int16_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  }
}

template <bool kHasIndices, typename DatasetViewT, typename IndexT,
          typename ResultElemT, typename CallbackT>
SCANN_HIGHWAY_OUTLINE void OneToManyUint4Int8Impl(
    const int8_t* __restrict__ query, DatasetViewT dataset_view,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dims = dataset_view.dimensionality();
  if (dims == 128) {
    OneToManyIntFloatTemplate<128, 1, kHasIndices, false, true, uint8_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  } else if (dims == 64) {
    OneToManyIntFloatTemplate<64, 1, kHasIndices, false, true, uint8_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  } else {
    OneToManyIntFloatTemplate<0, 1, kHasIndices, false, true, uint8_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  }
}

}  // namespace highway
}  // namespace research_scann
#endif
#endif
