// Copyright 2026 The Google Research Authors.
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



#ifndef SCANN_UTILS_INTRINSICS_HIGHWAY_H_
#define SCANN_UTILS_INTRINSICS_HIGHWAY_H_

#include "hwy/detect_targets.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_ALL_SVE
#endif

#endif

#if defined(SCANN_UTILS_INTRINSICS_HIGHWAY_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef SCANN_UTILS_INTRINSICS_HIGHWAY_TOGGLE
#undef SCANN_UTILS_INTRINSICS_HIGHWAY_TOGGLE
#else
#define SCANN_UTILS_INTRINSICS_HIGHWAY_TOGGLE
#endif

#include "hwy/highway.h"

#if !HWY_HAVE_CONSTEXPR_LANES

#include "scann/utils/intrinsics/fallback.h"

namespace research_scann {

namespace HWY_NAMESPACE = fallback;

}

#else

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "scann/utils/common.h"
#include "scann/utils/index_sequence.h"
#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/flags.h"

HWY_BEFORE_NAMESPACE();
namespace research_scann {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

static constexpr PlatformGeneration kPlatformGeneration = kHighway;

template <typename T, size_t kNumElementsRequired>
constexpr size_t InferNumRegisters() {
  return kNumElementsRequired / hn::Lanes(hn::ScalableTag<T>());
}

template <typename T, size_t kNumRegisters = 1, size_t... kTensorNumRegisters>
class Simd;

struct Zeros {};
struct Uninitialized {};

template <typename T, size_t kNumRegistersInferred>
class Simd<T, kNumRegistersInferred> {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(Simd);
  static_assert(IsSameAny<T, float, double, int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t>());

  static constexpr size_t kRegisterBytes =
      hn::Lanes(hn::ScalableTag<T>()) * sizeof(T);
  static constexpr size_t kRegisterBits = kRegisterBytes * 8;
  static constexpr size_t kNumRegisters = kNumRegistersInferred;
  static constexpr size_t kElementsPerRegister = kRegisterBytes / sizeof(T);
  static constexpr size_t kNumElements = kNumRegisters * kElementsPerRegister;

  using HwyType = decltype(hn::Zero(hn::ScalableTag<T>()));

  Simd(Uninitialized) {}
  Simd() : Simd(Uninitialized()) {}

  SCANN_HIGHWAY_INLINE Simd(Zeros) { Clear(); }

  SCANN_HIGHWAY_INLINE Simd(HwyType val) {
    static_assert(kNumRegisters == 1);
    *this = val;
  }

  SCANN_HIGHWAY_INLINE Simd(T val) { *this = Broadcast(val); }

  template <typename U, size_t kOther>
  SCANN_HIGHWAY_INLINE explicit Simd(const Simd<U, kOther>& other) {
    Simd& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        me[j] = hn::BitCast(hn::ScalableTag<T>(), *other[j]);
      } else if constexpr (kOther == 1) {
        me[j] = hn::BitCast(hn::ScalableTag<T>(), *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kOther == 1);
      }
    }
  }

  SCANN_HIGHWAY_INLINE Simd& operator=(Zeros val) {
    Clear();
    return *this;
  }

  SCANN_HIGHWAY_INLINE Simd& operator=(HwyType val) {
    static_assert(kNumRegisters == 1,
                  "To intentionally perform register-wise broadcast, "
                  "explicitly cast to an Simd<T>");
    registers_[0] = val;
    return *this;
  }

  SCANN_HIGHWAY_INLINE Simd& operator=(T val) {
    *this = Broadcast(val);
    return *this;
  }

  SCANN_HIGHWAY_INLINE HwyType operator*() const {
    static_assert(kNumRegisters == 1);
    return registers_[0];
  }

  SCANN_HIGHWAY_INLINE Simd<T, 1>& operator[](size_t idx) {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  SCANN_HIGHWAY_INLINE const Simd<T, 1>& operator[](size_t idx) const {
    if constexpr (kNumRegisters == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumRegisters);
      return registers_[idx];
    }
  }

  static SCANN_HIGHWAY_INLINE HwyType ZeroOneRegister() {
    return hn::Zero(hn::ScalableTag<T>());
  }

  static SCANN_HIGHWAY_INLINE Simd Zeros() {
    Simd<T, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ZeroOneRegister();
    }
    return ret;
  }

  SCANN_HIGHWAY_INLINE Simd& Clear() {
    for (size_t j : Seq(kNumRegisters)) {
      registers_[j] = ZeroOneRegister();
    }
    return *this;
  }

  static SCANN_HIGHWAY_INLINE HwyType BroadcastOneRegister(T x) {
    return hn::Set(hn::ScalableTag<T>(), x);
  }

  SCANN_HIGHWAY_INLINE static Simd Broadcast(T x) {
    Simd ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = BroadcastOneRegister(x);
    }
    return ret;
  }

  template <bool kAligned = false>
  static SCANN_HIGHWAY_INLINE HwyType LoadOneRegister(const T* address) {
    return hn::LoadU(hn::ScalableTag<T>(), address);
  }

  template <bool kAligned = false>
  SCANN_HIGHWAY_INLINE static Simd Load(const T* address) {
    Simd ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = LoadOneRegister<kAligned>(address + j * kElementsPerRegister);
    }
    return ret;
  }

  static SCANN_HIGHWAY_INLINE void StoreOneRegister(T* address, HwyType x) {
    hn::StoreU(x, hn::ScalableTag<T>(), address);
  }

  SCANN_HIGHWAY_INLINE void Store(T* address) const {
    const Simd& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      StoreOneRegister(address + j * kElementsPerRegister, *me[j]);
    }
  }

  SCANN_HIGHWAY_INLINE array<T, kNumElements> Store() const {
    array<T, kNumElements> ret;
    Store(ret.data());
    return ret;
  }

  template <size_t kOther, typename Op,
            size_t kOutput = std::max(kNumRegisters, kOther)>
  static SCANN_HIGHWAY_INLINE Simd<T, kOutput> BinaryOperatorImpl(
      const Simd& me, const Simd<T, kOther>& other, Op fn) {
    Simd<T, kOutput> ret;
    for (size_t j : Seq(Simd<T, kOutput>::kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        ret[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumRegisters == 1) {
        ret[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        ret[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kNumRegisters == 1 ||
                      kOther == 1);
      }
    }
    return ret;
  }

  static SCANN_HIGHWAY_INLINE HwyType Add(HwyType a, HwyType b) {
    return a + b;
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE auto operator+(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Add);
  }

  static SCANN_HIGHWAY_INLINE HwyType Subtract(HwyType a, HwyType b) {
    return a - b;
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE auto operator-(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Subtract);
  }

  static SCANN_HIGHWAY_INLINE HwyType Multiply(HwyType a, HwyType b) {
    return a * b;
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE auto operator*(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Multiply);
  }

  static SCANN_HIGHWAY_INLINE HwyType Divide(HwyType a, HwyType b) {
    return a / b;
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE auto operator/(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Divide);
  }

  static SCANN_HIGHWAY_INLINE auto BitwiseAnd(HwyType a, HwyType b) {
    return a & b;
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE auto operator&(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseAnd);
  }

  static SCANN_HIGHWAY_INLINE auto BitwiseOr(HwyType a, HwyType b) {
    return a | b;
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE auto operator|(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseOr);
  }

  static SCANN_HIGHWAY_INLINE HwyType ShiftRight(HwyType x, int count) {
    return x >> count;
  }

  SCANN_HIGHWAY_INLINE Simd operator>>(int count) const {
    const Simd& me = *this;
    Simd ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftRight(*me[j], count);
    }
    return ret;
  }

  static SCANN_HIGHWAY_INLINE HwyType ShiftLeft(HwyType x, int count) {
    return x << count;
  }

  SCANN_HIGHWAY_INLINE Simd operator<<(int count) const {
    const Simd& me = *this;
    Simd ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ShiftLeft(*me[j], count);
    }
    return ret;
  }

  template <size_t kOther, typename Op>
  SCANN_HIGHWAY_INLINE Simd& AccumulateOperatorImpl(
      const Simd<T, kOther>& other, Op fn) {
    Simd& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      if constexpr (kOther == kNumRegisters) {
        me[j] = fn(*me[j], *other[j]);
      } else if constexpr (kOther == 1) {
        me[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kOther == 1);
      }
    }
    return *this;
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE Simd& operator+=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Add);
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE Simd& operator-=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Subtract);
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE Simd& operator*=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Multiply);
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE Simd& operator/=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Divide);
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE Simd& operator&=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseAnd);
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE Simd& operator|=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseOr);
  }

  SCANN_HIGHWAY_INLINE Simd& operator<<=(int count) {
    Simd& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftLeft(*me[j], count);
    }
    return *this;
  }

  SCANN_HIGHWAY_INLINE Simd& operator>>=(int count) {
    Simd& me = *this;
    for (size_t j : Seq(kNumRegisters)) {
      me[j] = ShiftRight(*me[j], count);
    }
    return *this;
  }

  template <size_t kOther = kNumRegisters, typename Op>
  SCANN_HIGHWAY_INLINE auto ComparisonOperatorImpl(const Simd& me,
                                                   const Simd<T, kOther>& other,
                                                   Op fn) const {
    Simd<T, std::max(kNumRegisters, kOther)> masks;
    for (size_t j : Seq(std::max(kNumRegisters, kOther))) {
      if constexpr (kOther == kNumRegisters) {
        masks[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumRegisters == 1) {
        masks[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        masks[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumRegisters || kNumRegisters == 1 ||
                      kOther == 1);
      }
    }
    return masks;
  }

  static SCANN_HIGHWAY_INLINE auto LessThan(HwyType a, HwyType b) {
    return hn::VecFromMask(hn::ScalableTag<T>(), a < b);
  }

  template <size_t kOther = kNumRegisters>
  SCANN_HIGHWAY_INLINE auto operator<(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessThan);
  }

  static SCANN_HIGHWAY_INLINE HwyType LessOrEquals(HwyType a, HwyType b) {
    return hn::VecFromMask(hn::ScalableTag<T>(), a <= b);
  }

  template <size_t kOther = kNumRegisters>
  SCANN_HIGHWAY_INLINE auto operator<=(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessOrEquals);
  }

  static SCANN_HIGHWAY_INLINE HwyType Equals(HwyType a, HwyType b) {
    return hn::VecFromMask(hn::ScalableTag<T>(), a == b);
  }

  template <size_t kOther>
  SCANN_HIGHWAY_INLINE auto operator==(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &Equals);
  }

  static SCANN_HIGHWAY_INLINE HwyType GreaterOrEquals(HwyType a, HwyType b) {
    return hn::VecFromMask(hn::ScalableTag<T>(), a >= b);
  }

  template <size_t kOther = kNumRegisters>
  SCANN_HIGHWAY_INLINE auto operator>=(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterOrEquals);
  }

  static SCANN_HIGHWAY_INLINE HwyType GreaterThan(HwyType a, HwyType b) {
    return hn::VecFromMask(hn::ScalableTag<T>(), a > b);
  }

  template <size_t kOther = kNumRegisters>
  SCANN_HIGHWAY_INLINE auto operator>(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterThan);
  }

  SCANN_HIGHWAY_INLINE auto MaskFromHighBits() const {
    static_assert(kNumRegisters == 1);
    return BitsFromMask(registers_[0]);
  }

  SCANN_HIGHWAY_INLINE T GetLowElement() const {
    static_assert(kNumRegisters == 1);
    return hn::GetLane(registers_[0]);
  }

  template <typename U>
  static SCANN_HIGHWAY_INLINE typename Simd<U>::HwyType ConvertOneRegister(
      HwyType x) {
    return hn::ConvertTo(hn::ScalableTag<U>(), x);
  }

  template <typename U>
  SCANN_HIGHWAY_INLINE Simd<U, kNumRegisters> ConvertTo() const {
    const Simd& me = *this;
    Simd<U, kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      ret[j] = ConvertOneRegister<U>(*me[j]);
    }
    return ret;
  }

  static SCANN_HIGHWAY_INLINE auto InferExpansionType() {
    if constexpr (IsSame<T, float>()) {
      return double();
    }
    if constexpr (IsSame<T, double>()) {
      return double();
    }

    if constexpr (IsSame<T, int8_t>()) {
      return int16_t();
    }
    if constexpr (IsSame<T, int16_t>()) {
      return int32_t();
    }
    if constexpr (IsSame<T, int32_t>()) {
      return int64_t();
    }
    if constexpr (IsSame<T, int64_t>()) {
      return int64_t();
    }

    if constexpr (IsSame<T, uint8_t>()) {
      return uint16_t();
    }
    if constexpr (IsSame<T, uint16_t>()) {
      return uint32_t();
    }
    if constexpr (IsSame<T, uint32_t>()) {
      return uint64_t();
    }
    if constexpr (IsSame<T, uint64_t>()) {
      return uint64_t();
    }
  }
  using ExpansionType = decltype(InferExpansionType());
  using ExpansionHwyType = typename Simd<ExpansionType>::HwyType;
  using ExpandsTo = Simd<ExpansionType, 2 * kNumRegisters>;

  static SCANN_HIGHWAY_INLINE pair<ExpansionHwyType, ExpansionHwyType>
  ExpandOneRegister(HwyType x) {
    hn::ScalableTag<ExpansionType> d;
    return std::make_pair(hn::PromoteLowerTo(d, x), hn::PromoteUpperTo(d, x));
  }

  template <typename ValidateT>
  SCANN_HIGHWAY_INLINE ExpandsTo ExpandTo() const {
    static_assert(IsSame<ValidateT, ExpansionType>());
    const Simd& me = *this;
    Simd<ExpansionType, 2 * kNumRegisters> ret;
    for (size_t j : Seq(kNumRegisters)) {
      pair<ExpansionHwyType, ExpansionHwyType> expanded =
          ExpandOneRegister(*me[j]);
      ret[2 * j + 0] = expanded.first;
      ret[2 * j + 1] = expanded.second;
    }
    return ret;
  }

  SCANN_INLINE static auto BitsFromMask(HwyType vec) {
    const hn::ScalableTag<T> tag;
    constexpr size_t kLanes = hn::Lanes(tag);
    static_assert(kLanes <= 64,
                  "BitsFromMask is not implemented for >64 lanes SIMD.");

    constexpr bool kCanUseBitsFromMask = HWY_MAX_BYTES <= 64;
    if constexpr (kCanUseBitsFromMask) {
      using ResultT = std::conditional_t<kLanes <= 32, uint32_t, uint64_t>;
      return static_cast<ResultT>(hn::BitsFromMask(tag, hn::MaskFromVec(vec)));
    } else {
      constexpr size_t kExpectedBytesWritten = (hn::Lanes(tag) + 7) / 8;
      using ResultT =
          std::conditional_t<kExpectedBytesWritten <= 4, uint32_t, uint64_t>;
      uint8_t mask_bits[8];
      const size_t bytes_written =
          hn::StoreMaskBits(tag, hn::MaskFromVec(vec), mask_bits);
      DCHECK_EQ(bytes_written, kExpectedBytesWritten);
      ResultT result = 0;
      for (uint8_t i = 0; i < kExpectedBytesWritten; ++i) {
        result |= static_cast<ResultT>(mask_bits[i]) << (i * 8);
      }
      return result;
    }
  }

  SCANN_INLINE T ExtractLane(size_t lane) {
    static_assert(kNumRegisters == 1);
    return hn::ExtractLane(registers_[0], lane);
  }

 private:
  std::conditional_t<kNumRegisters == 1, HwyType, Simd<T, 1> >
      registers_[kNumRegisters];

  template <typename U, size_t kOther, size_t... kTensorOther>
  friend class Simd;
};

template <typename T, size_t kTensorNumRegisters0,
          size_t... kTensorNumRegisters>
class Simd {
 public:
  using SimdSubArray = Simd<T, kTensorNumRegisters...>;

  Simd(Uninitialized) {}
  Simd() : Simd(Uninitialized()) {}

  SCANN_HIGHWAY_INLINE Simd(Zeros) {
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j] = Zeros();
    }
  }

  SCANN_HIGHWAY_INLINE SimdSubArray& operator[](size_t idx) {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_HIGHWAY_INLINE const SimdSubArray& operator[](size_t idx) const {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_HIGHWAY_INLINE void Store(T* address) const {
    constexpr size_t kStride =
        sizeof(decltype(SimdSubArray().Store())) / sizeof(T);
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j].Store(address + j * kStride);
    }
  }

  using StoreResultType =
      array<decltype(SimdSubArray().Store()), kTensorNumRegisters0>;
  SCANN_HIGHWAY_INLINE StoreResultType Store() const {
    StoreResultType ret;
    for (size_t j : Seq(kTensorNumRegisters0)) {
      ret[j] = tensor_[j].Store();
    }
    return ret;
  }

 private:
  SimdSubArray tensor_[kTensorNumRegisters0];
};

template <typename T, size_t... kNumRegisters>
SCANN_HIGHWAY_INLINE Simd<T, index_sequence_sum_v<kNumRegisters...> >
HighwayConcat(const Simd<T, kNumRegisters>&... inputs) {
  Simd<T, index_sequence_sum_v<kNumRegisters...> > ret;

  size_t idx = 0;
  auto assign_one_input = [&](auto input) SCANN_INLINE_LAMBDA {
    for (size_t jj : Seq(decltype(input)::kNumRegisters)) {
      ret[idx++] = input[jj];
    }
  };
  (assign_one_input(inputs), ...);

  return ret;
}

template <typename T, typename AllButLastSeq, size_t kLast>
struct SimdForImpl;

template <typename T, size_t... kAllButLast, size_t kLast>
struct SimdForImpl<T, index_sequence<kAllButLast...>, kLast> {
  using type = Simd<T, kAllButLast..., InferNumRegisters<T, kLast>()>;
};

template <typename T, size_t... kTensorNumElements>
using SimdFor =
    typename SimdForImpl<T,
                         index_sequence_all_but_last_t<kTensorNumElements...>,
                         index_sequence_last_v<kTensorNumElements...> >::type;

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<int16_t> a, Simd<int16_t> b) {
  auto demoted = hn::OrderedDemote2To(hn::ScalableTag<int8_t>(), *a, *b);
  return Simd<int8_t>::BitsFromMask(demoted);
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<int16_t, 2> a) {
  return GetComparisonMask(a[0], a[1]);
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<int16_t> a) {
  return Simd<int16_t>::BitsFromMask(*a);
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<int16_t> a, Simd<int16_t> b,
                                            Simd<int16_t> c, Simd<int16_t> d) {
  if constexpr (hn::Lanes(hn::ScalableTag<int16_t>()) <= 16) {
    constexpr int kLanes = hn::Lanes(hn::ScalableTag<int16_t>());
    using ResultT = std::conditional_t<kLanes <= 8, uint32_t, uint64_t>;
    const ResultT m00 = GetComparisonMask(a, b);
    const ResultT m16 = GetComparisonMask(c, d);
    return m00 + (m16 << (kLanes * 2));
  } else {
    LOG(FATAL) << "This GetComparisonMask overload is not implemented for >256 "
                  "bit SIMD.";
  }
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<int16_t, 4> cmp) {
  return GetComparisonMask(cmp[0], cmp[1], cmp[2], cmp[3]);
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<int16_t> cmp[2]) {
  return GetComparisonMask(cmp[0], cmp[1]);
}

SCANN_HIGHWAY_INLINE uint32_t GetComparisonMask(Simd<float> cmp) {
  return Simd<float>::BitsFromMask(*cmp);
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<float> v00, Simd<float> v04,
                                            Simd<float> v08, Simd<float> v12) {
  constexpr int kLanes = hn::Lanes(hn::ScalableTag<float>());
  using ResultT = std::conditional_t<kLanes <= 8, uint32_t, uint64_t>;
  const ResultT m00 = GetComparisonMask(v00);
  const ResultT m04 = GetComparisonMask(v04);
  const ResultT m08 = GetComparisonMask(v08);
  const ResultT m12 = GetComparisonMask(v12);
  return m00 + (m04 << kLanes) + (m08 << (2 * kLanes)) + (m12 << (3 * kLanes));
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<float, 2> cmp) {
  constexpr int kLanes = hn::Lanes(hn::ScalableTag<float>());
  using ResultT = std::conditional_t<kLanes <= 8, uint32_t, uint64_t>;
  const ResultT m0 = GetComparisonMask(cmp[0]);
  const ResultT m1 = GetComparisonMask(cmp[1]);
  return m0 + (m1 << kLanes);
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<float, 4> cmp) {
  return GetComparisonMask(cmp[0], cmp[1], cmp[2], cmp[3]);
}

SCANN_HIGHWAY_INLINE auto GetComparisonMask(Simd<float, 8> cmp) {
  constexpr int kLanes = hn::Lanes(hn::ScalableTag<float>());
  if constexpr (kLanes <= 8) {
    using ResultT = std::conditional_t<kLanes <= 4, uint32_t, uint64_t>;
    ResultT mask0 = GetComparisonMask(cmp[0], cmp[1], cmp[2], cmp[3]);
    ResultT mask1 = GetComparisonMask(cmp[4], cmp[5], cmp[6], cmp[7]);
    return mask0 + (mask1 << (kLanes * 4));
  } else {
    LOG(FATAL)
        << "This GetComparisonMask overload is not implemented for >256 bit "
           "SIMD.";
  }
}
SCANN_INLINE string_view SimdName() { return hwy::TargetName(HWY_TARGET); }
SCANN_INLINE bool RuntimeSupportsSimd() { return true; }

}  // namespace HWY_NAMESPACE
}  // namespace research_scann

HWY_AFTER_NAMESPACE();

#endif
#endif
