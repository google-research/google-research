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

#ifndef SCANN_UTILS_INTRINSICS_FALLBACK_H_
#define SCANN_UTILS_INTRINSICS_FALLBACK_H_

#include <cstdint>

#include "scann/utils/index_sequence.h"
#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace fallback {

static constexpr PlatformGeneration kPlatformGeneration = kFallbackForNonX86;

template <typename T = void, size_t kTensorNumElements0 = 1,
          size_t... kTensorNumElements>
class Simd;

template <typename T, size_t kTensorNumElements0, size_t... kTensorNumElements>
using SimdFor = Simd<T, kTensorNumElements0, kTensorNumElements...>;

struct Zeros {};
struct Uninitialized {};

SCANN_INLINE string_view SimdName() { return "<Non-x86 Fallback>"; }
SCANN_INLINE bool RuntimeSupportsSimd() { return true; }

template <typename T, size_t kNumElementsRequired>
SCANN_INLINE constexpr size_t InferNumRegisters() {
  return kNumElementsRequired;
}

template <typename T>
struct FakeIntelType {
  T val;
  SCANN_INLINE FakeIntelType() {}
  SCANN_INLINE explicit FakeIntelType(T val) : val(val) {}
  SCANN_INLINE T operator[](size_t idx) const {
    DCHECK_EQ(idx, 0);
    return val;
  }
};

template <typename T, size_t kNumElementsArg>
class Simd<T, kNumElementsArg> {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(Simd);
  static_assert(IsSameAny<T, float, double, int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t>());

  static constexpr size_t kRegisterBits = sizeof(T) * 8;
  static constexpr size_t kRegisterBytes = sizeof(T);
  static constexpr size_t kNumRegisters = kNumElementsArg;
  static constexpr size_t kNumElements = kNumElementsArg;
  static constexpr size_t kElementsPerRegister = 1;

  using IntelType = FakeIntelType<T>;
  static_assert(sizeof(IntelType) == kRegisterBytes);

  Simd(Uninitialized) {}
  Simd() : Simd(Uninitialized()) {}

  SCANN_INLINE Simd(Zeros) { Clear(); }

  SCANN_INLINE Simd(IntelType val) {
    static_assert(kNumElements == 1);
    *this = val;
  }

  SCANN_INLINE Simd(T val) { *this = Broadcast(val); }

  template <typename U, size_t kOther>
  SCANN_INLINE explicit Simd(const Simd<U, kOther>& other) {
    Simd& me = *this;
    for (size_t j : Seq(kNumElements)) {
      if constexpr (kOther == kNumElements) {
        me[j] = *other[j];
      } else if constexpr (kOther == 1) {
        me[j] = *other[0];
      } else {
        static_assert(kOther == kNumElements || kOther == 1);
      }
    }
  }

  SCANN_INLINE Simd& operator=(Zeros val) {
    Clear();
    return *this;
  }

  SCANN_INLINE Simd& operator=(IntelType val) {
    static_assert(kNumElements == 1,
                  "To intentionally perform register-wise broadcast, "
                  "explicitly cast to an Simd<T>");
    registers_[0] = val;
    return *this;
  }

  SCANN_INLINE Simd& operator=(T val) {
    *this = Broadcast(val);
    return *this;
  }

  SCANN_INLINE IntelType operator*() const {
    static_assert(kNumElements == 1);
    return registers_[0];
  }

  SCANN_INLINE Simd<T, 1>& operator[](size_t idx) {
    if constexpr (kNumElements == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumElements);
      return registers_[idx];
    }
  }

  SCANN_INLINE const Simd<T, 1>& operator[](size_t idx) const {
    if constexpr (kNumElements == 1) {
      DCHECK_EQ(idx, 0);
      return *this;
    } else {
      DCHECK_LT(idx, kNumElements);
      return registers_[idx];
    }
  }

  static SCANN_INLINE Simd Zeros() {
    Simd<T, kNumElements> ret;
    for (size_t j : Seq(kNumElements)) {
      ret[j] = IntelType(0);
    }
    return ret;
  }

  SCANN_INLINE Simd& Clear() {
    for (size_t j : Seq(kNumElements)) {
      registers_[j] = IntelType(0);
    }
    return *this;
  }

  SCANN_INLINE static Simd Broadcast(T x) {
    Simd ret;
    for (size_t j : Seq(kNumElements)) {
      ret[j] = IntelType(x);
    }
    return ret;
  }

  template <bool kAligned = false>
  SCANN_INLINE static Simd Load(const T* address) {
    Simd ret;
    for (size_t j : Seq(kNumElements)) {
      ret[j] = address[j];
    }
    return ret;
  }

  SCANN_INLINE void Store(T* address) const {
    const Simd& me = *this;
    for (size_t j : Seq(kNumElements)) {
      address[j] = me[j].Unwrap();
    }
  }

  SCANN_INLINE array<T, kNumElements> Store() const {
    array<T, kNumElements> ret;
    Store(ret.data());
    return ret;
  }

  template <size_t kOther, typename Op,
            size_t kOutput = std::max(kNumElements, kOther)>
  static SCANN_INLINE Simd<T, kOutput> BinaryOperatorImpl(
      const Simd& me, const Simd<T, kOther>& other, Op fn) {
    Simd<T, kOutput> ret;
    for (size_t j : Seq(Simd<T, kOutput>::kNumElements)) {
      if constexpr (kOther == kNumElements) {
        ret[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumElements == 1) {
        ret[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        ret[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumElements || kNumElements == 1 ||
                      kOther == 1);
      }
    }
    return ret;
  }

  static SCANN_INLINE IntelType Add(IntelType a, IntelType b) {
    return IntelType(a.val + b.val);
  }

  template <size_t kOther>
  SCANN_INLINE auto operator+(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Add);
  }

  static SCANN_INLINE IntelType Subtract(IntelType a, IntelType b) {
    return IntelType(a.val - b.val);
  }

  template <size_t kOther>
  SCANN_INLINE auto operator-(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Subtract);
  }

  static SCANN_INLINE IntelType Multiply(IntelType a, IntelType b) {
    return IntelType(a.val * b.val);
  }

  template <size_t kOther>
  SCANN_INLINE auto operator*(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Multiply);
  }

  static SCANN_INLINE IntelType Divide(IntelType a, IntelType b) {
    return IntelType(a.val / b.val);
  }

  template <size_t kOther>
  SCANN_INLINE auto operator/(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &Divide);
  }

  static SCANN_INLINE IntelType BitwiseAnd(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return IntelType(a.val & b.val);
    }
    if constexpr (IsSame<T, double>()) {
      return IntelType(a.val & b.val);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return IntelType(a.val & b.val);
    }
  }

  template <size_t kOther>
  SCANN_INLINE auto operator&(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseAnd);
  }

  static SCANN_INLINE IntelType BitwiseOr(IntelType a, IntelType b) {
    if constexpr (IsSame<T, float>()) {
      return IntelType(a.val | b.val);
    }
    if constexpr (IsSame<T, double>()) {
      return IntelType(a.val | b.val);
    }
    if constexpr (IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t, int32_t,
                            uint32_t, int64_t, uint64_t>()) {
      return IntelType(a.val | b.val);
    }
  }

  template <size_t kOther>
  SCANN_INLINE auto operator|(const Simd<T, kOther>& other) const {
    return BinaryOperatorImpl(*this, other, &BitwiseOr);
  }

  static SCANN_INLINE IntelType ShiftRight(IntelType x, int count) {
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");
    return IntelType(x.val << count);
  }

  SCANN_INLINE Simd operator>>(int count) const {
    const Simd& me = *this;
    Simd ret;
    for (size_t j : Seq(kNumElements)) {
      ret[j] = ShiftRight(*me[j], count);
    }
    return ret;
  }

  static SCANN_INLINE IntelType ShiftLeft(IntelType x, int count) {
    static_assert(!IsSameAny<T, float, double>(),
                  "Bit shifting isn't defined for floating-point types.");
    return IntelType(x.val << count);
  }

  SCANN_INLINE Simd operator<<(int count) const {
    const Simd& me = *this;
    Simd ret;
    for (size_t j : Seq(kNumElements)) {
      ret[j] = ShiftLeft(*me[j], count);
    }
    return ret;
  }

  template <size_t kOther, typename Op>
  SCANN_INLINE Simd& AccumulateOperatorImpl(const Simd<T, kOther>& other,
                                            Op fn) {
    Simd& me = *this;
    for (size_t j : Seq(kNumElements)) {
      if constexpr (kOther == kNumElements) {
        me[j] = fn(*me[j], *other[j]);
      } else if constexpr (kOther == 1) {
        me[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumElements || kOther == 1);
      }
    }
    return *this;
  }

  template <size_t kOther>
  SCANN_INLINE Simd& operator+=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Add);
  }

  template <size_t kOther>
  SCANN_INLINE Simd& operator-=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Subtract);
  }

  template <size_t kOther>
  SCANN_INLINE Simd& operator*=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Multiply);
  }

  template <size_t kOther>
  SCANN_INLINE Simd& operator/=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &Divide);
  }

  template <size_t kOther>
  SCANN_INLINE Simd& operator&=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseAnd);
  }

  template <size_t kOther>
  SCANN_INLINE Simd& operator|=(const Simd<T, kOther>& other) {
    return AccumulateOperatorImpl(other, &BitwiseOr);
  }

  SCANN_INLINE Simd& operator<<=(int count) {
    Simd& me = *this;
    for (size_t j : Seq(kNumElements)) {
      me[j] = ShiftLeft(*me[j], count);
    }
    return *this;
  }

  SCANN_INLINE Simd& operator>>=(int count) {
    Simd& me = *this;
    for (size_t j : Seq(kNumElements)) {
      me[j] = ShiftRight(*me[j], count);
    }
    return *this;
  }

  template <size_t kOther = kNumElements, typename Op>
  SCANN_INLINE auto ComparisonOperatorImpl(const Simd& me,
                                           const Simd<T, kOther>& other,
                                           Op fn) const {
    array<bool, std::max(kNumElements, kOther)> masks;
    for (size_t j : Seq(std::max(kNumElements, kOther))) {
      if constexpr (kOther == kNumElements) {
        masks[j] = fn(*me[j], *other[j]);
      } else if constexpr (kNumElements == 1) {
        masks[j] = fn(*me[0], *other[j]);
      } else if constexpr (kOther == 1) {
        masks[j] = fn(*me[j], *other[0]);
      } else {
        static_assert(kOther == kNumElements || kNumElements == 1 ||
                      kOther == 1);
      }
    }
    return masks;
  }

  static SCANN_INLINE bool LessThan(IntelType a, IntelType b) {
    LOG(INFO) << StrFormat("Called: %s < %s", StrCat(a.val), StrCat(b.val));
    return a.val < b.val;
  }

  template <size_t kOther = kNumElements>
  SCANN_INLINE auto operator<(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessThan);
  }

  static SCANN_INLINE bool LessOrEquals(IntelType a, IntelType b) {
    return a.val <= b.val;
  }

  template <size_t kOther = kNumElements>
  SCANN_INLINE auto operator<=(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &LessOrEquals);
  }

  static SCANN_INLINE bool Equals(IntelType a, IntelType b) {
    return a.val == b.val;
  }

  template <size_t kOther>
  SCANN_INLINE auto operator==(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &Equals);
  }

  static SCANN_INLINE bool GreaterOrEquals(IntelType a, IntelType b) {
    return a.val >= b.val;
  }

  template <size_t kOther = kNumElements>
  SCANN_INLINE auto operator>=(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterOrEquals);
  }

  static SCANN_INLINE bool GreaterThan(IntelType a, IntelType b) {
    return a.val > b.val;
  }

  template <size_t kOther = kNumElements>
  SCANN_INLINE auto operator>(const Simd<T, kOther>& other) const {
    return ComparisonOperatorImpl(*this, other, &GreaterThan);
  }

  SCANN_INLINE int MaskFromHighBits() const {
    static_assert(0 != 1, "Not Implemented. TODO: implement if/when needed.");
  }

  SCANN_INLINE T GetLowElement() const {
    const Simd& me = *this;
    return (*me[0]).val;
  }

  SCANN_INLINE T Unwrap() const { return GetLowElement(); }

  template <typename U>
  SCANN_INLINE Simd<U, kNumElements> ConvertTo() const {
    static_assert(IsDivisibleBy(kNumElements, 2));
    const Simd& me = *this;
    Simd<U, kNumElements> ret;
    for (size_t j : Seq(kNumElements)) {
      ret[j] = static_cast<U>((*me[j]).val);
    }
    return ret;
  }

  static SCANN_INLINE auto InferExpansionType() {
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
  using ExpansionIntelType = typename Simd<ExpansionType>::IntelType;
  using ExpandsTo = Simd<ExpansionType, kNumElements>;

  template <typename ValidateT>
  SCANN_INLINE Simd<ExpansionType, kNumElements> ExpandTo() const {
    static_assert(IsSame<ValidateT, ExpansionType>());
    const Simd& me = *this;
    Simd<ExpansionType, kNumElements> ret;
    for (size_t j : Seq(kNumElements)) {
      ret[j] = static_cast<ExpansionType>(me[j].Unwrap());
    }
    return ret;
  }

 private:
  std::conditional_t<kNumElements == 1, IntelType, Simd<T, 1>>
      registers_[kNumElements];

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

  SCANN_INLINE Simd(Zeros) {
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j] = Zeros();
    }
  }

  SCANN_INLINE SimdSubArray& operator[](size_t idx) {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_INLINE const SimdSubArray& operator[](size_t idx) const {
    DCHECK_LT(idx, kTensorNumRegisters0);
    return tensor_[idx];
  }

  SCANN_INLINE void Store(T* address) const {
    constexpr size_t kStride =
        sizeof(decltype(SimdSubArray().Store())) / sizeof(T);
    for (size_t j : Seq(kTensorNumRegisters0)) {
      tensor_[j].Store(address + j * kStride);
    }
  }

  using StoreResultType =
      array<decltype(SimdSubArray().Store()), kTensorNumRegisters0>;
  SCANN_INLINE StoreResultType Store() const {
    StoreResultType ret;
    for (size_t j : Seq(kTensorNumRegisters0)) {
      ret[j] = tensor_[j].Store();
    }
    return ret;
  }

 private:
  SimdSubArray tensor_[kTensorNumRegisters0];
};

}  // namespace fallback

SCANN_INLINE uint32_t GetComparisonMask(array<bool, 32> cmp) {
  uint32_t ret = 0;
  for (size_t j : ReverseSeq(cmp.size())) {
    ret <<= 1;
    ret += (cmp[j] ? 1 : 0);
  }
  return ret;
}

SCANN_INLINE uint64_t GetComparisonMask(array<bool, 64> cmp) {
  uint64_t ret = 0;
  for (size_t j : ReverseSeq(cmp.size())) {
    ret <<= 1;
    ret |= (cmp[j] ? 1 : 0);
  }
  return ret;
}

}  // namespace research_scann

#endif
