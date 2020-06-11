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

#ifndef SCANN__UTILS_COMMON_H_
#define SCANN__UTILS_COMMON_H_

#include <stddef.h>

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_serialize.h"
#include "scann/oss_wrappers/scann_status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace scann_ops {

using ::std::array;
using ::std::make_shared;
using ::std::numeric_limits;
using ::std::pair;
using ::std::shared_ptr;
using ::std::tuple;
using ::std::unique_ptr;
using ::std::vector;

using std::vector;

using ::absl::make_unique;

using ::absl::flat_hash_map;
using ::absl::flat_hash_set;
using ::absl::node_hash_map;
using ::absl::node_hash_set;

using ::absl::PrintF;
using ::absl::StrAppendFormat;
using ::absl::StrCat;
using ::absl::StrFormat;
using ::absl::string_view;
using ::absl::StrJoin;

using ::absl::GetFlag;

using OkStatus = Status;
using internal::StatusOr;

#define MAKE_TF_ERROR_FORWARDER(ERRNAME)                                      \
  ABSL_MUST_USE_RESULT inline Status ERRNAME##Error(absl::string_view s) {    \
    return ::tensorflow::errors::ERRNAME(std::forward<absl::string_view>(s)); \
  }

MAKE_TF_ERROR_FORWARDER(Aborted);
MAKE_TF_ERROR_FORWARDER(AlreadyExists);
MAKE_TF_ERROR_FORWARDER(Cancelled);
MAKE_TF_ERROR_FORWARDER(FailedPrecondition);
MAKE_TF_ERROR_FORWARDER(Internal);
MAKE_TF_ERROR_FORWARDER(InvalidArgument);
MAKE_TF_ERROR_FORWARDER(NotFound);
MAKE_TF_ERROR_FORWARDER(OutOfRange);
MAKE_TF_ERROR_FORWARDER(Unauthenticated);
MAKE_TF_ERROR_FORWARDER(Unavailable);
MAKE_TF_ERROR_FORWARDER(Unimplemented);
MAKE_TF_ERROR_FORWARDER(Unknown);

#undef MAKE_TF_ERROR_FORWARDER

#define SCANN_INLINE inline ABSL_ATTRIBUTE_ALWAYS_INLINE

#define SCANN_INLINE_LAMBDA ABSL_ATTRIBUTE_ALWAYS_INLINE

#define SCANN_OUTLINE ABSL_ATTRIBUTE_NOINLINE

struct VirtualDestructor {
  virtual ~VirtualDestructor() {}
};

struct MoveOnly {
  MoveOnly() = default;

  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
};

#define SCANN_DECLARE_COPYABLE_CLASS(ClassName) \
  ClassName(ClassName&&) = default;             \
  ClassName& operator=(ClassName&&) = default;  \
  ClassName(const ClassName&) = default;        \
  ClassName& operator=(const ClassName&) = default

#define SCANN_DECLARE_MOVE_ONLY_CLASS(ClassName) \
  ClassName(ClassName&&) = default;              \
  ClassName& operator=(ClassName&&) = default;   \
  ClassName(const ClassName&) = delete;          \
  ClassName& operator=(const ClassName&) = delete

#define SCANN_DECLARE_MOVE_ONLY_CLASS_CUSTOM_IMPL(ClassName) \
  ClassName(const ClassName&) = delete;                      \
  ClassName& operator=(const ClassName&) = delete

#define SCANN_DECLARE_IMMOBILE_CLASS(ClassName) \
  ClassName(ClassName&&) = delete;              \
  ClassName& operator=(ClassName&&) = delete;   \
  ClassName(const ClassName&) = delete;         \
  ClassName& operator=(const ClassName&) = delete

template <typename T>
using ConstSpan = absl::Span<const T>;

using ::absl::MakeConstSpan;

template <typename T>
using MutableSpan = absl::Span<T>;

template <typename... Args>
auto MakeMutableSpan(Args&&... args)
    -> decltype(absl::MakeSpan(std::forward<Args>(args)...)) {
  return absl::MakeSpan(std::forward<Args>(args)...);
}

template <typename CollectionT>
bool NotEmpty(const CollectionT& c) {
  return !c.empty();
}

inline bool NotOk(const Status& status) { return !status.ok(); }

template <typename... Args>
ABSL_MUST_USE_RESULT Status FailedPreconditionError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return FailedPreconditionError(StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT Status InternalError(const absl::FormatSpec<Args...>& fmt,
                                          const Args&... args) {
  return InternalError(StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT Status InvalidArgumentError(
    const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return InvalidArgumentError(StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT Status NotFoundError(const absl::FormatSpec<Args...>& fmt,
                                          const Args&... args) {
  return NotFoundError(StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT Status
OutOfRangeError(const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return OutOfRangeError(StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT Status
UnavailableError(const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return UnavailableError(StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT Status
UnimplementedError(const absl::FormatSpec<Args...>& fmt, const Args&... args) {
  return UnimplementedError(StrFormat(fmt, args...));
}

template <typename... Args>
ABSL_MUST_USE_RESULT Status UnknownError(const absl::FormatSpec<Args...>& fmt,
                                         const Args&... args) {
  return UnknownError(StrFormat(fmt, args...));
}

template <typename ContainerT>
void FreeBackingStorage(ContainerT* c) {
  DCHECK(c);
  *c = ContainerT();
}

template <typename Int, typename DenomInt>
constexpr Int DivRoundUp(Int num, DenomInt denom) {
  return (num + static_cast<Int>(denom) - static_cast<Int>(1)) /
         static_cast<Int>(denom);
}

template <typename Int, typename DenomInt>
constexpr Int NextMultipleOf(Int num, DenomInt denom) {
  return DivRoundUp(num, denom) * denom;
}

template <typename T>
T ValueOrDie(StatusOr<T> statusor) {
  if (!statusor.ok()) {
    LOG(FATAL) << "VALUE_OR_DIE_FAILURE: " << statusor.status();
  }
  return std::move(statusor).ValueOrDie();
}

template <size_t kStride = 1>
class SeqWithStride {
 public:
  static constexpr size_t Stride() { return kStride; }

  SCANN_INLINE explicit SeqWithStride(size_t end) : begin_(0), end_(end) {}

  SCANN_INLINE SeqWithStride(size_t begin, size_t end)
      : begin_(begin), end_(end) {
    DCHECK_LE(begin, end);
  }

  class iterator {
   public:
    class SizeT {
     public:
      SCANN_INLINE SizeT(size_t val) : val_(val) {}

      SCANN_INLINE ~SizeT() {}

      operator size_t() const { return val_; }

     private:
      size_t val_;
    };

    SCANN_INLINE explicit iterator(size_t i) : i_(i) {}
    SCANN_INLINE SizeT operator*() const { return i_; }
    SCANN_INLINE iterator& operator++() {
      i_ += kStride;
      return *this;
    }
    SCANN_INLINE bool operator!=(iterator rhs) const { return i_ < rhs.i_; }

   private:
    size_t i_;
  };

  SCANN_INLINE iterator begin() const { return iterator(begin_); }
  SCANN_INLINE iterator end() const { return iterator(end_); }

 private:
  size_t begin_;
  size_t end_;
};

SCANN_INLINE SeqWithStride<1> Seq(size_t end) {
  return SeqWithStride<1>(0, end);
}

SCANN_INLINE SeqWithStride<1> Seq(size_t begin, size_t end) {
  return SeqWithStride<1>(begin, end);
}

template <typename Container>
SeqWithStride<1> IndicesOf(const Container& container) {
  return Seq(container.size());
}

template <typename FloatT>
Status VerifyAllFiniteImpl(ConstSpan<FloatT> span) {
  for (size_t i : IndicesOf(span)) {
    if (!ABSL_PREDICT_TRUE(std::isfinite(span[i]))) {
      return InternalError("Element not finite (dim idx = %d, value = %f)", i,
                           span[i]);
    }
  }
  return OkStatus();
}

inline Status VerifyAllFinite(ConstSpan<float> span) {
  return VerifyAllFiniteImpl<float>(span);
}

inline Status VerifyAllFinite(ConstSpan<double> span) {
  return VerifyAllFiniteImpl<double>(span);
}

using ::absl::conditional_t;
using ::absl::decay_t;
using ::absl::enable_if_t;
using ::absl::make_signed_t;
using ::absl::make_unsigned_t;
using ::absl::remove_cv_t;
using ::absl::remove_pointer_t;
using ::std::declval;

template <typename T, typename U>
inline constexpr bool IsSame() {
  return std::is_same<T, U>::value;
}

namespace is_same_any_impl {

template <typename T, typename... UU>
struct IsSameAny;

template <typename T>
struct IsSameAny<T> : std::false_type {};

template <typename T, typename U, typename... UU>
struct IsSameAny<T, U, UU...> {
  static constexpr bool value = IsSame<T, U>() || IsSameAny<T, UU...>::value;
};

}  // namespace is_same_any_impl

template <typename T, typename... UU>
inline constexpr bool IsSameAny() {
  return is_same_any_impl::IsSameAny<T, UU...>::value;
}

template <typename T>
inline constexpr bool IsUint8() {
  return IsSame<T, uint8_t>();
}

template <typename T>
inline constexpr bool IsFloat() {
  return IsSame<T, float>();
}

template <typename T>
inline constexpr bool IsDouble() {
  return IsSame<T, double>();
}

template <typename T>
inline constexpr bool IsFixedPointType() {
  return IsSameAny<T, int8_t, uint8_t, int16_t, uint16_t>();
}

template <typename T>
inline constexpr bool IsFloatingType() {
  return std::is_floating_point<decay_t<T>>::value;
}

template <typename T>
inline constexpr bool IsIntegerType() {
  return std::is_integral<T>::value;
}

template <typename T>
inline constexpr bool IsSignedType() {
  return std::is_signed<T>::value;
}

template <typename T>
constexpr bool IsStatusType() {
  return std::is_convertible<Status, T>::value;
}

template <typename T>
inline constexpr bool IsPod() {
  return std::is_pod<decay_t<T>>::value;
}

template <typename T>
using ConstRefOrPod = conditional_t<IsPod<T>(), decay_t<T>, const decay_t<T>&>;

template <typename T>
struct RecursivelyRemoveCVImpl {
  using type = remove_cv_t<T>;
};

template <typename T>
using RecursivelyRemoveCV = typename RecursivelyRemoveCVImpl<T>::type;

template <typename K, typename V>
struct RecursivelyRemoveCVImpl<pair<K, V>> {
  using type = pair<RecursivelyRemoveCV<K>, RecursivelyRemoveCV<V>>;
};

template <typename... Ts>
struct RecursivelyRemoveCVImpl<tuple<Ts...>> {
  using type = tuple<RecursivelyRemoveCV<Ts>...>;
};

static_assert(IsSame<pair<const int, volatile float>,
                     remove_cv_t<pair<const int, volatile float>>>(),
              "");
static_assert(IsSame<pair<int, float>,
                     RecursivelyRemoveCV<pair<const int, volatile float>>>(),
              "");

}  // namespace scann_ops
}  // namespace tensorflow

#endif
