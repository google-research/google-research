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
#include "absl/strings/str_split.h"
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
using ::absl::StrAppend;
using ::absl::StrAppendFormat;
using ::absl::StrCat;
using ::absl::StrFormat;
using ::absl::string_view;
using ::absl::StrJoin;
using ::absl::StrSplit;

using ::absl::GetFlag;

using ::absl::Mutex;
using ::absl::MutexLock;
using ::absl::ReaderMutexLock;

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

enum : uint32_t {
  kInvalidIdx32 = std::numeric_limits<uint32_t>::max(),
};
enum : uint64_t {
  kInvalidIdx64 = std::numeric_limits<uint64_t>::max(),
};

#define SCANN_INLINE inline ABSL_ATTRIBUTE_ALWAYS_INLINE

#define SCANN_INLINE_LAMBDA ABSL_ATTRIBUTE_ALWAYS_INLINE

#define SCANN_OUTLINE ABSL_ATTRIBUTE_NOINLINE

const std::string& EmptyString();

template <typename CollectionT>
bool IsEmpty(const CollectionT& c) {
  return c.empty();
}

inline bool IsEmpty(const absl::Flag<std::string>& flag) {
  return absl::GetFlag(flag).empty();
}

template <typename CollectionT>
bool NotEmpty(const CollectionT& c) {
  return !c.empty();
}

inline bool NotEmpty(const absl::Flag<std::string>& flag) {
  return !absl::GetFlag(flag).empty();
}

template <typename StatusT>
inline bool IsOk(const StatusT& status) {
  return status.ok();
}

template <typename T>
inline bool IsOk(const StatusOr<T>& statusor) {
  return statusor.ok();
}

template <typename StatusT>
inline bool NotOk(const StatusT& status) {
  return !status.ok();
}

template <typename T>
inline bool NotOk(const StatusOr<T>& statusor) {
  return !statusor.ok();
}

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

template <typename Int, typename DenomInt>
constexpr bool IsDivisibleBy(Int num, DenomInt denom) {
  return num % denom == 0;
}

template <typename T>
T ValueOrDie(StatusOr<T> statusor) {
  if (!statusor.ok()) {
    LOG(FATAL) << "VALUE_OR_DIE_FAILURE: " << statusor.status();
  }
  return std::move(statusor).ValueOrDie();
}

template <ssize_t kStride = 1>
class SeqWithStride {
 public:
  static constexpr size_t Stride() { return kStride; }

  SCANN_INLINE explicit SeqWithStride(size_t end) : begin_(0), end_(end) {}

  SCANN_INLINE SeqWithStride(size_t begin, size_t end)
      : begin_(begin), end_(end) {
    static_assert(kStride != 0);
    if (kStride > 0) {
      DCHECK_LE(begin, end);
      DCHECK_GE(static_cast<ssize_t>(begin), 0);
      DCHECK_GE(static_cast<ssize_t>(end), 0);
    } else {
      DCHECK_GE(begin, end);
      DCHECK_GE(static_cast<ssize_t>(begin + 1), 0);
      DCHECK_GE(static_cast<ssize_t>(end), 0);
    }
  }

  class SizeT {
   public:
    SCANN_INLINE SizeT(size_t val) : val_(val) {}
    SCANN_INLINE operator size_t() const { return val_; }

    SCANN_INLINE ~SizeT() {}

   private:
    size_t val_;
  };

  class Iterator {
   public:
    SCANN_INLINE explicit Iterator(size_t idx) : idx_(idx) {}

    SCANN_INLINE SizeT operator*() const {
      DCHECK_GE(static_cast<ssize_t>(idx_), 0);
      return idx_;
    }

    SCANN_INLINE Iterator& operator++() {
      idx_ += kStride;
      return *this;
    }

    SCANN_INLINE bool operator!=(Iterator end) const {
      if constexpr (kStride > 0) {
        return idx_ < end.idx_;
      }

      return static_cast<ssize_t>(idx_) >= static_cast<ssize_t>(end.idx_);
    }

   private:
    size_t idx_;
  };
  using iterator = Iterator;

  SCANN_INLINE Iterator begin() const { return Iterator(begin_); }
  SCANN_INLINE Iterator end() const { return Iterator(end_); }

 private:
  size_t begin_;
  size_t end_;
};

SCANN_INLINE auto Seq(size_t end) { return SeqWithStride<1>(0, end); }

SCANN_INLINE auto Seq(size_t begin, size_t end) {
  return SeqWithStride<1>(begin, end);
}

SCANN_INLINE auto ReverseSeq(size_t end) {
  return SeqWithStride<-1>(end - 1, 0);
}

SCANN_INLINE auto ReverseSeq(size_t begin, size_t end) {
  return SeqWithStride<-1>(end - 1, begin);
}

template <size_t kStride>
SCANN_INLINE auto ReverseSeqWithStride(size_t end) {
  end = (end - 1) / kStride * kStride;
  return SeqWithStride<-kStride>(end, 0);
}

template <size_t kStride>
SCANN_INLINE auto ReverseSeqWithStride(size_t begin, size_t end) {
  end = begin + (end - begin - 1) / kStride * kStride;
  return SeqWithStride<-kStride>(end, begin);
}

template <typename Container>
SeqWithStride<1> IndicesOf(const Container& container) {
  return Seq(container.size());
}

template <typename T, typename IdxType = size_t,
          typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto Enumerate(T&& iterable) {
  class IteratorWithIndex {
   public:
    IteratorWithIndex(IdxType idx, TIter it) : idx_(idx), it_(it) {}
    bool operator!=(const IteratorWithIndex& other) const {
      return it_ != other.it_;
    }
    void operator++() { idx_++, it_++; }
    auto operator*() const { return std::tie(idx_, *it_); }

   private:
    IdxType idx_;
    TIter it_;
  };
  struct iterator_wrapper {
    T iterable;
    auto begin() { return IteratorWithIndex{0, std::begin(iterable)}; }
    auto end() { return IteratorWithIndex{0, std::end(iterable)}; }
  };
  return iterator_wrapper{std::forward<T>(iterable)};
}

template <typename FloatT>
Status VerifyAllFiniteImpl(ConstSpan<FloatT> span) {
  for (size_t j : Seq(span.size())) {
    if (!ABSL_PREDICT_TRUE(std::isfinite(span[j]))) {
      return InternalError("Element not finite (dim idx = %d, value = %f)", j,
                           span[j]);
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
