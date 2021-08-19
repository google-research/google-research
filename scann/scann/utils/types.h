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



#ifndef SCANN_UTILS_TYPES_H_
#define SCANN_UTILS_TYPES_H_

#include <cstdint>
#include <limits>
#include <type_traits>

#include "scann/proto/input_output.pb.h"
#include "scann/utils/common.h"

namespace research_scann {

#define DCHECK_OK(val) DCHECK_EQ(OkStatus(), (val))

using DatapointIndex = uint32_t;
enum : DatapointIndex {
  kInvalidDatapointIndex = std::numeric_limits<DatapointIndex>::max(),
};

enum : size_t {
  kMaxNumDatapoints = 1 << 30,
};

using DimensionIndex = uint64_t;
enum : DimensionIndex {
  kInvalidDimension = std::numeric_limits<DimensionIndex>::max(),
};

inline DimensionIndex KeyToDimensionIndex(absl::string_view key) {
  return strings::KeyToUint64(key);
}
inline void KeyFromDimensionIndex(DimensionIndex di, std::string* result) {
  return strings::KeyFromUint64(di, result);
}
inline std::string DimensionIndexToKey(DimensionIndex di) {
  return strings::Uint64ToKey(di);
}

using NNResultsVector = std::vector<std::pair<DatapointIndex, float>>;

class NoValue {
 public:
  NoValue() {}
  explicit NoValue(int) {}
  explicit operator int() const { return 1; }
  explicit operator float() const { return 1; }

  bool operator==(const NoValue nv) const { return true; }
  bool operator!=(const NoValue nv) const { return false; }

  bool operator>(const NoValue nv) const { return false; }
  bool operator<(const NoValue nv) const { return false; }
  bool operator>=(const NoValue nv) const { return true; }
  bool operator<=(const NoValue nv) const { return true; }
};

template <typename T>
inline constexpr bool IsNoValue() {
  return IsSame<T, NoValue>();
}

namespace accum_type_impl {

template <typename T>
using AccumulatorTypeFor1 =
    conditional_t<IsFloatingType<T>(), decay_t<T>, int64_t>;

}

template <typename T, typename U = T, typename V = T>
using AccumulatorTypeFor =
    decltype(declval<accum_type_impl::AccumulatorTypeFor1<T>>() +
             declval<accum_type_impl::AccumulatorTypeFor1<U>>() +
             declval<accum_type_impl::AccumulatorTypeFor1<V>>());

static_assert(IsSame<AccumulatorTypeFor<int8_t>, int64_t>(), "");
static_assert(IsSame<AccumulatorTypeFor<uint8_t>, int64_t>(), "");
static_assert(IsSame<AccumulatorTypeFor<int16_t>, int64_t>(), "");
static_assert(IsSame<AccumulatorTypeFor<uint16_t>, int64_t>(), "");
static_assert(IsSame<AccumulatorTypeFor<int32_t>, int64_t>(), "");
static_assert(IsSame<AccumulatorTypeFor<uint32_t>, int64_t>(), "");
static_assert(IsSame<AccumulatorTypeFor<float>, float>(), "");
static_assert(IsSame<AccumulatorTypeFor<double>, double>(), "");
static_assert(IsSame<AccumulatorTypeFor<long double>, long double>(), "");

static_assert(IsSame<AccumulatorTypeFor<int8_t, int8_t>, int64_t>(), "");
static_assert(IsSame<AccumulatorTypeFor<int8_t, float>, float>(), "");
static_assert(IsSame<AccumulatorTypeFor<float, double>, double>(), "");

static_assert(IsSame<AccumulatorTypeFor<float&>, float>(), "");
static_assert(IsSame<AccumulatorTypeFor<const float&>, float>(), "");
static_assert(IsSame<AccumulatorTypeFor<volatile float&>, float>(), "");

template <typename T>
using FloatingTypeFor = conditional_t<IsDouble<T>(), double, float>;

enum Normalization : uint8_t {
  NONE = 0,
  UNITL2NORM = 1,
  STDGAUSSNORM = 2,
  UNITL1NORM = 3
};

inline const char* NormalizationString(Normalization normalization) {
  switch (normalization) {
    case NONE:
      return "NONE";
    case UNITL2NORM:
      return "UNITL2NORM";
    case STDGAUSSNORM:
      return "STDGAUSSNORM";
    case UNITL1NORM:
      return "UNITL1NORM";
    default:
      return "UNKNOWN";
  }
}

using TypeTag = InputOutputConfig::InMemoryTypes;

constexpr TypeTag kInvalidTypeTag = static_cast<TypeTag>(234);

template <typename T>
constexpr TypeTag TagForType();

template <>
inline constexpr TypeTag TagForType<NoValue>() {
  return InputOutputConfig::IN_MEMORY_DATA_TYPE_NOT_SPECIFIED;
}
template <>
inline constexpr TypeTag TagForType<int8_t>() {
  return InputOutputConfig::INT8;
}
template <>
inline constexpr TypeTag TagForType<uint8_t>() {
  return InputOutputConfig::UINT8;
}
template <>
inline constexpr TypeTag TagForType<int16_t>() {
  return InputOutputConfig::INT16;
}
template <>
inline constexpr TypeTag TagForType<uint16_t>() {
  return InputOutputConfig::UINT16;
}
template <>
inline constexpr TypeTag TagForType<int32_t>() {
  return InputOutputConfig::INT32;
}
template <>
inline constexpr TypeTag TagForType<uint32_t>() {
  return InputOutputConfig::UINT32;
}
template <>
inline constexpr TypeTag TagForType<int64_t>() {
  return InputOutputConfig::INT64;
}
template <>
inline constexpr TypeTag TagForType<uint64_t>() {
  return InputOutputConfig::UINT64;
}
template <>
inline constexpr TypeTag TagForType<float>() {
  return InputOutputConfig::FLOAT;
}
template <>
inline constexpr TypeTag TagForType<double>() {
  return InputOutputConfig::DOUBLE;
}

StatusOr<TypeTag> TypeTagFromName(absl::string_view type_name);

string_view TypeNameFromTag(TypeTag type_tag);

template <typename T>
string_view TypeName() {
  return TypeNameFromTag(TagForType<T>());
}

template <typename T>
struct IsTypeEnabled {
#ifdef SCANN_DISABLE_UNCOMMON_TYPES
  static constexpr bool value = IsSameAny<T, float, uint8_t>();
#else
  static constexpr bool value = true;
#endif
};

template <typename T>
enable_if_t<IsStatusType<T>(), T> ErrorOrCrash(string_view msg) {
  return InvalidArgumentError(msg);
}

template <typename T>
enable_if_t<!IsStatusType<T>(), T> ErrorOrCrash(string_view msg) {
  LOG(FATAL) << msg;
}

Status DisabledTypeError(TypeTag type_tag);

template <typename T>
T DisabledTagErrorOrCrash(uint8_t tag) {
  return ErrorOrCrash<T>(
      DisabledTypeError(static_cast<TypeTag>(tag)).error_message());
}

template <typename T>
T InvalidTagErrorOrCrash(uint8_t tag) {
  if (static_cast<TypeTag>(tag) == kInvalidTypeTag) {
    LOG(FATAL) << "\n\n\n"
               << "BUG_BUG_BUG: SCANN_CALL_FUNCTION_BY_TAG was invoked w/ "
                  "kInvalidTypeTag.\n"
               << "Your code has forgotten to initialize a TypeTag variable!"
               << "\n\n\n";

    return ErrorOrCrash<T>(
        "Invalid tag: kInvalidTag. This means that a "
        "SCANN_CALL_FUNCTION_BY_TAG "
        "macro was invoked with an uninitialized TypeTag variable. This is, by "
        "definition, a code bug. Please report it so that it can be fixed.");
  }
  return ErrorOrCrash<T>(
      absl::StrCat("Invalid tag: ", static_cast<uint32_t>(tag)));
}

template <typename T>
T NonFpTagErrorOrCrash(uint8_t tag) {
  const auto type_tag = static_cast<TypeTag>(tag);
  return ErrorOrCrash<T>(absl::StrCat(TypeNameFromTag(type_tag),
                                      " isn't a valid fixed-point type"));
}

#ifndef SCANN_DISABLE_UNCOMMON_TYPES

#define SCANN_CALL_FUNCTION_BY_TAG(tag, function, ...)                 \
  [&] {                                                                \
    using ReturnT = decltype(function<float>(__VA_ARGS__));            \
    switch (tag) {                                                     \
      case ::research_scann::InputOutputConfig::INT8:                  \
        return function<int8_t>(__VA_ARGS__);                          \
      case ::research_scann::InputOutputConfig::UINT8:                 \
        return function<uint8_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::INT16:                 \
        return function<int16_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::UINT16:                \
        return function<uint16_t>(__VA_ARGS__);                        \
      case ::research_scann::InputOutputConfig::INT32:                 \
        return function<int32_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::UINT32:                \
        return function<uint32_t>(__VA_ARGS__);                        \
      case ::research_scann::InputOutputConfig::INT64:                 \
        return function<int64_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::UINT64:                \
        return function<uint64_t>(__VA_ARGS__);                        \
      case ::research_scann::InputOutputConfig::FLOAT:                 \
        return function<float>(__VA_ARGS__);                           \
      case ::research_scann::InputOutputConfig::DOUBLE:                \
        return function<double>(__VA_ARGS__);                          \
      default:                                                         \
        return ::research_scann::InvalidTagErrorOrCrash<ReturnT>(tag); \
    }                                                                  \
  }()

#define SCANN_CALL_FUNCTION_BY_TAG_NV(tag, function, ...)              \
  [&] {                                                                \
    using ReturnT = decltype(function<float>(__VA_ARGS__));            \
    switch (tag) {                                                     \
      case ::research_scann::InputOutputConfig::                       \
          IN_MEMORY_DATA_TYPE_NOT_SPECIFIED:                           \
        return function<NoValue>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::INT8:                  \
        return function<int8_t>(__VA_ARGS__);                          \
      case ::research_scann::InputOutputConfig::UINT8:                 \
        return function<uint8_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::INT16:                 \
        return function<int16_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::UINT16:                \
        return function<uint16_t>(__VA_ARGS__);                        \
      case ::research_scann::InputOutputConfig::INT32:                 \
        return function<int32_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::UINT32:                \
        return function<uint32_t>(__VA_ARGS__);                        \
      case ::research_scann::InputOutputConfig::INT64:                 \
        return function<int64_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::UINT64:                \
        return function<uint64_t>(__VA_ARGS__);                        \
      case ::research_scann::InputOutputConfig::FLOAT:                 \
        return function<float>(__VA_ARGS__);                           \
      case ::research_scann::InputOutputConfig::DOUBLE:                \
        return function<double>(__VA_ARGS__);                          \
      default:                                                         \
        return ::research_scann::InvalidTagErrorOrCrash<ReturnT>(tag); \
    }                                                                  \
  }()

#else

#define SCANN_CALL_FUNCTION_BY_TAG(tag, function, ...)                  \
  [&] {                                                                 \
    using ReturnT = decltype(function<float>(__VA_ARGS__));             \
    switch (tag) {                                                      \
      case ::research_scann::InputOutputConfig::FLOAT:                  \
        return function<float>(__VA_ARGS__);                            \
      case ::research_scann::InputOutputConfig::UINT8:                  \
        return function<uint8_t>(__VA_ARGS__);                          \
      case ::research_scann::InputOutputConfig::INT8:                   \
      case ::research_scann::InputOutputConfig::INT16:                  \
      case ::research_scann::InputOutputConfig::UINT16:                 \
      case ::research_scann::InputOutputConfig::INT32:                  \
      case ::research_scann::InputOutputConfig::UINT32:                 \
      case ::research_scann::InputOutputConfig::INT64:                  \
      case ::research_scann::InputOutputConfig::UINT64:                 \
      case ::research_scann::InputOutputConfig::DOUBLE:                 \
        return ::research_scann::DisabledTagErrorOrCrash<ReturnT>(tag); \
      default:                                                          \
        return ::research_scann::InvalidTagErrorOrCrash<ReturnT>(tag);  \
    }                                                                   \
  }()

#define SCANN_CALL_FUNCTION_BY_TAG_NV(tag, function, ...)               \
  [&] {                                                                 \
    using ReturnT = decltype(function<float>(__VA_ARGS__));             \
    switch (tag) {                                                      \
      case ::research_scann::InputOutputConfig::                        \
          IN_MEMORY_DATA_TYPE_NOT_SPECIFIED:                            \
        return function<NoValue>(__VA_ARGS__);                          \
      case ::research_scann::InputOutputConfig::FLOAT:                  \
        return function<float>(__VA_ARGS__);                            \
      case ::research_scann::InputOutputConfig::UINT8:                  \
        return function<uint8_t>(__VA_ARGS__);                          \
      case ::research_scann::InputOutputConfig::INT8:                   \
      case ::research_scann::InputOutputConfig::INT16:                  \
      case ::research_scann::InputOutputConfig::UINT16:                 \
      case ::research_scann::InputOutputConfig::INT32:                  \
      case ::research_scann::InputOutputConfig::UINT32:                 \
      case ::research_scann::InputOutputConfig::INT64:                  \
      case ::research_scann::InputOutputConfig::UINT64:                 \
      case ::research_scann::InputOutputConfig::DOUBLE:                 \
        return ::research_scann::DisabledTagErrorOrCrash<ReturnT>(tag); \
      default:                                                          \
        return ::research_scann::InvalidTagErrorOrCrash<ReturnT>(tag);  \
    }                                                                   \
  }()

#endif

#define SCANN_CALL_FUNCTION_BY_FPTAG(tag, function, ...)               \
  [&] {                                                                \
    using ReturnT = decltype(function<float>(__VA_ARGS__));            \
    switch (tag) {                                                     \
      case ::research_scann::InputOutputConfig::INT8:                  \
        return function<int8_t>(__VA_ARGS__);                          \
      case ::research_scann::InputOutputConfig::UINT8:                 \
        return function<uint8_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::INT16:                 \
        return function<int16_t>(__VA_ARGS__);                         \
      case ::research_scann::InputOutputConfig::UINT16:                \
        return function<uint16_t>(__VA_ARGS__);                        \
      case ::research_scann::InputOutputConfig::INT32:                 \
      case ::research_scann::InputOutputConfig::UINT32:                \
      case ::research_scann::InputOutputConfig::INT64:                 \
      case ::research_scann::InputOutputConfig::UINT64:                \
      case ::research_scann::InputOutputConfig::FLOAT:                 \
      case ::research_scann::InputOutputConfig::DOUBLE:                \
        return ::research_scann::NonFpTagErrorOrCrash<ReturnT>(tag);   \
      default:                                                         \
        return ::research_scann::InvalidTagErrorOrCrash<ReturnT>(tag); \
    }                                                                  \
  }()

template <typename T>
size_t SizeOf() {
  return sizeof(T);
}

inline size_t SizeFromTag(InputOutputConfig::InMemoryTypes tag) {
  return SCANN_CALL_FUNCTION_BY_TAG(tag, SizeOf);
}

#define SCANN_INSTANTIATE_TYPED_CLASS(EXTERN_KEYWORD, ClassName) \
  EXTERN_KEYWORD template class ClassName<int8_t>;               \
  EXTERN_KEYWORD template class ClassName<uint8_t>;              \
  EXTERN_KEYWORD template class ClassName<int16_t>;              \
  EXTERN_KEYWORD template class ClassName<uint16_t>;             \
  EXTERN_KEYWORD template class ClassName<int32_t>;              \
  EXTERN_KEYWORD template class ClassName<uint32_t>;             \
  EXTERN_KEYWORD template class ClassName<int64_t>;              \
  EXTERN_KEYWORD template class ClassName<uint64_t>;             \
  EXTERN_KEYWORD template class ClassName<float>;                \
  EXTERN_KEYWORD template class ClassName<double>;

#define SCANN_INSTANTIATE_TYPED_CLASS_NV(EXTERN_KEYWORD, ClassName) \
  EXTERN_KEYWORD template class ClassName<NoValue>;                 \
  SCANN_INSTANTIATE_TYPED_CLASS(EXTERN_KEYWORD, ClassName)

#define SCANN_INSTANTIATE_TYPED_CLASS_COMMON_ONLY(EXTERN_KEYWORD, ClassName) \
  SCANN_INSTANTIATE_TYPED_CLASS(EXTERN_KEYWORD, ClassName)

#ifdef SCANN_DISABLE_UNCOMMON_TYPES
#undef SCANN_INSTANTIATE_TYPED_CLASS_COMMON_ONLY
#define SCANN_INSTANTIATE_TYPED_CLASS_COMMON_ONLY(EXTERN_KEYWORD, ClassName) \
  EXTERN_KEYWORD template class ClassName<float>;                            \
  EXTERN_KEYWORD template class ClassName<uint8_t>;
#endif

#define SCANN_COMMA ,

inline uint8_t ScannAbs(uint8_t num) { return num; }
inline int8_t ScannAbs(int8_t num) { return std::abs(num); }
inline uint16_t ScannAbs(uint16_t num) { return num; }
inline int16_t ScannAbs(int16_t num) { return std::abs(num); }
inline uint32_t ScannAbs(uint32_t num) { return num; }
inline int32_t ScannAbs(int32_t num) { return std::abs(num); }
inline uint64_t ScannAbs(uint64_t num) { return num; }
inline int64_t ScannAbs(int64_t num) { return std::abs(num); }
inline float ScannAbs(float num) { return std::abs(num); }
inline double ScannAbs(double num) { return std::abs(num); }
inline NoValue ScannAbs(NoValue num) { return num; }

template <typename T, typename U>
unique_ptr<T> unique_cast_unsafe(unique_ptr<U> ptr) {
  return absl::WrapUnique(down_cast<T*>(ptr.release()));
}

template <typename T>
using StatusOrPtr = StatusOr<unique_ptr<T>>;

template <typename T>
shared_ptr<T> ToShared(unique_ptr<T> uptr) {
  return shared_ptr<T>(std::move(uptr));
}

template <typename T, typename U>
StatusOr<unique_ptr<T>> unique_cast(unique_ptr<U> ptr) {
  if (!dynamic_cast<T*>(ptr.get())) {
    return InvalidArgumentError("Failed to down cast pointer.");
  }
  return unique_cast_unsafe<T>(std::move(ptr));
}

template <typename T>
constexpr T MaxOrInfinity() {
  if constexpr (std::is_floating_point_v<T>) {
    return numeric_limits<T>::infinity();
  } else {
    return numeric_limits<T>::max();
  }
}

}  // namespace research_scann

#endif
