// Copyright 2024 The Google Research Authors.
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

#include "scann/utils/io_npy.h"

#include <cstdint>
#include <string>
#include <type_traits>

namespace research_scann {

template <typename T>
struct assert_false : std::false_type {};

template <typename T>
std::string numpy_type_name() {
  static_assert(assert_false<T>::value, "Type incompatible with numpy");
}
template <>
std::string numpy_type_name<uint8_t>() {
  return "'<u1'";
}
template <>
std::string numpy_type_name<uint16_t>() {
  return "'<u2'";
}
template <>
std::string numpy_type_name<uint32_t>() {
  return "'<u4'";
}
template <>
std::string numpy_type_name<uint64_t>() {
  return "'<u8'";
}
template <>
std::string numpy_type_name<int8_t>() {
  return "'<i1'";
}
template <>
std::string numpy_type_name<int16_t>() {
  return "'<i2'";
}
template <>
std::string numpy_type_name<int32_t>() {
  return "'<i4'";
}
template <>
std::string numpy_type_name<int64_t>() {
  return "'<i8'";
}
template <>
std::string numpy_type_name<float>() {
  return "'<f4'";
}
template <>
std::string numpy_type_name<double>() {
  return "'<f8'";
}

}  // namespace research_scann
