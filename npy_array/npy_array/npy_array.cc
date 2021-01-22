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

#include "npy_array/npy_array.h"

#include <cstddef>
#include <cstdint>
#include <regex>  // NOLINT: ok to use std::regex in third_party code.
#include <string>
#include <vector>

#include "glog/logging.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace npy_array::internal {

namespace {

// Returns true if this machine is little-endian.
// Returns false if this machine is big-endian.
bool IsLittleEndian() {
  const int32_t x = 1;
  const char* bytes = reinterpret_cast<const char*>(&x);
  const char first_byte = bytes[0];
  return (first_byte != 0);
}

// 32-bit endianness swap (big <--> little) with no dependencies.
uint32_t SwapEndian(uint32_t host_int) {
  return (((host_int & uint32_t{0xFF}) << 24) |
          ((host_int & uint32_t{0xFF00}) << 8) |
          ((host_int & uint32_t{0xFF0000}) >> 8) |
          ((host_int & uint32_t{0xFF000000}) >> 24));
}

// 16-bit endianness swap (big <--> little) with no dependencies.
uint16_t SwapEndian(uint16_t host_int) {
  return (host_int >> 8) | (host_int << 8);
}

}  // namespace

template <>
std::string NpyDataTypeString<bool>() {
  return "b";
}

template <>
std::string NpyDataTypeString<NpyFloat16>() {
  return "f";
}

template <>
std::string NpyDataTypeString<float>() {
  return "f";
}

template <>
std::string NpyDataTypeString<double>() {
  return "f";
}

template <>
std::string NpyDataTypeString<int8_t>() {
  return "i";
}

template <>
std::string NpyDataTypeString<int16_t>() {
  return "i";
}

template <>
std::string NpyDataTypeString<int32_t>() {
  return "i";
}

template <>
std::string NpyDataTypeString<int64_t>() {
  return "i";
}

template <>
std::string NpyDataTypeString<uint8_t>() {
  return "u";
}

template <>
std::string NpyDataTypeString<uint16_t>() {
  return "u";
}

template <>
std::string NpyDataTypeString<uint32_t>() {
  return "u";
}

template <>
std::string NpyDataTypeString<unsigned long long_t>() {
  return "u";
}

std::string NpyEndiannessString() {
  return IsLittleEndian() ? "<" : ">";
}

std::string NpyShapeString(const std::vector<size_t>& shape) {
  if (shape.empty()) {
    return "()";
  }
  return "(" + absl::StrJoin(shape.begin(), shape.end(), ",") + ",)";
}

std::string NpyHeaderLengthString(absl::string_view header) {
  std::string header_length_string(4, '\0');
  const uint32_t header_length = header.length();
  uint32_t* header_length_ptr =
      reinterpret_cast<uint32_t*>(header_length_string.data());
  // The NPY 2.0 format requires that the header length be specified as four
  // bytes and little endian.
  *header_length_ptr =
      IsLittleEndian() ? header_length : SwapEndian(header_length);
  return header_length_string;
}

NpyHeader ReadHeader(absl::string_view src) {
  NpyHeader header;
  constexpr absl::string_view kMagic("\x93NUMPY");

  // Check for size - at least magic + 2 bytes of version. We are going to
  // adjust this value as we go through header parsing.
  size_t min_header_size = kMagic.length() + 2;
  if (src.length() < min_header_size) {
    LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse header, "
                  "expected size of at least "
               << min_header_size << ", got " << src.length() << ".";
    return NpyHeader();
  }

  // Check for magic.
  if (!absl::StartsWith(src, kMagic)) {
    LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse header, "
                  "header magic doesn't match.";
    return NpyHeader();
  }

  // Version is two bytes, where we expect it to be 1, 2, or 3.
  const int version = src.at(kMagic.length());
  if (version < 1 || version > 3) {
    LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse header, "
                  "expected version 1, 2, or 3, got "
               << version << ".";
    return NpyHeader();
  }

  // Version 1 encodes length as 16bits, version 2 and 3 as 32bits.
  size_t size_offset = version == 1 ? 2 : 4;
  min_header_size += size_offset;
  if (src.length() < min_header_size) {
    LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse header, "
                  "expected size of at least "
               << min_header_size << ", got " << src.length() << ".";
    return NpyHeader();
  }

  size_t header_length = 0;
  if (version == 1) {
    uint16_t header_length_16 = *reinterpret_cast<const uint16_t*>(
        src.substr(kMagic.length() + 2).data());
    header_length =
        IsLittleEndian() ? header_length_16 : SwapEndian(header_length_16);
  } else {
    uint32_t header_length_32 = *reinterpret_cast<const uint32_t*>(
        src.substr(kMagic.length() + 2).data());
    header_length =
        IsLittleEndian() ? header_length_32 : SwapEndian(header_length_32);
  }

  min_header_size += header_length;
  if (src.length() < min_header_size) {
    LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse header, "
                  "expected size of at least "
               << min_header_size << ", got " << src.length() << ".";
    return NpyHeader();
  }

  header.data_start_offset = min_header_size;
  std::string header_substr =
      std::string(src.substr(kMagic.length() + 2 + size_offset, header_length));
  {
    // Find the "fortran order" (whether dimensions are from innermost or
    // outermost).
    std::regex fortran_order_re("('fortran_order': (False|True))");
    std::smatch match;
    if (!std::regex_search(header_substr, match,
                           fortran_order_re)) {
      LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse "
                    "header, couldn't find fortran_order.";
      return NpyHeader();
    }
    header.fortran_order = match[1] == "True";
  }

  {
    // Find the "descr" - data type.
    std::regex descr_re(R"('descr':\s*'(<|>)(\w)(\d+)')");
    std::smatch match;
    if (!std::regex_search(header_substr, match, descr_re)) {
        LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse "
                      "header, couldn't find type descr.";
        return NpyHeader();
      }
      const bool little_endian = match[1].str() == "<";

      // We don't support endianness swapping at the moment.
      if (little_endian != IsLittleEndian()) {
        LOG(ERROR) << "DeserializeFromNpyString ReadHeader invalid header, we "
                      "don't support endianness swapping at the moment.";
        return NpyHeader();
      }

      header.type_char = match[2].str()[0];
      if (!absl::SimpleAtoi(match[3].str(), &header.word_size)) {
        return NpyHeader();
      }
    }

  {
    // Find the "shape" (array dimensions).
    std::regex shape_re(R"('shape': \(((?:\d*\s*,*)*)\))");
    std::smatch match;
    if (!std::regex_search(header_substr, match, shape_re)) {
      LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse "
                    "header, couldn't find shape.";
      return NpyHeader();
    }
    std::string shape_desc = match[1];
    header.total_element_count = 1;
    bool last_element = false;
    for (absl::string_view dim_s : absl::StrSplit(shape_desc, ',')) {
      if (dim_s == "") {
        // We allow a trailing "," (empty last element), but if it wasn't really
        // last, it's going to be an error next loop iteration.
        last_element = true;
        continue;
      }
      size_t dim;
      if (last_element || !absl::SimpleAtoi(dim_s, &dim)) {
        LOG(ERROR) << "DeserializeFromNpyString ReadHeader unable to parse "
                      "header, couldn't find parse shape string "
                   << shape_desc << ".";
        return NpyHeader();
      }
      header.shape.push_back(dim);
      header.total_element_count *= dim;
    }
  }

  header.valid = true;
  return header;
}

}  // namespace npy_array::internal
