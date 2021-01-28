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

#ifndef NPY_ARRAY_NPY_ARRAY_NPY_ARRAY_H_
#define NPY_ARRAY_NPY_ARRAY_NPY_ARRAY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "array/array.h"

namespace npy_array {

// This header exposes a single function, `SerializeToNpyString`, whose behavior
// can be configured with `NpySerializeOptions`.

struct NpySerializeOptions {
  // # In Numpy, shapes are indexed outermost to innermost. #
  // Let npy_shape = (480, 640, 3), with axis names (i, j, k).
  // i is the outermost axis with extent 480.
  // k is the innermost axis with extent 3.
  //
  // In the default layout (compact, C-contiguous), i changes least frequently
  // and k changes most frequently. (This corresponds to an interleaved image,
  // where k names a color channel.)
  //
  // In the alternative layout (compact, F-contiguous aka fortran_order), this
  // is reversed: i changes *most* frequently and k changes *least* frequently.
  //
  // # In nda, shapes are indexed innermost to outermost. #
  // Let nda_shape = (1920, 1080, 3) with axis names (x, y, c).
  // x is the innermost axis with extent 1920.
  // c is the outermost axis with extent 3.
  //
  // In the default layout (compact), x changes most frequently and c changes
  // least frequently. (This corresponds to an planar image, where c names a
  // color channel.)

  // If `reverse_axes` is true, then the ordering of the axes in the serialized
  // NPY file will be reverse that of nda. I.e., if the nda::shape is {1920,
  // 1080, 3}, the NPY shape will be (3, 1080, 1920). This leads to natural
  // interpretations if, in nda, we index:
  // - Planar images as (x, y, c):
  //   This yields an npy indexing of (c, y, x) aka (k, i, j).
  // - Interleaved images as (c, x, y):
  //   This yields an npy indexing of (y, x, c) aka (i, j, c).
  //   - Unfortunately, this does *not* lead to a natural ordering if in nda, we
  //     index interleaved images as (x, y, c). To reorder (x, y, c) as (c, x,
  //     y), call nda::reorder<2, 0, 1>(src).
  //
  // If `reverse_axes` is false, then the ordering of axes is preserved.
  bool reverse_axes = true;
};

// Serializes `src` to a std::string in the NPY file format
// (https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).
//
// DataType may be any type specialized by NpyDataTypeString(). This includes
// all signed and unsigned fixed width types in C++ (8, 16, 32, and 64 bits), as
// we as float (32 bits), double (64 bits), and float16 (see NpyFloat16 below).
//
// Caveats:
// - Non-zero mins are not preserved since the format does not support it.
// - If `src.empty()`, returns an empty string since the NPY format does not
//   support empty arrays (the shape "()" refers to a scalar).
//
// Future work:
// - Structured data.
// - Version 3.0, which uses optional utf8 strings to name structured data axes.
template <typename DataType, typename ShapeType>
std::string SerializeToNpyString(
    nda::array_ref<DataType, ShapeType> src,
    const NpySerializeOptions& options = NpySerializeOptions());

// A stand-in type for the purposes of template specialization that allows the
// client to use any float16 library. To serialize a float16 buffer, first
// reinterpret_cast the client pointer to const NpyFloat16* and wrap it in an
// nda::array_ref<const NpyFloat16>.
//
// If your source buffer is already an nda::array or nda::array_ref, use
// nda:reinterpret<const NpyFloat16>(src), which reinterprets the pointer type
// and preserves the shape.
struct NpyFloat16 {
  uint16_t u16;
};
static_assert(sizeof(NpyFloat16) == 2, "sizeof(NpyFloat16) should be 2 bytes.");

namespace internal {
// Returns "<" if this machine is little-endian and ">" if this machine is
// big-endian.
std::string NpyEndiannessString();

// Returns the single-character type string for the given DataType. See:
// https://numpy.org/devdocs/reference/arrays.interface.html#arrays-interface.
//
// This function template is explicitly specialized only for supported types.
template <typename DataType>
std::string NpyDataTypeString();

// Returns the NPY "descr string" describing DataType (it is a string that can
// be passed to the constructor of np.dtype).
//
// TODO(jiawen): This can be explicitly specialized too.
template <typename DataType>
std::string NpyDescrString() {
  return absl::StrCat(NpyEndiannessString(), NpyDataTypeString<DataType>(),
                      sizeof(DataType));
}

// Converts array shape to a vector.
template <typename ShapeType>
std::vector<size_t> NpyShapeVector(ShapeType shape) {
  std::vector<size_t> dims;
  for (size_t i = 0; i < ShapeType::rank(); ++i) {
    dims.push_back(shape.dim(i).extent());
  }
  return dims;
}

// Returns the NPY shape string for the given shape's extents.
// - Rank 0 (scalar) --> "()".
// - Rank 1 (vector) --> "(len,)". Trailing comma is intentional.
// - Rank >= 2 --> "(rows,cols,...)".
std::string NpyShapeString(const std::vector<size_t>& shape);

// Encodes the length of `header` in NPY format (four bytes, little endian).
std::string NpyHeaderLengthString(absl::string_view header);

// Returns the "full" NPY file header for the given DataType and ShapeType.
// The "full header" consists of the magic 6 bytes, version number, and the
// "NPY header" that describes the data.
//
// This returns a header for version 2.0.
template <typename DataType, typename ShapeType>
std::string NpyFullHeaderString(ShapeType shape, bool reverse_axes) {
  constexpr absl::string_view kMagic("\x93NUMPY");

  // The explicit count is required since the \x00 in the literal would be
  // interpreted to construct a string of length 1.
  constexpr absl::string_view kVersion("\x02\x00", /*count=*/2);

  // The NPY format says that:
  // - If fortran_order = False (the default NPY ordering):
  //   Then the data is stored with the innermost axis changing most frequently.
  // - If fortran_order = True:
  //   Then the data is stored with the outermost axis changing most frequently.
  // Since `NpyDataString` always serializes data with the innermost *nda* axis
  // changing most frequently, if we *do* reverse the axes so that in npy, the
  // last axis is the innermost, then fortran_order is False.
  const bool fortran_order = !reverse_axes;

  std::vector<size_t> shape_vector = NpyShapeVector(shape);
  if (reverse_axes) {
    std::reverse(shape_vector.begin(), shape_vector.end());
  }

  const std::string header =
      absl::StrCat("{'descr': ", "'", NpyDescrString<DataType>(), "'", ", ",
                   "'fortran_order': ", fortran_order ? "True, " : "False, ",
                   "'shape': ", NpyShapeString(shape_vector), "}");

  return absl::StrCat(kMagic, kVersion, NpyHeaderLengthString(header), header);
}

// Returns a copy of `src` serialized compactly. Pedentically:
// - The output buffer has the exact size needed to represent all elements of
//   `src`, no more, no less.
// - The output buffer stores data with the innermost axis of `src` changing
//   most frequently.
// - In Numpy parlance, this means the data is C-contiguous: the innermost
//   axis (the first axis in nda convention, the last axis in numpy convention)
//   changes most frequently.
template <typename DataType, typename ShapeType>
std::string NpyDataString(nda::array_ref<const DataType, ShapeType> src) {
  // Allocate a buffer with exactly the amount of space needed to compactly
  // store `src`.
  const size_t dst_buffer_size_bytes = src.size() * sizeof(DataType);
  std::string dst_buffer(dst_buffer_size_bytes, '\0');

  // Wrap `dst_buffer` in an nda::array_ref with the same shape as `src` but
  // compact. nda::make_compact won't remove known-at-compile-time padding, only
  // dynamic padding. So we first promote it to a shape_of_rank<R>.
  DataType* dst_ptr = reinterpret_cast<DataType*>(dst_buffer.data());
  const nda::shape_of_rank<ShapeType::rank()> dynamic_shape = src.shape();
  auto dst_ref = nda::make_array_ref(dst_ptr, nda::make_compact(dynamic_shape));

  // Now just do a copy.
  nda::copy(src, dst_ref);

  return dst_buffer;
}

template <typename DataType, typename ShapeType>
std::string SerializeToNpyString(nda::array_ref<const DataType, ShapeType> src,
                                 const NpySerializeOptions& options) {
  // TODO(jiawen): Write a test for this case.
  if (src.empty()) {
    return "";
  }

  return absl::StrCat(internal::NpyFullHeaderString<DataType>(
                          src.shape(), options.reverse_axes),
                      internal::NpyDataString(src));
}

struct NpyHeader {
  // Array shape and total element count derived from it. total_element_count is
  // product of all sizes if it's a non-empty vector (representing an array of
  // rank > 0), or 1 if it's empty.
  std::vector<size_t> shape;
  size_t total_element_count = 0;

  // A single character describing interpretation of the type, e.g. 'f' for
  // float etc.
  char type_char = 'x';

  // Word size, with interpretation forming full used type. e.g. 'f' and 16
  // means half float, 32 full float etc.
  size_t word_size = 0;

  // See NpyFullHeaderString above.
  bool fortran_order = false;

  // Offset in the byte stream that describes where the actual array data
  // starts.
  size_t data_start_offset = 0;

  // If this is false, parsing of the header has failed.
  bool valid = false;
};

NpyHeader ReadHeader(absl::string_view src);

template <class Shape, size_t... Is>
Shape ToShapeImpl(const std::vector<size_t>& sizes,
                  std::index_sequence<Is...>) {
  return Shape({sizes[Is]...});
}

template <class Shape>
Shape ToShape(const std::vector<size_t>& sizes) {
  return ToShapeImpl<Shape>(sizes, std::make_index_sequence<Shape::rank()>());
}

}  // namespace internal

template <typename DataType, typename ShapeType>
std::string SerializeToNpyString(nda::array_ref<DataType, ShapeType> src,
                                 const NpySerializeOptions& options) {
  // Ensure that we only work on const (read-only) arrays.
  return internal::SerializeToNpyString(src.cref(), options);
}

template <typename DataType, typename ShapeType>
nda::array<DataType, ShapeType> DeserializeFromNpyString(
    absl::string_view src) {
  if (src.empty()) {
    LOG(ERROR) << "DeserializeFromNpyString: unable to deserialize, got an "
                  "empty string.";
    return nda::array<DataType, ShapeType>();
  }

  internal::NpyHeader header = internal::ReadHeader(src);
  if (!header.valid) {
    LOG(ERROR) << "DeserializeFromNpyString: unable to deserialize, got an "
                  "invalid npy header.";
    return nda::array<DataType, ShapeType>();
  }

  // After parsing the header, perform expected data verification.
  const size_t expected_data_size =
      header.total_element_count * header.word_size;
  if (header.data_start_offset + expected_data_size != src.size()) {
    LOG(ERROR) << "DeserializeFromNpyString: unable to deserialize, expected "
                  "string of length "
               << header.data_start_offset + expected_data_size << ", got "
               << src.size() << ".";
    return nda::array<DataType, ShapeType>();
  }
  if (internal::NpyDataTypeString<DataType>()[0] != header.type_char ||
      sizeof(DataType) != header.word_size) {
    LOG(ERROR) << "DeserializeFromNpyString: unable to deserialize, npy "
                  "contains data type "
               << header.type_char << header.word_size << ", while requested "
               << internal::NpyDataTypeString<DataType>()[0]
               << sizeof(DataType) << ".";
    return nda::array<DataType, ShapeType>();
  }

  if (ShapeType::rank() != header.shape.size()) {
    LOG(ERROR) << "DeserializeFromNpyString: unable to deserialize, got rank "
               << header.shape.size() << ", while requested "
               << ShapeType::rank() << ".";
    return nda::array<DataType, ShapeType>();
  }

  // See the rationale for flipping in NpySerializeOptions.
  if (!header.fortran_order) {
    std::reverse(header.shape.begin(), header.shape.end());
  }

  nda::array<DataType, ShapeType> array = nda::array<DataType, ShapeType>(
      internal::ToShape<ShapeType>(header.shape));
  src.copy(reinterpret_cast<char*>(array.data()), expected_data_size,
           header.data_start_offset);
  return array;
}

}  // namespace npy_array

#endif  // NPY_ARRAY_NPY_ARRAY_NPY_ARRAY_H_
