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

#ifndef SCANN__UTILS_IO_NPY_H_
#define SCANN__UTILS_IO_NPY_H_

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/io_oss_wrapper.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
std::string numpy_type_name();

template <typename T>
Status SpanToNumpy(absl::string_view filename, ConstSpan<T> data,
                   ConstSpan<size_t> dim_sizes = {}) {
  std::string shape_str = "(";
  size_t dim_prod = 1;
  for (size_t dim_size : dim_sizes) {
    dim_prod *= dim_size;
    shape_str += std::to_string(dim_size) + ",";
  }
  if (dim_prod == 0 || data.size() % dim_prod != 0)
    return InvalidArgumentError(
        "Size of data isn't compatible with given shape");
  shape_str += std::to_string(data.size() / dim_prod) + ",)";

  if (shape_str.size() > 65000)
    return InvalidArgumentError("Shape string is too large for npy format: " +
                                shape_str);

  std::string pt1("\x93NUMPY\x01\x00  ", 10);
  std::string dict =
      absl::StrFormat("{'descr':%s, 'fortran_order':False, 'shape':%s}",
                      numpy_type_name<T>(), shape_str);
  while ((pt1.size() + dict.size()) % 64 != 63) dict += " ";
  dict += "\n";
  pt1[8] = dict.size() % 256;
  pt1[9] = dict.size() / 256;
  const std::string header = pt1 + dict;

  OpenSourceableFileWriter writer(filename);
  SCANN_RETURN_IF_ERROR(
      writer.Write(ConstSpan<char>(header.data(), header.size())));

  const char* ptr = reinterpret_cast<const char*>(data.data());
  return writer.Write(ConstSpan<char>(ptr, data.size() * sizeof(T)));
}

template <typename T>
Status VectorToNumpy(absl::string_view filename, const vector<T>& data,
                     const vector<size_t>& dim_sizes = {}) {
  return SpanToNumpy(filename, ConstSpan<T>(data.data(), data.size()),
                     ConstSpan<size_t>(dim_sizes.data(), dim_sizes.size()));
}

template <typename T>
Status DatasetToNumpy(absl::string_view filename, const DenseDataset<T>& data) {
  return SpanToNumpy(filename, data.data(), {data.size()});
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
