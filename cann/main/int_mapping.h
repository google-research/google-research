// Copyright 2025 The Google Research Authors.
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

#ifndef MAIN_INT_MAPPING_H_
#define MAIN_INT_MAPPING_H_

#include <vector>

#include "Eigen/Core"
#include "absl/types/span.h"

namespace visualmapping {

// IntMapping stores an immutable 1:N or N:M mapping between dense integer
// indices. It is equivalent to a std::vector<std::vector<int>>, but more
// memory-efficient. It can be serialized to an IntMappingProto.
class IntMapping {
 public:
  IntMapping() = default;

  // Initializes the IntMapping with the mapping from the nested vectors.
  explicit IntMapping(const std::vector<std::vector<int>>& data);

  // Initializes the IntMapping with the raw flattened data and range indices.
  // Consider this internal.
  explicit IntMapping(Eigen::VectorXi indices, Eigen::VectorXi data);

  // Explicitly copyable and movable.
  IntMapping(const IntMapping& other) = default;
  IntMapping& operator=(const IntMapping& other) = default;
  IntMapping(IntMapping&& other) = default;
  IntMapping& operator=(IntMapping&& other) = default;

  // Replaces the contents of the IntMapping with the mapping from the nested
  // vectors.
  void SetData(absl::Span<const std::vector<int>> data);

  // Returns the size of the mapping, i.e. the number of indices on the "from"
  // side or the number of outer vectors.
  int size() const;
  bool empty() const { return size() == 0; }

  // Returns the mapping for an index on the "from" side, i.e. the list of "to"
  // indices that the "from" index maps to.
  absl::Span<const int> operator[](int from_index) const {
    const int range_begin = indices_[from_index];
    const int range_end = indices_[from_index + 1];
    return absl::MakeConstSpan(data_.data() + range_begin,
                               range_end - range_begin);
  }

  // Returns the mapping for an index on the "from" side as a mutable span. The
  // caller can modify the "to" indices in the span, but cannot resize the
  // mapping. (Resizing would not be efficient.)
  absl::Span<int> operator[](int from_index) {
    const int range_begin = indices_[from_index];
    const int range_end = indices_[from_index + 1];
    return absl::MakeSpan(data_.data() + range_begin, range_end - range_begin);
  }

  // Iterator interface intentionally omitted since it's not needed so far.
  // Iteration of the entire mapping generally needs to know the current index,
  // and thus doesn't use range-based loops.

  // Builds and returns the inverse of this mapping. num_to_indices is the
  // number of indices on the "to" side, i.e. one more than the max of the
  // values in data_.
  IntMapping BuildInverse(int num_to_indices) const;

 private:
  // Contains the start indices of the inner vectors in below flattened array.
  // Vector has n + 1 elements where the last element contains data_.size().
  Eigen::VectorXi indices_;
  // Contains the flattened data of all vectors.
  Eigen::VectorXi data_;
};

// Builds and returns the inverse of a N:1 mapping given as a span of "to"
// indices. num_to_indices is the number of indices on the "to" side, i.e. one
// more than the max of the values in the span.
IntMapping BuildInverseIntMapping(absl::Span<const int> mapping,
                                  int num_to_indices);

}  // namespace visualmapping

#endif  // MAIN_INT_MAPPING_H_
