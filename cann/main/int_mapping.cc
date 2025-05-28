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

#include "main/int_mapping.h"

#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/types/span.h"

namespace visualmapping {

IntMapping::IntMapping(const std::vector<std::vector<int>>& data) {
  SetData(data);
}

IntMapping::IntMapping(Eigen::VectorXi indices, Eigen::VectorXi data)
    : indices_(std::move(indices)), data_(std::move(data)) {}

int IntMapping::size() const {
  if (indices_.size() == 0) {
    return 0;
  }
  return indices_.size() - 1;
}

void IntMapping::SetData(absl::Span<const std::vector<int>> data) {
  int total_num_elements = 0;
  for (const auto& inner : data) {
    total_num_elements += inner.size();
  }
  data_.resize(total_num_elements);
  indices_.resize(data.size() + 1);
  int offset = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    indices_[i] = offset;
    data_.segment(offset, data[i].size()) =
        Eigen::Map<const Eigen::VectorXi>(data[i].data(), data[i].size());
    offset += data[i].size();
  }
  indices_[data.size()] = total_num_elements;
}

namespace {

// Builds an "indices" vector for inverting the given mapping. This is the first
// pass of BuildInverse().
Eigen::VectorXi BuildInverseIndices(absl::Span<const int> mapping,
                                    int num_to_indices) {
  // Count the number of mapping elements per to_index. This is the same as the
  // "lengths" used in serialization. For efficiency, we collect them in the
  // vector that will turn into the "indices" vector of the new IntMapping.
  Eigen::VectorXi inverse_indices = Eigen::VectorXi::Zero(num_to_indices + 1);
  for (size_t i = 0; i < mapping.size(); ++i) {
    const int to_index = mapping[i];
    inverse_indices[to_index + 1]++;
  }

  // Convert lengths to start / end indices in the data vector.
  int range_index = 0;
  for (int i = 0; i < num_to_indices; ++i) {
    range_index += inverse_indices[i + 1];
    inverse_indices[i + 1] = range_index;
  }

  return inverse_indices;
}

}  // namespace

IntMapping IntMapping::BuildInverse(int num_to_indices) const {
  // First pass: Build the "indices" vector. The ranges of the source mapping
  // don't matter for this.
  Eigen::VectorXi inverse_indices = BuildInverseIndices(data_, num_to_indices);

  // Second pass: Build the "data" vector. For each to_index value we keep a
  // moving write_index that tells us where in the data vector the next element
  // for that to_index goes.
  Eigen::VectorXi inverse_data(data_.size());
  Eigen::VectorXi write_indices = inverse_indices.head(num_to_indices);
  for (int from_index = 0, data_index = 0; from_index < indices_.size() - 1;
       ++from_index) {
    const int data_end_index = indices_[from_index + 1];
    for (; data_index < data_end_index; ++data_index) {
      const int to_index = data_[data_index];
      inverse_data[write_indices[to_index]++] = from_index;
    }
  }

  return IntMapping(std::move(inverse_indices), std::move(inverse_data));
}

IntMapping BuildInverseIntMapping(absl::Span<const int> mapping,
                                  int num_to_indices) {
  // First pass: Build the "indices" vector.
  Eigen::VectorXi inverse_indices =
      BuildInverseIndices(mapping, num_to_indices);

  // Second pass: Build the "data" vector. For each to_index value we keep a
  // moving write_index that tells us where in the data vector the next element
  // for that to_index goes.
  Eigen::VectorXi inverse_data(mapping.size());
  Eigen::VectorXi write_indices = inverse_indices.head(num_to_indices);
  for (size_t i = 0; i < mapping.size(); ++i) {
    const int to_index = mapping[i];
    inverse_data[write_indices[to_index]++] = i;
  }

  return IntMapping(std::move(inverse_indices), std::move(inverse_data));
}

}  // namespace visualmapping
