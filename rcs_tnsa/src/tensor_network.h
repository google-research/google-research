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

#ifndef TENSOR_NETWORK_H
#define TENSOR_NETWORK_H

#include <array>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

class TensorNetwork {
 public:
  TensorNetwork() = default;

  TensorNetwork(const std::vector<std::vector<size_t>>& edges,
                const std::vector<double>& log2_weights);

  TensorNetwork(const std::string filename);

  TensorNetwork(const TensorNetwork& other) = default;

  size_t const GetNumEdges();

  size_t const GetNumNodes();

  std::vector<size_t> const& GetContractingEdgeNums();

  std::set<size_t> const& GetEnvironmentEdgeNums();

  std::vector<std::vector<std::vector<size_t>>> ContractionBreakdown(
      const std::vector<size_t>& ordering);

  double Log2Flops(
      const std::vector<std::vector<std::vector<size_t>>>& breakdown);

  double SlicedLog2FlopsGroupedSlicesSparseOutput(
      const std::vector<std::vector<std::vector<size_t>>>& breakdown,
      const std::vector<std::vector<size_t>>& sliced_grouped_edge_nums,
      size_t num_output_confs);

  double SlicedMemoizedLog2FlopsGroupedSlicesSparseOutput(
      const std::vector<std::vector<std::vector<size_t>>>& breakdown,
      const std::vector<std::vector<size_t>>& sliced_grouped_edge_nums,
      size_t num_output_confs);

  double Width(const std::vector<std::vector<std::vector<size_t>>>& breakdown);

  double SlicedWidthSparseOutput(
      const std::vector<std::vector<std::vector<size_t>>>& breakdown,
      const std::set<size_t>& sliced_edge_nums, size_t num_output_confs);

  double SlicedFullMemoryGroupedSlicesSparseOutput(
      const std::vector<std::vector<std::vector<size_t>>>& breakdown,
      const std::vector<std::vector<size_t>>& sliced_grouped_edge_nums,
      size_t num_output_confs);

  size_t NumSlicesGivenWidthGroupedSlicesSparseOutput(
      const std::vector<std::vector<std::vector<size_t>>>& breakdown,
      size_t max_width,
      const std::vector<std::vector<size_t>>& slices_groups_ordering,
      size_t num_output_confs);

  size_t NumSlicesGivenFullMemoryGroupedSlicesSparseOutput(
      const std::vector<std::vector<std::vector<size_t>>>& breakdown,
      double log2_max_memory,
      const std::vector<std::vector<size_t>>& slices_groups_ordering,
      size_t num_output_confs);

 private:
  std::vector<std::vector<size_t>> edges_;
  std::vector<size_t> contracting_edge_nums_;
  std::vector<double> log2_bds_;
  std::set<size_t> nodes_;
  size_t num_nodes_, num_contracting_nodes_;
  size_t num_edges_, num_contracting_edges_;
  std::vector<std::set<size_t>> node_to_edge_nums_;
  bool environment_;
  size_t environment_num_;
  std::set<size_t> environment_edge_nums_;

  void InitFromGraph(const std::string filename);
  void InitEncoding();
};

// External methods
std::vector<size_t> LoadOrdering(const std::string ordering_filename);

std::vector<size_t> LoadSlicedEdges(const std::string slicing_filename);

std::vector<size_t> GetSlicedEdges(const std::vector<size_t>& slicing_ordering,
                                   size_t num_sliced_edges);

template <typename T>
void SetUnion(const std::set<T>& s0, const std::set<T>& s1,
              std::set<T>& result);

template <typename T>
void SetIntersection(const std::set<T>& s0, const std::set<T>& s1,
                     std::set<T>& result);

template <typename T>
void SetDifference(const std::set<T>& s0, const std::set<T>& s1,
                   std::set<T>& result);

template <typename T>
void VectorUnion(const std::vector<T>& s0, const std::vector<T>& s1,
                 std::vector<T>& result);

#endif
