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

#include "main/colored_c_nn_random_grids_index.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/QR"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "main/int_mapping.h"
#include "main/thread_pool.h"

namespace visualmapping {
namespace processing {

typedef Eigen::Matrix<int8_t, Eigen::Dynamic, 1> VectorXs;

namespace {

// Builds an "indices" vector for inverting the given mapping. This is the first
// pass of BuildInverse().
std::pair<Eigen::VectorXi, std::vector<int>> BuildInverseIndices(
    absl::Span<const int> mapping, int num_to_indices,
    const Eigen::VectorXi& colors) {
  // Count the number of mapping elements per to_index. This is the same as the
  // "lengths" used in serialization. For efficiency, we collect them in the
  // vector that will turn into the "indices" vector of the new IntMapping.
  Eigen::VectorXi inverse_indices = Eigen::VectorXi::Zero(num_to_indices + 1);
  absl::flat_hash_set<std::pair<int, int>> seen_color_per_outer_idx;
  std::vector<int> relevant_mappings;
  relevant_mappings.reserve(mapping.size());
  for (size_t i = 0; i < mapping.size(); ++i) {
    const int to_index = mapping[i];
    // Skip over colors we already added for an element in the outer vector.
    if (!seen_color_per_outer_idx.insert({to_index, colors(i)}).second) {
      continue;
    }
    relevant_mappings.push_back(i);
    inverse_indices[to_index + 1]++;
  }

  // Convert lengths to start / end indices in the data vector.
  int range_index = 0;
  for (int i = 0; i < num_to_indices; ++i) {
    range_index += inverse_indices[i + 1];
    inverse_indices[i + 1] = range_index;
  }

  return std::make_pair(inverse_indices, relevant_mappings);
}

// Creates a random, fixed sized orthonormal matrix, i.e. columns and rows have
// unit norm and are orthogonal to each other. The determinant of the returned
// matrix det(Q) = (-1)^(NumDims-1).
// Note, that we do not guarantee that multiple calls to this methods generate
// orthonormal matrices that are uniformly distributed. This would be ensured by
// performing a QR decomposition of normally distributed random entries and
// ensuring that diag(R) > 0 via post processing. We violate the latter (but
// preserve the property on det(Q)). See, e.g.
// https://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization.
template <typename T, int NumDims>
Eigen::Matrix<T, NumDims, NumDims> RandomOrthonormalBasis(
    absl::BitGenRef gen, int num_dimensions = NumDims) {
  static_assert(std::is_floating_point<T>::value,
                "RandomOrthonormalBasis only supports floating point types.");
  using MatrixT = Eigen::Matrix<T, NumDims, NumDims>;
  const MatrixT mat = MatrixT::NullaryExpr(
      num_dimensions, num_dimensions, [&]() { return absl::Gaussian<T>(gen); });
  return Eigen::HouseholderQR<MatrixT>(mat).householderQ();
}

// Creates a random, fixed sized rotation matrix. The column space of the matrix
// defines a right-handed coordinate frame, i.e. the determinant of the returned
// matrix is 1.
template <typename T, int NumDims>
Eigen::Matrix<T, NumDims, NumDims> RandomRotationMatrix(
    absl::BitGenRef gen, int num_dimensions = NumDims) {
  Eigen::Matrix<T, NumDims, NumDims> q =
      RandomOrthonormalBasis<T, NumDims>(gen, num_dimensions);
  // det(Q) = -1 for matrices with an odd number of dimensions when using
  // Housholder transformations for the QR decomposition. Therefore, we ensure
  // to return an orthonormal matrix with det(Q) = +1 by changing direction of
  // one basis vector.
  if (num_dimensions % 2 == 0) {
    q.col(0) = -q.col(0);
  }
  return q;
}
// Creates a random, dynamically sized rotation matrix with 'num_dimensions'.
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> RandomRotationMatrix(
    absl::BitGenRef gen, int num_dimensions) {
  return RandomRotationMatrix<T, Eigen::Dynamic>(gen, num_dimensions);
}

}  // namespace

IntMapping BuildInverseIntMappingWithColors(absl::Span<const int> mapping,
                                            int num_to_indices,
                                            const Eigen::VectorXi& colors) {
  // First pass: Build the "indices" vector.
  std::pair<Eigen::VectorXi, std::vector<int>> inverse_indices =
      BuildInverseIndices(mapping, num_to_indices, colors);

  // Second pass: Build the "data" vector. For each to_index value we keep a
  // moving write_index that tells us where in the data vector the next element
  // for that to_index goes.
  Eigen::VectorXi inverse_data(mapping.size());
  Eigen::VectorXi write_indices = inverse_indices.first.head(num_to_indices);
  for (int i : inverse_indices.second) {
    const int to_index = mapping[i];
    inverse_data[write_indices[to_index]++] = colors(i);
  }

  return IntMapping(std::move(inverse_indices.first), std::move(inverse_data));
}

int ColoredCNNRandomGridsIndex::Grid::GetCellKey(
    Eigen::Ref<const Eigen::VectorXf> point) const {
  const Eigen::Matrix<float, kNumHashDimensions, 1> projected_point =
      rotation * point + shift;
  std::array<char, kNumHashDimensions> key_vector;
  Eigen::Map<Eigen::Matrix<char, kNumHashDimensions, 1>>(key_vector.data()) =
      projected_point.template cast<int>().template cast<char>();
  return static_cast<int>(
      absl::Hash<decltype(key_vector.data())>{}(key_vector.data()) % kNumBins);
}

std::array<int8_t, 128> ColoredCNNRandomGridsIndex::Grid::GetVecCellKey(
    Eigen::Ref<const Eigen::VectorXf> point) const {
  const Eigen::Matrix<float, kNumHashDimensions, 1> projected_point =
      rotation * point + shift;
  VectorXs key = projected_point.cast<int8_t>();
  std::array<int8_t, 128> ret;
  std::copy_n(key.data(), ret.size(), ret.begin());
  return ret;
}

void ColoredCNNRandomGridsIndex::BuildGrid(int seed, Grid* grid) const {
  for (int i = 0; i < points_.cols(); ++i) {
    const auto key_v = grid->GetVecCellKey(points_.col(i));
    auto& t = grid->hashed_bins[key_v];
    t.insert(colors_(i));
  }
}

absl::Status ColoredCNNRandomGridsIndex::Setup(Eigen::MatrixXf points,
                                               Eigen::VectorXi colors,
                                               int num_grids,
                                               float cell_size_factor,
                                               float radius) {
  if (points.rows() < kNumHashDimensions) {
    // Note: This isn't an actual limitation and could be changed. However one
    // would then need to make parts of the rotation and hashing dynamic size
    // which reduces performance.
    return absl::UnimplementedError(
        "Random grids doesn't support lower dimensional points.");
  }
  colors_ = std::move(colors);
  points_ = std::move(points);
  radius_sq_ = radius * radius;
  cell_size_ = cell_size_factor * radius;

  // Create a set of random shifted grids and store any given point in each
  // one of the hashed cells.
  std::mt19937 gen(1962);
  grids_.clear();
  grids_.resize(num_grids);
  for (auto& grid : grids_) {
    // Build a rotation matrix for the full point dimension and then truncate
    // it to the number of dimensions needed to form the hash-key.
    const Eigen::MatrixXf rotation =
        RandomRotationMatrix<float>(gen, points_.rows());
    grid.rotation = rotation.topRows<kNumHashDimensions>() / cell_size_;
    grid.shift = Eigen::Matrix<float, kNumHashDimensions, 1>::NullaryExpr(
        [&]() { return absl::Uniform(gen, 0.0, 1.0); });
  }

  {
    ThreadPool pool(-1);
    for (size_t grid_index = 0; grid_index < grids_.size(); ++grid_index) {
      pool.Schedule([this, grid_index, &grid = grids_[grid_index]]() {
        BuildGrid(grid_index, &grid);
      });
    }
  }

  return absl::OkStatus();
}

std::vector<int> ColoredCNNRandomGridsIndex::Query(
    Eigen::Ref<const Eigen::VectorXf> query_point) {
  absl::flat_hash_set<int> colors;
  for (auto& grid : grids_) {
    const auto key_v = grid.GetVecCellKey(query_point);
    const auto it = grid.hashed_bins.find(key_v);
    if (it == grid.hashed_bins.end()) continue;
    const auto& bin = it->second;

    for (auto& c : bin) {
      colors.insert(c);
    }
  }
  return std::vector<int>(colors.begin(), colors.end());
}

}  // namespace processing
}  // namespace visualmapping
