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

#ifndef MAIN_COLORED_C_NN_RANDOM_GRIDS_INDEX_H_
#define MAIN_COLORED_C_NN_RANDOM_GRIDS_INDEX_H_

#include <array>
#include <cstdint>
#include <vector>

#include "Eigen/Core"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "main/int_mapping.h"

namespace visualmapping {
namespace processing {

class ColoredCNNRandomGridsIndex {
 public:
  // Creates an empty index. Call Setup() to configure and build the index.
  ColoredCNNRandomGridsIndex() = default;

  const Eigen::MatrixXf& points() const { return points_; }

  absl::Status Setup(Eigen::MatrixXf points, Eigen::VectorXi colors,
                     int num_grids, float cell_size_factor,
                     float search_radius);

  // Queries arbitrary point and returns a set of nearest points in the
  // pre-specified radius.
  std::vector<int> Query(Eigen::Ref<const Eigen::VectorXf> query_point);

  int64_t GetMemorySizeInBytes() const;

 private:
  // Number of dimensions used to form the hash-int.
  static constexpr int kNumHashDimensions = 128;
  static constexpr int kNumBins = 8000;
  absl::Mutex mutex_;

  // Data for one random grid.
  struct Grid {
    // The random shift for this grid, with the cell size scaling pre-applied.
    Eigen::Matrix<float, kNumHashDimensions, 1> shift;
    // The random rotation and projection for this grid, applied before
    // shifting. Includes the cell-size scaling.
    Eigen::Matrix<float, kNumHashDimensions, Eigen::Dynamic> rotation;
    // Bins for the grid. Each entry maps the key as returned by GetCellKey to
    // the list of point indices in that bin.
    // IntMapping bins;
    IntMapping bins;
    absl::node_hash_map<std::array<int8_t, kNumHashDimensions>,
                        absl::flat_hash_set<int>>
        hashed_bins;

    // Calculates the hash int (== index into `bins`) for one point.
    int GetCellKey(Eigen::Ref<const Eigen::VectorXf> point) const;
    std::array<int8_t, 128> GetVecCellKey(
        Eigen::Ref<const Eigen::VectorXf> point) const;

    // Calculates the hash keys for multiple points in a batch.
    Eigen::VectorXi GetCellKeys(Eigen::Ref<const Eigen::MatrixXf> points) const;
  };

  // Builds the bins for one grid, picking a random subset of points for those
  // bins which have size larger than max_num_points_per_bin_.
  void BuildGrid(int seed, Grid* grid) const;

  // The indexed points. Used to calculate exact distances.
  Eigen::MatrixXf points_;
  // The random grids. Each grid includes the random shift and rotation, and a
  // map from cell id to the list of point indices.
  std::vector<Grid> grids_;
  // The square of the search radius.
  float radius_sq_ = 0.0;
  // Colors per each point.
  Eigen::VectorXi colors_;
  float cell_size_;
};

IntMapping BuildInverseIntMappingWithColors(absl::Span<const int> mapping,
                                            int num_to_indices,
                                            const Eigen::VectorXi& colors);

}  // namespace processing
}  // namespace visualmapping

#endif  // MAIN_COLORED_C_NN_RANDOM_GRIDS_INDEX_H_
