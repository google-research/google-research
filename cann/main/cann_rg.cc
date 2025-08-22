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

#include "main/cann_rg.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "file/file.h"
#include "main/colored_c_nn_random_grids_index.h"
#include "main/thread_pool.h"

namespace cann_rg {

std::string ScoredPair::ToString() const {
  return absl::StrCat(query, ", ", target, ", ", score);
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReadBinaryDescFile(const std::string &name, int dim, file::FileReader *reader) {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;
  auto data_or = reader->GetFileContents(name);
  if (!data_or.ok()) {
    std::cerr << "Error reading file " << name << ": " << data_or.status();
    return matrix;
  }
  int num_points =
      data_or->size() / dim / sizeof(typename Eigen::MatrixXf::Scalar);
  typename Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>::Index rows = num_points,
                                                 cols = dim;
  matrix.resize(rows, cols);
  memcpy(matrix.data(), data_or->data(), data_or->size());
  return matrix;
}

std::pair<std::vector<std::string>, std::vector<Eigen::MatrixXf>> ReadList(
    const std::vector<std::string> &file_list, int dim) {
  file::FileReader reader;
  std::pair<std::vector<std::string>, std::vector<Eigen::MatrixXf>> ret;
  for (const std::string &filename : file_list) {
    auto m = ReadBinaryDescFile(filename, dim, &reader);
    auto data = m.colwise().reverse().transpose();
    ret.first.emplace_back(filename);
    ret.second.push_back(std::move(data));
  }
  std::cout << "ReadList: read " << ret.first.size() << " images" << std::endl;
  return ret;
}

std::vector<ScoredPair> ComputeQueryScores(const DescriptorData &query_set,
                                           const DescriptorData &index_set,
                                           int num_features, int dim,
                                           double c_approx, int num_grids_init,
                                           double eps, double p0, double p1,
                                           double min_distance) {
  auto query_set_cpy = query_set;
  std::cout << "query images: " << query_set_cpy.first.size() << std::endl;
  std::cout << "index images: " << index_set.first.size() << std::endl;

  int num_all_points = 0;
  for (const auto &index : index_set.second) {
    const auto num_points = std::min<int>(index.cols(), num_features);
    num_all_points += num_points;
  }
  Eigen::MatrixXf points(dim, num_all_points);
  std::vector<std::string> image_per_points;
  int pos = 0;
  for (size_t i = 0; i < index_set.second.size(); i++) {
    const auto &index = index_set.second[i];
    const auto num_points = std::min<int>(index.cols(), num_features);
    for (int j = 0; j < num_points; j++) {
      image_per_points.push_back(index_set.first[i]);
    }
    points.block(0, pos, index.rows(), num_points) = index.leftCols(num_points);
    pos += num_points;
  }

  for (size_t i = 0; i < query_set_cpy.second.size(); i++) {
    auto &query = query_set_cpy.second[i];
    const auto num_points = std::min<int>(query.cols(), num_features);
    query.conservativeResize(dim, num_points);
  }

  std::cout << "points: " << points.cols() << " " << points.rows()
            << " images: " << image_per_points.size() << std::endl;

  absl::flat_hash_map<std::string, int> color_set;
  std::vector<std::string> index_to_string;
  for (size_t i = 0; i < index_set.first.size(); i++) {
    if (!color_set.contains(index_set.first[i])) {
      color_set[index_set.first[i]] = color_set.size();
      index_to_string.push_back(index_set.first[i]);
    }
  }
  Eigen::VectorXi colors(points.cols());
  for (int i = 0; i < points.cols(); i++) {
    colors(i) = color_set[image_per_points[i]];
  }
  const double w = c_approx / std::sqrt(dim);
  const double f =
      std::min(log(num_grids_init), static_cast<double>(dim) / c_approx);
  const int num_grids = std::exp(f);

  const double c = 1.0 / eps;
  const double a = p0;
  const double b = p1;
  const double rg_R = -(a * std::pow(1 - std::pow(c, b / (b - 1)), (1 / b))) /
                      (std::pow(c, b / (b - 1)) - 1);

  std::vector<float> radii;
  float current_r = std::min(min_distance, rg_R);
  while (current_r < rg_R) {
    radii.push_back(current_r);
    current_r *= 1.0 + (c_approx - 1.0) * 0.25;
  }
  std::cout << "p_0, p_1: " << p0 << "," << p1 << " eps: " << eps
            << " rg_R: " << rg_R << " grids: " << num_grids
            << " c: " << c_approx << " radiis: " << radii.size()
            << " min_distance: " << min_distance << std::endl;
  const int num_queries = query_set_cpy.first.size();
  std::vector<std::vector<std::pair<int, double>>> nn(num_queries);
  std::vector<std::vector<absl::flat_hash_set<int>>> feature_and_camera(
      num_queries);
  float all_index_time = 0;
  float all_query_time = 0;
  std::vector<absl::Mutex> nn_mutex(color_set.size());
  int radii_num = 0;
  for (auto rad : radii) {
    const double score_r =
        std::pow(1.0 - std::pow(rad / p0, p1 / (1 - p1)), (1 - p1) / p1);
    absl::Time t1 = absl::Now();
    visualmapping::processing::ColoredCNNRandomGridsIndex colored_c_nn_rg;
    CHECK_OK(colored_c_nn_rg.Setup(points, colors, num_grids, w, rad));
    const auto index_time = absl::ToDoubleSeconds(absl::Now() - t1);
    all_index_time += index_time;
    t1 = absl::Now();
    {
      ThreadPool pool(-1);
      for (int k = 0; k < num_queries; ++k) {
        pool.Schedule([k, query = query_set_cpy.second[k], radii_num,
                       &feature_and_camera, &nn, &color_set, &colored_c_nn_rg,
                       &nn_mutex, score_r] {
          if (radii_num == 0) {
            feature_and_camera[k].resize(query.cols());
            nn[k].resize(color_set.size());
            for (size_t i = 0; i < nn[k].size(); i++) {
              nn[k][i].first = i;
              nn[k][i].second = 0;
            }
          }
          for (int i = 0; i < query.cols(); i++) {
            const auto r = colored_c_nn_rg.Query(query.col(i));
            for (const auto &camera : r) {
              if (feature_and_camera[k][i].insert(camera).second) {
                absl::MutexLock lock(&nn_mutex[camera]);
                nn[k][camera].second += score_r;
              }
            }
          }
        });
      }
    }
    const float query_time = absl::ToDoubleSeconds(absl::Now() - t1);
    all_query_time += query_time;
    std::cout << "radii id: " << radii_num << "/" << radii.size() - 1
              << " index time: " << index_time << " sec ; radius: " << rad
              << " score: " << score_r
              << " image average query time: " << query_time / num_queries
              << std::endl;
    radii_num++;
  }
  std::cout << "num_queries: " << num_queries << std::endl;

  std::vector<ScoredPair> pairs;
  for (int k = 0; k < num_queries; ++k) {
    const auto query = query_set_cpy.second[k];
    const std::string query_string = query_set_cpy.first[k];
    absl::c_sort(nn[k], [](const auto &a, const auto &b) {
      return a.second > b.second;
    });
    for (int i = 0; i < std::min<int>(50, nn[k].size()); i++) {
      const auto color = nn[k][i].first;
      const auto count = nn[k][i].second;
      const auto image = index_to_string[color];
      double score =
          static_cast<double>(count) / static_cast<double>(query.cols());
      if (score == 0) score = 0.00000000001;
      pairs.push_back({.query = query_string, .target = image, .score = score});
    }
  }
  std::cout << " all index time: " << all_index_time
            << " all query time: " << all_query_time << " ("
            << all_query_time / query_set_cpy.first.size() << "/q)"
            << std::endl;
  return pairs;
}

std::vector<ScoredPair> MatchImages(
    const std::vector<std::string> &query_file_names,
    const std::vector<std::string> &mapping_file_names, int num_features,
    int dim, double c_approx, int num_grids_init, double eps, double p0,
    double p1, double min_distance) {
  std::cout << "reading features..." << std::endl;

  std::pair<std::vector<std::string>, std::vector<Eigen::MatrixXf>> query_set =
      cann_rg::ReadList(query_file_names, dim);
  std::pair<std::vector<std::string>, std::vector<Eigen::MatrixXf>> index_set =
      cann_rg::ReadList(mapping_file_names, dim);

  std::cout << "quering..." << std::endl;
  std::vector<ScoredPair> pairs = cann_rg::ComputeQueryScores(
      query_set, index_set, num_features, dim, c_approx, num_grids_init, eps,
      p0, p1, min_distance);
  return pairs;
}

}  // namespace cann_rg
