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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"

using SpVector = std::vector<std::pair<int, int>>;
using SpMatrix = std::unordered_map<int, SpVector>;

class Dataset {
 public:
  explicit Dataset(const std::string& filename);
  const SpMatrix& by_user() const { return by_user_; }
  const SpMatrix& by_item() const { return by_item_; }
  const int max_user() const { return max_user_; }
  const int max_item() const { return max_item_; }
  const int num_tuples() const { return num_tuples_; }

 private:
  SpMatrix by_user_;
  SpMatrix by_item_;
  int max_user_;
  int max_item_;
  int num_tuples_;
};

Dataset::Dataset(const std::string& filename) {
  max_user_ = -1;
  max_item_ = -1;
  num_tuples_ = 0;
  std::ifstream infile(filename);
  std::string line;

  // Discard header.
  assert(std::getline(infile, line));

  // Read the data.
  while (std::getline(infile, line)) {
    int pos = line.find(',');
    int user = std::atoi(line.substr(0, pos).c_str());
    int item = std::atoi(line.substr(pos + 1).c_str());
    by_user_[user].push_back({item, num_tuples_});
    by_item_[item].push_back({user, num_tuples_});
    max_user_ = std::max(max_user_, user);
    max_item_ = std::max(max_item_, item);
    ++num_tuples_;
  }
  std::cout << "max_user=" << max_user()
            << "\tmax_item=" << max_item()
            << "\tdistinct user=" << by_user_.size()
            << "\tdistinct item=" << by_item_.size()
            << "\tnum_tuples=" << num_tuples()
            << std::endl;
}

class Recommender {
 public:
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      MatrixXf;
  typedef Eigen::VectorXf VectorXf;

  virtual ~Recommender() {}

  virtual VectorXf Score(const int user_id, const SpVector& user_history) {
    return VectorXf::Zero(1);
  }

  virtual void Train(const Dataset& dataset) {}

  VectorXf EvaluateUser(const VectorXf& all_scores,
                        const SpVector& ground_truth,
                        const SpVector& exclude);

  // Templated implementation for evaluating a dataset. Requires a function that
  // scores all items for a given user or history.
  template <typename F>
      VectorXf EvaluateDatasetInternal(
      const Dataset& data, const SpMatrix& eval_by_user,
      F score_user_and_history) {
    std::mutex m;
    auto eval_by_user_iter = eval_by_user.begin();  // protected by m
    VectorXf metrics = VectorXf::Zero(3);

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&]{
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (eval_by_user_iter == eval_by_user.end()) {
            m.unlock();
            return;
          }
          int u = eval_by_user_iter->first;
          SpVector ground_truth = eval_by_user_iter->second;
          ++eval_by_user_iter;
          m.unlock();

          // Process the user.
          const SpVector& user_history = data.by_user().at(u);
          VectorXf scores = score_user_and_history(u, user_history);
          VectorXf this_metrics = this->EvaluateUser(scores, ground_truth,
                                                     user_history);

          // Update the metric.
          m.lock();
          metrics += this_metrics;
          m.unlock();
        }
      }));
    }

    // Join all threads.
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
    metrics /= eval_by_user.size();
    return metrics;
  }

  // Common implementation for evaluating a dataset. It uses the scoring
  // function of the class.
  virtual VectorXf EvaluateDataset(
      const Dataset& data, const SpMatrix& eval_by_user);
};

Recommender::VectorXf Recommender::EvaluateUser(
    const VectorXf& all_scores,
    const SpVector& ground_truth,
    const SpVector& exclude) {
  VectorXf scores = all_scores;
  for (int i = 0; i < exclude.size(); ++i) {
    assert(exclude[i].first < scores.size());
    scores[exclude[i].first] = std::numeric_limits<float>::lowest();
  }

  std::vector<size_t> topk(scores.size());
  std::iota(topk.begin(), topk.end(), 0);
  std::stable_sort(topk.begin(), topk.end(),
                   [&scores](size_t i1, size_t i2) {
                     return scores[i1] > scores[i2];
                   });
  auto recall = [](int k, const std::set<int>& gt_set,
                   const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0;
      }
    }
    return result / std::min<float>(k, gt_set.size());};

  auto ndcg = [](int k, const std::set<int>& gt_set,
                 const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0 / log2(i+2);
      }
    }
    double norm = 0.0;
    for (int i = 0; i < std::min<int>(k, gt_set.size()); ++i) {
      norm += 1.0 / log2(i+2);
    }
    return result / norm;};

  std::set<int> gt_set;
  std::transform(ground_truth.begin(), ground_truth.end(),
                 std::inserter(gt_set, gt_set.begin()),
                 [](const std::pair<int, int>& p) { return p.first; });
  VectorXf result(3);
  result << recall(20, gt_set, topk),
            recall(50, gt_set, topk),
            ndcg(100, gt_set, topk);
  return result;
}

Recommender::VectorXf Recommender::EvaluateDataset(
    const Dataset& data, const SpMatrix& eval_by_user) {
  return EvaluateDatasetInternal(
      data, eval_by_user,
      [&](const int user_id, const SpVector& history) -> VectorXf {
        return Score(user_id, history);
      });
}
