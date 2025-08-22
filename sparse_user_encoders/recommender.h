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

#ifndef RECOMMENDER_H
#define RECOMMENDER_H

// NOLINTBEGIN
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdlib.h>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"
// NOLINTEND

using SpVector = std::vector<std::pair<int, int>>;
using SpMatrix = std::unordered_map<int, SpVector>;
using SpFloatVector = std::vector<std::pair<int, float>>;

class Dataset {
 public:
  Dataset(const std::string& filename, const int filter_max_id);
  Dataset(const Dataset& source_data, int dropout_cases,
          float dropout_keep_prob);
  const SpMatrix& by_user() const { return by_user_; }
  const SpMatrix& by_item() const { return by_item_; }
  const SpMatrix& labels_by_item() const {
    return labels_by_item_.empty() ? by_item_ : labels_by_item_;
  }
  const int max_user() const { return max_user_; }
  const int max_item() const { return max_item_; }
  const int num_tuples() const { return num_tuples_; }

  void filterLabelsByFrequency(int num_items_to_keep);

 private:
  SpMatrix by_user_;
  SpMatrix by_item_;
  SpMatrix labels_by_item_;
  int max_user_;
  int max_item_;
  int num_tuples_;
};

Dataset::Dataset(const std::string& filename, const int filter_max_id = -1) {
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
    int user = std::atoi(line.substr(0, pos).c_str());  // NOLINT
    int item = std::atoi(line.substr(pos + 1).c_str());  // NOLINT
    if ((filter_max_id >= 0) && (item >= filter_max_id)) {
      continue;
    }
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

Dataset::Dataset(const Dataset& source_data, int dropout_cases,
                 float dropout_keep_prob) {
  printf("dropout_keep_prob=%f\n", dropout_keep_prob);
  printf("dropout_cases=%d\n", dropout_cases);
  int new_user_idx = 0;
  std::bernoulli_distribution bernoulli(dropout_keep_prob);
  auto rng = std::mt19937{std::random_device{}()};
  for (const auto& user_and_history : source_data.by_user()) {
    // tuple index values have no meaning. do not use.
    for (int j = 0; j < dropout_cases; ++j) {
      SpVector new_history;
      for (const auto& item_and_index : user_and_history.second) {
        if (bernoulli(rng)) {
          new_history.push_back({item_and_index.first, 0});
        }
      }
      if (!new_history.empty()) {
        by_user_[new_user_idx] = new_history;
        for (const auto& item_and_index : user_and_history.second) {
          labels_by_item_[item_and_index.first].push_back({new_user_idx, 0});
        }
        ++new_user_idx;
      }
    }
  }

  max_user_ = -1;
  max_item_ = -1;
  num_tuples_ = 0;
  for (const auto& user_and_history : by_user_) {
    int user = user_and_history.first;
    max_user_ = std::max(max_user_, user);
    num_tuples_ += user_and_history.second.size();
    for (const auto& item_and_index : user_and_history.second) {
      int item = item_and_index.first;
      max_item_ = std::max(max_item_, item);
      by_item_[item].push_back({user, item_and_index.second});
    }
  }
  std::cout << "max_user=" << max_user()
            << "\tmax_item=" << max_item()
            << "\tdistinct user=" << by_user_.size()
            << "\tdistinct item=" << by_item_.size()
            << "\tnum_tuples=" << num_tuples()
            << std::endl;
}

void Dataset::filterLabelsByFrequency(int num_items_to_keep) {
  std::cout << "filtering labels, keeping " << num_items_to_keep
            << " items" << std::endl;
  if (labels_by_item_.empty()) {
    labels_by_item_ = by_item_;
  }

  // Keep the num_item_to_keep most frequent items.
  std::unordered_set<int> keep;
  {
    size_t max_queue_length = num_items_to_keep;
    using WeightAndItem = std::pair<int, int>;
    std::priority_queue<WeightAndItem, std::vector<WeightAndItem>,
        std::greater<WeightAndItem>> best_items;
    if (max_queue_length > 0) {
      for (const auto& item_and_history : labels_by_item_) {
        const int item = item_and_history.first;
        const int value = item_and_history.second.size();
        if ((best_items.size() < max_queue_length) ||
            (value > best_items.top().first)) {
          best_items.push({value, {item}});
          while (best_items.size() > max_queue_length) {
            best_items.pop();
          }
        }
      }
    }
    while (!best_items.empty()) {
      const auto& weight_and_item = best_items.top();
      const int item = weight_and_item.second;
      keep.insert(item);
      best_items.pop();
    }
  }

  // Delete all labels that are not in "keep".
  for (auto& item_and_labels : labels_by_item_) {
    if (keep.find(item_and_labels.first) == keep.end()) {
      item_and_labels.second.clear();
    }
  }
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
    VectorXf metrics = VectorXf::Zero(3);

    parallel_iterate_over_map(
        eval_by_user,
        [&](const int thread_index, const SpVector& ground_truth,
            const int user, std::mutex* m) {
      // Process the user.
      const SpVector& user_history = data.by_user().at(user);
      VectorXf scores = score_user_and_history(user, user_history);
      VectorXf this_metrics = this->EvaluateUser(scores, ground_truth,
                                                 user_history);

      // Update the metric.
      m->lock();
      metrics += this_metrics;
      m->unlock();
    }, num_threads_);

    metrics /= eval_by_user.size();
    return metrics;
  }

  // Common implementation for evaluating a dataset. It uses the scoring
  // function of the class.
  virtual VectorXf EvaluateDataset(
      const Dataset& data, const SpMatrix& eval_by_user);

  void set_num_threads(int num_threads) {
    num_threads_ = num_threads;
  }

 protected:
  int num_threads_;
};

Recommender::VectorXf Recommender::EvaluateUser(
    const VectorXf& all_scores,
    const SpVector& ground_truth,
    const SpVector& exclude) {
  VectorXf scores = all_scores;
  for (const auto& item_and_index : exclude) {
    assert(item_and_index.first < scores.size());
    scores[item_and_index.first] = std::numeric_limits<float>::lowest();
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
#endif  // RECOMMENDER_H
