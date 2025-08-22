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

#ifndef SUE_RECOMMENDER_H_
#define SUE_RECOMMENDER_H_

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
#include <thread>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Core"
#include <Eigen/IterativeLinearSolvers>

#include "helper.h"
#include "recommender.h"
#include "trie.h"
// NOLINTEND

using SpFloatUniqueVector = std::unordered_map<int, float>;

// Implements "Algorithm 1: Solve Sparse User Encoder Model" from the paper
// "Efficient Optimization of Sparse User Encoder Recommenders"
template <typename HistoryToFeatures, typename HistoryAndItemToDelta,
          typename HistoryAndItemToDeltaPrime, typename ItemToPi>
Eigen::MatrixXf ComputeWeights(
    const Dataset& data, HistoryToFeatures history_to_features,
    HistoryAndItemToDelta history_and_item_to_delta,
    HistoryAndItemToDeltaPrime history_and_item_to_delta_prime,
    ItemToPi item_to_pi, const Eigen::VectorXf& regularization,
    const float frequency_reg, int feature_dim, int num_threads) {
  int num_items = data.max_item()+1;

  // Compute \Phi = \pi(H)
  std::cout << "Computing features...";
  Timer t_features;
  std::unordered_map<int, SpFloatVector> user_to_features;
  parallel_iterate_over_map(
      data.by_user(),
      [&](const int thread_index, const SpVector& history, const int user,
          std::mutex* m) {
        SpFloatUniqueVector features_unique = history_to_features(history);
        // Copy features into a std::vector to save memory for caching.
        SpFloatVector features(features_unique.size());
        size_t cntr = 0;
        for (const auto& feature_and_weight : features_unique) {
          features[cntr] = feature_and_weight;
          ++cntr;
        }
        assert(cntr == features_unique.size());
        m->lock();
        user_to_features[user] = features;
        m->unlock();
      }, num_threads);
  std::cout << " [done] in " << t_features.timeSinceStartAsString()
            << std::endl;

  // Calculate average length of \Phi as an additional metric:
  {
    size_t sum_of_lengths = 0;
    for (const auto& user_and_phi : user_to_features) {
      sum_of_lengths += user_and_phi.second.size();
    }
    std::cout << "number of nonzeros in Phi=" << sum_of_lengths
              << "\taverage size of phi="
              << ((float)sum_of_lengths / user_to_features.size())
              << std::endl;
  }

  // Add regularization proportional to the sum of feature squares.
  Eigen::VectorXf adjusted_regularization = regularization;
  if (frequency_reg != 0.0) {
    Eigen::VectorXd sum_of_feature_sqr(feature_dim);
    sum_of_feature_sqr.setZero();
    for (const auto& user_and_phi : user_to_features) {
      for (const auto& feature_and_weight : user_and_phi.second) {
        float weight = feature_and_weight.second;
        sum_of_feature_sqr[feature_and_weight.first] += weight * weight;
      }
    }
    adjusted_regularization += frequency_reg * sum_of_feature_sqr.cast<float>();
  }

  // Compute A = \Phi^t \Phi + \lambda I
  std::cout << "Computing Phi^t Phi...";
  Timer t_phitphi;
  Eigen::MatrixXf A(feature_dim, feature_dim);
  A.setZero();
  std::vector<std::mutex> lock_col_range(num_threads*4);
  parallel_iterate_over_map(
        data.by_user(),
        [&](const int thread_index, const SpVector& unused, const int user,
            std::mutex* m) {
    const SpFloatVector& features = user_to_features.at(user);
    for (const auto& feature_and_index : features) {
      int feature_i = feature_and_index.first;
      float weight_i = feature_and_index.second;

      lock_col_range.at(feature_i % lock_col_range.size()).lock();
      A(feature_i, feature_i) += weight_i * weight_i;
      for (const auto& feature_and_index_2 : features) {
        int feature_j = feature_and_index_2.first;
        float weight_j = feature_and_index_2.second;
        if (feature_i != feature_j) {
          A(feature_j, feature_i) += weight_i * weight_j;
        }
      }
      lock_col_range.at(feature_i % lock_col_range.size()).unlock();
    }
  }, num_threads);

  std::cout << " [done] in " << t_phitphi.timeSinceStartAsString() << std::endl;

  std::cout << "Cholesky+inverse of Phi^t Phi...";
  Timer t_cholesky;
  {
    A.diagonal().array() += adjusted_regularization.array();
    Eigen::LLT<Eigen::MatrixXf> A_llt(A);
    assert(A_llt.info() == Eigen::Success);
    // reuse A for the inverse:
    A.setZero();
    A.diagonal().array() = 1.0;
    A_llt.solveInPlace(A);
  }
  Eigen::MatrixXf& invA = A;
  std::cout << " [done] in " << t_cholesky.timeSinceStartAsString()
            << std::endl;

  std::cout << "Solving all items...";
  // The "solve \Psi" step from the paper is implemented using caching for
  // performance reasons. The multiplications with A^-1 are not done in
  // parallel but instead the Vs and bs for several items are collected in
  // parallel and then a batch of them is multiplied by A^-1. Then the results
  // of these multiplications are used to compute the solutions for the item
  // embeddings in parallel. The caching logic is implemented in "apply_cache".
  Eigen::MatrixXf result(num_items, feature_dim);
  result.setZero();
  Timer t_solving;
  // Cache for calculating a batch of embeddings.
  size_t cache_size = 1024 * 16;
  Eigen::MatrixXf cache(feature_dim, cache_size);
  size_t cache_next_location = 0;
  std::unordered_map<int, std::pair<int, int>> cache_item_to_begin_and_size;
  int total_time_emb = 0;
  int total_time_solutions = 0;
  auto apply_cache = [&](){
    Timer t_solutions;
    Eigen::MatrixXf solutions = invA * cache.leftCols(cache_next_location);
    total_time_solutions += t_solutions.timeSinceStartInMs();
    Timer t_emb;
    parallel_iterate_over_map(
        cache_item_to_begin_and_size,
        [&](const int thread_index, const std::pair<int, int>& begin_and_size,
            const int item, std::mutex* m) {
      int cache_begin = begin_and_size.first;
      int cache_size = begin_and_size.second;
      int cache_dim_i = (cache_size-1) / 2;
      assert(1+cache_dim_i*2 == cache_size);
      Eigen::VectorXf invA_b = solutions.col(cache_begin);
      Eigen::MatrixXf omega = solutions.middleCols(cache_begin + 1,
                                                   cache_dim_i * 2);
      Eigen::MatrixXf Vt = cache.middleCols(cache_begin + 1,
                                            cache_dim_i * 2).transpose();
      Eigen::VectorXf b = cache.col(cache_begin);

      Eigen::MatrixXf rot = Vt * omega;
      for (int i = 0; i < cache_dim_i; ++i) {
        rot(i, i) += 1.0;
        rot(i + cache_dim_i, i + cache_dim_i) -= 1.0;
      }
      Eigen::LDLT<Eigen::MatrixXf> small_llt(rot);
      assert(small_llt.info() == Eigen::Success);
      result.row(item).noalias() =
         invA_b - omega * small_llt.solve(omega.transpose() * b);
    }, num_threads);
    cache_next_location = 0;
    cache_item_to_begin_and_size.clear();
    total_time_emb += t_emb.timeSinceStartInMs();
  };

  // Main loop for the "solve \Psi" step.
  // V and b are collected in parallel and are cached. When the cache is full
  // they get solved with "apply_cache".
  parallel_iterate_over_map(
      data.by_item(),
      [&](const int thread_index, const SpVector& item_history, const int i,
          std::mutex* m) {
    Timer t_suffstats;
    const std::vector<SpFloatUniqueVector> pi_t = item_to_pi(i);
    const size_t dim_i = pi_t.size();   // dimension of subspace
    Eigen::MatrixXf Z(dim_i, feature_dim);
    Eigen::MatrixXf delta_delta(dim_i, dim_i);
    Z.setZero();
    delta_delta.setZero();
    for (const auto& user_and_weight : item_history) {
      const auto& phi = user_to_features[user_and_weight.first];
      const Eigen::VectorXf& delta_t = history_and_item_to_delta_prime(
          data.by_user().at(user_and_weight.first), i);
      // Compute Z += \phi \otimes delta_t
      for (const auto& feature_and_weight : phi) {
        float feature_weight = feature_and_weight.second;
        Z.col(feature_and_weight.first).noalias() += feature_weight * delta_t;
      }
      // delta_delta += delta_t \otimes delta_t
      delta_delta.noalias() += delta_t * delta_t.transpose();
    }

    Eigen::VectorXf b(feature_dim);
    b.setZero();
    const SpVector& users_with_nonzero_label = data.labels_by_item().at(i);

    // b = \sum_u (phi - delta)*y
    for (const auto& user_and_index : users_with_nonzero_label) {
      int user = user_and_index.first;
      const auto& phi = user_to_features[user];
      const float example_label = 1.0;  // note: uses a fixed label

      // b += phi * y
      for (const auto& feature_and_weight : phi) {
        float feature_weight = feature_and_weight.second;
        int feature_id = feature_and_weight.first;
        b(feature_id) += feature_weight * example_label;
      }

      // b += delta * y
      const SpVector& user_history = data.by_user().at(user);
      bool user_has_item_i = false;
      for (const auto& item_and_index : user_history) {
        if (item_and_index.first == i) {
          user_has_item_i = true;
          break;
        }
      }
      if (user_has_item_i) {
        const SpFloatUniqueVector& delta = history_and_item_to_delta(
            user_history, i);
        for (const auto& feature_and_weight : delta) {
          float delta_value = feature_and_weight.second;
          b(feature_and_weight.first) -= delta_value * example_label;
        }
      }
    }  // end for users with nonzero labels

    // Add a small constant to ensure that delta_delta is psd.
    if (dim_i > 1) {
      delta_delta += 0.01 * Eigen::MatrixXf::Identity(dim_i, dim_i);
    }
    Eigen::LLT<Eigen::MatrixXf> delta_delta_llt(delta_delta);
    assert(delta_delta_llt.info() == Eigen::Success);
    Eigen::MatrixXf V_top = delta_delta_llt.matrixL().solve(Z);
    Eigen::MatrixXf Vt(2 * dim_i, feature_dim);
    Vt.topRows(dim_i) = V_top;
    Vt.middleRows(dim_i, dim_i) = V_top;

    Eigen::MatrixXf lt = delta_delta_llt.matrixL().transpose();
    {
      int reduced_index = 0;
      for (const SpFloatUniqueVector& full_indices : pi_t) {
        for (const auto& index_and_weight : full_indices) {
          int full_index = index_and_weight.first;
          int pi_weight = index_and_weight.second;

          Vt.block(0, full_index, dim_i, 1).noalias() -=
              lt.col(reduced_index) * pi_weight;
        }
        ++reduced_index;
      }
    }
    Eigen::MatrixXf V = Vt.transpose().eval();

    // Write to cache
    m->lock();
    size_t size_of_entry = dim_i * 2 + 1;
    // if there is no space for this item, then apply all updates and clean the
    // cache:
    if (cache_next_location + size_of_entry >= cache_size) {
      apply_cache();
    }
    assert(cache_next_location + size_of_entry < cache_size);
    cache.col(cache_next_location) = b;
    cache.middleCols(cache_next_location + 1, dim_i * 2) = V;
    cache_item_to_begin_and_size.insert({i, {cache_next_location,
                                             size_of_entry}});
    cache_next_location += size_of_entry;
    m->unlock();
  }, num_threads);
  apply_cache();
  std::cout << " [done] in " << t_solving.timeSinceStartAsString()
            << std::endl;

  return result;
}

class SUERecommenderBase : public Recommender {
 public:
  SUERecommenderBase(int num_items, int num_features, float reg,
                     float frequency_regularization)
      : num_items_(num_items),
        num_features_(num_features),
        regularization_(reg),
        frequency_regularization_(frequency_regularization) {
    regularization_vec_ = Eigen::VectorXf::Constant(num_features, reg);
    std::cout
        << "num_items=" << num_items_ << std::endl
        << "num_features=" << num_features_ << std::endl
        << "reg=" << regularization_ << std::endl
        << "frequency_reg=" << frequency_regularization_ << std::endl;
  }

  VectorXf Score(const int user_id, const SpVector& user_history) override {
    Eigen::VectorXf results(weights_.rows());
    results.setZero();
    SpFloatUniqueVector phi = getPhi(user_history);
    for (const auto& item_and_index : phi) {
      results += weights_.col(item_and_index.first) * item_and_index.second;
    }
    return results;
  }

  virtual SpFloatUniqueVector getPhi(const SpVector& history) = 0;
  virtual SpFloatUniqueVector getDelta(const SpVector& history,
                                       const int item) = 0;
  virtual Eigen::VectorXf getDeltaPrime(const SpVector& history,
                                        const int item) = 0;
  virtual std::vector<SpFloatUniqueVector> getPi(const int item) = 0;

  void Train(const Dataset& data) override {
    weights_ = ComputeWeights(
        data,
        std::bind(&SUERecommenderBase::getPhi, this, std::placeholders::_1),
        std::bind(&SUERecommenderBase::getDelta, this, std::placeholders::_1,
                  std::placeholders::_2),
        std::bind(&SUERecommenderBase::getDeltaPrime, this,
                  std::placeholders::_1, std::placeholders::_2),
        std::bind(&SUERecommenderBase::getPi, this, std::placeholders::_1),
        this->regularization_vec_,
        this->frequency_regularization_,
        this->num_features_,
        this->num_threads_);
  }

 protected:
  Eigen::MatrixXf weights_;
  int num_items_;
  int num_features_;
  float regularization_;
  Eigen::VectorXf regularization_vec_;
  float frequency_regularization_;
};


// The model and learned parameters of this model are equivalent to EASE or
// equivalently to SLIM without constraints and L1 penalty=0.
class SUERecommenderIdentity : public SUERecommenderBase {
 public:
  SUERecommenderIdentity(int num_items, float reg, float frequency_reg)
      : SUERecommenderBase(num_items, num_items, reg, frequency_reg) {}

  SpFloatUniqueVector getPhi(const SpVector& history) {
    SpFloatUniqueVector result;
    for (const auto& item_and_index : history) {
      result[item_and_index.first] += 1.0;
    }
    return result;
  }

  SpFloatUniqueVector getDelta(const SpVector& history, const int item) {
    return {{item, 1.0}};
  }

  Eigen::VectorXf getDeltaPrime(const SpVector& history, const int item) {
    return Eigen::VectorXf::Constant(1, 1.0);
  }

  std::vector<SpFloatUniqueVector> getPi(const int item) {
    return std::vector<SpFloatUniqueVector>{SpFloatUniqueVector{{item, 1.0}}};
  }
};

class SUERecommenderFeatures : public SUERecommenderBase {
 private:
  int getMaxFeatureIndex(const SpMatrix& item_to_features) const {
    int result = 0;
    for (const auto& i_and_fs : item_to_features) {
      for (const auto& f_and_none : i_and_fs.second) {
        result = std::max(result, f_and_none.first);
      }
    }
    return result;
  }

 public:
  SUERecommenderFeatures(int num_items, float reg, float frequency_reg,
                         const SpMatrix& item_to_features)
      : SUERecommenderBase(
          num_items,
          getMaxFeatureIndex(item_to_features) + 1,
          reg,
          frequency_reg), item_to_features_(item_to_features) {}

  SpFloatUniqueVector getPhi(const SpVector& history) {
    SpFloatUniqueVector result;
    for (const auto& item_and_index : history) {
      for (const auto& features : item_to_features_[item_and_index.first]) {
        result[features.first] += 1.0;  // handles duplicates in
                                        // item_to_features
      }
    }
    return result;
  }

  SpFloatUniqueVector getDelta(const SpVector& history, const int item) {
    SpFloatUniqueVector result;
    for (const auto& features : item_to_features_[item]) {
      result[features.first] += 1.0;  // handles duplicates in item_to_features
    }
    return result;
  }

  Eigen::VectorXf getDeltaPrime(const SpVector& history, const int item) {
    // assumes that item is in history
    return Eigen::VectorXf::Constant(1, 1.0);
  }

  std::vector<SpFloatUniqueVector> getPi(const int item) {
    SpFloatUniqueVector result;
    for (const auto& features : item_to_features_[item]) {
      result[features.first] += 1.0;  // handles duplicates in
                                      // item_to_features
    }
    return {result};
  }

 private:
  SpMatrix item_to_features_;
};

class SUERecommenderHashing : public SUERecommenderFeatures {
 private:
  static SpMatrix CreateFeatures(int num_buckets, int num_hash_functions,
                                 int num_items) {
    SpMatrix result;
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<> d_int(0, num_buckets-1);
    for (int item = 0; item < num_items; ++item) {
      // Note: this can create duplicates. It is assumed that
      // SUERecommenderFeatures handles duplicates in the features correctly.
      for (int hash_function = 0; hash_function < num_hash_functions;
           ++hash_function) {
        result[item].push_back({d_int(gen), 1});
      }
    }
    return result;
  }

 public:
  SUERecommenderHashing(int num_items, int num_buckets, int num_hash_functions,
                        float reg, float frequency_reg)
      : SUERecommenderFeatures(num_items, reg, frequency_reg,
                               CreateFeatures(num_buckets, num_hash_functions,
                                              num_items)) {
    printf("num_buckets=%d\n", num_buckets);
    printf("num_hash_functions=%d\n", num_hash_functions);
  }
};

class SUERecommenderHO : public SUERecommenderBase {
 public:
  SUERecommenderHO(int num_items, const std::vector<int>& num_features_byorder,
                   const std::vector<float>& reg_byorder, float frequency_reg)
      : SUERecommenderBase(num_items,
                            std::reduce(num_features_byorder.begin(),
                                        num_features_byorder.end()),
                            /*reg=*/0.0, frequency_reg),
        num_features_byorder_(num_features_byorder),
        reg_byorder_(reg_byorder) {
    assert(reg_byorder_.size() == num_features_byorder_.size());
    for (size_t order = 0; order < reg_byorder_.size(); ++order) {
      std::cout
          << "order=" << (order+1) << "\t"
          << "num_features=" << num_features_byorder_.at(order) << "\t"
          << "regularization=" << reg_byorder_.at(order) << "\t"
          << std::endl;
    }
  }

  SpFloatUniqueVector getPhi(const SpVector& history) {
    SpFloatUniqueVector result;
    std::unordered_set<int> hist;
    for (const auto& item_and_weight : history) {
      hist.insert(item_and_weight.first);
    }
    const auto& trie_nodes = selected_item_sets_.root_.match(hist);
    for (const auto& node : trie_nodes) {
      result[node->data_] += 1;
    }
    return result;
  }

  SpFloatUniqueVector getDelta(const SpVector& history, const int item) {
    SpFloatUniqueVector result;
    auto& trie_nodes = item_to_feature_nodes_.at(item);
    std::unordered_set<int> hist_set;
    for (const auto& hi : history) {
      hist_set.insert(hi.first);
    }

    for (auto& node : trie_nodes) {
      // 1.) node is a match if all parents of node appear in history.
      // 2.) all descendents of node are matches if all parents of the
      //     descendent appear in the history.
      // We check first if all parents of node are in the history.
      if (!node->matchParents(hist_set)) {
        continue;
      }
      // Now we can just match all children of node.
      auto trie_nodes = node->match(hist_set);
      for (const auto& n : trie_nodes) {
        result[n->data_] += 1.0;
      }
    }
    return result;
  }

  std::unordered_map<int, int> getPiBijective(const int item) {
    std::unordered_map<int, int> result;
    const auto& trie_nodes = item_to_feature_nodes_.at(item);
    for (const auto& node : trie_nodes) {
      traverse_nodes(*node, {}, 0, [&result](
          const TrieNode& n, const std::vector<int>& prefix, int depth) {
        if (n.is_terminal_) {
          int cntr = result.size();
          result.try_emplace(n.data_, cntr);
        }
      });
    }
    return result;
  }

  std::vector<SpFloatUniqueVector> getPi(const int item) {
    const auto temp_pi = getPiBijective(item);

    std::vector<SpFloatUniqueVector> result(temp_pi.size());
    for (const auto& item_and_new_item : temp_pi) {
      result[item_and_new_item.second] =
          SpFloatUniqueVector{{item_and_new_item.first, 1.0}};
    }
    return result;
  }

  Eigen::VectorXf getDeltaPrime(const SpVector& history, const int item) {
    const SpFloatUniqueVector& delta = getDelta(history, item);
    const std::unordered_map<int, int> pi = getPiBijective(item);
    Eigen::VectorXf result(pi.size());
    result.setZero();
    for (const auto& index_and_weight : delta) {
      int old_index = index_and_weight.first;
      int new_index = pi.at(old_index);
      float delta_value = index_and_weight.second;
      result[new_index] += delta_value;
    }
    return result;
  }

  void InitHOFeatures(const Dataset& data) {
    std::cout << "Building XtX...";
    Timer t_xtx;
    Eigen::MatrixXf A(num_items_, num_items_);
    A.setZero();
    std::vector<std::mutex> lock_col_range(this->num_threads_*4);
    parallel_iterate_over_map(
        data.by_user(),
        [&](const int thread_index, const SpVector& history, const int user,
            std::mutex* m) {
      for (const auto& feature_and_index : history) {
        int feature_i = feature_and_index.first;
        float weight_i = 1.0;  // feature_index.second contains the index
        lock_col_range.at(feature_i % lock_col_range.size()).lock();
        A(feature_i, feature_i) += weight_i * weight_i;
        for (const auto& feature_and_index_2 : history) {
          int feature_j = feature_and_index_2.first;
          float weight_j = 1.0;   // see previous note
          if (feature_i != feature_j) {
            A(feature_j, feature_i) += weight_i * weight_j;
          }
        }
        lock_col_range.at(feature_i % lock_col_range.size()).unlock();
      }
    }, this->num_threads_);
    std::cout << "[done] in " << t_xtx.timeSinceStartAsString() << std::endl;

    int cntr = 0;
    std::cout << "Adding the most frequent items...";
    {
      Timer t_items;
      // Keep the highest scoring items
      size_t max_queue_length = num_features_byorder_.at(0);
      using WeightAndItem = std::pair<float, int>;
      std::priority_queue<WeightAndItem, std::vector<WeightAndItem>,
        std::greater<WeightAndItem>> best_items;
      if (max_queue_length > 0) {
        for (int item = 0; item < num_items_; ++item) {
          if ((best_items.size() < max_queue_length) ||
              (A.coeff(item, item) > best_items.top().first)) {
            best_items.push({A.coeff(item, item), {item}});
            while (best_items.size() > max_queue_length) {
              best_items.pop();
            }
          }
        }
      }

      int min_frequency = std::numeric_limits<int>::max();
      while (!best_items.empty()) {
        const auto& weight_and_item = best_items.top();
        const int item = weight_and_item.second;
        const float frequency = weight_and_item.first;
        min_frequency = std::min((int)frequency, min_frequency);

        TrieNode* n = selected_item_sets_.insert(std::vector<int>{item});
        n->data_ = cntr;
        regularization_vec_[cntr] = reg_byorder_.at(0);
        ++cntr;
        best_items.pop();
      }
      std::cout << "(min_freq=" << min_frequency << ") " << "[done] in "
                << t_items.timeSinceStartAsString() << std::endl;
    }

    Trie candidates;   // used for creating higher order interactions
    if (num_features_byorder_.size() >= 2) {
      std::cout << "Adding the most frequent pairs...";
      Timer t_bestpairs;
      // Keep the highest scoring pairs
      size_t max_queue_length = num_features_byorder_.at(1);
      using WeightAndIndexPair = std::pair<float, std::pair<int, int>>;
      std::priority_queue<WeightAndIndexPair, std::vector<WeightAndIndexPair>,
          std::greater<WeightAndIndexPair>> best_pairs;
      if (max_queue_length > 0) {
        for (int row = 0; row < num_items_; ++row) {
          for (int col = row+1; col < num_items_; ++col) {
            if ((best_pairs.size() < max_queue_length) ||
                (A.coeff(row, col) > best_pairs.top().first)) {
              best_pairs.push({A.coeff(row, col), {row, col}});
              while (best_pairs.size() > max_queue_length) {
                best_pairs.pop();
              }
            }
          }
        }
      }
      // Copy pairs into a trie.
      int min_frequency = std::numeric_limits<int>::max();
      while (!best_pairs.empty()) {
        const auto& weight_and_indexpair = best_pairs.top();
        const int item_a = weight_and_indexpair.second.first;
        const int item_b = weight_and_indexpair.second.second;
        assert(item_a != item_b);
        TrieNode* n = candidates.insert(
            std::unordered_set<int>{item_a, item_b});
        n->frequency_ = weight_and_indexpair.first;
        min_frequency = std::min(n->frequency_, min_frequency);

        TrieNode* n2 = selected_item_sets_.insert(
            std::unordered_set<int>{item_a, item_b});
        n2->data_ = cntr;
        regularization_vec_[cntr] = reg_byorder_.at(1);
        ++cntr;

        best_pairs.pop();
      }
      std::cout << "(min_freq=" << min_frequency << ") " << "[done] in "
                << t_bestpairs.timeSinceStartAsString() << std::endl;
    }

    // Create higher order candidates.
    for (size_t order = 2; order < num_features_byorder_.size(); ++order) {
      int candidate_size = order + 1;
      candidates = candidates.propose_candidates(candidate_size);
      if (candidates.root_.children_.empty()) {
        break;
      }
      // Count multithreaded.
      parallel_iterate_over_map(
        data.by_user(),
        [&](const int thread_index, const SpVector& history, const int user,
            std::mutex* m) {
          std::unordered_set<int> candidate_set;
          for (const auto& h : history) {
            candidate_set.insert(h.first);
          }
          const auto& matches = candidates.root_.match(candidate_set);
          m->lock();
          for (auto node : matches) {
            node->frequency_++;
          }
          m->unlock();
      }, this->num_threads_);
      int num_candidates = candidates.countTerminalNodes();
      std::cout << "Candidates of size " << candidate_size
                << ": " << num_candidates << std::endl;
      assert(num_candidates >= num_features_byorder_.at(order));

      // Keep the highest scoring nodes
      std::unordered_set<TrieNode*> best_nodes =
          find_n_most_frequent_terminal_nodes(&(candidates.root_),
                                              num_features_byorder_.at(order));
      candidates.root_.prune_nonmatching_nodes(best_nodes);
      num_candidates = candidates.countTerminalNodes();
      std::cout << "Frequent sets of size " << candidate_size
                << ": " << num_candidates << std::endl;
      assert(num_candidates == num_features_byorder_.at(order));

      // Add the frequent sets to the selected_items
      int min_frequency = std::numeric_limits<int>::max();
      for (TrieNode* node : best_nodes) {
        min_frequency = std::min(node->frequency_, min_frequency);

        std::unordered_set<int> set;
        TrieNode* temp = node;
        while (!temp->isRoot()) {
          set.insert(temp->key_);
          temp = temp->parent_;
        }
        TrieNode* n = selected_item_sets_.insert(set);
        n->data_ = cntr;
        regularization_vec_[cntr] = reg_byorder_.at(order);
        ++cntr;
      }
    }

    assert(cntr == num_features_);
    std::cout << "Building the mapping...";
    Timer t_mapping;
    // Build a mapping from items to nodes.
    item_to_feature_nodes_.resize(num_items_);
    traverse_nodes(&(selected_item_sets_.root_), {}, 0, [this](
        TrieNode* node, const std::vector<int>& prefix, int depth) {
      if (node->key_ >= 0) {
        item_to_feature_nodes_.at(node->key_).push_back(node);
      }
    });
    std::cout << "[done] in " << t_mapping.timeSinceStartAsString()
              << std::endl << "total number of features="
              << selected_item_sets_.countTerminalNodes() << std::endl;
  }

  void Train(const Dataset& data) override {
    InitHOFeatures(data);

    // Apply standard training
    SUERecommenderBase::Train(data);
  }

 private:
  Trie selected_item_sets_;
  std::vector<std::vector<TrieNode*>> item_to_feature_nodes_;
  std::vector<int> num_features_byorder_;
  std::vector<float> reg_byorder_;
};

#endif  // SUE_RECOMMENDER_H_
