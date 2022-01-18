// Copyright 2022 The Google Research Authors.
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

#include "recommender.h"

const Recommender::VectorXf Project(
    const SpVector& user_history,
    const Recommender::MatrixXf& item_embeddings,
    const Recommender::MatrixXf& gramian,
    const float reg, const float unobserved_weight) {
  assert(user_history.size() > 0);

  int embedding_dim = item_embeddings.cols();
  assert(embedding_dim > 0);

  Recommender::VectorXf new_value(embedding_dim);

  Eigen::MatrixXf matrix = unobserved_weight * gramian;

  for (int i = 0; i < embedding_dim; ++i) {
    matrix(i, i) += reg;
  }

  const int kMaxBatchSize = 128;
  auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
  Eigen::VectorXf rhs = Eigen::VectorXf::Zero(embedding_dim);
  const int batch_size = std::min(static_cast<int>(user_history.size()),
                                  kMaxBatchSize);
  int num_batched = 0;
  Eigen::MatrixXf factor_batch(embedding_dim, batch_size);
  for (const auto& item_and_rating_index : user_history) {
    const int cp = item_and_rating_index.first;
    assert(cp < item_embeddings.rows());
    const Recommender::VectorXf cp_v = item_embeddings.row(cp);

    factor_batch.col(num_batched) = cp_v;
    rhs += cp_v;

    ++num_batched;
    if (num_batched == batch_size) {
      matrix_symm.rankUpdate(factor_batch);
      num_batched = 0;
    }
  }
  if (num_batched != 0) {
    auto factor_block = factor_batch.block(0, 0, embedding_dim, num_batched);
    matrix_symm.rankUpdate(factor_block);
  }

  Eigen::LLT<Eigen::MatrixXf, Eigen::Lower> cholesky(matrix);
  assert(cholesky.info() == Eigen::Success);
  new_value = cholesky.solve(rhs);

  return new_value;
}

class IALSRecommender : public Recommender {
 public:
  IALSRecommender(int embedding_dim, int num_users, int num_items, float reg,
                  float reg_exp, float unobserved_weight, float stdev)
      : user_embedding_(num_users, embedding_dim),
        item_embedding_(num_items, embedding_dim) {
    // Initialize embedding matrices
    float adjusted_stdev = stdev / sqrt(embedding_dim);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d(0, adjusted_stdev);
    auto init_matrix = [&](Recommender::MatrixXf* matrix) {
      for (int i = 0; i < matrix->size(); ++i) {
        *(matrix->data() + i) = d(gen);
      }
    };
    init_matrix(&user_embedding_);
    init_matrix(&item_embedding_);

    regularization_ = reg;
    regularization_exp_ = reg_exp;
    embedding_dim_ = embedding_dim;
    unobserved_weight_ = unobserved_weight;
  }

  VectorXf Score(const int user_id, const SpVector& user_history) override {
    throw("Function 'Score' is not implemented");
  }

  // Custom implementation of EvaluateDataset that does the projection using the
  // iterative optimization algorithm.
  VectorXf EvaluateDataset(
      const Dataset& data, const SpMatrix& eval_by_user) override {
    std::unordered_map<int, VectorXf> user_to_emb;
    VectorXf prediction(data.num_tuples());

    // Initialize the user and predictions to 0.0. (Note: this code needs to
    // change if the embeddings would have biases).
    for (const auto& user_and_history : data.by_user()) {
      user_to_emb[user_and_history.first] = VectorXf::Zero(embedding_dim_);
      for (const auto& item_and_rating_index : user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) = 0.0;
      }
    }

    // Reproject the users.
    Step(data.by_user(),
         [&](const int user_id) -> VectorXf& {
           return user_to_emb[user_id];
         },
         item_embedding_, /*index_of_item_bias=*/1);

    // Evalute the dataset.
    return EvaluateDatasetInternal(
        data, eval_by_user,
        [&](const int user_id, const SpVector& history) -> VectorXf {
          return item_embedding_ * user_to_emb[user_id];
        });
  }

  void Train(const Dataset& data) override {
    Step(data.by_user(),
        [&](const int index) -> MatrixXf::RowXpr {
          return user_embedding_.row(index);
        },
        item_embedding_,
        /*index_of_item_bias=*/1);
    ComputeLosses(data);

    // Optimize the item embeddings
    Step(data.by_item(),
        [&](const int index) -> MatrixXf::RowXpr {
          return item_embedding_.row(index);
        },
        user_embedding_,
        /*index_of_item_bias=*/0);
    ComputeLosses(data);
  }

  void ComputeLosses(const Dataset& data) {
    if (!print_trainstats_) {
      return;
    }
    auto time_start = std::chrono::steady_clock::now();
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    for (const auto& user_and_history : data.by_user()) {
      VectorXf user_emb = user_embedding_.row(user_and_history.first);
      for (const auto& item_and_rating_index : user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) =
            item_embedding_.row(item_and_rating_index.first).dot(user_emb);
      }
    }
    int num_items = item_embedding_.rows();
    int num_users = user_embedding_.rows();

    // Compute observed loss.
    float loss_observed = (prediction.array() - 1.0).matrix().squaredNorm();

    // Compute regularizer.
    double loss_reg = 0.0;
    for (auto user_and_history : data.by_user()) {
      loss_reg += user_embedding_.row(user_and_history.first).squaredNorm() *
          RegularizationValue(user_and_history.second.size(), num_items);
    }
    for (auto item_and_history : data.by_item()) {
      loss_reg += item_embedding_.row(item_and_history.first).squaredNorm() *
          RegularizationValue(item_and_history.second.size(), num_users);
    }

    // Unobserved loss.
    MatrixXf user_gramian = user_embedding_.transpose() * user_embedding_;
    MatrixXf item_gramian = item_embedding_.transpose() * item_embedding_;
    float loss_unobserved = this->unobserved_weight_ * (
        user_gramian.array() * item_gramian.array()).sum();

    float loss = loss_observed + loss_unobserved + loss_reg;

    auto time_end = std::chrono::steady_clock::now();

    printf("Loss=%f, Loss_observed=%f Loss_unobserved=%f Loss_reg=%f Time=%d\n",
           loss, loss_observed, loss_unobserved, loss_reg,
           std::chrono::duration_cast<std::chrono::milliseconds>(
               time_end - time_start));
  }

  // Computes the regularization value for a user (or item). The value depends
  // on the number of observations for this user (or item) and the total number
  // of items (or users).
  const float RegularizationValue(int history_size, int num_choices) const {
    return this->regularization_ * pow(
              history_size + this->unobserved_weight_ * num_choices,
              this->regularization_exp_);
  }

  template <typename F>
  void Step(const SpMatrix& data_by_user,
            F get_user_embedding_ref,
            const MatrixXf& item_embedding,
            const int index_of_item_bias) {
    MatrixXf gramian = item_embedding.transpose() * item_embedding;

    // Used for per user regularization.
    int num_items = item_embedding.rows();

    std::mutex m;
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&]{
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (data_by_user_iter == data_by_user.end()) {
            m.unlock();
            return;
          }
          int u = data_by_user_iter->first;
          SpVector train_history = data_by_user_iter->second;
          ++data_by_user_iter;
          m.unlock();

          assert(!train_history.empty());
          float reg = RegularizationValue(train_history.size(), num_items);
          VectorXf new_user_emb = Project(
              train_history,
              item_embedding,
              gramian,
              reg, this->unobserved_weight_);
          // Update the user embedding.
          m.lock();
          get_user_embedding_ref(u) = new_user_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }

  const MatrixXf& item_embedding() const { return item_embedding_; }

  void SetPrintTrainStats(const bool print_trainstats) {
    print_trainstats_ = print_trainstats;
  }

 private:
  MatrixXf user_embedding_;
  MatrixXf item_embedding_;

  float regularization_;
  float regularization_exp_;
  int embedding_dim_;
  float unobserved_weight_;

  bool print_trainstats_;
};


int main(int argc, char* argv[]) {
  // Default flags.
  std::unordered_map<std::string, std::string> flags;
  flags["embedding_dim"] = "16";
  flags["unobserved_weight"] = "0.1";
  flags["regularization"] = "0.0001";
  flags["regularization_exp"] = "1.0";
  flags["stddev"] = "0.1";
  flags["print_train_stats"] = "0";
  flags["eval_during_training"] = "1";

  // Parse flags. This is a simple implementation to avoid external
  // dependencies.
  for (int i = 1; i < argc; ++i) {
    assert(i < (argc-1));
    std::string flag_name = argv[i];
    assert(flag_name.at(0) == '-');
    if (flag_name.at(1) == '-') {
      flag_name = flag_name.substr(2);
    } else {
      flag_name = flag_name.substr(1);
    }
    ++i;
    std::string flag_value = argv[i];
    flags[flag_name] = flag_value;
  }

  // Data related flags must exist.
  assert(flags.count("train_data") == 1);
  assert(flags.count("test_train_data") == 1);
  assert(flags.count("test_test_data") == 1);

  // Load the datasets
  Dataset train(flags.at("train_data"));
  Dataset test_tr(flags.at("test_train_data"));
  Dataset test_te(flags.at("test_test_data"));

  // Create the recommender.
  Recommender* recommender;
  recommender = new IALSRecommender(
    std::atoi(flags.at("embedding_dim").c_str()),
    train.max_user()+1,
    train.max_item()+1,
    std::atof(flags.at("regularization").c_str()),
    std::atof(flags.at("regularization_exp").c_str()),
    std::atof(flags.at("unobserved_weight").c_str()),
    std::atof(flags.at("stddev").c_str()));
  ((IALSRecommender*)recommender)->SetPrintTrainStats(
      std::atoi(flags.at("print_train_stats").c_str()));

  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);

  // Helper for evaluation.
  auto evaluate = [&](int epoch) {
    Recommender::VectorXf metrics =
        recommender->EvaluateDataset(test_tr, test_te.by_user());
    printf("Epoch %4d:\t Rec20=%.4f, Rec50=%.4f NDCG100=%.4f\n",
           epoch, metrics[0], metrics[1], metrics[2]);
  };

  bool eval_during_training =
      std::atoi(flags.at("eval_during_training").c_str());

  // Evaluate the model before training starts.
  if (eval_during_training) {
    evaluate(0);
  }

  // Train and evaluate.
  int num_epochs = std::atoi(flags.at("epochs").c_str());
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto time_train_start = std::chrono::steady_clock::now();
    recommender->Train(train);
    auto time_train_end = std::chrono::steady_clock::now();
    auto time_eval_start = std::chrono::steady_clock::now();
    if (eval_during_training) {
      evaluate(epoch + 1);
    }
    auto time_eval_end = std::chrono::steady_clock::now();
    printf("Timer: Train=%d\tEval=%d\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(
               time_train_end - time_train_start),
           std::chrono::duration_cast<std::chrono::milliseconds>(
               time_eval_end - time_eval_start));
  }
  if (!eval_during_training) {
    evaluate(num_epochs);
  }

  delete recommender;
  return 0;
}
