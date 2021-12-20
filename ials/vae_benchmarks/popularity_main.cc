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

#include "recommender.h"

class MostPopularRecommender : public Recommender {
 public:
  explicit MostPopularRecommender(const Dataset& data) {
    frequency_ = VectorXf::Zero(data.max_item() + 1);
    for (const auto& item_and_history : data.by_item()) {
      frequency_.coeffRef(item_and_history.first) =
          item_and_history.second.size();
    }
  }
  VectorXf Score(const int  user_id, const SpVector& user_history) override {
    return frequency_;
  }
 private:
  VectorXf frequency_;
};

int main(int argc, char* argv[]) {
  // Default flags.
  std::unordered_map<std::string, std::string> flags;
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
  recommender = new MostPopularRecommender(train);

  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);

  // Helper for evaluation.
  auto evaluate = [&](int epoch) {
    Recommender::VectorXf metrics =
        recommender->EvaluateDataset(test_tr, test_te.by_user());
    printf("Epoch %4d:\t Rec20=%.4f, Rec50=%.4f NDCG100=%.4f\n",
           epoch, metrics[0], metrics[1], metrics[2]);
  };

  evaluate(0);

  delete recommender;
  return 0;
}
