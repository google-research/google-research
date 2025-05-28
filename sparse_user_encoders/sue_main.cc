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

// NOLINTBEGIN
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <thread>

#include "helper.h"
#include "recommender.h"
#include "sue_recommender.h"
// NOLINTEND

int main(int argc, char* argv[]) {
  // Default flags.
  Flags flags(argc, argv);
  flags.setDefault("dropout_keep_prob", "1.0");
  flags.setDefault("dropout_num_cases", "0");
  flags.setDefault("encoder", "identity");
  flags.setDefault("filter_labels", "-1");
  flags.setDefault("filter_max_id", "-1");
  flags.setDefault("num_buckets", "1024");
  flags.setDefault("num_hash_functions", "1");
  flags.setDefault("num_features", "10000");
  flags.setDefault("num_threads",
                   std::to_string(std::thread::hardware_concurrency()));
  flags.setDefault("regularization", "500");
  flags.setDefault("frequency_regularization", "0");

  // Data related flags must exist.
  assert(flags.hasFlag("train_data"));
  assert(flags.hasFlag("test_train_data"));
  assert(flags.hasFlag("test_test_data"));

  // Load the datasets
  int filter_max_id = flags.getIntValue("filter_max_id");
  Dataset train(flags.getStrValue("train_data"), filter_max_id);
  if (flags.getFloatValue("dropout_num_cases") > 0) {
    // Creates a new version of the training set by duplicating users and
    // randomly dropping items from the histories.
    train = Dataset(
        train,
        flags.getIntValue("dropout_num_cases"),
        flags.getFloatValue("dropout_keep_prob"));
  }
  if ((flags.getIntValue("filter_labels") > 0) &&
      ((size_t)flags.getIntValue("filter_labels") < train.by_item().size())) {
    // Removes labels from the training dataset. Does not change the test sets.
    train.filterLabelsByFrequency(flags.getIntValue("filter_labels"));
  }
  Dataset test_tr(flags.getStrValue("test_train_data"), filter_max_id);
  Dataset test_te(flags.getStrValue("test_test_data"), filter_max_id);

  // Create the recommender.
  Recommender* recommender;
  if (flags.getStrValue("encoder") == "identity") {
    recommender = new SUERecommenderIdentity(
      train.max_item()+1,
      flags.getFloatValue("regularization"),
      flags.getFloatValue("frequency_regularization"));
  } else if (flags.getStrValue("encoder") == "features") {
    Dataset features(flags.getStrValue("features"));
    recommender = new SUERecommenderFeatures(
      train.max_item()+1,
      flags.getFloatValue("regularization"),
      flags.getFloatValue("frequency_regularization"),
      features.by_user());
  } else if (flags.getStrValue("encoder") == "hashing") {
    recommender = new SUERecommenderHashing(
      train.max_item()+1,
      flags.getIntValue("num_buckets"),
      flags.getIntValue("num_hash_functions"),
      flags.getFloatValue("regularization"),
      flags.getFloatValue("frequency_regularization"));
  } else if (flags.getStrValue("encoder") == "crosses") {
    recommender = new SUERecommenderHO(
      train.max_item()+1,
      flags.getIntValues("num_features"),
      flags.getFloatValues("regularization"),
      flags.getFloatValue("frequency_regularization"));
  } else {
    throw "unknown encoder";
  }
  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);

  // Training
  recommender->set_num_threads(flags.getIntValue("num_threads"));
  Timer t_train;
  recommender->Train(train);
  std::cout << "Total training time: " << t_train.timeSinceStartAsString()
            << std::endl;

  // Evaluation
  std::cout << "Evaluating...";
  Timer t_eval;
  Recommender::VectorXf metrics =
      recommender->EvaluateDataset(test_tr, test_te.by_user());
  std::cout << " [done] in " << t_eval.timeSinceStartAsString() << std::endl;
  printf("Rec20=%.4f, Rec50=%.4f NDCG100=%.4f\n",
         metrics[0], metrics[1], metrics[2]);

  delete recommender;
  return 0;
}
