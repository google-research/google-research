// Copyright 2020 The Google Research Authors.
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

#include "fec_cache.h"

#include <limits>

#include "definitions.h"
#include "fec_cache.pb.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"

namespace automl_zero {

using ::absl::StrCat;
using ::std::pair;
using ::testing::Test;

constexpr IntegerT kNumTrainExamples = 10;
constexpr IntegerT kNumValidExamples = 5;
constexpr IntegerT kCacheSize = 10;

class FECCacheTest : public Test {
 protected:
  void InsertAndVerify(
      const size_t hash, const double fitness, const bool expect_new,
      FECCache* cache) {
    pair<double, bool> fitness_and_found = cache->Find(hash);
    if (expect_new) {
      EXPECT_FALSE(fitness_and_found.second);
      cache->InsertOrDie(hash, fitness);
    } else {
      EXPECT_TRUE(fitness_and_found.second);
      EXPECT_EQ(fitness_and_found.first, fitness);
    }
  }
};

TEST_F(FECCacheTest, InsertsCorrectly) {
  FECCache cache(ParseTextFormat<FECSpec>(StrCat(
      "num_train_examples: ", kNumTrainExamples, " "
      "num_valid_examples: ", kNumValidExamples, " "
      "cache_size: ", kCacheSize, " "
      "forget_every: ", 0, " "
      )));

  // New Algorithms.
  InsertAndVerify(1, 0.1, true, &cache);
  InsertAndVerify(2, 0.2, true, &cache);
  InsertAndVerify(3, 0.3, true, &cache);

  // Previously seen Algorithm.
  InsertAndVerify(2, 0.2, false, &cache);

  // New Algorithm.
  InsertAndVerify(4, 0.4, true, &cache);

  // Previously seen Algorithm.
  InsertAndVerify(3, 0.3, false, &cache);

  // Insert same repeatedly.
  InsertAndVerify(2, 0.2, false, &cache);
  InsertAndVerify(2, 0.2, false, &cache);
}

TEST_F(FECCacheTest, DiscardsByLRUPolicy) {
  FECCache cache(ParseTextFormat<FECSpec>(StrCat(
      "num_train_examples: ", kNumTrainExamples, " "
      "num_valid_examples: ", kNumValidExamples, " "
      "cache_size: ", 5, " "
      "forget_every: ", 0, " "
      )));

  // Insert just enough to fill cache.
  InsertAndVerify(5, 0.5, true, &cache);
  InsertAndVerify(2, 0.2, true, &cache);
  InsertAndVerify(4, 0.4, true, &cache);
  InsertAndVerify(1, 0.1, true, &cache);
  InsertAndVerify(3, 0.3, true, &cache);

  // Insert two more.
  InsertAndVerify(6, 0.6, true, &cache);
  InsertAndVerify(7, 0.7, true, &cache);

  // Check oldest two were discarded.
  EXPECT_TRUE(cache.Find(1).second);
  EXPECT_FALSE(cache.Find(2).second);
  EXPECT_TRUE(cache.Find(3).second);
  EXPECT_TRUE(cache.Find(4).second);
  EXPECT_FALSE(cache.Find(5).second);
  EXPECT_TRUE(cache.Find(6).second);
  EXPECT_TRUE(cache.Find(7).second);
}

TEST_F(FECCacheTest, LRUPolicyConsidersReinserts) {
  FECCache cache(ParseTextFormat<FECSpec>(StrCat(
      "num_train_examples: ", kNumTrainExamples, " "
      "num_valid_examples: ", kNumValidExamples, " "
      "cache_size: ", 5, " "
      "forget_every: ", 0, " "
      )));

  // Insert just enough to fill cache.
  InsertAndVerify(5, 0.5, true, &cache);
  InsertAndVerify(2, 0.2, true, &cache);
  InsertAndVerify(4, 0.4, true, &cache);
  InsertAndVerify(1, 0.1, true, &cache);
  InsertAndVerify(3, 0.3, true, &cache);

  // Reinsert a couple.
  InsertAndVerify(5, 0.5, false, &cache);
  InsertAndVerify(4, 0.4, false, &cache);

  // Insert two new ones.
  InsertAndVerify(6, 0.6, true, &cache);
  InsertAndVerify(7, 0.7, true, &cache);

  // Check oldest were discarded but the not reinserted ones.
  EXPECT_FALSE(cache.Find(1).second);
  EXPECT_FALSE(cache.Find(2).second);
  EXPECT_TRUE(cache.Find(3).second);
  EXPECT_TRUE(cache.Find(4).second);
  EXPECT_TRUE(cache.Find(5).second);
  EXPECT_TRUE(cache.Find(6).second);
  EXPECT_TRUE(cache.Find(7).second);
}

TEST_F(FECCacheTest, NumTrainExamplesWorks) {
  FECCache cache(ParseTextFormat<FECSpec>(StrCat(
      "num_train_examples: ", kNumTrainExamples, " "
      "num_valid_examples: ", kNumValidExamples, " "
      "cache_size: ", kCacheSize, " "
      "forget_every: ", 0, " "
      )));
  EXPECT_EQ(cache.NumTrainExamples(), kNumTrainExamples);
}

TEST_F(FECCacheTest, NumValidExamplesWorks) {
  FECCache cache(ParseTextFormat<FECSpec>(StrCat(
      "num_train_examples: ", kNumTrainExamples, " "
      "num_valid_examples: ", kNumValidExamples, " "
      "cache_size: ", kCacheSize, " "
      "forget_every: ", 0, " "
      )));
  EXPECT_EQ(cache.NumValidExamples(), kNumValidExamples);
}

}  // namespace automl_zero
