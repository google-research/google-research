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

#include "evaluator.h"

#include <algorithm>
#include <iomanip>
#include <ios>
#include <limits>
#include <memory>
#include <string>

#include "dataset.h"
#include "dataset_util.h"
#include "datasets.proto.h"
#include "definitions.h"
#include "executor.h"
#include "random_generator.h"
#include "train_budget.h"
#include "google/protobuf/text_format.h"
#include "absl/algorithm/container.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace automl_zero {

using ::absl::c_linear_search;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::fixed;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::min;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::nth_element;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::setprecision;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using internal::CombineFitnesses;

constexpr IntegerT kMinNumTrainExamples = 10;
constexpr RandomSeedT kFunctionalCacheRandomSeed = 235732282;

Evaluator::Evaluator(const FitnessCombinationMode fitness_combination_mode,
                     const TaskCollection& task_collection,
                     RandomGenerator* rand_gen,
                     FECCache* functional_cache,
                     TrainBudget* train_budget,
                     const double max_abs_error)
    : fitness_combination_mode_(fitness_combination_mode),
      task_collection_(task_collection),
      train_budget_(train_budget),
      rand_gen_(rand_gen),
      functional_cache_(functional_cache),
      functional_cache_bit_gen_owned_(
          make_unique<mt19937>(kFunctionalCacheRandomSeed)),
      functional_cache_rand_gen_owned_(make_unique<RandomGenerator>(
          functional_cache_bit_gen_owned_.get())),
      functional_cache_rand_gen_(functional_cache_rand_gen_owned_.get()),
      best_fitness_(-1.0),
      max_abs_error_(max_abs_error) {
  FillDatasets(task_collection_, &datasets_);
  CHECK_GT(datasets_.size(), 0);
}

double Evaluator::Evaluate(const Algorithm& algorithm) {
  // Compute the mean fitness across all datasets.
  vector<double> dataset_fitnesses;
  dataset_fitnesses.reserve(datasets_.size());
  vector<double> debug_fitnesses;
  vector<IntegerT> debug_num_train_examples;
  vector<IntegerT> dataset_indexes;  // Datasets to use.
  // Use all the datasets.
  for (IntegerT i = 0; i < datasets_.size(); ++i) {
    dataset_indexes.push_back(i);
  }
  for (IntegerT dataset_index : dataset_indexes) {
    const unique_ptr<TaskInterface>& dataset = datasets_[dataset_index];
    CHECK_GE(dataset->MaxTrainExamples(), kMinNumTrainExamples);
    const IntegerT num_train_examples =
        train_budget_ == nullptr ?
        dataset->MaxTrainExamples() :
        train_budget_->TrainExamples(algorithm, dataset->MaxTrainExamples());
    double curr_fitness = -1.0;
    curr_fitness = Execute(*dataset, num_train_examples, algorithm);
    dataset_fitnesses.push_back(curr_fitness);
  }
  double combined_fitness =
      CombineFitnesses(dataset_fitnesses, fitness_combination_mode_);

  CHECK_GE(combined_fitness, kMinFitness);
  CHECK_LE(combined_fitness, kMaxFitness);

  return combined_fitness;
}

double Evaluator::Execute(const TaskInterface& dataset,
                          const IntegerT num_train_examples,
                          const Algorithm& algorithm) {
  switch (dataset.FeaturesSize()) {
    case 2: {
      const Dataset<2>& downcasted_dataset = *SafeDowncast<2>(&dataset);
      return ExecuteImpl<2>(downcasted_dataset, num_train_examples, algorithm);
    }
    case 4: {
      const Dataset<4>& downcasted_dataset = *SafeDowncast<4>(&dataset);
      return ExecuteImpl<4>(downcasted_dataset, num_train_examples, algorithm);
    }
    case 8: {
      const Dataset<8>& downcasted_dataset = *SafeDowncast<8>(&dataset);
      return ExecuteImpl<8>(downcasted_dataset, num_train_examples, algorithm);
    }
    case 16: {
      const Dataset<16>& downcasted_dataset = *SafeDowncast<16>(&dataset);
      return ExecuteImpl<16>(downcasted_dataset, num_train_examples, algorithm);
    }
    case 32: {
      const Dataset<32>& downcasted_dataset = *SafeDowncast<32>(&dataset);
      return ExecuteImpl<32>(downcasted_dataset, num_train_examples, algorithm);
    }
    default:
      LOG(FATAL) << "Unsupported features size." << endl;
  }
}

template <FeatureIndexT F>
double Evaluator::ExecuteImpl(const Dataset<F>& dataset,
                              const IntegerT num_train_examples,
                              const Algorithm& algorithm) {
  if (functional_cache_ != nullptr) {
    CHECK_LE(functional_cache_->NumTrainExamples(), dataset.MaxTrainExamples());
    CHECK_LE(functional_cache_->NumValidExamples(), dataset.ValidSteps());
    functional_cache_bit_gen_owned_->seed(kFunctionalCacheRandomSeed);
    Executor<F> functional_cache_executor(
        algorithm, dataset, functional_cache_->NumTrainExamples(),
        functional_cache_->NumValidExamples(), functional_cache_rand_gen_,
        max_abs_error_);
    vector<double> train_errors;
    vector<double> valid_errors;
    functional_cache_executor.Execute(&train_errors, &valid_errors);
    const size_t hash = functional_cache_->Hash(
        train_errors, valid_errors, dataset.index_, num_train_examples);
    pair<double, bool> fitness_and_found = functional_cache_->Find(hash);
    if (fitness_and_found.second) {
      // Cache hit.
      functional_cache_->UpdateOrDie(hash, fitness_and_found.first);
      return fitness_and_found.first;
    } else {
      // Cache miss.
      Executor<F> executor(algorithm, dataset, num_train_examples,
                           dataset.ValidSteps(), rand_gen_, max_abs_error_);
      double fitness = executor.Execute();
      functional_cache_->InsertOrDie(hash, fitness);
      return fitness;
    }
  } else {
    Executor<F> executor(
        algorithm, dataset, num_train_examples, dataset.ValidSteps(),
        rand_gen_, max_abs_error_);
    const double fitness = executor.Execute();
    return fitness;
  }
}

namespace internal {

double Median(vector<double> values) {  // Intentional copy.
  const size_t half_num_values = values.size() / 2;
  nth_element(values.begin(), values.begin() + half_num_values, values.end());
  return values[half_num_values];
}

double CombineFitnesses(
    const vector<double>& dataset_fitnesses,
    const FitnessCombinationMode mode) {
  if (mode == MEAN_FITNESS_COMBINATION) {
    double combined_fitness = 0.0;
    for (const double fitness : dataset_fitnesses) {
      combined_fitness += fitness;
    }
    combined_fitness /= static_cast<double>(dataset_fitnesses.size());
    return combined_fitness;
  } else if (mode == MEDIAN_FITNESS_COMBINATION) {
    return Median(dataset_fitnesses);
  } else {
    LOG(FATAL) << "Unsupported fitness combination." << endl;
  }
}

}  // namespace internal

}  // namespace automl_zero
