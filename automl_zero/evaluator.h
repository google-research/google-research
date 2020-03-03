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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_EVALUATOR_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_EVALUATOR_H_

#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "algorithm.h"
#include "dataset.h"
#include "datasets.proto.h"
#include "definitions.h"
#include "experiment.proto.h"
#include "fec_cache.h"
#include "random_generator.h"
#include "train_budget.h"

namespace brain {
namespace evolution {
namespace amlz {

class Algorithm;

// See base class.
class Evaluator {
 public:
  Evaluator(
      const FitnessCombinationMode fitness_combination_mode,
      // Datasets to use. Will be filtered to only keep datasets targeted
      // to this worker.
      const DatasetCollection& dataset_collection,
      // The random generator seed to use for any random operations that
      // may be executed by the component_function (e.g. VectorRandomInit).
      RandomGenerator* rand_gen,
      // An cache to avoid reevaluating models that are functionally
      // identical. Can be nullptr.
      FECCache* functional_cache,
      // A train budget to use.
      TrainBudget* train_budget,
      // Errors larger than this trigger early stopping, as they signal
      // models that likely have runnaway behavior.
      double max_abs_error,
      // If false, suppresses all logging output. Finer grain control
      // available through logging flags.
      bool verbose);

  Evaluator(const Evaluator& other) = delete;
  Evaluator& operator=(const Evaluator& other) = delete;

  // Evaluates a Algorithm by executing it on the datasets. Returns the mean
  // fitness.
  double Evaluate(const Algorithm& algorithm);

 private:
  double Execute(const DatasetInterface& dataset, IntegerT num_train_examples,
                 const Algorithm& algorithm);

  template <FeatureIndexT F>
  double ExecuteImpl(const Dataset<F>& dataset, IntegerT num_train_examples,
                     const Algorithm& algorithm);

  double CapFitness(double fitness);

  const FitnessCombinationMode fitness_combination_mode_;

  // Contains only dataset specifications targeted to his worker.
  const DatasetCollection dataset_collection_;

  TrainBudget* train_budget_;
  RandomGenerator* rand_gen_;
  std::vector<std::unique_ptr<DatasetInterface>> datasets_;
  FECCache* functional_cache_;
  std::unique_ptr<std::mt19937> functional_cache_bit_gen_owned_;
  std::unique_ptr<RandomGenerator> functional_cache_rand_gen_owned_;
  RandomGenerator* functional_cache_rand_gen_;
  const std::vector<RandomSeedT> first_param_seeds_;
  const std::vector<RandomSeedT> first_data_seeds_;
  const bool verbose_;

  double best_fitness_;
  std::shared_ptr<Algorithm> best_algorithm_;

  const double max_abs_error_;
};

namespace internal {

double CombineFitnesses(
    const std::vector<double>& dataset_fitnesses,
    const FitnessCombinationMode mode);

}  // namespace internal

}  // namespace amlz
}  // namespace evolution
}  // namespace brain

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_EVALUATOR_H_
