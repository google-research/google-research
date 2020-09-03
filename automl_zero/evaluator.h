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

#ifndef AUTOML_ZERO_EVALUATOR_H_
#define AUTOML_ZERO_EVALUATOR_H_

#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "algorithm.h"
#include "task.h"
#include "task.pb.h"
#include "definitions.h"
#include "experiment.pb.h"
#include "fec_cache.h"
#include "random_generator.h"
#include "train_budget.h"

namespace automl_zero {

class Algorithm;

// See base class.
class Evaluator {
 public:
  Evaluator(
      const FitnessCombinationMode fitness_combination_mode,
      // Tasks to use. Will be filtered to only keep tasks targeted
      // to this worker.
      const TaskCollection& task_collection,
      // The random generator seed to use for any random operations that
      // may be executed by the component function (e.g. VectorRandomInit).
      RandomGenerator* rand_gen,
      // An cache to avoid reevaluating models that are functionally
      // identical. Can be nullptr.
      FECCache* functional_cache,
      // A train budget to use.
      TrainBudget* train_budget,
      // Errors larger than this trigger early stopping, as they signal
      // models that likely have runnaway behavior.
      double max_abs_error);
      // If false, suppresses all logging output. Finer grain control
      // available through logging flags.

  Evaluator(const Evaluator& other) = delete;
  Evaluator& operator=(const Evaluator& other) = delete;

  // Evaluates a Algorithm by executing it on the tasks. Returns the mean
  // fitness.
  double Evaluate(const Algorithm& algorithm);
  // Get the number of train steps this evaluator has performed.
  IntegerT GetNumTrainStepsCompleted() const;

 private:
  double Execute(const TaskInterface& task, IntegerT num_train_examples,
                 const Algorithm& algorithm);

  template <FeatureIndexT F>
  double ExecuteImpl(const Task<F>& task, IntegerT num_train_examples,
                     const Algorithm& algorithm);

  double CapFitness(double fitness);

  const FitnessCombinationMode fitness_combination_mode_;

  // Contains only task specifications targeted to his worker.
  const TaskCollection task_collection_;

  TrainBudget* train_budget_;
  RandomGenerator* rand_gen_;
  std::vector<std::unique_ptr<TaskInterface>> tasks_;
  FECCache* functional_cache_;
  std::unique_ptr<std::mt19937> functional_cache_bit_gen_owned_;
  std::unique_ptr<RandomGenerator> functional_cache_rand_gen_owned_;
  RandomGenerator* functional_cache_rand_gen_;
  const std::vector<RandomSeedT> first_param_seeds_;
  const std::vector<RandomSeedT> first_data_seeds_;

  double best_fitness_;
  std::shared_ptr<Algorithm> best_algorithm_;

  const double max_abs_error_;
  IntegerT num_train_steps_completed_;
};

namespace internal {

double CombineFitnesses(
    const std::vector<double>& task_fitnesses,
    const FitnessCombinationMode mode);

}  // namespace internal

}  // namespace automl_zero

#endif  // AUTOML_ZERO_EVALUATOR_H_
