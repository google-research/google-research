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

#include "regularized_evolution.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <utility>

#include "algorithm.h"
#include "algorithm.pb.h"
#include "task_util.h"
#include "definitions.h"
#include "executor.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace automl_zero {

namespace {

using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::absl::Seconds;  // NOLINT
using ::std::abs;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::fixed;  // NOLINT
using ::std::make_pair;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::setprecision;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT

//constexpr double kLn2 = 0.69314718056;

}  // namespace

RegularizedEvolution::RegularizedEvolution(
    RandomGenerator* rand_gen, const IntegerT population_size,
    const IntegerT tournament_size, const IntegerT progress_every,
    Generator* generator, Evaluator* evaluator, Mutator* mutator)
    : evaluator_(evaluator),
      rand_gen_(rand_gen),
      start_secs_(GetCurrentTimeNanos() / kNanosPerSecond),
      epoch_secs_(start_secs_),
      epoch_secs_last_progress_(epoch_secs_),
      num_individuals_last_progress_(std::numeric_limits<IntegerT>::min()),
      tournament_size_(tournament_size),
      progress_every_(progress_every),
      initialized_(false),
      generator_(generator),
      mutator_(mutator),
      population_size_(population_size),
      algorithms_(population_size_, make_shared<Algorithm>()),
      fitnesses_(population_size_),
      num_individuals_(0) {}

IntegerT RegularizedEvolution::Init() {
  // Otherwise, initialize the population from scratch.
  const IntegerT start_individuals = num_individuals_;
  std::vector<double>::iterator fitness_it = fitnesses_.begin();
  for (shared_ptr<const Algorithm>& algorithm : algorithms_) {
    InitAlgorithm(&algorithm);
    *fitness_it = Execute(algorithm);
    ++fitness_it;
  }
  CHECK(fitness_it == fitnesses_.end());

  MaybePrintProgress();
  initialized_ = true;
  return num_individuals_ - start_individuals;
}

IntegerT RegularizedEvolution::Run(const IntegerT max_train_steps,
                                   const IntegerT max_nanos) {
  CHECK(initialized_) << "RegularizedEvolution not initialized."
                      << std::endl;
  const IntegerT start_nanos = GetCurrentTimeNanos();
  const IntegerT start_train_steps = evaluator_->GetNumTrainStepsCompleted();
  while (evaluator_->GetNumTrainStepsCompleted() - start_train_steps <
             max_train_steps &&
         GetCurrentTimeNanos() - start_nanos < max_nanos) {
    vector<double>::iterator next_fitness_it = fitnesses_.begin();
    for (shared_ptr<const Algorithm>& next_algorithm : algorithms_) {
      SingleParentSelect(&next_algorithm);
      mutator_->Mutate(1, &next_algorithm);
      *next_fitness_it = Execute(next_algorithm);
      ++next_fitness_it;
    }
    MaybePrintProgress();
  }
  return evaluator_->GetNumTrainStepsCompleted() - start_train_steps;
}

IntegerT RegularizedEvolution::NumIndividuals() const {
  return num_individuals_;
}

IntegerT RegularizedEvolution::PopulationSize() const {
  return population_size_;
}

IntegerT RegularizedEvolution::NumTrainSteps() const {
  return evaluator_->GetNumTrainStepsCompleted();
}

shared_ptr<const Algorithm> RegularizedEvolution::Get(
    double* fitness) {
  const IntegerT indiv_index =
      rand_gen_->UniformPopulationSize(population_size_);
  CHECK(fitness != nullptr);
  *fitness = fitnesses_[indiv_index];
  return algorithms_[indiv_index];
}

shared_ptr<const Algorithm> RegularizedEvolution::GetBest(
    double* fitness) {
  double best_fitness = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || fitnesses_[index] > best_fitness) {
      best_index = index;
      best_fitness = fitnesses_[index];
    }
  }
  CHECK_NE(best_index, -1);
  *fitness = best_fitness;
  return algorithms_[best_index];
}

void RegularizedEvolution::PopulationStats(
    double* pop_mean, double* pop_stdev,
    shared_ptr<const Algorithm>* pop_best_algorithm,
    double* pop_best_fitness) const {
  double total = 0.0;
  double total_squares = 0.0;
  double best_fitness = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || fitnesses_[index] > best_fitness) {
      best_index = index;
      best_fitness = fitnesses_[index];
    }
    const double fitness_double = static_cast<double>(fitnesses_[index]);
    total += fitness_double;
    total_squares += fitness_double * fitness_double;
  }
  CHECK_NE(best_index, -1);
  double size = static_cast<double>(population_size_);
  const double pop_mean_double = total / size;
  *pop_mean = static_cast<double>(pop_mean_double);
  double var = total_squares / size - pop_mean_double * pop_mean_double;
  if (var < 0.0) var = 0.0;
  *pop_stdev = static_cast<double>(sqrt(var));
  *pop_best_algorithm = algorithms_[best_index];
  *pop_best_fitness = best_fitness;
}

void RegularizedEvolution::InitAlgorithm(
    shared_ptr<const Algorithm>* algorithm) {
  *algorithm = make_shared<Algorithm>(generator_->TheInitModel());
  // TODO(ereal): remove next line. Affects random number generation.
  mutator_->Mutate(0, algorithm);
}

double RegularizedEvolution::Execute(shared_ptr<const Algorithm> algorithm) {
  ++num_individuals_;
  epoch_secs_ = GetCurrentTimeNanos() / kNanosPerSecond;
  const double fitness = evaluator_->Evaluate(*algorithm);
  return fitness;
}

shared_ptr<const Algorithm>
    RegularizedEvolution::BestFitnessTournament() {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1;
  for (IntegerT tour_idx = 0; tour_idx < tournament_size_; ++tour_idx) {
    const IntegerT algorithm_index =
        rand_gen_->UniformPopulationSize(population_size_);
    const double curr_fitness = fitnesses_[algorithm_index];
    if (best_index == -1 || curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness;
      best_index = algorithm_index;
    }
  }
  return algorithms_[best_index];
}

void RegularizedEvolution::SingleParentSelect(
    shared_ptr<const Algorithm>* algorithm) {
  *algorithm = BestFitnessTournament();
}

void RegularizedEvolution::MaybePrintProgress() {
  if (num_individuals_ < num_individuals_last_progress_ + progress_every_) {
    return;
  }
  num_individuals_last_progress_ = num_individuals_;
  double pop_mean, pop_stdev, pop_best_fitness;
  shared_ptr<const Algorithm> pop_best_algorithm;
  PopulationStats(
      &pop_mean, &pop_stdev, &pop_best_algorithm, &pop_best_fitness);
  std::cout << "indivs=" << num_individuals_ << ", " << setprecision(0) << fixed
            << "elapsed_secs=" << epoch_secs_ - start_secs_ << ", "
            << "mean=" << setprecision(6) << fixed << pop_mean << ", "
            << "stdev=" << setprecision(6) << fixed << pop_stdev << ", "
            << "best fit=" << setprecision(6) << fixed << pop_best_fitness
            << "," << std::endl;
  std::cout.flush();
}

}  // namespace automl_zero
