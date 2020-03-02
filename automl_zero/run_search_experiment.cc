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

// Runs the RegularizedEvolution algorithm locally.

#include <algorithm>
#include <iostream>
#include <memory>


#include "glog/logging.h"
#include "algorithm.h"
#include "dataset_util.h"
#include "datasets.proto.h"
#include "definitions.h"
#include "definitions.proto.h"
#include "evaluator.h"
#include "experiment.proto.h"
#include "experiment_util.h"
#include "fec_cache.h"
#include "generator.h"
#include "mutator.h"
#include "random_generator.h"
#include "regularized_evolution.h"
#include "train_budget.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/time/time.h"
#include "util/random/mt_random.h"
#include "util/scaffolding/util/flag_types.h"

typedef brain::evolution::amlz::IntegerT IntegerT;
typedef brain::evolution::amlz::RandomSeedT RandomSeedT;
typedef brain::evolution::amlz::InstructionIndexT InstructionIndexT;

ABSL_FLAG(std::string, experiment_spec, "",
          "A text-format ExperimentSpec proto. Required.");
ABSL_FLAG(
    IntegerT, init_model, 0,
    "Which model to use to initialize the population.");
ABSL_FLAG(
    IntegerT, init_mutations, 0,
    "How many mutations to apply to the initial model(s) before putting them "
    "into the population.");
ABSL_FLAG(
    InstructionIndexT,
    learn_size_init,
    9,
    "Number of instructions for seed Algorithms. With op-op or random seeds, "
    "this will be the exact number of instructions. With hand-designed models, "
    "the component_function will be padded with no-ops to this number. If this "
    "is too small, the minimum number necessary will be used.");
ABSL_FLAG(
    double, max_abs_error, 100.0,
    "Maximum absolute error. If a component_function produces errors larger "
    "than this after any example during in its training or validation, it "
    "will be assigned the minimum fitness if early stopping is used.");
ABSL_FLAG(
    IntegerT, max_individuals, 1000,
    "Total number of individuals in the experiment.");
ABSL_FLAG(std::string, dataset_collection, "",
          "The text format of a DatasetCollection proto.");
ABSL_FLAG(
    InstructionIndexT,
    mutate_learn_size_max,
    10,
    "Maximum size of learn component_functions that mutations will attempt to "
    "produce. If initialization is way off, mutations may produce results "
    "outside the specified range.");
ABSL_FLAG(
    InstructionIndexT,
    mutate_learn_size_min,
    9,
    "Minimum size of learn component_functions that mutations will attempt to "
    "produce. If initialization is way off, mutations may produce results "
    "outside the specified range.");
ABSL_FLAG(
    InstructionIndexT,
    mutate_predict_size_max,
    4,
    "Maximum size of predict component_functions that mutations will attempt "
    "to produce. If initialization is way off, mutations may produce results "
    "outside the specified range.");
ABSL_FLAG(
    InstructionIndexT,
    mutate_predict_size_min,
    3,
    "Minimum size of setup component_functions that mutations will attempt "
    "to produce. If initialization is way off, mutations may produce "
    "results outside the specified range.");
ABSL_FLAG(
    double, mutate_prob, 1.0,
    "Probability of mutation after each selection.");
ABSL_FLAG(
    InstructionIndexT,
    mutate_setup_size_max,
    7,
    "Maximum size of setup component_functions that mutations will attempt to "
    "produce. If initialization is way off, mutations may produce results "
    "outside the specified range.");
ABSL_FLAG(
    InstructionIndexT,
    mutate_setup_size_min,
    6,
    "Minimum size of setup component_functions that mutations will attempt "
    "to produce. If initialization is way off, mutations may produce results "
    "outside the specified range.");
ABSL_FLAG(
    scaffolding::FlagList<IntegerT>,
    mutation_actions,
    scaffolding::FlagList<IntegerT>({
        brain::evolution::amlz::kAlterParamMutationAction,
        brain::evolution::amlz::kRandomizeInstructionMutationAction,
        brain::evolution::amlz::kRandomizeComponentFunctionMutationAction}),
    "See the MutationAction enum.");
ABSL_FLAG(
    InstructionIndexT,
    predict_size_init,
    3,
    "Number of instructions for seed Algorithms. With op-op or random seeds, "
    "this will be the exact number of instructions. With hand-designed models, "
    "the component_function will be padded with no-ops to this number. If this "
    "is too small, the minimum number necessary will be used.");
ABSL_FLAG(
    IntegerT, progress_every, 1000000000000,
    "The period between progress reports to stdout, as the experiment "
    "progresses.");
ABSL_FLAG(
    bool, progress_every_by_time, false,
    "Whether `progress_every` is measured in seconds (true) or in number of "
    "individuals (false).");
ABSL_FLAG(
    RandomSeedT, random_seed, 100000,
    "Random seed for everything other than dataset parameters.");
ABSL_FLAG(
    bool, randomize_dataset_seeds, false,
    "If true, the seeds in the data and param seeds in the curriculum are "
    "randomized. This is done by setting a random seed as the first value for "
    "the param and data seed in each DatasetSpec. Data or param seeds must not "
    "be set, or else this feature will cause an intentional crash. The test "
    "datasets are not affected.");
ABSL_FLAG(
    RandomSeedT, dataset_randomization_seed, 0,
    "If randomize_dataset_seeds is set to true, this seed will be used"
    " as the seed for the randomization of param and data seed, and other"
    " dataset specs, for example, the positive and negative class in"
    " ProjectedBinaryClassificationDataset.");
ABSL_FLAG(
    InstructionIndexT,
    setup_size_init,
    6,
    "Number of instructions for seed Algorithms. With op-op or random seeds, "
    "this will be the exact number of instructions. With hand-designed models, "
    "the component_function will be padded with no-ops to this number. If this "
    "is too small, the minimum number necessary will be used.");
ABSL_FLAG(std::string, test_dataset_collection, "",
          "The text format of a DatasetCollection proto.");
ABSL_FLAG(
    IntegerT, tournament_size, 2,
    "Tournament size.");
ABSL_FLAG(
    IntegerT, train_budget_type, 0,
    "Type of TrainBudget to use. See TrainBudgetType enum.");
ABSL_FLAG(
    IntegerT, train_budget_baseline, 29,
    "Which model to use to as the baseline for the time budget. See "
    "TimeTrainBudget.");
ABSL_FLAG(
    double, train_budget_threshold_factor, 2.0,
    "Determines the maximum number of examples to train for when using certain "
    "training budgets. See TimeTrainBudget.");

namespace brain {
namespace evolution {
namespace amlz {

namespace {
using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT
}  // namespace

void run() {
  // Finalize datasets.
  DatasetCollection dataset_collection =
      ParseTextFormat<DatasetCollection>(GetFlag(FLAGS_dataset_collection));

  if (GetFlag(FLAGS_randomize_dataset_seeds)) {
    RandomizeDatasetSeeds(
        &dataset_collection,
        CustomHashMix(GetFlag(FLAGS_dataset_randomization_seed),
                      static_cast<RandomSeedT>(0)));  // TODO(ereal): simplify.
  }

  // Set up.
  CHECK(!GetFlag(FLAGS_experiment_spec).empty());
  const auto parsed_experiment_spec = ParseTextFormat<ExperimentSpec>(
      GetFlag(FLAGS_experiment_spec));
  RandomSeedT random_seed;
  // TODO(ereal): consider random_seed = GenerateRandomSeed().
  random_seed = GetFlag(FLAGS_random_seed);
  cout << "main: random_seed = " << random_seed << endl;
  MTRandom bit_gen(random_seed);
  RandomGenerator rand_gen(&bit_gen);

  Generator generator(
      static_cast<ModelT>(GetFlag(FLAGS_init_model)),
      GetFlag(FLAGS_setup_size_init),
      GetFlag(FLAGS_predict_size_init), GetFlag(FLAGS_learn_size_init),
      ExtractOps(parsed_experiment_spec.setup_ops()),
      ExtractOps(parsed_experiment_spec.predict_ops()),
      ExtractOps(parsed_experiment_spec.learn_ops()), &bit_gen,
      &rand_gen);
  unique_ptr<FECCache> functional_cache =
      parsed_experiment_spec.has_fec_cache() ?
          make_unique<FECCache>(
              parsed_experiment_spec.fec_cache()) :
          nullptr;
  unique_ptr<TrainBudget> train_budget;
  if (parsed_experiment_spec.has_train_budget()) {
    train_budget =
        BuildTrainBudget(parsed_experiment_spec.train_budget(), &generator);
  }

  Evaluator evaluator(
      parsed_experiment_spec.fitness_combination_mode(), dataset_collection,
      &rand_gen, functional_cache.get(), train_budget.get(),
      GetFlag(FLAGS_max_abs_error),
      true);  // verbose
  Mutator mutator(GetFlag(FLAGS_mutation_actions), GetFlag(FLAGS_mutate_prob),
                  ExtractOps(parsed_experiment_spec.setup_ops()),
                  ExtractOps(parsed_experiment_spec.predict_ops()),
                  ExtractOps(parsed_experiment_spec.learn_ops()),
                  GetFlag(FLAGS_mutate_setup_size_min),
                  GetFlag(FLAGS_mutate_setup_size_max),
                  GetFlag(FLAGS_mutate_predict_size_min),
                  GetFlag(FLAGS_mutate_predict_size_max),
                  GetFlag(FLAGS_mutate_learn_size_min),
                  GetFlag(FLAGS_mutate_learn_size_max), &bit_gen, &rand_gen);
  RegularizedEvolution regularized_evolution(
      &rand_gen, parsed_experiment_spec.local_population_size(),
      GetFlag(FLAGS_tournament_size), GetFlag(FLAGS_init_mutations),
      GetFlag(FLAGS_progress_every),
      GetFlag(FLAGS_progress_every_by_time), &generator, &evaluator, &mutator);
  cout << "Local algorithm ready." << endl;

  cout << "Initializing..." << endl;
  const IntegerT max_individuals = GetFlag(FLAGS_max_individuals);
  IntegerT evolved_num_individuals = 0;
  regularized_evolution.Init();
  cout << "Initialization done." << endl;
  cout << "Evolving..." << endl;
  const IntegerT remaining_individuals =
      max_individuals - regularized_evolution.NumIndividuals();
  IntegerT start_time = GetCurrentTimeNanos();
  const IntegerT evaluated_individuals = regularized_evolution.Run(
      remaining_individuals, kUnlimitedTime);
  evolved_num_individuals += evaluated_individuals;
  CHECK_EQ(regularized_evolution.NumIndividuals(), max_individuals);
  double time_per_model_us =
      static_cast<double>(GetCurrentTimeNanos() - start_time) /
      static_cast<double>(evolved_num_individuals * kNanosPerMicro);
  cout << "Evolution done." << endl;
  cout << "Time per model = " << time_per_model_us << " us." << endl;
  cout << "Evaluated " << regularized_evolution.NumIndividuals()
       << " individuals." << endl;

  // Evaluate and print details about the best global model.
  cout << endl << "Testing best model..." << endl;
  double pop_mean, pop_stdev, best_fitness;
  shared_ptr<const Algorithm> best_algorithm = make_shared<const Algorithm>();
  regularized_evolution.PopulationStats(
      &pop_mean, &pop_stdev, &best_algorithm, &best_fitness);

  const auto test_dataset_collection = ParseTextFormat<DatasetCollection>(
      GetFlag(FLAGS_test_dataset_collection));

  // Construct the train budget.
  unique_ptr<TrainBudget> test_train_budget;
  MTRandom test_bit_gen(random_seed + 426082123);
  RandomGenerator test_rand_gen(&test_bit_gen);
  Evaluator test_evaluator(
      MEAN_FITNESS_COMBINATION, test_dataset_collection, &test_rand_gen,
      nullptr,  // functional_cache
      nullptr,  // train_budget
      GetFlag(FLAGS_max_abs_error),
      false);  // verbose
  const double test_fitness = test_evaluator.Evaluate(*best_algorithm);

  cout << "test fitness = " << test_fitness << endl;
  cout << "test algorithm = " << best_algorithm->ToReadable() << endl;
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  brain::evolution::amlz::run();
  return 0;
}
