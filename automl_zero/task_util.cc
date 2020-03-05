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

#include "task_util.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "algorithm.h"
#include "task.h"
#include "task.pb.h"
#include "definitions.h"
#include "executor.h"
#include "generator.h"
#include "memory.h"
#include "random_generator.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace automl_zero {

using ::absl::make_unique;  // NOLINT
using ::std::enable_if;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::max;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::is_same;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::set;  // NOLINT

// The values of the seeds below were chosen so that they span tasks of
// varying difficulties (the difficulties are for the nonlinear tasks).
vector<RandomSeedT> DefaultFirstParamSeeds() {
  return {
      1001,  // Easy.
      1012,  // Medium (on easier side).
      1010,  // Medium (on harder side).
      1000,  // Hard.
      1006,  // Easy.
      1008,  // Medium (on easier side).
      1007,  // Medium (on harder side).
      1003,  // Hard.
  };
}

vector<RandomSeedT> DefaultFirstDataSeeds() {
  return {11001, 11012, 11010, 11000, 11006, 11008, 11007, 11003};
}

void FillTasksFromTaskSpec(
    const TaskSpec& task_spec,
    vector<unique_ptr<TaskInterface>>* return_tasks) {
  const IntegerT num_tasks = task_spec.num_tasks();
  CHECK_GT(num_tasks, 0);
  vector<RandomSeedT> first_param_seeds =
      task_spec.param_seeds_size() == 0
          ? DefaultFirstParamSeeds()
          : vector<RandomSeedT>(task_spec.param_seeds().begin(),
                                task_spec.param_seeds().end());
  vector<RandomSeedT> first_data_seeds =
      task_spec.data_seeds_size() == 0
          ? DefaultFirstDataSeeds()
          : vector<RandomSeedT>(task_spec.data_seeds().begin(),
                                task_spec.data_seeds().end());
  CHECK(!first_param_seeds.empty());
  CHECK(!first_data_seeds.empty());

  RandomSeedT param_seed;
  RandomSeedT data_seed;
  for (IntegerT i = 0; i < num_tasks; ++i) {
    param_seed =
        i < first_param_seeds.size() ? first_param_seeds[i] : param_seed + 1;
    data_seed =
        i < first_data_seeds.size() ? first_data_seeds[i] : data_seed + 1;

    const IntegerT task_index = return_tasks->size();
    switch (task_spec.features_size()) {
      case 2:
        return_tasks->push_back(CreateTask<2>(task_index, param_seed,
                                                    data_seed, task_spec));
        break;
      case 4:
        return_tasks->push_back(CreateTask<4>(task_index, param_seed,
                                                    data_seed, task_spec));
        break;
      case 8:
        return_tasks->push_back(CreateTask<8>(task_index, param_seed,
                                                    data_seed, task_spec));
        break;
      case 16:
        return_tasks->push_back(CreateTask<16>(task_index, param_seed,
                                                     data_seed, task_spec));
        break;
      case 32:
        return_tasks->push_back(CreateTask<32>(task_index, param_seed,
                                                     data_seed, task_spec));
        break;
      default:
        LOG(FATAL) << "Unsupported features size: "
                   << task_spec.features_size() << std::endl;
    }
  }
}

void FillTasks(
    const TaskCollection& task_collection,
    vector<unique_ptr<TaskInterface>>* return_tasks) {
  // Check return targets are empty.
  CHECK(return_tasks->empty());
  for (const TaskSpec& task_spec : task_collection.tasks()) {
    FillTasksFromTaskSpec(task_spec, return_tasks);
  }
}

void RandomizeTaskSeeds(TaskCollection* task_collection,
                           const RandomSeedT seed) {
  RandomSeedT base_param_seed =
      HashMix(static_cast<RandomSeedT>(85652777), seed);
  mt19937 param_seed_bit_gen(base_param_seed);
  RandomGenerator param_seed_gen = RandomGenerator(
      &param_seed_bit_gen);

  RandomSeedT base_data_seed =
      HashMix(static_cast<RandomSeedT>(38272328), seed);
  mt19937 data_seed_bit_gen(base_data_seed);
  RandomGenerator data_seed_gen = RandomGenerator(
      &data_seed_bit_gen);

  for (TaskSpec& task : *task_collection->mutable_tasks()) {
    task.clear_param_seeds();
    task.clear_data_seeds();
    for (IntegerT i = 0; i < task.num_tasks(); i++) {
      task.add_param_seeds(param_seed_gen.UniformRandomSeed());
    }
    for (IntegerT i = 0; i < task.num_tasks(); i++) {
      task.add_data_seeds(data_seed_gen.UniformRandomSeed());
    }
  }
}

}  // namespace automl_zero
