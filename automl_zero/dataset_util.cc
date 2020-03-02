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

#include "dataset_util.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "file/base/path.h"
#include "algorithm.h"
#include "dataset.h"
#include "datasets.proto.h"
#include "definitions.h"
#include "executor.h"
#include "generator.h"
#include "memory.h"
#include "random_generator.h"
#include "google/protobuf/text_format.h"
#include "sstable/public/sstable.h"
#include "absl/base/integral_types.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::absl::make_unique;  // NOLINT
using ::std::enable_if;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::max;  // NOLINT
using ::std::is_same;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::set;  // NOLINT

// The values of the seeds below were chosen so that they span datasets of
// varying difficulties (the difficulties are for the nonlinear datasets).
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

void FillDatasetsFromDatasetSpec(
    const DatasetSpec& dataset_spec,
    vector<unique_ptr<DatasetInterface>>* return_datasets) {
  const IntegerT num_datasets = dataset_spec.num_datasets();
  CHECK_GT(num_datasets, 0);
  vector<RandomSeedT> first_param_seeds =
      dataset_spec.param_seeds_size() == 0
          ? DefaultFirstParamSeeds()
          : vector<RandomSeedT>(dataset_spec.param_seeds().begin(),
                                dataset_spec.param_seeds().end());
  vector<RandomSeedT> first_data_seeds =
      dataset_spec.data_seeds_size() == 0
          ? DefaultFirstDataSeeds()
          : vector<RandomSeedT>(dataset_spec.data_seeds().begin(),
                                dataset_spec.data_seeds().end());
  CHECK(!first_param_seeds.empty());
  CHECK(!first_data_seeds.empty());

  RandomSeedT param_seed;
  RandomSeedT data_seed;
  for (IntegerT i = 0; i < num_datasets; ++i) {
    param_seed =
        i < first_param_seeds.size() ? first_param_seeds[i] : param_seed + 1;
    data_seed =
        i < first_data_seeds.size() ? first_data_seeds[i] : data_seed + 1;

    const IntegerT dataset_index = return_datasets->size();
    switch (dataset_spec.features_size()) {
      case 2:
        return_datasets->push_back(CreateDataset<2>(dataset_index, param_seed,
                                                    data_seed, dataset_spec));
        break;
      case 4:
        return_datasets->push_back(CreateDataset<4>(dataset_index, param_seed,
                                                    data_seed, dataset_spec));
        break;
      case 8:
        return_datasets->push_back(CreateDataset<8>(dataset_index, param_seed,
                                                    data_seed, dataset_spec));
        break;
      case 16:
        return_datasets->push_back(CreateDataset<16>(dataset_index, param_seed,
                                                     data_seed, dataset_spec));
        break;
      case 32:
        return_datasets->push_back(CreateDataset<32>(dataset_index, param_seed,
                                                     data_seed, dataset_spec));
        break;
      default:
        LOG(FATAL) << "Unsupported features size: "
                   << dataset_spec.features_size() << std::endl;
    }
  }
}

void FillDatasets(
    const DatasetCollection& dataset_collection,
    vector<unique_ptr<DatasetInterface>>* return_datasets) {
  // Check return targets are empty.
  CHECK(return_datasets->empty());
  for (const DatasetSpec& dataset_spec : dataset_collection.datasets()) {
    FillDatasetsFromDatasetSpec(dataset_spec, return_datasets);
  }
}

void RandomizeDatasetSeeds(DatasetCollection* dataset_collection,
                           const RandomSeedT seed) {
  RandomSeedT base_param_seed =
      CustomHashMix(static_cast<RandomSeedT>(85652777), seed);
  MTRandom param_seed_bit_gen(base_param_seed);
  RandomGenerator param_seed_gen = RandomGenerator(
      &param_seed_bit_gen);

  RandomSeedT base_data_seed =
      CustomHashMix(static_cast<RandomSeedT>(38272328), seed);
  MTRandom data_seed_bit_gen(base_data_seed);
  RandomGenerator data_seed_gen = RandomGenerator(
      &data_seed_bit_gen);

  for (DatasetSpec& dataset : *dataset_collection->mutable_datasets()) {
    CHECK_EQ(dataset.param_seeds_size(), 0)
        << "Seed randomization requested but param seed was provided."
        << std::endl;
    dataset.clear_param_seeds();
    for (IntegerT i = 0; i < dataset.num_datasets(); i++) {
      dataset.add_param_seeds(param_seed_gen.UniformRandomSeed());
    }

    CHECK_EQ(dataset.data_seeds_size(), 0)
        << "Seed randomization requested but data seed was provided."
        << std::endl;
    dataset.clear_data_seeds();
    for (IntegerT i = 0; i < dataset.num_datasets(); i++) {
      dataset.add_data_seeds(data_seed_gen.UniformRandomSeed());
    }
  }
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
