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

#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/partitioning/kmeans_tree_partitioner.h"

#include "scann/base/single_machine_base.h"
#include "scann/brute_force/brute_force.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace scann_ops {

namespace internal {

StatusOr<unique_ptr<SingleMachineSearcherBase<float>>>
CreateRecommendedAsymmetricSearcher(
    shared_ptr<DenseDataset<float>> dataset,
    shared_ptr<const DistanceMeasure> quantization_distance,
    int32_t num_neighbors, float epsilon = numeric_limits<float>::infinity(),
    bool with_exact_reordering = true,
    shared_ptr<thread::ThreadPool> pool = nullptr,
    int num_clusters_per_block = 16, int num_dimension_per_block = 2) {
  DCHECK(dataset);

  if (dataset->size() < num_clusters_per_block ||
      dataset->dimensionality() < num_dimension_per_block) {
    unique_ptr<SingleMachineSearcherBase<float>> bf_searcher(
        new BruteForceSearcher<float>(quantization_distance, dataset,
                                      num_neighbors, epsilon));
    return std::move(bf_searcher);
  }

  AsymmetricHasherConfig hasher_config;
  hasher_config.set_num_clusters_per_block(num_clusters_per_block);

  DimensionIndex dim = dataset->dimensionality();
  hasher_config.mutable_projection()->set_input_dim(dim);
  hasher_config.mutable_projection()->set_projection_type(
      ProjectionConfig::CHUNK);
  hasher_config.mutable_projection()->set_num_blocks(
      (dim + num_dimension_per_block - 1) / num_dimension_per_block);
  hasher_config.mutable_projection()->set_num_dims_per_block(
      num_dimension_per_block);
  if (num_clusters_per_block == 16) {
    hasher_config.set_lookup_type(AsymmetricHasherConfig::INT8_LUT16);
  } else {
    hasher_config.set_lookup_type(AsymmetricHasherConfig::FLOAT);
  }
  hasher_config.mutable_quantization_distance()->set_distance_measure(
      std::string(quantization_distance->name()));
  asymmetric_hashing2::TrainingOptions<float> training_opts(
      hasher_config, quantization_distance, *dataset);
  TF_ASSIGN_OR_RETURN(shared_ptr<asymmetric_hashing2::Model<float>> model,
                      asymmetric_hashing2::TrainSingleMachine<float>(
                          *dataset, training_opts, pool));
  auto indexer = make_unique<asymmetric_hashing2::Indexer<float>>(
      training_opts.projector(), quantization_distance, model);

  auto hashed_dataset = std::make_shared<DenseDataset<uint8_t>>();
  hashed_dataset->Reserve(dataset->size());
  for (int i = 0; i < dataset->size(); ++i) {
    Datapoint<uint8_t> dp;
    SCANN_RETURN_IF_ERROR(indexer->Hash((*dataset)[i], &dp));
    hashed_dataset->AppendOrDie(dp.ToPtr());
  }

  auto queryer = make_unique<asymmetric_hashing2::AsymmetricQueryer<float>>(
      training_opts.projector(), quantization_distance, model);
  asymmetric_hashing2::SearcherOptions<float> searcher_opts;
  searcher_opts.set_asymmetric_lookup_type(hasher_config.lookup_type());
  searcher_opts.EnableAsymmetricQuerying(std::move(queryer));
  unique_ptr<SingleMachineSearcherBase<float>> ah_searcher(
      new asymmetric_hashing2::Searcher<float>(
          std::move(dataset), std::move(hashed_dataset), searcher_opts,
          num_neighbors, numeric_limits<float>::infinity()));

  if (with_exact_reordering) {
    auto helper = std::make_shared<ExactReorderingHelper<float>>(
        quantization_distance, ah_searcher->shared_dataset());
    ah_searcher->EnableReordering(
        std::make_shared<ExactReorderingHelper<float>>(
            quantization_distance, ah_searcher->shared_dataset()),
        num_neighbors, epsilon);
  }

  return std::move(ah_searcher);
}

}  // namespace internal
}  // namespace scann_ops
}  // namespace tensorflow
