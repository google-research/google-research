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

#include "scann/utils/single_machine_retraining.h"

#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/internal/single_machine_factory_impl.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/base/single_machine_factory_scann.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/scann_config_utils.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
StatusOrSearcherUntyped RetrainAndReindexSearcherImpl(
    UntypedSingleMachineSearcherBase* untyped_searcher,
    absl::Mutex* searcher_pointer_mutex, ScannConfig config,
    shared_ptr<ThreadPool> parallelization_pool) {
  if (searcher_pointer_mutex) searcher_pointer_mutex->AssertNotHeld();
  SingleMachineSearcherBase<T>* searcher =
      down_cast<SingleMachineSearcherBase<T>*>(untyped_searcher);

  SCANN_ASSIGN_OR_RETURN(auto dataset, searcher->ReconstructFloatDataset());
  if (!dataset) {
    return FailedPreconditionError(
        "Searchers passed to RetrainAndReindexSearcher must contain the "
        "original, uncompressed dataset, i.e. dataset() must not return null.");
  }
  RetrainAndReindexFixup<float>(
      searcher, std::const_pointer_cast<DenseDataset<float>>(dataset));

  StripPreprocessedArtifacts(&config);
  SingleMachineFactoryOptions opts;
  opts.parallelization_pool = std::move(parallelization_pool);
  SCANN_ASSIGN_OR_RETURN(
      auto result,
      SingleMachineFactoryScann<T>(
          config, std::const_pointer_cast<TypedDataset<T>>(searcher->dataset_),
          opts));

  auto lock_mutex = [&searcher_pointer_mutex]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
    if (searcher_pointer_mutex) searcher_pointer_mutex->WriterLock();
  };
  lock_mutex();

  result->docids_ = std::move(searcher->docids_);
  result->retraining_requires_dataset_ =
      untyped_searcher->retraining_requires_dataset_;
  return result;
}

StatusOrSearcherUntyped RetrainAndReindexSearcher(
    UntypedSingleMachineSearcherBase* searcher,
    absl::Mutex* searcher_pointer_mutex, const ScannConfig& config,
    shared_ptr<ThreadPool> parallelization_pool)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (searcher_pointer_mutex) searcher_pointer_mutex->AssertNotHeld();
  SCANN_RET_CHECK(searcher);
  return SCANN_CALL_FUNCTION_BY_TAG(
      searcher->TypeTag(), RetrainAndReindexSearcherImpl, searcher,
      searcher_pointer_mutex, config, std::move(parallelization_pool));
}

}  // namespace research_scann
