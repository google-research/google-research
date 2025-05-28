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

#ifndef SCANN_UTILS_SINGLE_MACHINE_RETRAINING_H_
#define SCANN_UTILS_SINGLE_MACHINE_RETRAINING_H_

#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"

namespace research_scann {

StatusOrSearcherUntyped RetrainAndReindexSearcher(
    UntypedSingleMachineSearcherBase* searcher,
    absl::Mutex* searcher_pointer_mutex, const ScannConfig& config,
    shared_ptr<ThreadPool> parallelization_pool = nullptr);

}

#endif
