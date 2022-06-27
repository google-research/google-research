// Copyright 2022 The Google Research Authors.
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



#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/many_to_many/many_to_many_templates.h"

namespace research_scann {
namespace mm_internal {

template void DenseDistanceManyToManyImpl(
    const DistanceMeasure &dist, const DenseDataset<double> &queries,
    const DenseDataset<double> &database, ThreadPool *pool,
    ManyToManyResultsCallback<double> callback);

}
}  // namespace research_scann
