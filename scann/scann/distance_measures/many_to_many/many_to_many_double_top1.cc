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



#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/many_to_many/many_to_many_templates.h"

namespace tensorflow {
namespace scann_ops {
namespace mm_internal {

template void DenseDistanceManyToManyImpl(
    const DistanceMeasure &dist, const DenseDataset<double> &queries,
    const DenseDataset<double> &database, thread::ThreadPool *pool,
    ManyToManyTop1Callback<double> callback);
template void DenseDistanceManyToManyImpl(
    const DistanceMeasure &dist, const DenseDataset<double> &queries,
    const DenseDataset<double> &database, thread::ThreadPool *pool,
    ManyToManyTop1SquaredL2Callback<double, double, double> callback);

}  // namespace mm_internal
}  // namespace scann_ops
}  // namespace tensorflow
