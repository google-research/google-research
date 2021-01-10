// Copyright 2021 The Google Research Authors.
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

#include "scann/utils/factory_helpers.h"

#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/proto/compressed_reordering.pb.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/utils/types.h"

namespace research_scann {

Status GenericSearchParameters::PopulateValuesFromScannConfig(
    const ScannConfig& config) {
  if (!config.has_num_neighbors() && !config.has_epsilon_distance()) {
    return InvalidArgumentError(
        "Must specify num_neighbors and/or epsilon_distance.");
  }

  if (config.has_num_single_shard_neighbors()) {
    if (!config.has_num_neighbors()) {
      return InvalidArgumentError(
          "ScannConfig must have num_neighbors if it has "
          "num_single_shard_neighbors.");
    }

    if (config.num_neighbors() < config.num_single_shard_neighbors()) {
      return InvalidArgumentError(
          "num_neighbors must be >= num_single_shard_neighbors if "
          "both are set.");
    }

    post_reordering_num_neighbors = config.num_single_shard_neighbors();
  } else if (config.has_num_neighbors()) {
    post_reordering_num_neighbors = config.num_neighbors();
  } else {
    post_reordering_num_neighbors = numeric_limits<int32_t>::max();
  }

  post_reordering_epsilon = config.has_epsilon_distance()
                                ? config.epsilon_distance()
                                : numeric_limits<float>::infinity();
  if (post_reordering_num_neighbors <= 0) {
    return InvalidArgumentError("num_neighbors must be > 0.");
  }

  TF_ASSIGN_OR_RETURN(reordering_dist,
                      GetDistanceMeasure(config.distance_measure()));

  if (config.has_compressed_reordering()) {
    const auto& cr = config.compressed_reordering();
    if (!cr.has_approx_num_neighbors() && !cr.has_approx_epsilon_distance()) {
      return InvalidArgumentError(
          "Must specify approx_num_neighbors and/or approx_epsilon if "
          "performing compressed reordering.");
    }

    pre_reordering_num_neighbors = cr.has_approx_num_neighbors()
                                       ? cr.approx_num_neighbors()
                                       : numeric_limits<int32_t>::max();
    if (pre_reordering_num_neighbors <= 0) {
      return InvalidArgumentError("approx_num_neighbors must be > 0.");
    }

    pre_reordering_epsilon = cr.has_approx_epsilon_distance()
                                 ? cr.approx_epsilon_distance()
                                 : numeric_limits<float>::infinity();
  }

  if (config.has_exact_reordering()) {
    const auto& er = config.exact_reordering();
    if (er.has_approx_distance_measure()) {
      TF_ASSIGN_OR_RETURN(pre_reordering_dist,
                          GetDistanceMeasure(er.approx_distance_measure()));
    } else {
      pre_reordering_dist = reordering_dist;
    }

    if (!er.has_approx_num_neighbors() && !er.has_approx_epsilon_distance()) {
      return InvalidArgumentError(
          "Must specify approx_num_neighbors and/or approx_epsilon if "
          "performing exact reordering.");
    }

    pre_reordering_num_neighbors = (er.has_approx_num_neighbors())
                                       ? (er.approx_num_neighbors())
                                       : (numeric_limits<int32_t>::max());
    if (pre_reordering_num_neighbors <= 0) {
      return InvalidArgumentError("approx_num_neighbors must be > 0.");
    }

    pre_reordering_epsilon = er.has_approx_epsilon_distance()
                                 ? er.approx_epsilon_distance()
                                 : numeric_limits<float>::infinity();
  } else if (config.has_compressed_reordering()) {
    const auto& cr = config.compressed_reordering();
    if (!cr.has_approx_num_neighbors() && !cr.has_approx_epsilon_distance()) {
      return InvalidArgumentError(
          "Must specify approx_num_neighbors and/or approx_epsilon if "
          "performing compressed reordering.");
    }
    pre_reordering_num_neighbors = cr.has_approx_num_neighbors()
                                       ? cr.approx_num_neighbors()
                                       : numeric_limits<int32_t>::max();
    if (pre_reordering_num_neighbors <= 0) {
      return InvalidArgumentError("approx_num_neighbors must be > 0.");
    }
    pre_reordering_epsilon = cr.has_approx_epsilon_distance()
                                 ? cr.approx_epsilon_distance()
                                 : numeric_limits<float>::infinity();
    pre_reordering_dist = reordering_dist;
  } else {
    pre_reordering_dist = reordering_dist;
    pre_reordering_num_neighbors = post_reordering_num_neighbors;
    pre_reordering_epsilon = post_reordering_epsilon;
  }

  return OkStatus();
}

}  // namespace research_scann
