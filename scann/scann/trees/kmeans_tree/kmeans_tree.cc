// Copyright 2024 The Google Research Authors.
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

#include "scann/trees/kmeans_tree/kmeans_tree.h"

#include <cstdint>
#include <numeric>

#include "absl/log/check.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/trees/kmeans_tree/kmeans_tree.pb.h"
#include "scann/trees/kmeans_tree/kmeans_tree_node.h"
#include "scann/trees/kmeans_tree/training_options.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

namespace {

int32_t CountLeaves(const KMeansTreeNode& node) {
  if (node.IsLeaf()) {
    return 1;
  } else {
    int32_t result = 0;
    for (const KMeansTreeNode& child : node.Children()) {
      result += CountLeaves(child);
    }

    return result;
  }
}

}  // namespace

KMeansTree::KMeansTree() = default;

KMeansTree::KMeansTree(const SerializedKMeansTree& serialized) {
  learned_spilling_type_ = serialized.learned_spilling_type();
  max_spill_centers_ = serialized.max_centers_for_learned_spilling();
  root_.BuildFromProto(serialized.root());
  n_tokens_ = CountLeaves(root_);
  root_.PopulateCurNodeCenters();
  root_.CreateFixedPointCenters();
  CheckIfFlat();
}

KMeansTree KMeansTree::CreateFlat(DenseDataset<float> centers) {
  KMeansTree result;
  result.root_ = KMeansTreeNode::CreateFlat(std::move(centers));
  result.n_tokens_ = CountLeaves(result.root_);
  result.root_.PopulateCurNodeCenters();
  result.root_.CreateFixedPointCenters();
  result.CheckIfFlat();
  CHECK(result.is_flat_);
  return result;
}

Status KMeansTree::Train(const Dataset& training_data,
                         const DistanceMeasure& training_distance,
                         int32_t k_per_level,
                         KMeansTreeTrainingOptions* training_options) {
  DCHECK(training_options);

  vector<DatapointIndex> subset(training_data.size());
  std::iota(subset.begin(), subset.end(), static_cast<DatapointIndex>(0));
  Status status = root_.Train(training_data, subset, training_distance,
                              k_per_level, 0, training_options);
  if (!status.ok()) return status;
  n_tokens_ = root_.NumberLeaves(0);
  root_.PopulateCurNodeCenters();
  learned_spilling_type_ = training_options->learned_spilling_type;
  max_spill_centers_ = training_options->max_spill_centers;
  root_.CreateFixedPointCenters();
  CheckIfFlat();
  return status;
}

void KMeansTree::Serialize(SerializedKMeansTree* result) const {
  CHECK(result != nullptr);
  result->set_learned_spilling_type(learned_spilling_type_);
  root_.CopyToProto(result->mutable_root(), true);
}

void KMeansTree::SerializeWithoutIndices(SerializedKMeansTree* result) const {
  CHECK(result != nullptr);
  result->set_learned_spilling_type(learned_spilling_type_);
  root_.CopyToProto(result->mutable_root(), false);
}

void KMeansTree::CheckIfFlat() {
  if (root_.IsLeaf()) return;

  bool all_children_are_leaves = true;
  for (const KMeansTreeNode& node : root_.Children()) {
    all_children_are_leaves = all_children_are_leaves && node.IsLeaf();
  }
  if (all_children_are_leaves) is_flat_ = true;
}

}  // namespace research_scann
