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

#ifndef ZEBRAIX_BASE_DOMINANCE_CONVERSION_H_
#define ZEBRAIX_BASE_DOMINANCE_CONVERSION_H_

#include "base/dominance.h"
#include "base/zebraix_graph.proto.h"

namespace zebraix {
namespace base {

// Fill a dominance graph with the information from the proto graph's nodes.
void FillGraphFromProto(const zebraix_proto::ZebraixGraph& p_graph,
                        DominanceGraph* d_graph);

// Apply default ("base") values to proto nodes by replacing each node with the
// base values and merging the originals over the result.
void SelfMergeLayoutIntoBaseValues(zebraix_proto::ZebraixGraph* layout);

}  // namespace base
}  // namespace zebraix

#endif  // ZEBRAIX_BASE_DOMINANCE_CONVERSION_H_
