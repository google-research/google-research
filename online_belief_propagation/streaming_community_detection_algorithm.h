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

#ifndef ONLINE_BELIEF_PROPAGATION_STREAMING_COMMUNITY_DETECTION_ALGORITHM_H_
#define ONLINE_BELIEF_PROPAGATION_STREAMING_COMMUNITY_DETECTION_ALGORITHM_H_

#include <assert.h>

#include "graph_stream.h"

// Abstract base class for label propagation algorithms. Any implemented label
// propagation algorithm should be inherited from this.
//
// Optionally, side information can be provided along with each vertex of the
// graph. Label information is stored in each vertex; the aggregate labels of
// all vertices define the state of the algorithm.
template <typename SideInfoType, typename LabelInfoType>
class StreamingCommunityDetectionAlgorithm {
 public:
  StreamingCommunityDetectionAlgorithm() {}
  virtual ~StreamingCommunityDetectionAlgorithm() {}

  // Runs the algorithm on a given input graph stream. Calls Initialize, then
  // steps through the input stream and calls UpdateLabels after each step.
  void Run(GraphStream* graph_stream,
           const std::vector<SideInfoType>* side_info_vector = NULL) {
    assert(
        ("Incorrect side information vector length.",
         side_info_vector == NULL ||
             side_info_vector->size() == graph_stream->GetTotalVertexNumber()));
    graph_stream->Restart();
    Initialize();
    for (int i = 0; i < graph_stream->GetTotalVertexNumber(); ++i) {
      graph_stream->Step();
      UpdateLabels(i, graph_stream,
                   side_info_vector == NULL ? NULL : &((*side_info_vector)[i]));
    }
  }
  // After Run has terminated, GenerateClusters generates the output clustering
  // based on the current labels.
  virtual std::vector<int> GenerateClusters() const = 0;

 protected:
  // labels_ stores the current label of each vertex at every point in time.
  std::vector<LabelInfoType> labels_;
  // The following two pure virtual functions are the core of any derived
  // algorithm.
  // 'UpdateLabels' is ran after each vertex insertion. 'vertex' represents the
  // id of the newly added vertex. 'graph_stream' represents the input graph
  // stream. 'side_information' points to the side information associated with
  // the newly added vertex, or it is NULL if no side information is available.
  virtual void UpdateLabels(int vertex, const GraphStream* graph_stream,
                            const SideInfoType* side_information) = 0;
  // 'Initialize' is ran before the arrival of the first vertex.
  virtual void Initialize() = 0;
};

#endif  // ONLINE_BELIEF_PROPAGATION_STREAMING_COMMUNITY_DETECTION_ALGORITHM_H_
