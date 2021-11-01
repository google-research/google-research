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

#include "overlap_evaluate.h"

#include <assert.h>

double OverlapEvaluate::operator()(
    const std::vector<int>& clusters,
    const std::vector<int>& ground_truth_communities) {
  int n = clusters.size();
  assert(n > 0);

  double score = 0;
  for (int i = 0; i < n; i++) {
    if (clusters[i] == ground_truth_communities[i]) {
      score += 1.0;
    }
  }
  return score / n;
}
