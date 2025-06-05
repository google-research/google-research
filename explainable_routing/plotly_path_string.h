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

#ifndef EXPERIMENTAL_GEO_ALGORITHMS_EXPLAINABLE_PLOTLY_PATH_STRING_H_
#define EXPERIMENTAL_GEO_ALGORITHMS_EXPLAINABLE_PLOTLY_PATH_STRING_H_

#include <string>
#include <vector>

#include "graph.h"

namespace geo_algorithms {

bool IsNull(const AOrRIndex& e);

std::vector<std::string> GetPlotlyPathStrings(
    const Graph& graph,
    const std::vector<AOrRIndex>& path,  // may or may not have null edge at end
    const std::string& name, const std::string& color);

std::string GetPlotlyPathString(
    const Graph& graph,
    const std::vector<AOrRIndex>& path,  // may or may not have null edge at end
    const std::string& name, const std::string& color);

}  // namespace geo_algorithms

#endif  // EXPERIMENTAL_GEO_ALGORITHMS_EXPLAINABLE_PLOTLY_PATH_STRING_H_
