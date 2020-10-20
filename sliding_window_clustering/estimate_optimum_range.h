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

// In the paper we described how our algorithm can be implemented without the
// need of an upper and lower bound on the optimum value. For simplicity, in the
// experimental evaluation (as state in the extended version of the paper) we
// use instead a heuristic, which we observe to be highly effective, to set up
// an approximate upper and lower bounds on the optimum. The heuristic works as
// follows, we sample a few sliding window start points and we run an
// approximation algorithm on the sampled window. This algorithm could be any
// insertion only algorithm or any algorithm that is able to run on the window.
// Then, we output as guess of the optimum range the mean cost of the
// solutions found by the algorithm +/- 3 standard deviations. We
// notice that the solution output by our algorithm is not very sensitive to the
// details of this heuristic.
#ifndef SLIDING_WINDOW_CLUSTERING_ESTIMATE_OPTIMUM_RANGE_H_
#define SLIDING_WINDOW_CLUSTERING_ESTIMATE_OPTIMUM_RANGE_H_

#include <math.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "base.h"

namespace sliding_window {

// Takes as input a path to a file encoding the stream as described in io.h
// the window size, the number of samples to obtain, the number of centers and
// a random bit generator. Outputs a pair containing the lower and upper bounds
// for optimum that are guessed by the algorithm using the heuristic.
std::pair<double, double> guess_optimum_range_bounds(
    const string& stream_file_path, int32_t window_size, int32_t num_samples,
    int32_t k, absl::BitGen* gen);

}  //  namespace sliding_window

#endif  // SLIDING_WINDOW_CLUSTERING_ESTIMATE_OPTIMUM_RANGE_H_
