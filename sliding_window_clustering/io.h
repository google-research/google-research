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

// Methods to read from a text file the input of the algorithm.
// The stream is represented as a text file where each row is a data point in
// order of the stream. Each row has d fields separated by tab '\t' character
// where d is the dimension of the points. The d fields are the dimensions of
// the point. All points must have the same number of dimensions.
#ifndef SLIDING_WINDOW_CLUSTERING_IO_H_
#define SLIDING_WINDOW_CLUSTERING_IO_H_

#include "absl/random/random.h"
#include "base.h"

namespace sliding_window {

// Reads (at most) num_sample consecutive and disjoint sliding windows from the
// input file after skipping prefix_to_skip elements from the beginning of the
// stream. Outputs the size of the stream read. Each window given in output is
// complete (assuming the stream has >= window_size + prefix_to_skip elements)
// but the number of samples in output will be lower than num_samples if the
// stream is shorter than num_samples*window_size+prefix_to_skip elements
std::vector<std::vector<TimePointPair>> get_windows_samples(
    const string& stream_file_path, int32_t window_size, int32_t num_samples,
    int32_t prefix_to_skip, int32_t* stream_size);

// Given a line from a file outputs a vector.
std::vector<double> parse_point_from_string(const std::string& file_line);

}  // namespace sliding_window

#endif  // SLIDING_WINDOW_CLUSTERING_IO_H_
