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

// Main program to run the experiments of the paper.
// The program reads a file representing the stream of points as a sequence of
// rows one point per row. All rows contain d+1 fields separated by tab '\t'
// where the first field is the timestamp of the point (which must start from 0
// and be a consecutive integer). The remaining fields are the dimensions of the
// point.
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using std::string;
using std::vector;

#include "absl/flags/commandlineflag.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "base.h"
#include "estimate_optimum_range.h"
#include "io.h"
#include "meyerson_algorithm.h"
#include "sliding_window_framework.h"

ABSL_FLAG(std::string, input_file, "",
          "Path to the input file containing the stream of the points. The "
          "file format is a text file where each line is a point (in order of "
          "the stream). Each line contains d floating point numbers seprated "
          "by tab character defining the d dimensions of the point.");
ABSL_FLAG(
    std::string, output_file, "",
    "Path to the output text file with statistics on the experiment. The "
    "file will contain one line for every solution computed (i.e., one "
    "every output_every_n points). Each line contains in order the following "
    "tab-separated numerical fields: position in the stream; number of dist "
    "calls for one update, number of items stored, cost of the solution of our "
    "algorithm; cost of the baseline algorithm storing the entire window; cost "
    "of the baseline algorithm storing the same number of points of our "
    "algorithm.");
ABSL_FLAG(
    std::string, output_centers_file, "",
    "(Optional) Path to the output text file with the centers computed by our "
    "algorithm. If the path is non-empty, the file will contain one line for "
    "every solution computed (i.e., one every output_every_n points). Each line"
    " contains at most k comma-separated vectors. Each vector is a sequence of "
    "d tab-separated numbers.");
ABSL_FLAG(int32_t, window_size, 10000, "The size of the sliding window.");
ABSL_FLAG(int32_t, output_every_n, 100, "print solution every n steps.");
ABSL_FLAG(double, delta_grid, 0.2,
          "The delta parameter used int the grid for guessing the optimum.");
ABSL_FLAG(int32_t, k, 10, "the number of centers.");

namespace sliding_window {

// Define the global counters.
// global  count of the *current* number of points stored in any summary.
int64_t ITEMS_STORED;
// global count of all calls to the distance function.
int64_t CALLS_DIST_FN;

const char SEPARATOR[] = "\t";

void Main() {
  CHECK_GT(absl::GetFlag(FLAGS_window_size), 0);
  CHECK(!absl::GetFlag(FLAGS_input_file).empty());
  CHECK(!absl::GetFlag(FLAGS_output_file).empty());
  CHECK_GT(absl::GetFlag(FLAGS_k), 0);
  CHECK_GT(absl::GetFlag(FLAGS_delta_grid), 0);
  CHECK_LT(absl::GetFlag(FLAGS_k), absl::GetFlag(FLAGS_window_size));

  // Initialize the globals counters.
  ITEMS_STORED = 0;
  CALLS_DIST_FN = 0;

  absl::BitGen shared_genetor;

  const auto& [lower_bound, upper_bound] = guess_optimum_range_bounds(
      absl::GetFlag(FLAGS_input_file), absl::GetFlag(FLAGS_window_size),
      /*num_samples=*/100, absl::GetFlag(FLAGS_k), &shared_genetor);
  CHECK_GT(lower_bound, 0);
  CHECK_GT(upper_bound, 0);

  std::cout << "Estimated range for opt: [" << lower_bound << ", "
            << upper_bound << "]" << std::endl;

  std::ofstream output_file(absl::GetFlag(FLAGS_output_file), std::ios::out);

  absl::optional<std::ofstream> output_centers_file;
  if (!absl::GetFlag(FLAGS_output_centers_file).empty()) {
    output_centers_file.emplace(absl::GetFlag(FLAGS_output_centers_file),
                                std::ios::out);
  }

  // Solution found
  std::vector<TimePointPair> centers;

  // Our algorithm.
  FrameworkAlg<KMeansSummary> framework(
      absl::GetFlag(FLAGS_window_size), absl::GetFlag(FLAGS_k),
      absl::GetFlag(FLAGS_delta_grid), lower_bound, upper_bound,
      &shared_genetor);

  // The sampling based baseline.
  BaselineKmeansOverSampleWindow baseline(
      absl::GetFlag(FLAGS_window_size), absl::GetFlag(FLAGS_k),
      /* number_tries=*/10, &shared_genetor);

  std::string line;
  std::ifstream input_file(absl::GetFlag(FLAGS_input_file), std::ios::in);
  int32_t lines_read = 0;
  CHECK(input_file.is_open());
  while (std::getline(input_file, line)) {
    const auto& point = parse_point_from_string(line);

    // Execute the main algorithm.
    // Stores the number of calls before execting the algorithm.
    int64_t num_calls_dist_before = CALLS_DIST_FN;
    framework.process_point(lines_read, point);
    // New distance calls.
    int64_t num_calls_dist = CALLS_DIST_FN - num_calls_dist_before;

    // Executes the baseline.
    baseline.process_point(lines_read, point);

    // The current solution is evalauted only every FLAGS_print_every_n steps
    // for efficiency reasons (computing the correct cost of the solution
    // requires O(W) operations).
    if (lines_read >= absl::GetFlag(FLAGS_window_size) &&
        lines_read % absl::GetFlag(FLAGS_output_every_n) == 0) {
      // Estimate of the cost of the solution.
      double cost_estimate = 0;
      framework.solution(&centers, &cost_estimate);

      // If the path is passed, output the solution.
      if (output_centers_file.has_value()) {
        CHECK(output_centers_file.value().is_open());
        CHECK(!centers.empty());
        string centers_output;
        for (int i = 0; i < centers.size(); i++) {
          for (int j = 0; j < centers[i].second.size(); j++) {
            absl::StrAppend(&centers_output, centers[i].second[j]);
            if (j < centers[i].second.size() - 1) {
              absl::StrAppend(&centers_output, "\t");
            }
          }
          if (i < centers.size() - 1) {
            absl::StrAppend(&centers_output, ",");
          }
        }
        output_centers_file.value() << centers_output << "\n";
      }

      // Get the actual points in the window.
      const auto& points_window = baseline.points_window();

      // Actual cost of the solution for our algorithm.
      double cost = cost_solution(points_window, centers);
      CHECK_LE(centers.size(), absl::GetFlag(FLAGS_k));

      std::vector<TimePointPair> unused_centers;
      // Cost of the solution of the baseline algorithm using the whole window.
      double baseline_cost = 0;
      baseline.solution(&unused_centers, &baseline_cost);

      // Cost of the solution of the baseline algorithm using the same space of
      // our algorithm.
      double cost_baseline_subsampling;
      baseline.solution_subsampling(ITEMS_STORED, &unused_centers,
                                    &cost_baseline_subsampling);

      string output_str =
          absl::StrCat(lines_read, SEPARATOR, num_calls_dist, SEPARATOR,
                       ITEMS_STORED, SEPARATOR, cost, SEPARATOR, baseline_cost,
                       SEPARATOR, cost_baseline_subsampling, "\n");

      output_file << output_str.c_str();
    }

    if (lines_read % 500 == 0) {
      std::cout << "Points processed: " << lines_read << std::endl;
    }

    ++lines_read;
  }

  output_file.close();
  if (output_centers_file.has_value()) {
    output_centers_file.value().close();
  }
}

}  // namespace sliding_window

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  sliding_window::Main();
  return 0;
}
