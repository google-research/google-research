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

#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "main/cann_rg.h"

ABSL_FLAG(double, c, 1.165939651397022,
          "Approximation factor (c-approximation, see the paper)");
ABSL_FLAG(int, dim, 128, "The dimentionality of the points (descriptors).");
ABSL_FLAG(int, num_grids, 20, "Number of grids.");
ABSL_FLAG(int, num_features, 1000,
          "Number of features (it clips the input file to this size).");
ABSL_FLAG(double, eps, 1e-6, "Minimum point contribution to score.");
ABSL_FLAG(double, p0, 120,
          "The absolute radius for which distances are counted.");
ABSL_FLAG(double, p1, 0.4591763556652752,
          "The modulation factor to the metric (see the paper).");
ABSL_FLAG(double, min_distance, 28.28296718361788,
          "Minimum distance that determines the number of distances used in "
          "the c-approximation (see the paper).");

ABSL_FLAG(std::string, index_descriptor_files, "",
          "File containing a list of descriptor files for the index, one "
          "file per line.");
ABSL_FLAG(std::string, query_descriptor_files, "",
          "File containing a list of descriptor files for the queries, one "
          "file per line.");
ABSL_FLAG(std::string, pairs_file, "",
          "Output file containing the top matches for each query.");

using cann_rg::ScoredPair;

using std::pair;
using std::string;
using std::vector;

vector<string> ReadFileLines(const string &file_path) {
  string line;
  vector<string> file_lines;
  std::ifstream file(file_path);
  while (std::getline(file, line)) {
    file_lines.push_back(line);
    std::cout << line << "\n";
  }
  std::cout << "Read " << file_lines.size() << " lines from " << file_path
            << "\n";
  return file_lines;
}

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::cout << "reading names..." << std::endl;
  const vector<string> index_filenames =
      ReadFileLines(absl::GetFlag(FLAGS_index_descriptor_files));
  const vector<string> query_filenames =
      ReadFileLines(absl::GetFlag(FLAGS_query_descriptor_files));

  std::cout << "matching images..." << std::endl;

  const auto pairs = cann_rg::MatchImages(
      query_filenames, index_filenames, absl::GetFlag(FLAGS_num_features),
      absl::GetFlag(FLAGS_dim), absl::GetFlag(FLAGS_c),
      absl::GetFlag(FLAGS_num_grids), absl::GetFlag(FLAGS_eps),
      absl::GetFlag(FLAGS_p0), absl::GetFlag(FLAGS_p1),
      absl::GetFlag(FLAGS_min_distance));

  std::ofstream pairs_file(absl::GetFlag(FLAGS_pairs_file));
  for (const auto &pair : pairs) {
    pairs_file << pair.query << ", " << pair.target << ", " << pair.score
               << std::endl;
  }
  pairs_file.close();
  return 0;
}
