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

// Copyright 2023 The Google Research Authors.
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

// Reads answers from a CSV files and prints ELO scores and other information.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/strings/string_view.h"
#include "csv-parser/parser.hpp"
#include "elo.h"

ABSL_FLAG(std::string, csv_path, "", "path to csv file");

enum class Winner { kA, kB, kDraw };

struct CSVAnswer {
  std::string rater;
  bool is_golden;
  bool golden_correct;
  std::string method_a;
  std::string method_b;
  Winner winner;
};

void Run() {
  std::ifstream f(absl::GetFlag(FLAGS_csv_path));
  aria::csv::CsvParser parser(f);
  absl::flat_hash_map<std::string, size_t> field_map;

  std::vector<CSVAnswer> answers;

  for (const auto& record : parser) {
    // Row 1 is headers.
    if (field_map.empty()) {
      for (size_t i = 0; i < record.size(); i++) {
        field_map.emplace(record[i], i);
      }
      continue;
    }

    auto f = [&](absl::string_view fname) {
      return record[field_map.at(fname)];
    };

    CSVAnswer answer;
    auto raw_answer = f("answerValue");
    if (raw_answer == "1" || raw_answer == "true" || raw_answer == "B") {
      answer.winner = Winner::kB;
    } else if (raw_answer == "draw") {
      answer.winner = Winner::kDraw;
    } else {
      answer.winner = Winner::kA;
    }
    answer.method_a = f("methodA");
    answer.method_b = f("methodB");
    answer.is_golden = f("isGolden") == "1" || f("isGolden") == "true";
    answer.golden_correct =
        f("modelProbability") == "1" || f("modelProbability") == "true";
    answer.rater = f("answerer");
    answers.push_back(std::move(answer));
  }

  absl::flat_hash_set<std::string> methods_hs;
  for (const auto& ans : answers) {
    if (ans.is_golden) continue;
    methods_hs.insert(ans.method_a);
    methods_hs.insert(ans.method_b);
  }
  std::vector<std::string> methods(methods_hs.begin(), methods_hs.end());
  std::sort(methods.begin(), methods.end());

  absl::flat_hash_map<std::string, size_t> raters;
  std::vector<std::string> rater_names;
  ELOData elo_data(ELOSettings(), 1, methods);
  for (const auto& ans : answers) {
    if (!raters.count(ans.rater)) {
      raters.emplace(ans.rater, raters.size());
      rater_names.push_back(ans.rater);
    }
    size_t rater_index = raters.at(ans.rater);
    if (ans.is_golden) {
      elo_data.AddQuestion({{true}}, "A", "B", {ans.golden_correct ? "A" : "B"},
                           rater_index);
    } else {
      std::string choice = ans.winner == Winner::kA   ? ans.method_a
                           : ans.winner == Winner::kB ? ans.method_b
                                                      : "";
      elo_data.AddQuestion(std::nullopt, ans.method_a, ans.method_b, {choice},
                           rater_index);
    }
  }

  const auto& computed_elos = elo_data.ComputeELOScores()[0];
  std::vector<std::pair<ELOScore, std::string>> elos(computed_elos.size());
  for (size_t i = 0; i < computed_elos.size(); i++) {
    elos[i] = std::make_pair(computed_elos[i], methods[i]);
  }
  std::sort(elos.begin(), elos.end(), [](const auto& a, const auto& b) {
    return a.first.elo > b.first.elo;
  });

  for (const auto& [score, method] : elos) {
    printf("%30s: %10.2f [%10.2f - %10.2f]\n", method.c_str(), score.elo,
           score.p99_low, score.p99_hi);
  }

  auto suggestions = elo_data.SuggestQuestions();
  std::sort(suggestions.begin(), suggestions.end(),
            [](const auto& a, const auto& b) { return a.weight > b.weight; });

  printf("\n\n\nRater random probabilities\n");
  const auto& rater_random_probabilities = elo_data.RaterRandomProbability(0);
  std::vector<size_t> rater_sorted_indices(raters.size());
  std::iota(rater_sorted_indices.begin(), rater_sorted_indices.end(), 0);
  std::sort(rater_sorted_indices.begin(), rater_sorted_indices.end(),
            [&](auto a, auto b) {
              return rater_random_probabilities[a] <
                     rater_random_probabilities[b];
            });
  for (const auto& idx : rater_sorted_indices) {
    printf("%20s: %7.7f\n", rater_names[idx].c_str(),
           rater_random_probabilities[idx]);
  }

  printf("\n\n\nSuggestions\n");
  for (const auto& suggestion : suggestions) {
    printf("%30s vs %30s: %7.7f\n", suggestion.method_a.c_str(),
           suggestion.method_b.c_str(), suggestion.weight);
  }
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  Run();
  return 0;
}
