// Copyright 2023 The Authors.
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

// Experiments for fair submodular maximization.

#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "algorithm.h"
#include "bank_data.h"
#include "better_greedy_algorithm.h"
#include "clustering_function.h"
#include "fairness_constraint.h"
#include "graph.h"
#include "graph_utility.h"
#include "laminar_matroid.h"
#include "matroid_intersection.h"
#include "movies_data.h"
#include "movies_mixed_utility_function.h"
#include "partition_matroid.h"
#include "submodular_function.h"
#include "two_pass_algorithm_with_conditioned_matroid.h"
#include "utilities.h"

std::vector<int> Range(int n) {
  std::vector<int> range(n);
  for (int i = 0; i < n; ++i) {
    range[i] = i;
  }
  return range;
}

void PrintSolutionVector(const std::vector<int>& solution, std::ofstream& fout,
                         absl::string_view alg_name, const int rank,
                         bool verbose = true) {
  if (verbose) {
    fout << "Solution for " << alg_name << " for rank = " << rank << std::endl;
  }
  for (auto& p : solution) {
    fout << p + 1 << " ";  // add 1 for matlab indexing
  }
  fout << std::endl;
}

bool FeasibleSolutionExists(std::unique_ptr<Matroid>& matroid,
                            FairnessConstraint& fairness,
                            const std::vector<int>& universe) {
  MaxIntersection(matroid.get(), fairness.LowerBoundsToMatroid().get(),
                  universe);
  std::vector<int> solution = matroid->GetCurrent();
  std::cout << "Is feasible: " << matroid->IsFeasible(solution) << " "
            << fairness.IsFeasible(solution) << std::endl;
  return fairness.IsFeasible(solution);
}

// GetSolutionValue() should always be called once, before GetSolutionVector()
void SingleKBaseExperiment(
    SubmodularFunction& f, const int rank,
    const std::unique_ptr<Matroid>& matroid, const FairnessConstraint& fairness,
    const std::vector<std::reference_wrapper<Algorithm>>& algorithms,
    std::vector<std::ofstream>& result_files,
    std::vector<std::ofstream>& solutions_files,
    std::ofstream& general_log_file) {
  int number_of_colors = fairness.GetColorNum();
  // Fixing the parameters.
  // Running all the algorithms. For the random algorithms, we repeat 20 times.
  for (int idx = 0; idx < algorithms.size(); idx++) {
    // Reset seed for fair comparison
    RandomHandler::generator_.seed(1);

    Algorithm& alg = algorithms[idx];
    std::ofstream& of = result_files[idx];

    of << rank << " ";
    int num_rep = 1;
    // For the random algorithm, we repeat 20 times.
    if (alg.GetAlgorithmName() == "Totally random algorithm" ||
        alg.GetAlgorithmName() == "Fair random algorithm") {
      num_rep = 20;
    }

    std::vector<double> values;
    std::vector<int> errors;
    std::vector<double> lb_ratios;
    std::vector<int> solution;

    for (int j = 0; j < num_rep; j++) {
      std::cout << "Now running " << alg.GetAlgorithmName()
                << " with rank=" << rank << "...\n";
      alg.Init(f, fairness, *matroid);
      const std::vector<int>& universe = f.GetUniverse();
      for (int i = 0; i < universe.size(); i++) {
        alg.Insert(universe[i]);
      }
      double solution_value = alg.GetSolutionValue();
      std::cout << "Value: " << solution_value << std::endl;
      values.push_back(solution_value);
      solution = alg.GetSolutionVector();
      if (num_rep == 1) {
        PrintSolutionVector(solution, general_log_file, alg.GetAlgorithmName(),
                            rank);
      }
      PrintSolutionVector(solution, solutions_files[idx],
                          alg.GetAlgorithmName(), rank, false);

      std::vector<int> occurance(number_of_colors, 0);
      std::vector<std::pair<int, int>> bounds = fairness.GetBounds();
      for (int i = 0; i < solution.size(); i++) {
        occurance[fairness.GetColor(solution[i])]++;
      }
      int error = 0;
      double ratio = 1;
      std::cout << "Color distribution: ";
      for (int i = 0; i < occurance.size(); i++) {
        std::cout << occurance[i] << " ";
        error += std::max(0, occurance[i] - bounds[i].second);
        error += std::max(0, -occurance[i] + bounds[i].first);
        ratio = std::min(
            ratio, static_cast<double>(occurance[i]) / (bounds[i].first / 2));
      }
      std::cout << std::endl << "error :" << error << std::endl << std::endl;
      std::cout << "worst lower bound ratio :" << ratio << std::endl
                << std::endl;
      errors.push_back(error);
      lb_ratios.push_back(ratio);
    }
    // Computing average and variance.
    if (num_rep != 1) {
      double average_value = 0., average_error = 0., average_ratio = 0.;
      double var_value = 0., var_error = 0., var_ratio = 0.;
      for (auto& value : values) average_value += value;
      for (auto& error : errors) average_error += error;
      for (auto& ratio : lb_ratios) average_ratio += ratio;
      average_value /= values.size();
      average_error /= errors.size();
      average_ratio /= lb_ratios.size();
      for (auto& value : values)
        var_value += (value - average_value) * (value - average_value);
      for (auto& error : errors)
        var_error += (error - average_error) * (error - average_error);
      for (auto& ratio : lb_ratios)
        var_ratio += (ratio - average_ratio) * (ratio - average_ratio);
      std::cout << "Average value: " << average_value << " "
                << sqrt(var_value / (values.size() - 1)) << std::endl;
      of << average_value << " ";
      general_log_file << "Variance in % for rank = " << rank << " "
                       << sqrt(var_value / (values.size() - 1)) / average_value
                       << std::endl;
      std::cout << "Average error: " << average_error << " "
                << sqrt(var_error / (errors.size() - 1)) << std::endl;
      of << average_error << " ";
      std::cout << "Average ratio: " << average_ratio << " "
                << sqrt(var_ratio / (lb_ratios.size() - 1)) << std::endl;
      of << average_ratio << " ";
    } else {
      of << values[0] << " ";
      of << errors[0] << " ";
      of << lb_ratios[0] << " ";
    }
    of << f.oracle_calls_ << std::endl;
    f.oracle_calls_ = 0;
  }
}

void BaseExperiment(SubmodularFunction& f, std::vector<int>& ranks,
                    std::vector<std::unique_ptr<Matroid>>& matroids,
                    std::vector<FairnessConstraint>& fairness,
                    absl::string_view exp_name) {
  // Algorithms to run
  TwoPassAlgorithmWithConditionedMatroid two_pass;
  BetterGreedyAlgorithm greedy;

  std::vector<std::reference_wrapper<Algorithm>> algorithms = {greedy,
                                                               two_pass};

  // Create files to output results and save solution sets.
  std::vector<std::ofstream> result_files;
  std::vector<std::ofstream> solutions_files;
  std::string exp_base_path = "";  // Add path to dataset here (see README).

  for (Algorithm& alg : algorithms) {
    result_files.emplace_back(
        absl::StrCat(exp_base_path, "_", alg.GetAlgorithmName(), ".txt"));
    result_files.back() << "rank f error ratio OC" << std::endl;
    solutions_files.emplace_back(
        absl::StrCat(exp_base_path, "_sols_", alg.GetAlgorithmName(), ".txt"));
  }
  std::ofstream general_log_file(exp_base_path + "_general.txt");

  for (int i = 0; i < ranks.size(); i++) {
    // Skip rank if no feasible solution
    if (!FeasibleSolutionExists(matroids[i], fairness[i], f.GetUniverse())) {
      std::cerr << "No feasible solution for " << exp_name
                << " with rank = " << ranks[i] << std::endl;
      continue;
    }

    SingleKBaseExperiment(f, ranks[i], matroids[i], fairness[i], algorithms,
                          result_files, solutions_files, general_log_file);
  }
  for (std::ofstream& of : result_files) of.close();

  for (std::ofstream& of : solutions_files) of.close();

  general_log_file.close();
}

void ClusteringExperiment(int lower_i, int upper_i) {
  char input_path[] = "";  // Add input path here (see README).
  BankData data(input_path);
  ClusteringFunction f(data.input_);
  int ngrps = (int)data.balance_grpcards_.size();
  int ncolors = (int)data.age_grpcards_.size();

  std::vector<int> ranks;
  std::vector<std::unique_ptr<Matroid>> matroids;
  std::vector<FairnessConstraint> fairness;

  for (int i = 3; i <= 12; i++) {
    int rank = 5 * i;
    ranks.push_back(rank);
    std::vector<int> groups_bounds(ngrps, i);
    std::cout << "group bound: " << ngrps << " " << groups_bounds[0]
              << std::endl;
    matroids.emplace_back(
        new PartitionMatroid(data.balance_map_, groups_bounds));

    std::vector<std::pair<int, int>> color_bounds(ncolors, {i / 2 + 2, 2 * i});
    std::cout << "color bounds: " << color_bounds[0].first << " "
              << color_bounds[0].second << std::endl;
    fairness.emplace_back(data.age_map_, color_bounds);
  }
  std::cout << "ranks size " << ranks.size() << " " << ranks[0] << std::endl;
  BaseExperiment(f, ranks, matroids, fairness,
                 absl::StrCat("clustering", lower_i, "_", upper_i));
}

void CoverageExperiment(int lower_i, int upper_i) {
  Graph graph("pokec_age_BMI");  // "pokec_BMI_age"
  GraphUtility f(graph);
  int n = graph.GetUniverseVertices().size();
  std::cout << "n = " << n << std::endl;
  int ncolors = graph.GetColorsCards().size();
  std::cout << "ncolors = " << ncolors << std::endl;
  int ngrps = graph.GetGroupsCards().size();
  std::cout << "ngrps = " << ngrps << std::endl;

  std::vector<int> ranks;
  std::vector<std::unique_ptr<Matroid>> matroids;
  std::vector<FairnessConstraint> fairness;
  constexpr double lower_coeff = 0.9, upper_coeff = 1.5;

  for (int i = lower_i; i <= upper_i; i++) {
    int rank = 10 * i;
    std::cout << "rank = " << rank << std::endl;
    ranks.push_back(rank);
    double bound;
    std::vector<int> groups_bounds;
    std::cout << "group bounds:" << std::endl;
    for (int card : graph.GetGroupsCards()) {
      groups_bounds.emplace_back(rank * static_cast<double>(card) / n + 0.999);
      std::cout << groups_bounds.back() << std::endl;
    }
    matroids.emplace_back(
        new PartitionMatroid(graph.GetGroupsMap(), groups_bounds));

    int lower_bd;
    std::vector<std::pair<int, int>> color_bounds;
    std::cout << "color bounds:" << std::endl;
    for (int j = 0; j < ncolors; j++) {
      bound = rank * static_cast<double>(graph.GetColorsCards()[j]) / n;
      lower_bd = (int)(lower_coeff * bound + 0.001);
      color_bounds.emplace_back(lower_bd, upper_coeff * bound + 0.999);

      std::cout << " " << color_bounds.back().first << " "
                << color_bounds.back().second << std::endl;
    }
    fairness.emplace_back(graph.GetColorsMap(), color_bounds);
  }
  BaseExperiment(f, ranks, matroids, fairness,
                 absl::StrCat("coverage", lower_i, "_", upper_i));
}

void MovieExperiment(bool laminar) {
  MoviesMixedUtilityFunction f(444, 0.85);
  std::vector<int> ranks;
  std::vector<std::unique_ptr<Matroid>> matroids;
  std::vector<FairnessConstraint> fairness;
  for (int r = 10; r <= 200; r += 10) {
    // r is kinda like rank, but not really
    ranks.push_back(r);
    std::cerr << std::endl << std::endl << "r = " << r << std::endl;

    std::cerr << "group (matroid) bounds:";
    std::vector<int> groups_bounds;
    for (double p :
         MoviesData::GetInstance().GetMovieYearBandBoundPercentages()) {
      groups_bounds.push_back(r * p + 0.999);
      std::cerr << " " << groups_bounds.back();
    }
    if (!laminar) {
      matroids.emplace_back(new PartitionMatroid(
          MoviesData::GetInstance().GetMovieIdToYearBandMap(), groups_bounds));
    } else {
      const int noYearBands = groups_bounds.size();
      // small groups: 0 .. noYearBands-1
      // large groups: noYearBands onwards (one large group consists of L small
      // groups)
      constexpr int L = 3;
      absl::flat_hash_map<int, std::vector<int>> group_map;
      for (const auto& p :
           MoviesData::GetInstance().GetMovieIdToYearBandMap()) {
        // p == {element, its small group id}
        group_map[p.first] = {p.second, noYearBands + p.second / L};
      }
      std::cerr << " |";
      for (int gr = 0; gr < noYearBands; gr += L) {
        // make new large group
        int sumOfGroupBounds = 0;
        for (int g = gr; g < noYearBands && g < gr + L; ++g) {
          sumOfGroupBounds += groups_bounds[g];
        }
        groups_bounds.push_back(0.8 * sumOfGroupBounds + 0.999);
        std::cerr << " " << groups_bounds.back();
      }
      matroids.emplace_back(new LaminarMatroid(group_map, groups_bounds));
    }

    std::cerr << std::endl;

    std::cerr << "color bounds:";
    std::vector<std::pair<int, int>> color_bounds;
    for (const std::pair<double, double>& p :
         MoviesData::GetInstance().GetMovieGenreBoundPercentages()) {
      color_bounds.emplace_back(r * p.first + 0.001, r * p.second + 0.999);
      std::cerr << " " << color_bounds.back().first << "-"
                << color_bounds.back().second;
    }
    fairness.emplace_back(MoviesData::GetInstance().GetMovieIdToGenreIdMap(),
                          color_bounds);
    std::cerr << std::endl;
  }

  BaseExperiment(
      f, ranks, matroids, fairness,
      absl::StrCat("movies_exp_444_0.85", (laminar ? "_laminar" : "")));
}

int main() {
  ClusteringExperiment(1, 1);
  // CoverageExperiment(1,10);
  // MovieExperiment(false);
  // MovieExperiment(true);
}
