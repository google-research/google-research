// Copyright 2020 The Authors.
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
#include "algorithm.h"
#include "clustering_function.h"
#include "fair_algorithm.h"
#include "fair_algorithm_ck.h"
#include "fair_greedy_algorithm.h"
#include "fair_random_subset_algorithm.h"
#include "greedy_algorithm.h"
#include "matroid_algorithm.h"
#include "random_subset_algorithm.h"
#include "sieve_streaming_algorithm.h"
#include "submodular_function.h"

// GetSolutionValue() should always be called once, before GetSolutionVector()
void BankExperiment() {
  for (int k = 10; k <= 70; k += 10) {
    ClusteringFunction oracle;
    auto bounds = oracle.Init("bank");
    // Fixing the parameters.
    for (auto& bound : bounds) {
      bound.first = k / 10;
      bound.second = k / 5;
    }
    // Running all the algorithms. For the random algorithms, we repeat 20
    // times.
    SieveStreaming sieve;
    FairAlgorithm our_algo;
    RandomSubsetAlgorithm random;
    Greedy greedy;
    FairGreedy fair_greedy;
    FairRandomSubsetAlgorithm fair_random;
    MatroidAlgorithm matroid_algorithm;
    FairAlgorithmCK fa_kale;
    std::vector<std::reference_wrapper<Algorithm>> algorithms = {
        fa_kale, our_algo,    sieve,  matroid_algorithm,
        random,  fair_random, greedy, fair_greedy};
    for (Algorithm& alg : algorithms) {
      SubmodularFunction::oracle_calls_ = 0;
      std::vector<double> costs;
      std::vector<int> errors;
      std::cerr << "Now running " << alg.GetAlgorithmName() << "...\n";
      alg.Init(oracle, bounds, k);
      auto universe = oracle.GetUniverse();
      for (int i = 0; i < universe.size(); i++) {
        alg.Insert(universe[i]);
      }
      // In case of clustering oracle, the clustering cost is
      // oracle.GetMaxValue() - solution_value
      double solution_value = alg.GetSolutionValue();
      std::cerr << "Cost: " << solution_value << " "
                << oracle.GetMaxValue() - solution_value << std::endl;
      costs.push_back(solution_value);
      auto solution = alg.GetSolutionVector();
      std::vector<int> occurance(6, 0);
      for (int i = 0; i < solution.size(); i++) {
        occurance[solution[i].second]++;
      }
      int error = 0;
      std::cerr << "Color distribution: ";
      for (int i = 0; i < occurance.size(); i++) {
        std::cerr << occurance[i] << " ";
        error += std::max(0, occurance[i] - bounds[i].second);
        error += std::max(0, -occurance[i] + bounds[i].first);
      }
      std::cerr << std::endl << "error :" << error << std::endl << std::endl;
      errors.push_back(error);
      std::cout << alg.GetAlgorithmName() << " " << solution_value << " "
                << oracle.GetMaxValue() - solution_value << " " << error << " "
                << SubmodularFunction::oracle_calls_ << std::endl;
    }
  }
}

int main(int argc, char* argv[]) {
  BankExperiment();
  return 0;
}
