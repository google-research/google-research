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

//
// The main file for running experiments
//

// compile this as C++14 (or later)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dynamic_submodular_algorithm.h"
#include "graph.h"
#include "graph_utility.h"
#include "greedy_algorithm.h"
#include "sieve_streaming_algorithm.h"
#include "simple_greedy.h"
#include "submodular_function.h"
#include "utilities.h"

using std::ifstream;
using std::ostream;
using std::pair;
using std::unordered_set;
using std::vector;

template <class T>
ostream& operator<<(ostream& s, const vector<T>& t) {
  for (const auto& x : t) s << x << " ";
  return s;
}
template <class T>
ostream& operator<<(ostream& s, const unordered_set<T>& t) {
  for (const auto& x : t) s << x << " ";
  return s;
}
template <class T, class R>
ostream& operator<<(ostream& s, const pair<T, R>& p) {
  return s << p.first << ' ' << p.second;
}

//
//  Experiments
//
// An experiment is a function that takes parameters:
// - Submodular function
// - Algorithm
// - Any number of other parameters
// And returns the (average) objective function value.

// Window experiment:
// Process elements one by one (in the universe-order),
// at all times maintaining a window of size at most windowSize
// return the average objective value.
double windowExperiment(SubmodularFunction& sub_func_f, Algorithm& alg,
                        int windowSize) {
  vector<double> values;
  windowSize =
      std::min(windowSize, static_cast<int>(sub_func_f.GetUniverse().size()));
  for (int i = 0;
       i < static_cast<int>(sub_func_f.GetUniverse().size()) + windowSize;
       ++i) {
    // There are more elements to insert.
    if (i < static_cast<int>(sub_func_f.GetUniverse().size())) {
      alg.Insert(sub_func_f.GetUniverse()[i]);
    }
    // We should start deleting elements when the window reaches them.
    if (i >= windowSize) {
      alg.Erase(sub_func_f.GetUniverse()[i - windowSize]);
    }
    values.push_back(alg.GetSolutionValue());
  }
  // Returns average.
  return accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

// An experiment where we first insert elements as they come in the universe,
// then remove them largest-to-smallest and return the average objective value.
double insertInOrderThenDeleteLargeToSmall(SubmodularFunction& sub_func_f,
                                           Algorithm& alg) {
  // Sort elements from largest marginal value to smallest
  vector<pair<double, int>> sorter;
  for (int e : sub_func_f.GetUniverse()) {
    sorter.emplace_back(sub_func_f.DeltaAndIncreaseOracleCall(e), e);
    SubmodularFunction::oracle_calls_--;
  }
  sort(sorter.begin(), sorter.end());
  reverse(sorter.begin(), sorter.end());

  vector<double> values;

  // First insert elements in arbitrary order (the order they come in the
  // universe).
  for (int e : sub_func_f.GetUniverse()) {
    alg.Insert(e);
    const double val = alg.GetSolutionValue();
    values.push_back(val);
  }

  // Then delete elements in the order from largest to smallest.
  for (auto it : sorter) {
    alg.Erase(it.second);
    const double val = alg.GetSolutionValue();
    values.push_back(val);
  }

  // Return average.
  return accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

// Helper function that runs an experiment on a given submodular function,
// a set of algorithms, and a set of k-values.
template <typename Function, typename... Args>
void runExperimentForAlgorithms(
    const Function& experiment, SubmodularFunction& sub_func_f,
    const vector<std::reference_wrapper<Algorithm>>& algs,
    const vector<int>& cardinality_ks, Args... args) {
  std::cout << "submodular function: " << sub_func_f.GetName() << "\n";
  for (Algorithm& alg : algs) {
    std::cout << "now running " << alg.GetAlgorithmName() << "...\n";
    vector<double> valuesPerK;
    vector<int64_t> oracleCallsPerK;
    for (int cardinality_k : cardinality_ks) {
      std::cerr << "running k = " << cardinality_k << std::endl;
      // Reseed reproducible randomness.
      RandomHandler::generator_.seed();
      alg.Init(sub_func_f, cardinality_k);
      const int64_t oracleCallsAtStart = SubmodularFunction::oracle_calls_;

      double value = experiment(sub_func_f, alg, args...);
      // We could also pass k, but none of our experiments needs it
      // (that is, k was only needed to init the algorithm).

      valuesPerK.push_back(value);
      oracleCallsPerK.push_back(SubmodularFunction::oracle_calls_ -
                                oracleCallsAtStart);
      // Simple trick, just to reduce memory usage.
      alg.Init(sub_func_f, cardinality_k);
    }
    std::cout << "k f\n";
    for (int i = 0; i < static_cast<int>(cardinality_ks.size()); ++i) {
      std::cout << cardinality_ks[i] << " " << valuesPerK[i] << "\n";
    }
    std::cout << std::endl;
    std::cout << "k OC\n";
    for (int i = 0; i < static_cast<int>(cardinality_ks.size()); ++i) {
      std::cout << cardinality_ks[i] << " " << oracleCallsPerK[i] << "\n";
    }
    std::cout << std::endl;
  }
}

int main() {
  RandomHandler::CheckRandomNumberGenerator();
  SieveStreaming sieveStreaming;
  SimpleGreedy simpleGreedy;
  Greedy greedy;
  OurSimpleAlgorithm ourSimpleAlgorithmEps00(0.0);
  OurSimpleAlgorithm ourSimpleAlgorithmEps02(0.2);

  GraphUtility f_pokec("pokec");
  // Potential values of cardinality constraint that can be used.
  const vector<int> from10to200 = {10,  20,  30,  40,  50,  60,  70,
                                   80,  90,  100, 110, 120, 130, 140,
                                   150, 160, 170, 180, 190, 200};
  // Potential values of cardinality constraint that can be used.
  const vector<int> from20to200 = {20,  40,  60,  80,  100,
                                   120, 140, 160, 180, 200};

  std::cout << "window experiment\n";
  for (int windowSize : {2000000, 1300000}) {
    std::cout << "window size = " << windowSize << std::endl;
    runExperimentForAlgorithms(
        windowExperiment, f_pokec,
        {ourSimpleAlgorithmEps00, ourSimpleAlgorithmEps02, sieveStreaming},
        from10to200, windowSize);
  }
}
