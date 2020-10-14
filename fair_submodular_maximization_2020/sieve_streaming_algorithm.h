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

//
//  Sieve Streaming Algorithm
//
// The Sieve-Streaming algorithm of Badanidiyuru et al. a faithful simulation:
//

#ifndef FAIR_SUBMODULAR_MAXIMIZATION_2020_SIEVE_STREAMING_ALGORITHM_H_
#define FAIR_SUBMODULAR_MAXIMIZATION_2020_SIEVE_STREAMING_ALGORITHM_H_

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "algorithm.h"
#include "utilities.h"

// Implements the virtual functions of class Algorithm.
class SieveStreaming : public Algorithm {
 public:
  void Init(SubmodularFunction& sub_func_f,
            std::vector<std::pair<int, int>> bounds, int cardinality_k);
  double GetSolutionValue();
  std::vector<std::pair<int, int>> GetSolutionVector();
  void Insert(std::pair<int, int> element, bool non_monotone = false);
  std::string GetAlgorithmName() const;

  // Sieve-Streaming uses multiple sub-algorithms, each of a single guess of OPT
  // this class holds the state of one such algorithm.
  class SingleThresholdSieve {
   public:
    SingleThresholdSieve(const SubmodularFunction& sub_func_f,
                         int cardinality_k, double gamma);

    // Process the arrival of a new element e on the stream.
    void Process(std::pair<int, int> element);
    void Reset();
    bool IsInSolution(std::pair<int, int> element) const;
    double GetSolutionValue() const;
    std::vector<std::pair<int, int>> GetSolutionVector() const;

   private:
    std::unique_ptr<SubmodularFunction> sub_func_f_;
    const int cardinality_k_;
    std::vector<std::pair<int, int>> solution_;

    // obj_val[i] = f(solution[0..i-1]).
    std::vector<double> obj_vals_;

    // One can think of it as an estimate for OPT, but it can be far away from
    // it. There exists an instance of SingleThresholdSieve so that gamma is
    // close to the optimum value. More details are provided in the paper.
    const double gamma_;
  };

 private:
  // A vector of sub-algorithms.
  std::vector<SingleThresholdSieve> sieves_;
  static constexpr int deletedElement = -1;
};

#endif  // FAIR_SUBMODULAR_MAXIMIZATION_2020_SIEVE_STREAMING_ALGORITHM_H_
