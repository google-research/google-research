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
//             Our algorithm
//
// The simple algorithm from Section 3 of the paper.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_DYNAMIC_SUBMODULAR_ALGORITHM_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_DYNAMIC_SUBMODULAR_ALGORITHM_H_

#include <unordered_set>
#include <vector>

#include "algorithm.h"
#include "utilities.h"

class OurSimpleAlgorithm : public Algorithm {
  // A single sub-algorithm, like for SieveStreaming.
  class OurSimpleAlgorithmSingleThreshold {
   public:
    OurSimpleAlgorithmSingleThreshold(int num_T, SubmodularFunction& sub_func_f,
                                      int cardinality_k, double gamma,
                                      double eps);
    double GetSolutionValue();
    std::vector<int> GetSolutionVector();
    void Insert(int element);
    void Erase(int element);

   private:
    // Number of thresholds.
    // Same at variable T in the paper.
    int num_T_;

    // Stores the elements that are not yet considered in the level construct.
    // Same as the B data structure used in the paper.
    std::vector<std::unordered_set<int>> buffer_B_;

    // Stores the elements that are not yet considered in the level construct.
    // Same as the A data structure used in the paper.
    std::vector<std::unordered_set<int>> levels_A_;

    // Stores the elements that are not yet considered in the level construct.
    // Same as the S data structure used in the paper.
    std::vector<std::unordered_set<int>> solutions_S_;

    // Size of the union of solutions_S_-sets.
    int size_of_S_;

    // Reference to a shared copy that every  method can reset, use, and leave
    // dirty.
    SubmodularFunction& sub_func_f_;

    // The lowest level where there was a deletion since the.
    // Last time levelConstruct() was run (-1: none).
    int lowest_level_;

    // Number of elements the algorithm picks (cardinality constraint).
    const int cardinality_k_;

    // One can think of it as an estimate for OPT, but it can be far away from
    // it. There exists an instance of SingleThresholdSieve so that gamma is
    // close to the optimum value. More details are provided in the paper.
    const double gamma_;

    // The epsilon used to set the threshold of S-value reduction.
    const double eps_;

    // Reconstruct starting from level l_begin.
    // The LevelConstruct algorithm explained in the paper.
    void LevelConstruct(int l_begin);
  };

 public:
  OurSimpleAlgorithm(double eps);
  void Init(const SubmodularFunction& sub_func_f, int cardinality_k);
  double GetSolutionValue();
  std::vector<int> GetSolutionVector();
  void Insert(int element);
  void Erase(int element);
  std::string GetAlgorithmName() const;

 private:
  // Vector of sub-algorithms.
  std::vector<OurSimpleAlgorithmSingleThreshold> singles_;

  // A shared copy that every method can reset, use, and leave dirty.
  std::unique_ptr<SubmodularFunction> sub_func_f_;

  // The epsilon used to set the threshold of S-value reduction.
  const double eps_;
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_DYNAMIC_SUBMODULAR_ALGORITHM_H_
