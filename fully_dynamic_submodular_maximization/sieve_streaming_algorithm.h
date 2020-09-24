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
//  Sieve Streaming Algorithm
//
// The Sieve-Streaming algorithm of Badanidiyuru et al. a faithful simulation:
// Whenever an element is deleted from the solution, the sieve is rerun.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_SIEVE_STREAMING_ALGORITHM_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_SIEVE_STREAMING_ALGORITHM_H_

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "algorithm.h"
#include "utilities.h"

// Implements the virtual functions of class Algorithm.
class SieveStreaming : public Algorithm {
 public:
  void Init(const SubmodularFunction& sub_func_f, int cardinality_k);
  double GetSolutionValue();
  std::vector<int> GetSolutionVector();
  void Insert(int element);
  void Erase(int element);
  std::string GetAlgorithmName() const;

  // Sieve-Streaming uses multiple sub-algorithms, each of a single guess of OPT
  // this class holds the state of one such algorithm.
  class SingleThresholdSieve {
   public:
    SingleThresholdSieve(const SubmodularFunction& sub_func_f,
                         int cardinality_k, double gamma);

    // Process the arrival of a new element e on the stream.
    void Process(int element);
    void Reset();
    bool IsInSolution(int element) const;
    double GetSolutionValue() const;
    std::vector<int> GetSolutionVector() const;

   private:
    std::unique_ptr<SubmodularFunction> sub_func_f_;
    const int cardinality_k_;
    std::vector<int> solution_;

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

  // What came on the stream if it has value = deletedElement, then its deleted
  // (maybe should be replaced by doubly-linked list, or just delete in linear
  // time) (one could also replace with unordered_set, but then we would lose
  // the ordering  and the simulation would no longer be faithful).
  std::vector<int> stream_;

  std::unordered_map<int, int> position_on_stream_;
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_SIEVE_STREAMING_ALGORITHM_H_
