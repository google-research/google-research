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
// Clustering Oracle Function.
//

#ifndef FAIR_SUBMODULAR_MAXIMIZATION_2020_CLUSTERING_FUNCTION_H_
#define FAIR_SUBMODULAR_MAXIMIZATION_2020_CLUSTERING_FUNCTION_H_

#include "submodular_function.h"

class ClusteringFunction : public SubmodularFunction {
 public:
  // Resets the oracle by clearing the solution stored.
  void Reset();

  // Initiation function for this submodular function.
  std::vector<std::pair<int, int>> Init(std::string experiment_name = {});

  // Returns all the points in the universe.
  const std::vector<std::pair<int, int>>& GetUniverse() const;

  // Gets name of utility function.
  std::string GetName() const;

  // Clones the object (see e.g. GraphUtility for an example).
  std::unique_ptr<SubmodularFunction> Clone() const;

  // Gets the maximum value;
  double GetMaxValue();

  ~ClusteringFunction() {}

 protected:
  // Adds a new element to set S.
  void Add(std::pair<int, int> element);

  // Computes f(S u {e}) - f(S).
  double Delta(std::pair<int, int> element);

  // Returns the value of the given elements.
  double Objective(const std::vector<std::pair<int, int>>& elements) const;

 private:
  // Distance squared between two points.
  double distance(int x, int y) const;

  // Prepares the instance for bank experiment.
  std::vector<std::pair<int, int>> BankPrep();

  // Coordinate of input points.
  std::vector<std::vector<double>> input_;

  // Elements of the universe.
  std::vector<std::pair<int, int>> universe_;

  // Current solution
  std::vector<std::pair<int, int>> solution_;

  // The maximum possible solution value;
  double max_value_ = 0;
};

#endif  // FAIR_SUBMODULAR_MAXIMIZATION_2020_CLUSTERING_FUNCTION_H_
