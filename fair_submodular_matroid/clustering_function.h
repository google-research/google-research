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

#ifndef FAIR_SUBMODULAR_MATROID_CLUSTERING_FUNCTION_H_
#define FAIR_SUBMODULAR_MATROID_CLUSTERING_FUNCTION_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "submodular_function.h"

class ClusteringFunction : public SubmodularFunction {
 public:
  explicit ClusteringFunction(const std::vector<std::vector<double>>& input);

  void Reset() override;

  void Add(int element) override;

  void Remove(int element) override;

  double RemoveAndIncreaseOracleCall(int element) override;

  const std::vector<int>& GetUniverse() const override;

  std::string GetName() const override;

  std::unique_ptr<SubmodularFunction> Clone() const override;

  // Gets the maximum value;
  double GetMaxValue();

  ~ClusteringFunction() override = default;

 protected:
  double Delta(int element) override;

  double RemovalDelta(int element) override;

  // F(S) = sum_{i in V} dist(i,-1) - min_{j in S U {-1}} dist(i, j)
  double Objective(const std::vector<int>& elements) const override;

 private:
  // current distances: min_dist[i] = dist(i,-1) U {dist(i, j) : j in S}
  std::vector<std::multiset<double, std::less<double>>> min_dist_;

  // Distance squared between two points, or to origin if y=-1
  double distance(int x, int y) const;

  // Coordinate of input points.
  std::vector<std::vector<double>> input_;

  // Elements of the universe.
  std::vector<int> universe_;

  // The maximum possible solution value;
  double max_value_ = 0;
};

#endif  // FAIR_SUBMODULAR_MATROID_CLUSTERING_FUNCTION_H_
