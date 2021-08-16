// Copyright 2021 The Google Research Authors.
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

#ifndef FAIRNESS_AND_BIAS_SELECTION_DISTRIBUTIONS_H_
#define FAIRNESS_AND_BIAS_SELECTION_DISTRIBUTIONS_H_

#include <random>
#include <vector>

#include "random_handler.h"

namespace fair_secretary {

// Provides statistics about Random, Uniform and Binomial distribution.
class RandomDistribution {
 public:
  virtual double Reverse(double x) { return 1.0; }
  virtual double Sample() { return 1.0; }
  virtual double Middle(int n) { return 1.0; }
  virtual double PThreshold(int index) { return 1.0; }
  virtual ~RandomDistribution() {}
};

class UniformDistribution : public RandomDistribution {
 public:
  double Reverse(double x) override;
  double Sample() override;
  double Middle(int n) override;
  void Init(int n, double range);
  double PThreshold(int index) override;

 private:
  double range_;
  std::vector<double> p_th_;
};

class BinomialDistribution : public RandomDistribution {
 public:
  double Reverse(double x) override;
  double Sample() override;
  double Middle(int n) override;
  void Init(int n, double p);
  void ComputeMaxDist(double num_dists);
  double PThreshold(int index) override;
  double Expected(double lower_bound);

 private:
  std::vector<std::vector<double>> choose_;
  std::vector<double> probability_;
  std::vector<double> r_probability_;
  std::vector<double> max_dist_;
  std::vector<double> p_th_;
  std::binomial_distribution<int> distribution_;
};

}  // namespace fair_secretary

#endif  // FAIRNESS_AND_BIAS_SELECTION_DISTRIBUTIONS_H_
