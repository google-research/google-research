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

#ifndef ELO_FAST_MATH_H_
#define ELO_FAST_MATH_H_

#include "Eigen/Core"
#include "Eigen/Dense"

constexpr static float kLog10 = 2.302585093;
constexpr static float kEloN = 400;

// Probability that a wins against b.
float WinProbability(float elo_a, float elo_b);

// log_e of the probability that a wins against b.
float LogWinProbability(float elo_a, float elo_b);

// Given a vector of ELO scores, computes the matrix of the logarithm of the
// winning probabilities.
Eigen::MatrixXd LogWinProbabilityMatrix(const Eigen::VectorXd& elos);

// Given a vector of ELO scores, computes the matrix of the
// winning probabilities.
Eigen::MatrixXd WinProbabilityMatrix(const Eigen::VectorXd& elos);

// Increments accum by the bernoulli entropy of p, scaled by weight.
void AccumulateH2(const Eigen::MatrixXd& p, double weight,
                  Eigen::MatrixXd& accum);

// Returns the sum of KL divergences between each entry, considered as a
// Bernoulli variable.
double KLDivergence(const Eigen::MatrixXd& true_p, const Eigen::MatrixXd& p);

#endif  // ELO_FAST_MATH_H_
