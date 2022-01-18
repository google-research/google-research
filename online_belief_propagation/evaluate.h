// Copyright 2022 The Google Research Authors.
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

#ifndef ONLINE_BELIEF_PROPAGATION_EVALUATE_H_
#define ONLINE_BELIEF_PROPAGATION_EVALUATE_H_

#include <vector>

// Abstract base class used to evaluate the output of a label propagation
// algorithm. Any specific evaluation function should be inherited from it.
template <typename... AdditionalArgTypes>
class Evaluate {
 public:
  virtual ~Evaluate() {}
  // clusters should be the output of a label propagation algorithm, generated
  // by LabelPropagationAlgorithm::GenerateClusters. additional_args can be
  // defined to be any additional arguments required by the evaluation function,
  // such as the groud truth communities or the graph itself.
  virtual double operator()(const std::vector<int>& clusters,
                            AdditionalArgTypes... additional_args) = 0;
};

#endif  // ONLINE_BELIEF_PROPAGATION_EVALUATE_H_
