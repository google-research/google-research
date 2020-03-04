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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_GENERATOR_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_GENERATOR_H_

#include <memory>
#include <random>

#include "algorithm.h"
#include "definitions.h"
#include "instruction.proto.h"
#include "generator.proto.h"
#include "instruction.h"
#include "randomizer.h"

namespace automl_zero {

class RandomGenerator;

constexpr double kDefaultLearningRate = 0.01;
constexpr double kDefaultInitScale = 0.1;

// A class to generate Algorithms.
class Generator {
 public:
  Generator(
      // The model used to initialize the population. See HardcodedAlgorithmID
      // enum. Used by TheInitModel() and ignored by other methods.
      HardcodedAlgorithmID init_model,
      // The sizes of the component_functions. Can be zero if only using
      // deterministic models without padding.
      IntegerT setup_size_init,
      IntegerT predict_size_init,
      IntegerT learn_size_init,
      // Ops that can be introduced into the setup component_function. Can be
      // empty if only deterministic models will be generated.
      const std::vector<Op>& allowed_setup_ops,
      // Ops that can be introduced into the predict component_function. Can be
      // empty if only deterministic models will be generated.
      const std::vector<Op>& allowed_predict_ops,
      // Ops that can be introduced into the learn component_function. Can be
      // empty if deterministic models will be generated.
      const std::vector<Op>& allowed_learn_ops,
      // Can be a nullptr if only deterministic models will be generated.
      std::mt19937* bit_gen,
      // Can be a nullptr if only deterministic models will be generated.
      RandomGenerator* rand_gen);
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  // Returns Algorithm for initialization.
  Algorithm TheInitModel();

  // Returns Algorithm of the given model type. This will be one of the ones
  // below.
  Algorithm ModelByID(HardcodedAlgorithmID model);

  // A Algorithm with no-op instructions.
  Algorithm NoOp();

  // Returns Algorithm with fixed-size component functions with random
  // instructions.
  Algorithm Random();

  // A linear model with learning by gradient descent.
  static constexpr AddressT LINEAR_ALGORITHMWeightsAddress = 1;
  Algorithm LinearModel(double learning_rate);

  // A 2-layer neural network with one nonlinearity, where both layers implement
  // learning by gradient descent. The weights are initialized randomly.
  Algorithm NeuralNet(
      double learning_rate, double first_init_scale, double final_init_scale);

  // A 2-layer neural network without bias and no learning.
  static constexpr AddressT
      kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress = 1;
  static constexpr AddressT
      kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress = 0;
  Algorithm UnitTestNeuralNetNoBiasNoGradient(const double learning_rate);

 private:
  friend Generator SimpleGenerator();

  // Used to create a simple generator for tests. See SimpleGenerator.
  Generator();

  const HardcodedAlgorithmID init_model_;
  const IntegerT setup_size_init_;
  const IntegerT predict_size_init_;
  const IntegerT learn_size_init_;
  const std::vector<Op> allowed_setup_ops_;
  const std::vector<Op> allowed_predict_ops_;
  const std::vector<Op> allowed_learn_ops_;
  std::unique_ptr<std::mt19937> bit_gen_owned_;
  std::unique_ptr<RandomGenerator> rand_gen_owned_;
  RandomGenerator* rand_gen_;
  Randomizer randomizer_;
  std::shared_ptr<const Instruction> no_op_instruction_;
};

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_GENERATOR_H_
