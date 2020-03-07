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

#include "generator_test_util.h"

#include "random_generator.h"
#include "absl/memory/memory.h"

namespace automl_zero {

using ::absl::make_unique;
using ::std::mt19937;  // NOLINT
using ::std::shared_ptr;
using ::std::vector;

Algorithm SimpleNoOpAlgorithm() {
  Generator generator(NO_OP_ALGORITHM,                         // Irrelevant.
                      6,                                  // setup_size_init
                      3,                                  // predict_size_init
                      9,                                  // learn_size_init
                      {}, {}, {}, nullptr, nullptr);  // Irrelevant.
  return generator.NoOp();
}

Algorithm SimpleRandomAlgorithm() {
  mt19937 bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Generator generator(
      RANDOM_ALGORITHM,  // Irrelevant.
      6,  // setup_size_init
      3,  // predict_size_init
      9,  // learn_size_init
      {SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      &bit_gen,
      &rand_gen);
  return generator.Random();
}

Algorithm SimpleGz() {
  Generator generator(NO_OP_ALGORITHM, 0, 0, 0, {}, {}, {}, nullptr, nullptr);
  return generator.LinearModel(kDefaultLearningRate);
}

Algorithm SimpleGrTildeGrWithBias() {
  Generator generator(NO_OP_ALGORITHM, 0, 0, 0, {}, {}, {}, nullptr, nullptr);
  return generator.NeuralNet(
      kDefaultLearningRate, kDefaultInitScale, kDefaultInitScale);
}

void SetIncreasingDataInComponentFunction(
    vector<shared_ptr<const Instruction>>* component_function) {
  for (IntegerT position = 0;
    position < component_function->size();
    ++position) {
    auto instruction =
        make_unique<Instruction>(*(*component_function)[position]);
    instruction->SetIntegerData(position);
    (*component_function)[position].reset(instruction.release());
  }
}

Algorithm SimpleIncreasingDataAlgorithm() {
  Algorithm algorithm = SimpleNoOpAlgorithm();
  SetIncreasingDataInComponentFunction(&algorithm.setup_);
  SetIncreasingDataInComponentFunction(&algorithm.predict_);
  SetIncreasingDataInComponentFunction(&algorithm.learn_);
  return algorithm;
}

}  // namespace automl_zero
