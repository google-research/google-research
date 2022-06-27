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

#include "randomizer.h"

#include <memory>

#include "algorithm.h"
#include "random_generator.h"

namespace automl_zero {

using ::std::make_shared;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::vector;  // NOLINT

Randomizer::Randomizer(
    vector<Op> allowed_setup_ops,
    vector<Op> allowed_predict_ops,
    vector<Op> allowed_learn_ops,
    mt19937* bit_gen,
    RandomGenerator* rand_gen)
    : allowed_setup_ops_(allowed_setup_ops),
      allowed_predict_ops_(allowed_predict_ops),
      allowed_learn_ops_(allowed_learn_ops),
      bit_gen_(bit_gen),
      rand_gen_(rand_gen) {}

void Randomizer::Randomize(Algorithm* algorithm) {
  if (!allowed_setup_ops_.empty()) {
    RandomizeSetup(algorithm);
  }
  if (!allowed_predict_ops_.empty()) {
    RandomizePredict(algorithm);
  }
  if (!allowed_learn_ops_.empty()) {
    RandomizeLearn(algorithm);
  }
}

void Randomizer::RandomizeSetup(Algorithm* algorithm) {
  for (shared_ptr<const Instruction>& instruction : algorithm->setup_) {
    instruction = make_shared<const Instruction>(SetupOp(), rand_gen_);
  }
}

void Randomizer::RandomizePredict(Algorithm* algorithm) {
  for (shared_ptr<const Instruction>& instruction : algorithm->predict_) {
    instruction = make_shared<const Instruction>(PredictOp(), rand_gen_);
  }
}

void Randomizer::RandomizeLearn(Algorithm* algorithm) {
  for (shared_ptr<const Instruction>& instruction : algorithm->learn_) {
    instruction = make_shared<const Instruction>(LearnOp(), rand_gen_);
  }
}

Op Randomizer::SetupOp() {
  IntegerT op_index = absl::Uniform<DeprecatedOpIndexT>(
      *bit_gen_, 0, allowed_setup_ops_.size());
  return allowed_setup_ops_[op_index];
}

Op Randomizer::PredictOp() {
  IntegerT op_index = absl::Uniform<DeprecatedOpIndexT>(
      *bit_gen_, 0, allowed_predict_ops_.size());
  return allowed_predict_ops_[op_index];
}

Op Randomizer::LearnOp() {
  IntegerT op_index = absl::Uniform<DeprecatedOpIndexT>(
      *bit_gen_, 0, allowed_learn_ops_.size());
  return allowed_learn_ops_[op_index];
}

}  // namespace automl_zero
