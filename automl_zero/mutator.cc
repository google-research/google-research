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

#include "mutator.h"

#include <memory>
#include <vector>

#include "definitions.h"
#include "random_generator.h"
#include "absl/memory/memory.h"

namespace automl_zero {

using ::absl::make_unique;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::vector;  // NOLINT

Mutator::Mutator(
    const MutationTypeList& allowed_actions,
    const double mutate_prob,
    const vector<Op>& allowed_setup_ops,
    const vector<Op>& allowed_predict_ops,
    const vector<Op>& allowed_learn_ops,
    const IntegerT setup_size_min,
    const IntegerT setup_size_max,
    const IntegerT predict_size_min,
    const IntegerT predict_size_max,
    const IntegerT learn_size_min,
    const IntegerT learn_size_max,
    mt19937* bit_gen,
    RandomGenerator* rand_gen)
    : allowed_actions_(allowed_actions),
      mutate_prob_(mutate_prob),
      allowed_setup_ops_(allowed_setup_ops),
      allowed_predict_ops_(allowed_predict_ops),
      allowed_learn_ops_(allowed_learn_ops),
      mutate_setup_(!allowed_setup_ops_.empty()),
      mutate_predict_(!allowed_predict_ops_.empty()),
      mutate_learn_(!allowed_learn_ops_.empty()),
      setup_size_min_(setup_size_min),
      setup_size_max_(setup_size_max),
      predict_size_min_(predict_size_min),
      predict_size_max_(predict_size_max),
      learn_size_min_(learn_size_min),
      learn_size_max_(learn_size_max),
      bit_gen_(bit_gen),
      rand_gen_(rand_gen),
      randomizer_(
          allowed_setup_ops_,
          allowed_predict_ops_,
          allowed_learn_ops_,
          bit_gen_,
          rand_gen_) {}

vector<MutationType> ConvertToMutationType(
    const vector<IntegerT>& mutation_actions_as_ints) {
  vector<MutationType> mutation_actions;
  mutation_actions.reserve(mutation_actions_as_ints.size());
  for (const IntegerT action_as_int : mutation_actions_as_ints) {
    mutation_actions.push_back(static_cast<MutationType>(action_as_int));
  }
  return mutation_actions;
}

void Mutator::Mutate(shared_ptr<const Algorithm>* algorithm) {
  if (mutate_prob_ >= 1.0 || rand_gen_->UniformProbability() < mutate_prob_) {
    auto mutated = make_unique<Algorithm>(**algorithm);
    MutateImpl(mutated.get());
    algorithm->reset(mutated.release());
  }
}

void Mutator::Mutate(const IntegerT num_mutations,
                     shared_ptr<const Algorithm>* algorithm) {
  if (mutate_prob_ >= 1.0 || rand_gen_->UniformProbability() < mutate_prob_) {
    auto mutated = make_unique<Algorithm>(**algorithm);
    for (IntegerT i = 0; i < num_mutations; ++i) {
      MutateImpl(mutated.get());
    }
    algorithm->reset(mutated.release());
  }
}

Mutator::Mutator()
    : allowed_actions_(ParseTextFormat<MutationTypeList>(
        "mutation_types: [ "
        "  ALTER_PARAM_MUTATION_TYPE, "
        "  RANDOMIZE_INSTRUCTION_MUTATION_TYPE, "
        "  RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE "
        "]")),
      mutate_prob_(0.5),
      allowed_setup_ops_(
          {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
      allowed_predict_ops_(
          {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
      allowed_learn_ops_(
          {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
      mutate_setup_(!allowed_setup_ops_.empty()),
      mutate_predict_(!allowed_predict_ops_.empty()),
      mutate_learn_(!allowed_learn_ops_.empty()),
      setup_size_min_(2),
      setup_size_max_(4),
      predict_size_min_(3),
      predict_size_max_(5),
      learn_size_min_(4),
      learn_size_max_(6),
      bit_gen_owned_(make_unique<mt19937>(GenerateRandomSeed())),
      bit_gen_(bit_gen_owned_.get()),
      rand_gen_owned_(make_unique<RandomGenerator>(bit_gen_)),
      rand_gen_(rand_gen_owned_.get()),
      randomizer_(
          allowed_setup_ops_,
          allowed_predict_ops_,
          allowed_learn_ops_,
          bit_gen_,
          rand_gen_) {}

void Mutator::MutateImpl(Algorithm* algorithm) {
  CHECK(!allowed_actions_.mutation_types().empty());
  const size_t action_index =
      absl::Uniform<size_t>(*bit_gen_, 0,
                            allowed_actions_.mutation_types_size());
  const MutationType action = allowed_actions_.mutation_types(action_index);
  switch (action) {
    case ALTER_PARAM_MUTATION_TYPE:
      AlterParam(algorithm);
      return;
    case RANDOMIZE_INSTRUCTION_MUTATION_TYPE:
      RandomizeInstruction(algorithm);
      return;
    case RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE:
      RandomizeComponentFunction(algorithm);
      return;
    case IDENTITY_MUTATION_TYPE:
      return;
    case INSERT_INSTRUCTION_MUTATION_TYPE:
      InsertInstruction(algorithm);
      return;
    case REMOVE_INSTRUCTION_MUTATION_TYPE:
      RemoveInstruction(algorithm);
      return;
    case TRADE_INSTRUCTION_MUTATION_TYPE:
      TradeInstruction(algorithm);
      return;
    case RANDOMIZE_ALGORITHM_MUTATION_TYPE:
      RandomizeAlgorithm(algorithm);
      return;
    // Do not add a default clause here. All actions should be supported.
  }
}

void Mutator::AlterParam(Algorithm* algorithm) {
  switch (ComponentFunction()) {
    case kSetupComponentFunction: {
      if (!algorithm->setup_.empty()) {
        InstructionIndexT index = InstructionIndex(algorithm->setup_.size());
        algorithm->setup_[index] =
            make_shared<const Instruction>(
                *algorithm->setup_[index], rand_gen_);
      }
      return;
    }
    case kPredictComponentFunction: {
      if (!algorithm->predict_.empty()) {
        InstructionIndexT index = InstructionIndex(algorithm->predict_.size());
        algorithm->predict_[index] =
            make_shared<const Instruction>(
                *algorithm->predict_[index], rand_gen_);
      }
      return;
    }
    case kLearnComponentFunction: {
      if (!algorithm->learn_.empty()) {
        InstructionIndexT index = InstructionIndex(algorithm->learn_.size());
        algorithm->learn_[index] =
            make_shared<const Instruction>(
                *algorithm->learn_[index], rand_gen_);
      }
      return;
    }
  }
  LOG(FATAL) << "Control flow should not reach here.";
}

void Mutator::RandomizeInstruction(Algorithm* algorithm) {
  switch (ComponentFunction()) {
    case kSetupComponentFunction: {
      if (!algorithm->setup_.empty()) {
        InstructionIndexT index = InstructionIndex(algorithm->setup_.size());
        algorithm->setup_[index] =
            make_shared<const Instruction>(SetupOp(), rand_gen_);
      }
      return;
    }
    case kPredictComponentFunction: {
      if (!algorithm->predict_.empty()) {
        InstructionIndexT index = InstructionIndex(algorithm->predict_.size());
        algorithm->predict_[index] =
            make_shared<const Instruction>(PredictOp(), rand_gen_);
      }
      return;
    }
    case kLearnComponentFunction: {
      if (!algorithm->learn_.empty()) {
        InstructionIndexT index = InstructionIndex(algorithm->learn_.size());
        algorithm->learn_[index] =
            make_shared<const Instruction>(LearnOp(), rand_gen_);
      }
      return;
    }
  }
  LOG(FATAL) << "Control flow should not reach here.";
}

void Mutator::RandomizeComponentFunction(Algorithm* algorithm) {
  switch (ComponentFunction()) {
    case kSetupComponentFunction: {
      randomizer_.RandomizeSetup(algorithm);
      return;
    }
    case kPredictComponentFunction: {
      randomizer_.RandomizePredict(algorithm);
      return;
    }
    case kLearnComponentFunction: {
      randomizer_.RandomizeLearn(algorithm);
      return;
    }
  }
  LOG(FATAL) << "Control flow should not reach here.";
}

void Mutator::InsertInstruction(Algorithm* algorithm) {
  Op op;  // Operation for the new instruction.
  vector<shared_ptr<const Instruction>>* component_function;  // To modify.
  switch (ComponentFunction()) {
    case kSetupComponentFunction: {
      if (algorithm->setup_.size() >= setup_size_max_ - 1) return;
      op = SetupOp();
      component_function = &algorithm->setup_;
      break;
    }
    case kPredictComponentFunction: {
      if (algorithm->predict_.size() >= predict_size_max_ - 1) return;
      op = PredictOp();
      component_function = &algorithm->predict_;
      break;
    }
    case kLearnComponentFunction: {
      if (algorithm->learn_.size() >= learn_size_max_ - 1) return;
      op = LearnOp();
      component_function = &algorithm->learn_;
      break;
    }
  }
  InsertInstructionUnconditionally(op, component_function);
}

void Mutator::RemoveInstruction(Algorithm* algorithm) {
  vector<shared_ptr<const Instruction>>* component_function;  // To modify.
  switch (ComponentFunction()) {
    case kSetupComponentFunction: {
      if (algorithm->setup_.size() <= setup_size_min_) return;
      component_function = &algorithm->setup_;
      break;
    }
    case kPredictComponentFunction: {
      if (algorithm->predict_.size() <= predict_size_min_) return;
      component_function = &algorithm->predict_;
      break;
    }
    case kLearnComponentFunction: {
      if (algorithm->learn_.size() <= learn_size_min_) return;
      component_function = &algorithm->learn_;
      break;
    }
  }
  RemoveInstructionUnconditionally(component_function);
}

void Mutator::TradeInstruction(Algorithm* algorithm) {
  Op op;  // Operation for the new instruction.
  vector<shared_ptr<const Instruction>>* component_function;  // To modify.
  switch (ComponentFunction()) {
    case kSetupComponentFunction: {
      op = SetupOp();
      component_function = &algorithm->setup_;
      break;
    }
    case kPredictComponentFunction: {
      op = PredictOp();
      component_function = &algorithm->predict_;
      break;
    }
    case kLearnComponentFunction: {
      op = LearnOp();
      component_function = &algorithm->learn_;
      break;
    }
  }
  InsertInstructionUnconditionally(op, component_function);
  RemoveInstructionUnconditionally(component_function);
}

void Mutator::RandomizeAlgorithm(Algorithm* algorithm) {
  if (mutate_setup_) {
    randomizer_.RandomizeSetup(algorithm);
  }
  if (mutate_predict_) {
    randomizer_.RandomizePredict(algorithm);
  }
  if (mutate_learn_) {
    randomizer_.RandomizeLearn(algorithm);
  }
}

void Mutator::InsertInstructionUnconditionally(
    const Op op, vector<shared_ptr<const Instruction>>* component_function) {
  const InstructionIndexT position =
      InstructionIndex(component_function->size() + 1);
  component_function->insert(
      component_function->begin() + position,
      make_shared<const Instruction>(op, rand_gen_));
}

void Mutator::RemoveInstructionUnconditionally(
    vector<shared_ptr<const Instruction>>* component_function) {
  CHECK_GT(component_function->size(), 0);
  const InstructionIndexT position =
      InstructionIndex(component_function->size());
  component_function->erase(component_function->begin() + position);
}

Op Mutator::SetupOp() {
  IntegerT op_index = absl::Uniform<DeprecatedOpIndexT>(
      *bit_gen_, 0, allowed_setup_ops_.size());
  return allowed_setup_ops_[op_index];
}

Op Mutator::PredictOp() {
  IntegerT op_index = absl::Uniform<DeprecatedOpIndexT>(
      *bit_gen_, 0, allowed_predict_ops_.size());
  return allowed_predict_ops_[op_index];
}

Op Mutator::LearnOp() {
  IntegerT op_index = absl::Uniform<DeprecatedOpIndexT>(
      *bit_gen_, 0, allowed_learn_ops_.size());
  return allowed_learn_ops_[op_index];
}

InstructionIndexT Mutator::InstructionIndex(
    const InstructionIndexT component_function_size) {
  return absl::Uniform<InstructionIndexT>(
      *bit_gen_, 0, component_function_size);
}

ComponentFunctionT Mutator::ComponentFunction() {
  vector<ComponentFunctionT> allowed_component_functions;
  allowed_component_functions.reserve(4);
  if (mutate_setup_) {
    allowed_component_functions.push_back(kSetupComponentFunction);
  }
  if (mutate_predict_) {
    allowed_component_functions.push_back(kPredictComponentFunction);
  }
  if (mutate_learn_) {
    allowed_component_functions.push_back(kLearnComponentFunction);
  }
  CHECK(!allowed_component_functions.empty())
      << "Must mutate at least one component function." << endl;
  const IntegerT index =
      absl::Uniform<IntegerT>(*bit_gen_, 0, allowed_component_functions.size());
  return allowed_component_functions[index];
}

}  // namespace automl_zero
