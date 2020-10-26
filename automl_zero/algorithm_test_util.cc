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

#include "algorithm_test_util.h"

#include <memory>
#include <unordered_set>

#include "absl/container/node_hash_set.h"
#include "absl/memory/memory.h"
#include "algorithm.h"
#include "definitions.h"
#include "generator_test_util.h"
#include "instruction.h"

namespace automl_zero {

using ::absl::make_unique;
using ::std::make_shared;
using ::std::shared_ptr;
using ::std::string;
using ::std::vector;

Algorithm DnaFromId(const IntegerT algorithm_id) {
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.predict_[0] =
      make_shared<const Instruction>(IntegerDataSetter(algorithm_id));
  return algorithm;
}

shared_ptr<const Algorithm> DnaSharedPtrFromId(const IntegerT algorithm_id) {
  return make_shared<const Algorithm>(DnaFromId(algorithm_id));
}

SerializedAlgorithm SerializedAlgorithmFromId(const IntegerT algorithm_id) {
  return DnaFromId(algorithm_id).ToProto();
}

void SwitchDnaId(shared_ptr<const Algorithm>* algorithm,
                 const IntegerT new_algorithm_id) {
  auto mutable_algorithm = make_unique<Algorithm>(**algorithm);
  mutable_algorithm->predict_[0] =
      make_shared<const Instruction>(IntegerDataSetter(new_algorithm_id));
  algorithm->reset(mutable_algorithm.release());
}

void IncrementDnaId(shared_ptr<const Algorithm>* algorithm, const IntegerT by) {
  auto mutable_algorithm = make_unique<Algorithm>(**algorithm);
  const IntegerT algorithm_id =
      mutable_algorithm->predict_[0]->GetIntegerData();
  mutable_algorithm->predict_[0] =
      make_shared<const Instruction>(IntegerDataSetter(algorithm_id + by));
  algorithm->reset(mutable_algorithm.release());
}

IntegerT CountDifferentSetupInstructions(const Algorithm& algorithm1,
                                         const Algorithm& algorithm2) {
  IntegerT num_diff_instructions = 0;
  vector<shared_ptr<const Instruction>>::const_iterator instruction1_it =
      algorithm1.setup_.begin();
  for (const shared_ptr<const Instruction>& instruction2 : algorithm2.setup_) {
    if (*instruction2 != **instruction1_it) {
      ++num_diff_instructions;
    }
    ++instruction1_it;
  }
  CHECK(instruction1_it == algorithm1.setup_.end());
  return num_diff_instructions;
}

IntegerT CountDifferentPredictInstructions(const Algorithm& algorithm1,
                                           const Algorithm& algorithm2) {
  IntegerT num_diff_instructions = 0;
  vector<shared_ptr<const Instruction>>::const_iterator instruction1_it =
      algorithm1.predict_.begin();
  for (const shared_ptr<const Instruction>& instruction2 :
       algorithm2.predict_) {
    if (*instruction2 != **instruction1_it) {
      ++num_diff_instructions;
    }
    ++instruction1_it;
  }
  CHECK(instruction1_it == algorithm1.predict_.end());
  return num_diff_instructions;
}

IntegerT CountDifferentLearnInstructions(const Algorithm& algorithm1,
                                         const Algorithm& algorithm2) {
  IntegerT num_diff_instructions = 0;
  vector<shared_ptr<const Instruction>>::const_iterator instruction1_it =
      algorithm1.learn_.begin();
  for (const shared_ptr<const Instruction>& instruction2 : algorithm2.learn_) {
    if (*instruction2 != **instruction1_it) {
      ++num_diff_instructions;
    }
    ++instruction1_it;
  }
  CHECK(instruction1_it == algorithm1.learn_.end());
  return num_diff_instructions;
}

IntegerT CountDifferentInstructions(const Algorithm& algorithm1,
                                    const Algorithm& algorithm2) {
  return CountDifferentSetupInstructions(algorithm1, algorithm2) +
         CountDifferentPredictInstructions(algorithm1, algorithm2) +
         CountDifferentLearnInstructions(algorithm1, algorithm2);
}

IntegerT ScalarSumOpPosition(
    const vector<shared_ptr<const Instruction>>& component_function) {
  vector<IntegerT> positions;
  for (IntegerT position = 0;
       position < component_function.size();
       ++position) {
    if (component_function[position]->op_ == SCALAR_SUM_OP) {
      positions.push_back(position);
    }
  }
  if (positions.empty()) {
    return -1;
  } else if (positions.size() == 1) {
    return positions[0];
  } else {
    return -2;
  }
}

IntegerT DifferentComponentFunction(const Algorithm& algorithm1,
                                    const Algorithm& algorithm2) {
  vector<IntegerT> component_functions;
  if (algorithm1.setup_.size() != algorithm2.setup_.size() ||
      CountDifferentSetupInstructions(algorithm1, algorithm2) > 0) {
    component_functions.push_back(kSetupComponentFunction);
  }
  if (algorithm1.predict_.size() != algorithm2.predict_.size() ||
      CountDifferentPredictInstructions(algorithm1, algorithm2) > 0) {
    component_functions.push_back(kPredictComponentFunction);
  }
  if (algorithm1.learn_.size() != algorithm2.learn_.size() ||
      CountDifferentLearnInstructions(algorithm1, algorithm2) > 0) {
    component_functions.push_back(kLearnComponentFunction);
  }
  if (component_functions.empty()) {
    return -1;
  } else if (component_functions.size() == 1) {
    return component_functions[0];
  } else {
    return -2;
  }
}

IntegerT MissingDataInComponentFunction(
    const vector<shared_ptr<const Instruction>>& component_function1,
    const vector<shared_ptr<const Instruction>>& component_function2) {
  absl::node_hash_set<IntegerT> data2;
  for (const shared_ptr<const Instruction>& instruction : component_function2) {
    data2.insert(instruction->GetIntegerData());
  }
  vector<IntegerT> missing;
  for (const shared_ptr<const Instruction>& instruction : component_function1) {
    const IntegerT data1_value = instruction->GetIntegerData();
    if (data2.find(data1_value) == data2.end()) {
      missing.insert(missing.end(), data1_value);
    }
  }
  if (missing.empty()) {
    return -1;
  } else if (missing.size() == 1) {
    return missing[0];
  } else {
    return -2;
  }
}

}  // namespace automl_zero
