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

#include "algorithm.h"

#include <sstream>
#include <string>
#include <vector>

#include "definitions.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/flags/flag.h"

namespace automl_zero {

using ::std::istringstream;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::ostream;  // NOLINT
using ::std::ostringstream;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::string;  // NOLINT
using ::std::stringstream;  // NOLINT
using ::std::vector;  // NOLINT

Algorithm::Algorithm(const SerializedAlgorithm& checkpoint_algorithm) {
  this->FromProto(checkpoint_algorithm);
}

inline void ShallowCopyComponentFunction(
    const vector<shared_ptr<const Instruction>>& src,
    vector<shared_ptr<const Instruction>>* dest) {
  dest->reserve(src.size());
  dest->clear();
  for (const shared_ptr<const Instruction>& src_instr : src) {
    dest->emplace_back(src_instr);
  }
}

Algorithm::Algorithm(const Algorithm& other) {
  ShallowCopyComponentFunction(other.setup_, &this->setup_);
  ShallowCopyComponentFunction(other.predict_, &this->predict_);
  ShallowCopyComponentFunction(other.learn_, &this->learn_);
}

Algorithm& Algorithm::operator=(const Algorithm& other) {
  if (&other != this) {
    ShallowCopyComponentFunction(other.setup_, &this->setup_);
    ShallowCopyComponentFunction(other.predict_, &this->predict_);
    ShallowCopyComponentFunction(other.learn_, &this->learn_);
  }
  return *this;
}

Algorithm::Algorithm(Algorithm&& other) {
  setup_ = std::move(other.setup_);
  predict_ = std::move(other.predict_);
  learn_ = std::move(other.learn_);
}

Algorithm& Algorithm::operator=(Algorithm&& other) {
  if (&other != this) {
    setup_ = std::move(other.setup_);
    predict_ = std::move(other.predict_);
    learn_ = std::move(other.learn_);
  }
  return *this;
}

inline bool IsComponentFunctionEqual(
    const vector<shared_ptr<const Instruction>>& component_function1,
    const vector<shared_ptr<const Instruction>>& component_function2) {
  if (component_function1.size() != component_function2.size()) {
    return false;
  }
  vector<shared_ptr<const Instruction>>::const_iterator instruction1_it =
      component_function1.begin();
  for (const shared_ptr<const Instruction>& instruction2 :
       component_function2) {
    if (*instruction2 != **instruction1_it) return false;
    ++instruction1_it;
  }
  CHECK(instruction1_it == component_function1.end());
  return true;
}

bool Algorithm::operator==(const Algorithm& other) const {
  if (!IsComponentFunctionEqual(setup_, other.setup_)) return false;
  if (!IsComponentFunctionEqual(predict_, other.predict_)) return false;
  if (!IsComponentFunctionEqual(learn_, other.learn_)) return false;
  return true;
}

string Algorithm::ToReadable() const {
  ostringstream stream;
  stream << "\n### Start component function. ###\n" << std::endl;
  stream << "# s is a list of scalars." << std::endl;
  stream << "# v is a list of vectors." << std::endl;
  stream << "# m is a list of matrices.\n" << std::endl;
  stream << "def setup(s, v, m):" << std::endl;
  for (const shared_ptr<const Instruction>& instruction : setup_) {
    stream << instruction->ToString();
  }
  stream << std::endl;

  stream << "def predict(s, v, m, features):" << std::endl;
  stream << "  v[" << kFeaturesVectorAddress << "] = features" << std::endl;
  stream << "  s[" << kLabelsScalarAddress << "] = 0.0" << std::endl;
  for (const shared_ptr<const Instruction>& instruction : predict_) {
    stream << instruction->ToString();
  }
  stream << "  scalar_pred = s[" << kPredictionsScalarAddress << "]"
         << std::endl;
  stream << "  vector_pred = v[" << kPredictionsVectorAddress << "]"
         << std::endl;
  stream << "  return scalar_pred, vector_pred\n" << std::endl;

  stream << "def learn(s, v, m, features, label):" << std::endl;
  stream << "  v[" << kFeaturesVectorAddress << "] = features" << std::endl;
  stream << "  s[" << kLabelsScalarAddress << "] = label" << std::endl;
  for (const shared_ptr<const Instruction>& instruction : learn_) {
    stream << instruction->ToString();
  }
  stream << std::endl;
  stream << "### End component function. ###\n" << std::endl;
  return stream.str();
}

SerializedAlgorithm Algorithm::ToProto() const {
  SerializedAlgorithm checkpoint_algorithm;
  for (const shared_ptr<const Instruction>& instr : setup_) {
    *checkpoint_algorithm.add_setup_instructions() = instr->Serialize();
  }
  for (const shared_ptr<const Instruction>& instr : predict_) {
    *checkpoint_algorithm.add_predict_instructions() = instr->Serialize();
  }
  for (const shared_ptr<const Instruction>& instr : learn_) {
    *checkpoint_algorithm.add_learn_instructions() = instr->Serialize();
  }
  return checkpoint_algorithm;
}

void Algorithm::FromProto(const SerializedAlgorithm& checkpoint_algorithm) {
  setup_.reserve(checkpoint_algorithm.setup_instructions_size());
  setup_.clear();
  for (const SerializedInstruction& checkpoint_instruction :
       checkpoint_algorithm.setup_instructions()) {
    setup_.emplace_back(
        make_shared<const Instruction>(checkpoint_instruction));
  }

  predict_.reserve(checkpoint_algorithm.predict_instructions_size());
  predict_.clear();
  for (const SerializedInstruction& checkpoint_instruction :
       checkpoint_algorithm.predict_instructions()) {
    predict_.emplace_back(
        make_shared<const Instruction>(checkpoint_instruction));
  }

  learn_.reserve(checkpoint_algorithm.learn_instructions_size());
  learn_.clear();
  for (const SerializedInstruction& checkpoint_instruction :
       checkpoint_algorithm.learn_instructions()) {
    learn_.emplace_back(
        make_shared<const Instruction>(checkpoint_instruction));
  }
}

const vector<shared_ptr<const Instruction>>& Algorithm::ComponentFunction(
    const ComponentFunctionT component_function_type) const {
  switch (component_function_type) {
    case kSetupComponentFunction:
      return setup_;
    case kPredictComponentFunction:
      return predict_;
    case kLearnComponentFunction:
      return learn_;
  }
}

vector<shared_ptr<const Instruction>>* Algorithm::MutableComponentFunction(
    const ComponentFunctionT component_function_type) {
  switch (component_function_type) {
    case kSetupComponentFunction:
      return &setup_;
    case kPredictComponentFunction:
      return &predict_;
    case kLearnComponentFunction:
      return &learn_;
  }
}

}  // namespace automl_zero
