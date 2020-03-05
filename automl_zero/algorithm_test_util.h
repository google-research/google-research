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

// Tools to test code that moves Algorithms around.

#ifndef ALGORITHM_TEST_UTIL_H_
#define ALGORITHM_TEST_UTIL_H_

#include "algorithm.h"
#include "algorithm.pb.h"
#include "testing/base/public/gmock.h"

namespace automl_zero {

// Convenience method to generate a Algorithm with an ID tag.
Algorithm DnaFromId(const IntegerT algorithm_id);
std::shared_ptr<const Algorithm> DnaSharedPtrFromId(
    const IntegerT algorithm_id);

// Convenience method to generate a Algorithm with an ID tag and serialize it.
SerializedAlgorithm SerializedAlgorithmFromId(const IntegerT algorithm_id);

// Convenience method to generate a Algorithm with an ID tag and serialize it
// as a string.
std::string BytesAlgorithmStringFromId(const IntegerT algorithm_id);

// Matcher to check the ID tag of a Algorithm.
MATCHER_P(DnaSharedPtrHasId, algorithm_id, "") {
  return arg->predict_[0]->GetIntegerData() == algorithm_id;
}

// Matcher to check the ID tag of a Algorithm.
MATCHER_P(DnaSharedPtrPtrHasId, algorithm_id, "") {
  return (*arg)->predict_[0]->GetIntegerData() == algorithm_id;
}

// Matcher to check the ID tag of a serialized Algorithm.
MATCHER_P(SerializedAlgorithmHasId, algorithm_id, "") {
  Algorithm algorithm;
  algorithm.FromProto(arg);
  return algorithm.predict_[0]->GetIntegerData() == algorithm_id;
}

// Method to change the ID tag of a Algorithm in place.
void SwitchDnaId(std::shared_ptr<const Algorithm>* algorithm,
                 const IntegerT new_algorithm_id);

// Method to increment the ID tag of a Algorithm in place.
void IncrementDnaId(std::shared_ptr<const Algorithm>* algorithm,
                    IntegerT by = 1);

// Counts how many different instructions there are in the setup
// component function between the two given Algorithm instances. Assumes both
// component functions are the same size.
IntegerT CountDifferentSetupInstructions(const Algorithm& algorithm1,
                                         const Algorithm& algorithm2);

// Counts how many different instructions there are in the predict
// component function between the two given Algorithm instances. Assumes both
// component functions are the same size.
IntegerT CountDifferentPredictInstructions(const Algorithm& algorithm1,
                                           const Algorithm& algorithm2);

// Counts how many different instructions there are in the learn
// component function between the two given Algorithm instances. Assumes both
// component functions are the same size.
IntegerT CountDifferentLearnInstructions(const Algorithm& algorithm1,
                                         const Algorithm& algorithm2);

// Counts how many different instructions there are between the two given
// Algorithm instances. Assumes matching component functions are the same size.
IntegerT CountDifferentInstructions(const Algorithm& algorithm1,
                                    const Algorithm& algorithm2);

// Returns the position of the instruction with SCALAR_SUM_OP in the given
// component function. Returns -1 if it doesn't appear in that
// component function and -2 if it appears multiple times.
IntegerT ScalarSumOpPosition(
    const std::vector<std::shared_ptr<const Instruction>>& component_function);

// Returns which component function is different (as a casted
// ComponentFunctionT integer). Returns -1 if none of the component functions
// are different and -2 if more than 1 is.
IntegerT DifferentComponentFunction(const Algorithm& algorithm1,
                                    const Algorithm& algorithm2);

// Assumes that each position in both component functions has integer data
// (one integer per position). Returns which integer is present in
// component function 1 that is not present in component function 2. Returns -1
// if none is missing and -2 if multiple are missing.
IntegerT MissingDataInComponentFunction(
    const std::vector<std::shared_ptr<const Instruction>>& component_function1,
    const std::vector<std::shared_ptr<const Instruction>>& component_function2);

}  // namespace automl_zero

#endif  // ALGORITHM_TEST_UTIL_H_
