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

#ifndef AUTOML_ZERO_INSTRUCTION_H_
#define AUTOML_ZERO_INSTRUCTION_H_

#include <math.h>

#include <cmath>
#include <cstring>

#include "algorithm.pb.h"
#include "definitions.h"
#include "random_generator.h"

namespace automl_zero {

// Used for unit tests.
constexpr double kActivationDataTolerance = 0.00001;
constexpr float kFloatDataTolerance = 0.00001;
constexpr double kVectorDataTolerance = 0.001;
constexpr double kMatrixRowDataTolerance = 0.001;

constexpr double kActivationMutationFixedScale = 0.1;
constexpr double kSignFlipProb = 0.1;

// Used in unit tests.
class IntegerDataSetter {
 public:
  explicit IntegerDataSetter(const IntegerT value) : value_(value) {}
  const IntegerT value_;
};

class ActivationDataSetter {
 public:
  explicit ActivationDataSetter(const double value) : value_(value) {}
  const double value_;
};

class FloatDataSetter {
 public:
  explicit FloatDataSetter(const float value) : value_(value) {}
  const float value_;
};

class IndexDataSetter {
 public:
  explicit IndexDataSetter(const FeatureIndexT value) : value_(value) {}
  const FeatureIndexT value_;
};

// Within the Instructions/Algorithm, we represent the index in a
// vector/matrix as a float. This float is interpreted as the fraction of the
// size of the vector. Example: a float coordinate (0.501, 0.251) is
// interpreted as the 8,4-entry in a 16x16 matrix; the same coordinate is
// interpreted as the 4,2-entry in an 8x8 matrix. This is because the features
// size is not know at the time the Algorithm is initialized/mutated. The
// details of the conversion are defined by these two functions. Note that
// FloatToIndex(IndexToFloat(i)) == i, but that that
// IndexToFloat(FloatToIndex(f)) is only similar to f.
inline FeatureIndexT FloatToIndex(
    const float value, const FeatureIndexT features_size) {
  const float size = static_cast<float>(features_size);
  return static_cast<FeatureIndexT>(size * value);
}
inline float IndexToFloat(
    const FeatureIndexT index, const FeatureIndexT features_size) {
  return (static_cast<float>(index) + 0.5) / static_cast<float>(features_size);
}

// An instruction (eg. sum two vectors at given addresses into a third address).
// NOTE: the default constructor does NOT serve as a way to initialize the
// Instruction.
class Instruction {
 public:
  // Constructor that initializes the instruction to a no-op.
  Instruction();

  // Constructors that initialize parameters explicitly.
  explicit Instruction(const IntegerDataSetter& integer_data_setter);
  Instruction(Op op, AddressT in, AddressT out);
  Instruction(Op op, AddressT in1, AddressT in2, AddressT out);
  Instruction(
      Op op, AddressT out,
      const ActivationDataSetter& activation_data_setter);
  Instruction(
      Op op, AddressT out,
      const IntegerDataSetter& integer_data_setter);
  Instruction(
      Op op, AddressT out,
      const FloatDataSetter& float_data_setter_0,
      const FloatDataSetter& float_data_setter_1);
  Instruction(
      Op op, AddressT out,
      const FloatDataSetter& float_data_setter_0,
      const FloatDataSetter& float_data_setter_1,
      const FloatDataSetter& float_data_setter_2);

  // Constructor that randomizes all parameters.
  Instruction(Op op, RandomGenerator* rand_gen);

  // Copy constructor that randomly alters a parameter.
  Instruction(const Instruction& other, RandomGenerator* rand_gen);

  // Deserializing constructor.
  explicit Instruction(const SerializedInstruction& serialized);

  inline void SetIntegerData(const IntegerT value) {
    activation_data_ = static_cast<double>(value);
  }

  // Instruction data accessors. Setting is mainly through the constructors.
  IntegerT GetIntegerData() const;  // Used in unit tests.
  inline double GetActivationData() const {return activation_data_;}
  inline float GetFloatData0() const {return float_data_0_;}
  inline float GetFloatData1() const {return float_data_1_;}
  inline float GetFloatData2() const {return float_data_2_;}

  bool operator ==(const Instruction& other) const;
  bool operator !=(const Instruction& other) const {
    return !(*this == other);
  }

  // Clears the instruction, setting it to a no-op. Serves as a way to
  // initialize the instruction.
  void FillWithNoOp();

  // Sets an op and randomizes all the parameters of the instruction. The
  // operation is passed as an argument because it's choice must be decided
  // based on the component function (setup / learn / predict), and that is
  // not known at this point. Serves as a way to initialize the instruction.
  void SetOpAndRandomizeParams(Op op, RandomGenerator* rand_gen);

  // Alters one parameter a small amount (if it makes sense) or randomizes it
  // (otherwise), depending on the parameter. The choice of parameter is random.
  // Does not serve as a way to initialize the Instruction; typically used after
  // copy-construction.
  void AlterParam(RandomGenerator* rand_gen);

  // Randomizes one parameter in the instruction. Internal use and tests only.
  void RandomizeIn1(RandomGenerator* rand_gen);
  void RandomizeIn2(RandomGenerator* rand_gen);
  void RandomizeOut(RandomGenerator* rand_gen);
  void RandomizeData(RandomGenerator* rand_gen);
  void AlterData(RandomGenerator* rand_gen);

  std::string ToString() const;
  SerializedInstruction Serialize() const;

  void Deserialize(const SerializedInstruction& checkpoint_instruction);

  Op op_;
  AddressT in1_;  // First input address.
  AddressT in2_;  // Second input address.
  AddressT out_;  // Output address.

 private:
  double activation_data_;
  float float_data_0_;
  float float_data_1_;
  float float_data_2_;
};

inline void MutateActivationLogScale(
    RandomGenerator* rand_gen, double* value) {
  if (*value > 0) {
    *value = std::exp(
        std::log(*value) + rand_gen->GaussianActivation(0.0, 1.0));
    return;
  } else {
    *value = -std::exp(
        std::log(-*value) + rand_gen->GaussianActivation(0.0, 1.0));
    return;
  }
}

inline void MutateFloatUnitInterval(
    RandomGenerator* rand_gen, float* value) {
  *value += rand_gen->UniformFloat(0.0, 0.1);
  if (*value < 0.0) {
    *value = 0.0;
  }
  if (*value > 1.0) {
    *value = 1.0;
  }
}

inline void MutateFloatLogScale(
    RandomGenerator* rand_gen, float* value) {
  if (*value > 0) {
    *value = std::exp(
        std::log(*value) + rand_gen->GaussianFloat(0.0, 1.0));
    return;
  } else {
    *value = -std::exp(
        std::log(-*value) + rand_gen->GaussianFloat(0.0, 1.0));
    return;
  }
}

inline void MutateActivationLogScaleOrFlip(
    RandomGenerator* rand_gen, double* value) {
  if (rand_gen->UniformProbability() < kSignFlipProb) {
    *value = -*value;
    return;
  } else {
    MutateActivationLogScale(rand_gen, value);
    return;
  }
}

inline void MutateFloatLogScaleOrFlip(
    RandomGenerator* rand_gen, float* value) {
  if (rand_gen->UniformProbability() < kSignFlipProb) {
    *value = -*value;
    return;
  } else {
    MutateFloatLogScale(rand_gen, value);
    return;
  }
}

}  // namespace automl_zero

#endif  // AUTOML_ZERO_INSTRUCTION_H_
