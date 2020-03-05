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

#include "instruction.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>

#include "definitions.h"
#include "instruction.pb.h"
#include "random_generator.h"
#include "absl/strings/str_replace.h"

namespace automl_zero {

using ::std::abs;  // NOLINT
using ::std::max;  // NOLINT
using ::std::min;  // NOLINT
using ::std::numeric_limits;  // NOLINT
using ::std::ostringstream;  // NOLINT
using ::std::round;  // NOLINT

Instruction::Instruction()
    : op_(NO_OP),
      in1_(0),
      in2_(0),
      out_(0),
      activation_data_(0.0),
      float_data_0_(0.0),
      float_data_1_(0.0),
      float_data_2_(0.0) {}

Instruction::Instruction(const IntegerDataSetter& integer_data_setter)
    : op_(NO_OP),
      in1_(0),
      in2_(0),
      out_(0),
      activation_data_(static_cast<double>(integer_data_setter.value_)),
      float_data_0_(0.0),
      float_data_1_(0.0),
      float_data_2_(0.0) {}

Instruction::Instruction(const Op op, const AddressT in, const AddressT out)
    : op_(op),
      in1_(in),
      in2_(0),
      out_(out),
      activation_data_(0.0),
      float_data_0_(0.0),
      float_data_1_(0.0),
      float_data_2_(0.0) {}

Instruction::Instruction(
    const Op op, const AddressT in1, const AddressT in2, const AddressT out)
    : op_(op),
      in1_(in1),
      in2_(in2),
      out_(out),
      activation_data_(0.0),
      float_data_0_(0.0),
      float_data_1_(0.0),
      float_data_2_(0.0) {}

Instruction::Instruction(
    const Op op, const AddressT out,
    const ActivationDataSetter& activation_data_setter)
    : op_(op),
      in1_(0),
      in2_(0),
      out_(out),
      activation_data_(activation_data_setter.value_),
      float_data_0_(0.0),
      float_data_1_(0.0),
      float_data_2_(0.0) {}

Instruction::Instruction(
    const Op op, const AddressT out,
    const IntegerDataSetter& integer_data_setter)
    : op_(op),
      in1_(0),
      in2_(0),
      out_(out),
      activation_data_(static_cast<double>(integer_data_setter.value_)),
      float_data_0_(0.0),
      float_data_1_(0.0),
      float_data_2_(0.0) {}

Instruction::Instruction(
    const Op op, const AddressT out,
    const FloatDataSetter& float_data_setter_0,
    const FloatDataSetter& float_data_setter_1)
    : op_(op),
      in1_(0),
      in2_(0),
      out_(out),
      activation_data_(0.0),
      float_data_0_(float_data_setter_0.value_),
      float_data_1_(float_data_setter_1.value_),
      float_data_2_(0.0) {}

Instruction::Instruction(
    const Op op, const AddressT out,
    const FloatDataSetter& float_data_setter_0,
    const FloatDataSetter& float_data_setter_1,
    const FloatDataSetter& float_data_setter_2)
    : op_(op),
      in1_(0),
      in2_(0),
      out_(out),
      activation_data_(0.0),
      float_data_0_(float_data_setter_0.value_),
      float_data_1_(float_data_setter_1.value_),
      float_data_2_(float_data_setter_2.value_) {}

Instruction::Instruction(Op op, RandomGenerator* rand_gen) {
  SetOpAndRandomizeParams(op, rand_gen);
}

Instruction::Instruction(
    const Instruction& other, RandomGenerator* rand_gen)
    : op_(other.op_),
      in1_(other.in1_),
      in2_(other.in2_),
      out_(other.out_),
      activation_data_(other.activation_data_),
      float_data_0_(other.float_data_0_),
      float_data_1_(other.float_data_1_),
      float_data_2_(other.float_data_2_) {
  AlterParam(rand_gen);
}

Instruction::Instruction(
    const SerializedInstruction& serialized) {
  Deserialize(serialized);
}

IntegerT Instruction::GetIntegerData() const {
  const double rounded = round(activation_data_);
  CHECK_GE(rounded, std::numeric_limits<IntegerT>::min());
  CHECK_LE(rounded, std::numeric_limits<IntegerT>::max());
  return static_cast<IntegerT>(rounded);
}

bool Instruction::operator==(const Instruction& other) const {
  return op_ == other.op_ &&
         in1_ == other.in1_ &&
         in2_ == other.in2_ &&
         out_ == other.out_ &&
         abs(activation_data_ - other.activation_data_) <
             kActivationDataTolerance &&
         abs(float_data_0_ - other.float_data_0_) <
             kFloatDataTolerance &&
         abs(float_data_1_ - other.float_data_1_) <
             kFloatDataTolerance &&
         abs(float_data_2_ - other.float_data_2_) <
             kFloatDataTolerance;
}

void Instruction::FillWithNoOp() {
  op_ = NO_OP;
  in1_ = 0;
  in2_ = 0;
  out_ = 0;
  activation_data_ = 0.0;
  float_data_0_ = 0.0;
  float_data_1_ = 0.0;
  float_data_2_ = 0.0;
}

void Instruction::SetOpAndRandomizeParams(
    Op op, RandomGenerator* rand_gen) {
  FillWithNoOp();
  op_ = op;
  switch (op_) {
    case NO_OP:
      return;
    case SCALAR_CONST_SET_OP:
    case VECTOR_CONST_SET_OP:
    case MATRIX_CONST_SET_OP:
    case SCALAR_GAUSSIAN_SET_OP:
    case VECTOR_GAUSSIAN_SET_OP:
    case MATRIX_GAUSSIAN_SET_OP:
    case SCALAR_UNIFORM_SET_OP:
    case VECTOR_UNIFORM_SET_OP:
    case MATRIX_UNIFORM_SET_OP:
      RandomizeOut(rand_gen);
      RandomizeData(rand_gen);
      return;
    case SCALAR_ABS_OP:
    case SCALAR_HEAVYSIDE_OP:
    case SCALAR_SIN_OP:
    case SCALAR_COS_OP:
    case SCALAR_TAN_OP:
    case SCALAR_ARCSIN_OP:
    case SCALAR_ARCCOS_OP:
    case SCALAR_ARCTAN_OP:
    case SCALAR_EXP_OP:
    case SCALAR_LOG_OP:
    case SCALAR_RECIPROCAL_OP:
    case SCALAR_BROADCAST_OP:
    case VECTOR_ABS_OP:
    case VECTOR_HEAVYSIDE_OP:
    case VECTOR_RECIPROCAL_OP:
    case MATRIX_RECIPROCAL_OP:
    case MATRIX_ROW_NORM_OP:
    case MATRIX_COLUMN_NORM_OP:
    case VECTOR_COLUMN_BROADCAST_OP:
    case VECTOR_ROW_BROADCAST_OP:
    case MATRIX_ABS_OP:
    case MATRIX_HEAVYSIDE_OP:
    case VECTOR_NORM_OP:
    case MATRIX_NORM_OP:
    case MATRIX_TRANSPOSE_OP:
    case VECTOR_MEAN_OP:
    case VECTOR_ST_DEV_OP:
    case MATRIX_MEAN_OP:
    case MATRIX_ST_DEV_OP:
    case MATRIX_ROW_MEAN_OP:
    case MATRIX_ROW_ST_DEV_OP:
      RandomizeIn1(rand_gen);
      RandomizeOut(rand_gen);
      return;
    case SCALAR_SUM_OP:
    case SCALAR_DIFF_OP:
    case SCALAR_PRODUCT_OP:
    case SCALAR_DIVISION_OP:
    case SCALAR_MIN_OP:
    case SCALAR_MAX_OP:
    case VECTOR_SUM_OP:
    case VECTOR_DIFF_OP:
    case VECTOR_PRODUCT_OP:
    case VECTOR_DIVISION_OP:
    case VECTOR_MIN_OP:
    case VECTOR_MAX_OP:
    case MATRIX_SUM_OP:
    case MATRIX_DIFF_OP:
    case MATRIX_PRODUCT_OP:
    case MATRIX_DIVISION_OP:
    case MATRIX_MIN_OP:
    case MATRIX_MAX_OP:
    case SCALAR_VECTOR_PRODUCT_OP:
    case VECTOR_INNER_PRODUCT_OP:
    case VECTOR_OUTER_PRODUCT_OP:
    case SCALAR_MATRIX_PRODUCT_OP:
    case MATRIX_VECTOR_PRODUCT_OP:
    case MATRIX_MATRIX_PRODUCT_OP:
      RandomizeIn1(rand_gen);
      RandomizeIn2(rand_gen);
      RandomizeOut(rand_gen);
      return;
    // Do not add default clause. All ops should be supported here.
  }
}

void Instruction::AlterParam(
    RandomGenerator* rand_gen) {
  switch (op_) {
    case NO_OP:
      return;
    case SCALAR_CONST_SET_OP:
    case VECTOR_CONST_SET_OP:
    case MATRIX_CONST_SET_OP:
    case SCALAR_GAUSSIAN_SET_OP:
    case VECTOR_GAUSSIAN_SET_OP:
    case MATRIX_GAUSSIAN_SET_OP:
    case SCALAR_UNIFORM_SET_OP:
    case VECTOR_UNIFORM_SET_OP:
    case MATRIX_UNIFORM_SET_OP:
      switch (rand_gen->Choice2()) {
        case kChoice0of2:
          RandomizeOut(rand_gen);
          return;
        case kChoice1of2:
          AlterData(rand_gen);
          return;
      }
    case SCALAR_ABS_OP:
    case SCALAR_HEAVYSIDE_OP:
    case SCALAR_SIN_OP:
    case SCALAR_COS_OP:
    case SCALAR_TAN_OP:
    case SCALAR_ARCSIN_OP:
    case SCALAR_ARCCOS_OP:
    case SCALAR_ARCTAN_OP:
    case SCALAR_EXP_OP:
    case SCALAR_LOG_OP:
    case SCALAR_RECIPROCAL_OP:
    case SCALAR_BROADCAST_OP:
    case VECTOR_RECIPROCAL_OP:
    case MATRIX_RECIPROCAL_OP:
    case MATRIX_ROW_NORM_OP:
    case MATRIX_COLUMN_NORM_OP:
    case VECTOR_COLUMN_BROADCAST_OP:
    case VECTOR_ROW_BROADCAST_OP:
    case VECTOR_ABS_OP:
    case VECTOR_HEAVYSIDE_OP:
    case MATRIX_ABS_OP:
    case MATRIX_HEAVYSIDE_OP:
    case VECTOR_NORM_OP:
    case MATRIX_NORM_OP:
    case MATRIX_TRANSPOSE_OP:
    case VECTOR_MEAN_OP:
    case VECTOR_ST_DEV_OP:
    case MATRIX_MEAN_OP:
    case MATRIX_ST_DEV_OP:
    case MATRIX_ROW_MEAN_OP:
    case MATRIX_ROW_ST_DEV_OP:
      switch (rand_gen->Choice2()) {
        case kChoice0of2:
          RandomizeIn1(rand_gen);
          return;
        case kChoice1of2:
          RandomizeOut(rand_gen);
          return;
      }
    case SCALAR_SUM_OP:
    case SCALAR_DIFF_OP:
    case SCALAR_PRODUCT_OP:
    case SCALAR_DIVISION_OP:
    case SCALAR_MIN_OP:
    case SCALAR_MAX_OP:
    case VECTOR_SUM_OP:
    case VECTOR_DIFF_OP:
    case VECTOR_PRODUCT_OP:
    case VECTOR_DIVISION_OP:
    case VECTOR_MIN_OP:
    case VECTOR_MAX_OP:
    case MATRIX_SUM_OP:
    case MATRIX_DIFF_OP:
    case MATRIX_PRODUCT_OP:
    case MATRIX_DIVISION_OP:
    case MATRIX_MIN_OP:
    case MATRIX_MAX_OP:
    case SCALAR_VECTOR_PRODUCT_OP:
    case VECTOR_INNER_PRODUCT_OP:
    case VECTOR_OUTER_PRODUCT_OP:
    case SCALAR_MATRIX_PRODUCT_OP:
    case MATRIX_VECTOR_PRODUCT_OP:
    case MATRIX_MATRIX_PRODUCT_OP:
      switch (rand_gen->Choice3()) {
        case kChoice0of3:
          RandomizeIn1(rand_gen);
          return;
        case kChoice1of3:
          RandomizeIn2(rand_gen);
          return;
        case kChoice2of3:
          RandomizeOut(rand_gen);
          return;
      }
    // Do not add default clause. All ops should be supported here.
  }
}

void Instruction::RandomizeIn1(RandomGenerator* rand_gen) {
  switch (op_) {
    case NO_OP:
    case SCALAR_CONST_SET_OP:
    case VECTOR_CONST_SET_OP:
    case MATRIX_CONST_SET_OP:
    case SCALAR_GAUSSIAN_SET_OP:
    case VECTOR_GAUSSIAN_SET_OP:
    case MATRIX_GAUSSIAN_SET_OP:
    case SCALAR_UNIFORM_SET_OP:
    case VECTOR_UNIFORM_SET_OP:
    case MATRIX_UNIFORM_SET_OP:
      LOG(FATAL) << "Invalid op: " << static_cast<IntegerT>(op_) << std::endl;
    case SCALAR_SUM_OP:
    case SCALAR_DIFF_OP:
    case SCALAR_PRODUCT_OP:
    case SCALAR_DIVISION_OP:
    case SCALAR_MIN_OP:
    case SCALAR_MAX_OP:
    case SCALAR_ABS_OP:
    case SCALAR_HEAVYSIDE_OP:
    case SCALAR_SIN_OP:
    case SCALAR_COS_OP:
    case SCALAR_TAN_OP:
    case SCALAR_ARCSIN_OP:
    case SCALAR_ARCCOS_OP:
    case SCALAR_ARCTAN_OP:
    case SCALAR_EXP_OP:
    case SCALAR_LOG_OP:
    case SCALAR_RECIPROCAL_OP:
    case SCALAR_BROADCAST_OP:
    case SCALAR_VECTOR_PRODUCT_OP:
    case SCALAR_MATRIX_PRODUCT_OP:
      in1_ = rand_gen->ScalarInAddress();
      return;
    case VECTOR_SUM_OP:
    case VECTOR_DIFF_OP:
    case VECTOR_PRODUCT_OP:
    case VECTOR_DIVISION_OP:
    case VECTOR_MIN_OP:
    case VECTOR_MAX_OP:
    case VECTOR_ABS_OP:
    case VECTOR_HEAVYSIDE_OP:
    case VECTOR_INNER_PRODUCT_OP:
    case VECTOR_OUTER_PRODUCT_OP:
    case VECTOR_NORM_OP:
    case VECTOR_MEAN_OP:
    case VECTOR_ST_DEV_OP:
    case VECTOR_RECIPROCAL_OP:
    case VECTOR_COLUMN_BROADCAST_OP:
    case VECTOR_ROW_BROADCAST_OP:
      in1_ = rand_gen->VectorInAddress();
      return;
    case MATRIX_SUM_OP:
    case MATRIX_DIFF_OP:
    case MATRIX_PRODUCT_OP:
    case MATRIX_DIVISION_OP:
    case MATRIX_MIN_OP:
    case MATRIX_MAX_OP:
    case MATRIX_ABS_OP:
    case MATRIX_HEAVYSIDE_OP:
    case MATRIX_VECTOR_PRODUCT_OP:
    case MATRIX_NORM_OP:
    case MATRIX_TRANSPOSE_OP:
    case MATRIX_MATRIX_PRODUCT_OP:
    case MATRIX_MEAN_OP:
    case MATRIX_ST_DEV_OP:
    case MATRIX_ROW_MEAN_OP:
    case MATRIX_ROW_ST_DEV_OP:
    case MATRIX_RECIPROCAL_OP:
    case MATRIX_ROW_NORM_OP:
    case MATRIX_COLUMN_NORM_OP:
      in1_ = rand_gen->MatrixInAddress();
      return;
    // Do not add default clause. All ops should be supported here.
  }
}

void Instruction::RandomizeIn2(RandomGenerator* rand_gen) {
  switch (op_) {
    case NO_OP:
    case SCALAR_ABS_OP:
    case SCALAR_HEAVYSIDE_OP:
    case SCALAR_CONST_SET_OP:
    case SCALAR_SIN_OP:
    case SCALAR_COS_OP:
    case SCALAR_TAN_OP:
    case SCALAR_ARCSIN_OP:
    case SCALAR_ARCCOS_OP:
    case SCALAR_ARCTAN_OP:
    case SCALAR_EXP_OP:
    case SCALAR_LOG_OP:
    case SCALAR_RECIPROCAL_OP:
    case SCALAR_BROADCAST_OP:
    case VECTOR_RECIPROCAL_OP:
    case MATRIX_RECIPROCAL_OP:
    case MATRIX_ROW_NORM_OP:
    case MATRIX_COLUMN_NORM_OP:
    case VECTOR_COLUMN_BROADCAST_OP:
    case VECTOR_ROW_BROADCAST_OP:
    case VECTOR_ABS_OP:
    case VECTOR_HEAVYSIDE_OP:
    case VECTOR_CONST_SET_OP:
    case MATRIX_ABS_OP:
    case MATRIX_HEAVYSIDE_OP:
    case MATRIX_CONST_SET_OP:
    case VECTOR_NORM_OP:
    case MATRIX_NORM_OP:
    case MATRIX_TRANSPOSE_OP:
    case VECTOR_MEAN_OP:
    case VECTOR_ST_DEV_OP:
    case MATRIX_MEAN_OP:
    case MATRIX_ST_DEV_OP:
    case MATRIX_ROW_MEAN_OP:
    case MATRIX_ROW_ST_DEV_OP:
    case SCALAR_GAUSSIAN_SET_OP:
    case VECTOR_GAUSSIAN_SET_OP:
    case MATRIX_GAUSSIAN_SET_OP:
    case SCALAR_UNIFORM_SET_OP:
    case VECTOR_UNIFORM_SET_OP:
    case MATRIX_UNIFORM_SET_OP:
      LOG(FATAL) << "Invalid op: " << static_cast<IntegerT>(op_) << std::endl;
    case SCALAR_SUM_OP:
    case SCALAR_DIFF_OP:
    case SCALAR_PRODUCT_OP:
    case SCALAR_DIVISION_OP:
    case SCALAR_MIN_OP:
    case SCALAR_MAX_OP:
      in2_ = rand_gen->ScalarInAddress();
      return;
    case VECTOR_SUM_OP:
    case VECTOR_DIFF_OP:
    case VECTOR_PRODUCT_OP:
    case VECTOR_DIVISION_OP:
    case VECTOR_MIN_OP:
    case VECTOR_MAX_OP:
    case SCALAR_VECTOR_PRODUCT_OP:
    case VECTOR_INNER_PRODUCT_OP:
    case VECTOR_OUTER_PRODUCT_OP:
    case MATRIX_VECTOR_PRODUCT_OP:
      in2_ = rand_gen->VectorInAddress();
      return;
    case MATRIX_SUM_OP:
    case MATRIX_DIFF_OP:
    case MATRIX_PRODUCT_OP:
    case MATRIX_DIVISION_OP:
    case MATRIX_MIN_OP:
    case MATRIX_MAX_OP:
    case SCALAR_MATRIX_PRODUCT_OP:
    case MATRIX_MATRIX_PRODUCT_OP:
      in2_ = rand_gen->MatrixInAddress();
      return;
    // Do not add default clause. All ops should be supported here.
  }
}

void Instruction::RandomizeOut(RandomGenerator* rand_gen) {
  switch (op_) {
    case NO_OP:
      LOG(FATAL) << "Invalid op: " << static_cast<IntegerT>(op_) << std::endl;
    case SCALAR_SUM_OP:
    case SCALAR_DIFF_OP:
    case SCALAR_PRODUCT_OP:
    case SCALAR_DIVISION_OP:
    case SCALAR_MIN_OP:
    case SCALAR_MAX_OP:
    case SCALAR_ABS_OP:
    case SCALAR_HEAVYSIDE_OP:
    case SCALAR_CONST_SET_OP:
    case SCALAR_SIN_OP:
    case SCALAR_COS_OP:
    case SCALAR_TAN_OP:
    case SCALAR_ARCSIN_OP:
    case SCALAR_ARCCOS_OP:
    case SCALAR_ARCTAN_OP:
    case SCALAR_EXP_OP:
    case SCALAR_LOG_OP:
    case SCALAR_RECIPROCAL_OP:
    case VECTOR_INNER_PRODUCT_OP:
    case VECTOR_NORM_OP:
    case MATRIX_NORM_OP:
    case VECTOR_MEAN_OP:
    case VECTOR_ST_DEV_OP:
    case MATRIX_MEAN_OP:
    case MATRIX_ST_DEV_OP:
    case SCALAR_GAUSSIAN_SET_OP:
    case SCALAR_UNIFORM_SET_OP:
      out_ = rand_gen->ScalarOutAddress();
      return;
    case VECTOR_SUM_OP:
    case VECTOR_DIFF_OP:
    case VECTOR_PRODUCT_OP:
    case VECTOR_DIVISION_OP:
    case VECTOR_MIN_OP:
    case VECTOR_MAX_OP:
    case VECTOR_ABS_OP:
    case VECTOR_HEAVYSIDE_OP:
    case VECTOR_CONST_SET_OP:
    case SCALAR_VECTOR_PRODUCT_OP:
    case MATRIX_VECTOR_PRODUCT_OP:
    case MATRIX_ROW_MEAN_OP:
    case MATRIX_ROW_ST_DEV_OP:
    case VECTOR_GAUSSIAN_SET_OP:
    case VECTOR_UNIFORM_SET_OP:
    case SCALAR_BROADCAST_OP:
    case VECTOR_RECIPROCAL_OP:
    case MATRIX_ROW_NORM_OP:
    case MATRIX_COLUMN_NORM_OP:
      out_ = rand_gen->VectorOutAddress();
      return;
    case MATRIX_SUM_OP:
    case MATRIX_DIFF_OP:
    case MATRIX_PRODUCT_OP:
    case MATRIX_DIVISION_OP:
    case MATRIX_MIN_OP:
    case MATRIX_MAX_OP:
    case MATRIX_ABS_OP:
    case MATRIX_HEAVYSIDE_OP:
    case MATRIX_CONST_SET_OP:
    case VECTOR_OUTER_PRODUCT_OP:
    case SCALAR_MATRIX_PRODUCT_OP:
    case MATRIX_TRANSPOSE_OP:
    case MATRIX_MATRIX_PRODUCT_OP:
    case MATRIX_GAUSSIAN_SET_OP:
    case MATRIX_UNIFORM_SET_OP:
    case MATRIX_RECIPROCAL_OP:
    case VECTOR_COLUMN_BROADCAST_OP:
    case VECTOR_ROW_BROADCAST_OP:
      out_ = rand_gen->MatrixOutAddress();
      return;
    // Do not add default clause. All ops should be supported here.
  }
}

void Instruction::RandomizeData(RandomGenerator* rand_gen) {
  switch (op_) {
    case NO_OP:
    case SCALAR_SUM_OP:
    case SCALAR_DIFF_OP:
    case SCALAR_PRODUCT_OP:
    case SCALAR_DIVISION_OP:
    case SCALAR_MIN_OP:
    case SCALAR_MAX_OP:
    case SCALAR_ABS_OP:
    case SCALAR_HEAVYSIDE_OP:
    case SCALAR_SIN_OP:
    case SCALAR_COS_OP:
    case SCALAR_TAN_OP:
    case SCALAR_ARCSIN_OP:
    case SCALAR_ARCCOS_OP:
    case SCALAR_ARCTAN_OP:
    case SCALAR_EXP_OP:
    case SCALAR_LOG_OP:
    case SCALAR_RECIPROCAL_OP:
    case SCALAR_BROADCAST_OP:
    case VECTOR_RECIPROCAL_OP:
    case MATRIX_RECIPROCAL_OP:
    case MATRIX_ROW_NORM_OP:
    case MATRIX_COLUMN_NORM_OP:
    case VECTOR_COLUMN_BROADCAST_OP:
    case VECTOR_ROW_BROADCAST_OP:
    case VECTOR_SUM_OP:
    case VECTOR_DIFF_OP:
    case VECTOR_PRODUCT_OP:
    case VECTOR_DIVISION_OP:
    case VECTOR_MIN_OP:
    case VECTOR_MAX_OP:
    case VECTOR_ABS_OP:
    case VECTOR_HEAVYSIDE_OP:
    case MATRIX_SUM_OP:
    case MATRIX_DIFF_OP:
    case MATRIX_PRODUCT_OP:
    case MATRIX_DIVISION_OP:
    case MATRIX_MIN_OP:
    case MATRIX_MAX_OP:
    case MATRIX_ABS_OP:
    case MATRIX_HEAVYSIDE_OP:
    case SCALAR_VECTOR_PRODUCT_OP:
    case VECTOR_INNER_PRODUCT_OP:
    case VECTOR_OUTER_PRODUCT_OP:
    case SCALAR_MATRIX_PRODUCT_OP:
    case MATRIX_VECTOR_PRODUCT_OP:
    case VECTOR_NORM_OP:
    case MATRIX_NORM_OP:
    case MATRIX_TRANSPOSE_OP:
    case MATRIX_MATRIX_PRODUCT_OP:
    case VECTOR_MEAN_OP:
    case VECTOR_ST_DEV_OP:
    case MATRIX_MEAN_OP:
    case MATRIX_ST_DEV_OP:
    case MATRIX_ROW_MEAN_OP:
    case MATRIX_ROW_ST_DEV_OP:
      LOG(FATAL) << "Invalid op: " << static_cast<IntegerT>(op_) << std::endl;
    case SCALAR_CONST_SET_OP: {
      activation_data_ = rand_gen->UniformActivation(-1.0, 1.0);
      return;
    }
    case VECTOR_CONST_SET_OP: {
      // float_data_0_ represents the index. See FloatToIndex for more details.
      float_data_0_ = rand_gen->UniformFloat(0.0, 1.0);
      // float_data_1_ represents the value to store.
      float_data_1_ = rand_gen->UniformFloat(-1.0, 1.0);
      return;
    }
    case MATRIX_CONST_SET_OP: {
      float_data_0_ = rand_gen->UniformFloat(0.0, 1.0);
      float_data_1_ = rand_gen->UniformFloat(0.0, 1.0);
      float_data_2_ = rand_gen->UniformFloat(-1.0, 1.0);
      return;
    }
    case SCALAR_GAUSSIAN_SET_OP:
    case VECTOR_GAUSSIAN_SET_OP:
    case MATRIX_GAUSSIAN_SET_OP: {
      float_data_0_ = rand_gen->UniformFloat(-1.0, 1.0);  // Mean.
      float_data_1_ = rand_gen->UniformFloat(0.0, 1.0);  // St. dev.
      return;
    }
    case SCALAR_UNIFORM_SET_OP:
    case VECTOR_UNIFORM_SET_OP:
    case MATRIX_UNIFORM_SET_OP: {
      float_data_0_ = rand_gen->UniformFloat(-1.0, 1.0);
      float_data_1_ = rand_gen->UniformFloat(-1.0, 1.0);
      return;
    }
    // Do not add default clause. All ops should be supported here.
  }
}

void Instruction::AlterData(RandomGenerator* rand_gen) {
  switch (op_) {
    case NO_OP:
    case SCALAR_SUM_OP:
    case SCALAR_DIFF_OP:
    case SCALAR_PRODUCT_OP:
    case SCALAR_DIVISION_OP:
    case SCALAR_MIN_OP:
    case SCALAR_MAX_OP:
    case SCALAR_ABS_OP:
    case SCALAR_HEAVYSIDE_OP:
    case SCALAR_SIN_OP:
    case SCALAR_COS_OP:
    case SCALAR_TAN_OP:
    case SCALAR_ARCSIN_OP:
    case SCALAR_ARCCOS_OP:
    case SCALAR_ARCTAN_OP:
    case SCALAR_EXP_OP:
    case SCALAR_LOG_OP:
    case VECTOR_SUM_OP:
    case VECTOR_DIFF_OP:
    case VECTOR_PRODUCT_OP:
    case VECTOR_DIVISION_OP:
    case VECTOR_MIN_OP:
    case VECTOR_MAX_OP:
    case VECTOR_ABS_OP:
    case VECTOR_HEAVYSIDE_OP:
    case MATRIX_SUM_OP:
    case MATRIX_DIFF_OP:
    case MATRIX_PRODUCT_OP:
    case MATRIX_DIVISION_OP:
    case MATRIX_MIN_OP:
    case MATRIX_MAX_OP:
    case MATRIX_ABS_OP:
    case MATRIX_HEAVYSIDE_OP:
    case SCALAR_VECTOR_PRODUCT_OP:
    case VECTOR_INNER_PRODUCT_OP:
    case VECTOR_OUTER_PRODUCT_OP:
    case SCALAR_MATRIX_PRODUCT_OP:
    case MATRIX_VECTOR_PRODUCT_OP:
    case VECTOR_NORM_OP:
    case MATRIX_NORM_OP:
    case MATRIX_TRANSPOSE_OP:
    case MATRIX_MATRIX_PRODUCT_OP:
    case VECTOR_MEAN_OP:
    case VECTOR_ST_DEV_OP:
    case MATRIX_MEAN_OP:
    case MATRIX_ST_DEV_OP:
    case MATRIX_ROW_MEAN_OP:
    case MATRIX_ROW_ST_DEV_OP:
    case SCALAR_RECIPROCAL_OP:
    case SCALAR_BROADCAST_OP:
    case VECTOR_RECIPROCAL_OP:
    case MATRIX_RECIPROCAL_OP:
    case MATRIX_ROW_NORM_OP:
    case MATRIX_COLUMN_NORM_OP:
    case VECTOR_COLUMN_BROADCAST_OP:
    case VECTOR_ROW_BROADCAST_OP:
      LOG(FATAL) << "Invalid op: " << static_cast<IntegerT>(op_) << std::endl;
    case SCALAR_CONST_SET_OP: {
      MutateActivationLogScaleOrFlip(rand_gen, &activation_data_);
      return;
    }
    case VECTOR_CONST_SET_OP: {
      switch (rand_gen->Choice2()) {
        case kChoice0of2:
          // Mutate index. See FloatToIndex for more details.
          float_data_0_ = rand_gen->UniformFloat(0.0, 1.0);
          break;
        case kChoice1of2:
          // Mutate value.
          MutateFloatLogScaleOrFlip(rand_gen, &float_data_1_);
          break;
      }
      return;
    }
    case MATRIX_CONST_SET_OP: {
      switch (rand_gen->Choice3()) {
        case kChoice0of3:
          // Mutate first index.
          float_data_0_ = rand_gen->UniformFloat(0.0, 1.0);
          break;
        case kChoice1of3:
          // Mutate second index.
          float_data_1_ = rand_gen->UniformFloat(0.0, 1.0);
          break;
        case kChoice2of3:
          // Mutate value.
          MutateFloatLogScaleOrFlip(rand_gen, &float_data_2_);
          break;
      }
      return;
    }
    case SCALAR_GAUSSIAN_SET_OP:
    case VECTOR_GAUSSIAN_SET_OP:
    case MATRIX_GAUSSIAN_SET_OP: {
      switch (rand_gen->Choice2()) {
        case kChoice0of2:
          // Mutate mean.
          MutateFloatLogScaleOrFlip(rand_gen, &float_data_0_);
          break;
        case kChoice1of2:
          // Mutate stdev.
          MutateFloatLogScale(rand_gen, &float_data_1_);
          break;
      }
      return;
    }
    case SCALAR_UNIFORM_SET_OP:
    case VECTOR_UNIFORM_SET_OP:
    case MATRIX_UNIFORM_SET_OP: {
      float value0 = float_data_0_;
      float value1 = float_data_1_;
      switch (rand_gen->Choice2()) {
        case kChoice0of2:
          // Mutate low.
          MutateFloatLogScaleOrFlip(rand_gen, &value0);
          break;
        case kChoice1of2:
          // Mutate high.
          MutateFloatLogScaleOrFlip(rand_gen, &value1);
          break;
      }
      float_data_0_ = min(value0, value1);
      float_data_1_ = max(value0, value1);
      return;
    }
    // Do not add default clause. All ops should be supported here.
  }
}

std::string Instruction::ToString() const {
  ostringstream stream;
  switch (op_) {
    case NO_OP:
      stream << "  NoOp()" << std::endl;
      break;
    case SCALAR_SUM_OP:
      stream << "  s" << out_ << " = s" << in1_ << " + s" << in2_
             << std::endl;
      break;
    case SCALAR_DIFF_OP:
      stream << "  s" << out_ << " = s" << in1_ << " - s" << in2_
             << std::endl;
      break;
    case SCALAR_PRODUCT_OP:
      stream << "  s" << out_ << " = s" << in1_ << " * s" << in2_
             << std::endl;
      break;
    case SCALAR_DIVISION_OP:
      stream << "  s" << out_ << " = s" << in1_ << " / s" << in2_
             << std::endl;
      break;
    case SCALAR_MIN_OP:
      stream << "  s" << out_ << " = minimum(s" << in1_ << ", s" << in2_
             << ")" << std::endl;
      break;
    case SCALAR_MAX_OP:
      stream << "  s" << out_ << " = maximum(s" << in1_ << ", s" << in2_
             << ")" << std::endl;
      break;
    case SCALAR_ABS_OP:
      stream << "  s" << out_ << " = abs(s" << in1_ << ")" << std::endl;
      break;
    case SCALAR_HEAVYSIDE_OP:
      stream << "  s" << out_ << " = heaviside(s" << in1_ << ", 1.0)"
             << std::endl;
      break;
    case SCALAR_CONST_SET_OP: {
      stream << "  s" << out_ << " = " << activation_data_ << std::endl;
      break;
    }
    case SCALAR_SIN_OP:
      stream << "  s" << out_ << " = sin(s" << in1_ << ")" << std::endl;
      break;
    case SCALAR_COS_OP:
      stream << "  s" << out_ << " = cos(s" << in1_ << ")" << std::endl;
      break;
    case SCALAR_TAN_OP:
      stream << "  s" << out_ << " = tan(s" << in1_ << ")" << std::endl;
      break;
    case SCALAR_ARCSIN_OP:
      stream << "  s" << out_ << " = arcsin(s" << in1_ << ")"
             << std::endl;
      break;
    case SCALAR_ARCCOS_OP:
      stream << "  s" << out_ << " = arccos(s" << in1_ << ")"
             << std::endl;
      break;
    case SCALAR_ARCTAN_OP:
      stream << "  s" << out_ << " = arctan(s" << in1_ << ")"
             << std::endl;
      break;
    case SCALAR_EXP_OP:
      stream << "  s" << out_ << " = exp(s" << in1_ << ")" << std::endl;
      break;
    case SCALAR_LOG_OP:
      stream << "  s" << out_ << " = log(s" << in1_ << ")" << std::endl;
      break;
    case SCALAR_RECIPROCAL_OP:
      stream << "  s" << out_ << " = 1 / s" << in1_ << std::endl;
      break;
    case SCALAR_BROADCAST_OP:
      stream << "  v" << out_ << " = bcast(s" << in1_ << ")"
             << std::endl;
      break;
    case VECTOR_RECIPROCAL_OP:
      stream << "  v" << out_ << " = 1 / v" << in1_ << std::endl;
      break;
    case MATRIX_RECIPROCAL_OP:
      stream << "  m" << out_ << " = 1 / m" << in1_ << std::endl;
      break;
    case MATRIX_ROW_NORM_OP:
      stream << "  v" << out_ << " = norm(m" << in1_ << ", axis=1)"
             << std::endl;
      break;
    case MATRIX_COLUMN_NORM_OP:
      stream << "  v" << out_ << " = norm(m" << in1_ << ", axis=0)"
             << std::endl;
      break;
    case VECTOR_COLUMN_BROADCAST_OP:
      stream << "  m" << out_ << " = bcast(v" << in1_ << ", axis=0)"
             << std::endl;
      break;
    case VECTOR_ROW_BROADCAST_OP:
      stream << "  m" << out_ << " = bcast(v" << in1_ << ", axis=1)"
             << std::endl;
      break;
    case VECTOR_SUM_OP:
      stream << "  v" << out_ << " = v" << in1_ << " + v" << in2_
             << std::endl;
      break;
    case VECTOR_DIFF_OP:
      stream << "  v" << out_ << " = v" << in1_ << " - v" << in2_
             << std::endl;
      break;
    case VECTOR_PRODUCT_OP:
      stream << "  v" << out_ << " = v" << in1_ << " * v" << in2_
             << std::endl;
      break;
    case VECTOR_DIVISION_OP:
      stream << "  v" << out_ << " = v" << in1_ << " / v" << in2_
             << std::endl;
      break;
    case VECTOR_MIN_OP:
      stream << "  v" << out_ << " = minimum(v" << in1_ << ", v" << in2_
             << ")" << std::endl;
      break;
    case VECTOR_MAX_OP:
      stream << "  v" << out_ << " = maximum(v" << in1_ << ", v" << in2_
             << ")" << std::endl;
      break;
    case VECTOR_ABS_OP:
      stream << "  v" << out_ << " = abs(v" << in1_ << ")" << std::endl;
      break;
    case VECTOR_HEAVYSIDE_OP:
      stream << "  v" << out_ << " = heaviside(v" << in1_ << ", 1.0)"
             << std::endl;
      break;
    case VECTOR_CONST_SET_OP: {
      stream << "  v" << out_ << "[" << float_data_0_ << "]"
             << " = " << float_data_1_ << std::endl;
      break;
    }
    case MATRIX_SUM_OP:
      stream << "  m" << out_ << " = m" << in1_ << " + m" << in2_
             << std::endl;
      break;
    case MATRIX_DIFF_OP:
      stream << "  m" << out_ << " = m" << in1_ << " - m" << in2_
             << std::endl;
      break;
    case MATRIX_PRODUCT_OP:
      stream << "  m" << out_ << " = m" << in1_ << " * m" << in2_
             << std::endl;
      break;
    case MATRIX_DIVISION_OP:
      stream << "  m" << out_ << " = m" << in1_ << " / m" << in2_
             << std::endl;
      break;
    case MATRIX_MIN_OP:
      stream << "  m" << out_ << " = minimum(m" << in1_ << ", m" << in2_
             << ")" << std::endl;
      break;
    case MATRIX_MAX_OP:
      stream << "  m" << out_ << " = maximum(m" << in1_ << ", m" << in2_
             << ")" << std::endl;
      break;
    case MATRIX_ABS_OP:
      stream << "  m" << out_ << " = abs(m" << in1_ << ")" << std::endl;
      break;
    case MATRIX_HEAVYSIDE_OP:
      stream << "  m" << out_ << " = heaviside(m" << in1_ << ", 1.0)"
             << std::endl;
      break;
    case MATRIX_CONST_SET_OP: {
      stream << "  m" << out_
             << "[" << float_data_0_ << ", " << float_data_1_ << "]"
             << " = " << float_data_2_ << std::endl;
      break;
    }
    case SCALAR_VECTOR_PRODUCT_OP:
      stream << "  v" << out_ << " = s" << in1_ << " * v" << in2_
             << std::endl;
      break;
    case VECTOR_INNER_PRODUCT_OP:
      stream << "  s" << out_ << " = "
             << "dot(v" << in1_ << ", v" << in2_ << ")" << std::endl;
      break;
    case VECTOR_OUTER_PRODUCT_OP:
      stream << "  m" << out_ << " = "
             << "outer(v" << in1_ << ", v" << in2_ << ")" << std::endl;
      break;
    case SCALAR_MATRIX_PRODUCT_OP:
      stream << "  m" << out_ << " = s" << in1_ << " * m" << in2_
             << std::endl;
      break;
    case MATRIX_VECTOR_PRODUCT_OP:
      stream << "  v" << out_ << " = dot(m" << in1_ << ", v" << in2_
             << ")" << std::endl;
      break;
    case VECTOR_NORM_OP:
      stream << "  s" << out_ << " = norm(v" << in1_ << ")"
             << std::endl;
      break;
    case MATRIX_NORM_OP:
      stream << "  s" << out_ << " = norm(m" << in1_ << ")"
             << std::endl;
      break;
    case MATRIX_TRANSPOSE_OP:
      stream << "  m" << out_ << " = transpose(m" << in1_ << ")"
             << std::endl;
      break;
    case MATRIX_MATRIX_PRODUCT_OP:
      stream << "  m" << out_ << " = matmul(m" << in1_ << ", m" << in2_
             << ")" << std::endl;
      break;
    case VECTOR_MEAN_OP:
      stream << "  s" << out_ << " = mean(v" << in1_ << ")" << std::endl;
      break;
    case VECTOR_ST_DEV_OP:
      stream << "  s" << out_ << " = std(v" << in1_ << ")" << std::endl;
      break;
    case MATRIX_MEAN_OP:
      stream << "  s" << out_ << " = mean(m" << in1_ << ")" << std::endl;
      break;
    case MATRIX_ST_DEV_OP:
      stream << "  s" << out_ << " = std(m" << in1_ << ")" << std::endl;
      break;
    case MATRIX_ROW_MEAN_OP:
      stream << "  v" << out_ << " = mean(m" << in1_ << ", axis=1)"
             << std::endl;
      break;
    case MATRIX_ROW_ST_DEV_OP:
      stream << "  v" << out_ << " = std(m" << in1_ << ", axis=1)"
             << std::endl;
      break;
    case SCALAR_GAUSSIAN_SET_OP: {
      stream << "  s" << out_ << " = gaussian(" << float_data_0_
             << ", " << float_data_1_ << ")" << std::endl;
      break;
    }
    case VECTOR_GAUSSIAN_SET_OP: {
      stream << "  v" << out_ << " = gaussian(" << float_data_0_
             << ", " << float_data_1_ << ", n_features)" << std::endl;
      break;
    }
    case MATRIX_GAUSSIAN_SET_OP: {
      stream << "  m" << out_ << " = gaussian(" << float_data_0_
             << ", " << float_data_1_ << ", (n_features, n_features))"
             << std::endl;
      break;
    }
    case SCALAR_UNIFORM_SET_OP: {
      stream << "  s" << out_ << " = uniform(" << float_data_0_
             << ", " << float_data_1_ << ")" << std::endl;
      break;
    }
    case VECTOR_UNIFORM_SET_OP: {
      stream << "  v" << out_ << " = uniform(" << float_data_0_
             << ", " << float_data_1_ << ", n_features)" << std::endl;
      break;
    }
    case MATRIX_UNIFORM_SET_OP: {
      stream << "  m" << out_ << " = uniform(" << float_data_0_
             << ", " << float_data_1_ << ", (n_features, n_features))"
             << std::endl;
      break;
    }
    // Do not add default clause. All ops should be supported here.
  }
  std::string instr_str = stream.str();
  return instr_str;
}

SerializedInstruction Instruction::Serialize() const {
  SerializedInstruction checkpoint_instruction;
  checkpoint_instruction.set_op(op_);
  checkpoint_instruction.set_in1(
      static_cast<int32_t>(in1_));  // Convert to higher precision.
  checkpoint_instruction.set_in2(
      static_cast<int32_t>(in2_));  // Convert to higher precision.
  checkpoint_instruction.set_out(
      static_cast<int32_t>(out_));  // Convert to higher precision.
  checkpoint_instruction.set_activation_data(activation_data_);
  checkpoint_instruction.set_float_data_0(float_data_0_);
  checkpoint_instruction.set_float_data_1(float_data_1_);
  checkpoint_instruction.set_float_data_2(float_data_2_);
  return checkpoint_instruction;
}

void Instruction::Deserialize(
    const SerializedInstruction& checkpoint_instruction) {
  op_ = checkpoint_instruction.op();

  // Convert to lower precision.
  CHECK_GE(checkpoint_instruction.in1(), numeric_limits<AddressT>::min());
  CHECK_LE(checkpoint_instruction.in1(), numeric_limits<AddressT>::max());
  in1_ = static_cast<AddressT>(checkpoint_instruction.in1());

  // Convert to lower precision.
  CHECK_GE(checkpoint_instruction.in2(), numeric_limits<AddressT>::min());
  CHECK_LE(checkpoint_instruction.in2(), numeric_limits<AddressT>::max());
  in2_ = static_cast<AddressT>(checkpoint_instruction.in2());

  // Convert to lower precision.
  CHECK_GE(checkpoint_instruction.out(), numeric_limits<AddressT>::min());
  CHECK_LE(checkpoint_instruction.out(), numeric_limits<AddressT>::max());
  out_ = static_cast<AddressT>(checkpoint_instruction.out());

  activation_data_ = checkpoint_instruction.activation_data();

  float_data_0_ = checkpoint_instruction.float_data_0();

  float_data_1_ = checkpoint_instruction.float_data_1();

  float_data_2_ = checkpoint_instruction.float_data_2();
}

}  // namespace automl_zero
