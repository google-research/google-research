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

#include <array>
#include <cstdint>
#include <random>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "definitions.h"
#include "definitions.proto.h"
#include "random_generator.h"
#include "random_generator_test_util.h"
#include "test_util.h"
#include "google/protobuf/enum-utils.h"
#include "gtest/gtest.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::proto2::contrib::utils::EnumerateEnumValues;
using ::std::abs;  // NOLINT
using ::std::find;  // NOLINT
using ::std::function;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::round;  // NOLINT
using ::std::unordered_set;  // NOLINT
using ::std::vector;  // NOLINT
using ::testing::Test;

constexpr double kTolerance = 0.0001;

enum DiffId : IntegerT {
  kNoDifference = 0,
  kDifferentOp = 1,
  kDifferentIn1 = 2,
  kDifferentIn2 = 3,
  kDifferentOut = 4,
  kDifferentActivationData = 5,
  kDifferentIndexData0 = 6,
  kDifferentFloatData0 = 7,
  kDifferentFloatData1 = 8,
  kDifferentFloatData2 = 9,
  kDifferentVectorData0 = 10,
  kDifferentVectorData1 = 11,
  kDifferentVectorData2 = 12,
  kDifferentVectorData3 = 13,
};

vector<Op> TestableOps() {
  auto all_ops = EnumerateEnumValues<Op>();
  vector<Op> testable_ops(all_ops.begin(), all_ops.end());
  testable_ops.erase(
      find(testable_ops.begin(), testable_ops.end(), UNSUPPORTED_OP));
  CHECK(!testable_ops.empty());
  return testable_ops;
}

Instruction NoOpInstruction() {
  Instruction instruction;
  return instruction;
}

Instruction BlankInstruction(const Op op) {
  Instruction instruction;
  instruction.op_ = op;
  return instruction;
}

Instruction SetOpAndRandomizeParams(
    const Op op, RandomGenerator* rand_gen) {
  Instruction instruction = NoOpInstruction();
  instruction.SetOpAndRandomizeParams(op, rand_gen);
  return instruction;
}

Instruction AlterParam(
    const Instruction& instruction, RandomGenerator* rand_gen) {
  Instruction modified_instruction = instruction;
  modified_instruction.AlterParam(rand_gen);
  return modified_instruction;
}

unordered_set<DiffId> Differences(
    const Instruction& instr1, const Instruction& instr2) {
  unordered_set<DiffId> differences;
  if (instr1.op_ != instr2.op_) {
    differences.insert(kDifferentOp);
  }
  if (instr1.in1_ != instr2.in1_) {
    differences.insert(kDifferentIn1);
  }
  if (instr1.in2_ != instr2.in2_) {
    differences.insert(kDifferentIn2);
  }
  if (instr1.out_ != instr2.out_) {
    differences.insert(kDifferentOut);
  }
  if (abs(instr1.GetActivationData() - instr2.GetActivationData()) >
      kActivationDataTolerance) {
    differences.insert(kDifferentActivationData);
  }
  if (instr1.GetIndexData0() != instr2.GetIndexData0()) {
    differences.insert(kDifferentIndexData0);
  }
  if (abs(instr1.GetFloatData0() - instr2.GetFloatData0()) >
      kFloatDataTolerance) {
    differences.insert(kDifferentFloatData0);
  }
  if (abs(instr1.GetFloatData1() - instr2.GetFloatData1()) >
      kFloatDataTolerance) {
    differences.insert(kDifferentFloatData1);
  }
  if (abs(instr1.GetFloatData2() - instr2.GetFloatData2()) >
      kFloatDataTolerance) {
    differences.insert(kDifferentFloatData2);
  }
  if (abs(instr1.GetVectorData()(0) - instr2.GetVectorData()(0)) >
      kActivationDataTolerance) {
    differences.insert(kDifferentVectorData0);
  }
  if (abs(instr1.GetVectorData()(1) - instr2.GetVectorData()(1)) >
      kActivationDataTolerance) {
    differences.insert(kDifferentVectorData1);
  }
  if (abs(instr1.GetVectorData()(2) - instr2.GetVectorData()(2)) >
      kActivationDataTolerance) {
    differences.insert(kDifferentVectorData2);
  }
  if (abs(instr1.GetVectorData()(3) - instr2.GetVectorData()(3)) >
      kActivationDataTolerance) {
    differences.insert(kDifferentVectorData3);
  }
  return differences;
}

DiffId RandomDifference(
    const Instruction& instr1, const Instruction& instr2) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  const unordered_set<DiffId> differences = Differences(instr1, instr2);
  if (differences.empty()) {
    return kNoDifference;
  }
  vector<DiffId> differences_list(
      differences.begin(), differences.end());
  return differences_list[
      rand_gen.UniformInteger(0, differences_list.size())];
}

IntegerT CountDifferences(
    const Instruction& instr1, const Instruction& instr2) {
  const unordered_set<DiffId> differences = Differences(instr1, instr2);
  return differences.size();
}

AddressT RandomizeIn1(Op op, RandomGenerator* rand_gen) {
  Instruction instr;
  instr.op_ = op;
  instr.RandomizeIn1(rand_gen);
  CHECK_EQ(instr.in2_, 0);
  CHECK_EQ(instr.out_, 0);
  CHECK_EQ(instr.GetActivationData(), 0.0);
  CHECK_EQ(instr.GetIndexData0(), 0);
  CHECK_EQ(instr.GetFloatData0(), 0.0);
  CHECK_EQ(instr.GetFloatData1(), 0.0);
  CHECK_EQ(instr.GetFloatData2(), 0.0);
  CHECK_EQ(instr.GetVectorData().norm(), 0.0);
  return instr.in1_;
}

AddressT RandomizeIn2(Op op, RandomGenerator* rand_gen) {
  Instruction instr;
  instr.op_ = op;
  instr.RandomizeIn2(rand_gen);
  CHECK_EQ(instr.in1_, 0);
  CHECK_EQ(instr.out_, 0);
  CHECK_EQ(instr.GetActivationData(), 0.0);
  CHECK_EQ(instr.GetIndexData0(), 0);
  CHECK_EQ(instr.GetFloatData0(), 0.0);
  CHECK_EQ(instr.GetFloatData1(), 0.0);
  CHECK_EQ(instr.GetFloatData2(), 0.0);
  CHECK_EQ(instr.GetVectorData().norm(), 0.0);
  return instr.in2_;
}

AddressT RandomizeOut(Op op, RandomGenerator* rand_gen) {
  Instruction instr;
  instr.op_ = op;
  instr.RandomizeOut(rand_gen);
  CHECK_EQ(instr.in1_, 0);
  CHECK_EQ(instr.in2_, 0);
  CHECK_EQ(instr.GetActivationData(), 0.0);
  CHECK_EQ(instr.GetIndexData0(), 0);
  CHECK_EQ(instr.GetFloatData0(), 0.0);
  CHECK_EQ(instr.GetFloatData1(), 0.0);
  CHECK_EQ(instr.GetFloatData2(), 0.0);
  CHECK_EQ(instr.GetVectorData().norm(), 0.0);
  return instr.out_;
}

void RandomizeData(Op op, RandomGenerator* rand_gen, Instruction* instr) {
  instr->FillWithNoOp();
  instr->op_ = op;
  instr->RandomizeData(rand_gen);
  CHECK_EQ(instr->in1_, 0);
  CHECK_EQ(instr->in2_, 0);
  CHECK_EQ(instr->out_, 0);
}

TEST(FloatIndexConversion, RecoversIndex) {
  EXPECT_EQ(FloatToIndex(IndexToFloat(0, 4), 4), 0);
  EXPECT_EQ(FloatToIndex(IndexToFloat(1, 4), 4), 1);
  EXPECT_EQ(FloatToIndex(IndexToFloat(2, 4), 4), 2);
  EXPECT_EQ(FloatToIndex(IndexToFloat(3, 4), 4), 3);
  EXPECT_EQ(FloatToIndex(IndexToFloat(11, 16), 16), 11);
}

TEST(InstructionTest, IsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copy_assignable<Instruction>::value);
  EXPECT_TRUE(std::is_trivially_copy_constructible<Instruction>::value);
}

TEST(InstructionTest, InstructionIsSmall) {
  EXPECT_LE(sizeof(Instruction), 48);
}

TEST(InstructionTest, Constructor_Default) {
  Instruction instruction;
  EXPECT_EQ(instruction.op_, NO_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 0);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_EQ(instruction.GetFloatData0(), 0.0);
  EXPECT_EQ(instruction.GetFloatData1(), 0.0);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest, Constructor_IntegerData) {
  Instruction instruction(IntegerDataSetter(10));
  EXPECT_EQ(instruction.op_, NO_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 0);
  EXPECT_EQ(instruction.GetIntegerData(), 10);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_EQ(instruction.GetFloatData0(), 0.0);
  EXPECT_EQ(instruction.GetFloatData1(), 0.0);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest, Constructor_Op_Address_Address) {
  Instruction instruction(SCALAR_ABS_OP, 10, 20);
  EXPECT_EQ(instruction.op_, SCALAR_ABS_OP);
  EXPECT_EQ(instruction.in1_, 10);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 20);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_EQ(instruction.GetFloatData0(), 0.0);
  EXPECT_EQ(instruction.GetFloatData1(), 0.0);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest, Constructor_Op_Address_Address_Address) {
  Instruction instruction(SCALAR_SUM_OP, 10, 20, 30);
  EXPECT_EQ(instruction.op_, SCALAR_SUM_OP);
  EXPECT_EQ(instruction.in1_, 10);
  EXPECT_EQ(instruction.in2_, 20);
  EXPECT_EQ(instruction.out_, 30);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_EQ(instruction.GetFloatData0(), 0.0);
  EXPECT_EQ(instruction.GetFloatData1(), 0.0);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest, Constructor_Op_Address_ActivationData) {
  Instruction instruction(SCALAR_CONST_SET_OP, 10, ActivationDataSetter(2.2));
  EXPECT_EQ(instruction.op_, SCALAR_CONST_SET_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 10);
  EXPECT_LE(abs(instruction.GetActivationData() - 2.2),
            kActivationDataTolerance);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_EQ(instruction.GetFloatData0(), 0.0);
  EXPECT_EQ(instruction.GetFloatData1(), 0.0);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest, Constructor_Op_Address_FloatData_FloatData) {
  Instruction instruction(SCALAR_GAUSSIAN_SET_OP, 10,
                          FloatDataSetter(2.2), FloatDataSetter(3.3));
  EXPECT_EQ(instruction.op_, SCALAR_GAUSSIAN_SET_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 10);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_LE(abs(instruction.GetFloatData0() - 2.2),
            kFloatDataTolerance);
  EXPECT_LE(abs(instruction.GetFloatData1() - 3.3),
            kFloatDataTolerance);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest,
     ConstructorCanStoreIndex_Op_Address_FloatData_FloatData) {
  Instruction instruction(
      MATRIX_GAUSSIAN_SET_OP, 10,
      FloatDataSetter(IndexToFloat(3, 4)),
      FloatDataSetter(IndexToFloat(2, 4)));
  EXPECT_EQ(instruction.op_, MATRIX_GAUSSIAN_SET_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 10);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_LE(FloatToIndex(instruction.GetFloatData0(), 4), 3);
  EXPECT_LE(FloatToIndex(instruction.GetFloatData1(), 4), 2);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest, Constructor_Op_Address_FloatData_FloatData_FloatData) {
  Instruction instruction(
      MATRIX_GAUSSIAN_SET_OP, 10, FloatDataSetter(2.2), FloatDataSetter(3.3),
      FloatDataSetter(4.4));
  EXPECT_EQ(instruction.op_, MATRIX_GAUSSIAN_SET_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 10);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_LE(abs(instruction.GetFloatData0() - 2.2),
            kFloatDataTolerance);
  EXPECT_LE(abs(instruction.GetFloatData1() - 3.3),
            kFloatDataTolerance);
  EXPECT_LE(abs(instruction.GetFloatData2() - 4.4),
            kFloatDataTolerance);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest,
     ConstructorCanStoreIndexes_Op_Address_FloatData_FloatData_FloatData) {
  Instruction instruction(
      MATRIX_GAUSSIAN_SET_OP, 10,
      FloatDataSetter(IndexToFloat(1, 4)),
      FloatDataSetter(IndexToFloat(0, 4)),
      FloatDataSetter(IndexToFloat(4, 4)));
  EXPECT_EQ(instruction.op_, MATRIX_GAUSSIAN_SET_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 10);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_LE(FloatToIndex(instruction.GetFloatData0(), 4), 1);
  EXPECT_LE(FloatToIndex(instruction.GetFloatData1(), 4), 0);
  EXPECT_LE(FloatToIndex(instruction.GetFloatData2(), 4), 4);
  EXPECT_EQ(instruction.GetVectorData().norm(), 0.0);
}

TEST(InstructionTest, Constructor_Op_Address_VectorData) {
  Vector<4> vector;
  vector << kActivationAsDataMin, -1.6875, 0.0, kActivationAsDataMax;
  Instruction instruction(
      VECTOR_CONST_SET_OLD_OP, 10, VectorDataSetter(vector));
  EXPECT_EQ(instruction.op_, VECTOR_CONST_SET_OLD_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 10);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 0);
  EXPECT_EQ(instruction.GetFloatData0(), 0.0);
  EXPECT_EQ(instruction.GetFloatData1(), 0.0);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_LE((instruction.GetVectorData() - vector).norm(),
            kVectorDataTolerance);
}

TEST(InstructionTest, Constructor_Op_Address_IndexData_VectorData) {
  Vector<4> vector;
  vector << kActivationAsDataMin, -1.6875, 0.0, kActivationAsDataMax;
  Instruction instruction(
      VECTOR_CONST_SET_OLD_OP, 10, IndexDataSetter(20),
      VectorDataSetter(vector));
  EXPECT_EQ(instruction.op_, VECTOR_CONST_SET_OLD_OP);
  EXPECT_EQ(instruction.in1_, 0);
  EXPECT_EQ(instruction.in2_, 0);
  EXPECT_EQ(instruction.out_, 10);
  EXPECT_EQ(instruction.GetActivationData(), 0.0);
  EXPECT_EQ(instruction.GetIndexData0(), 20);
  EXPECT_EQ(instruction.GetFloatData0(), 0.0);
  EXPECT_EQ(instruction.GetFloatData1(), 0.0);
  EXPECT_EQ(instruction.GetFloatData2(), 0.0);
  EXPECT_LE((instruction.GetVectorData() - vector).norm(),
            kVectorDataTolerance);
}

TEST(InstructionTest, CopyConstructor) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  Instruction instruction1;
  instruction1.SetOpAndRandomizeParams(SCALAR_SUM_OP, &rand_gen);
  Instruction instruction2 = instruction1;
  EXPECT_TRUE(instruction1 == instruction2);
}

TEST(InstructionTest, CopyAssignmentOperator) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  Instruction instruction1;
  instruction1.SetOpAndRandomizeParams(SCALAR_SUM_OP, &rand_gen);
  Instruction instruction2;
  instruction2 = instruction1;
  EXPECT_TRUE(instruction1 == instruction2);
}

TEST(InstructionTest, EqualsOperatorThroughSomeExamples) {
  Instruction instruction(VECTOR_SUM_OP, 1, 2, 3);
  Instruction instruction_same(VECTOR_SUM_OP, 1, 2, 3);
  Instruction instruction_diff_op(SCALAR_DIFF_OP, 1, 2, 3);
  Instruction instruction_diff_in1(VECTOR_SUM_OP, 2, 2, 3);
  Instruction instruction_diff_in2(VECTOR_SUM_OP, 1, 0, 3);
  Instruction instruction_diff_out(VECTOR_SUM_OP, 1, 2, 2);
  EXPECT_TRUE(instruction == instruction_same);
  EXPECT_TRUE(instruction != instruction_diff_op);
  EXPECT_TRUE(instruction != instruction_diff_in1);
  EXPECT_TRUE(instruction != instruction_diff_in2);
  EXPECT_TRUE(instruction != instruction_diff_out);
}

TEST(InstructionTest, EqualsOperatorForDataThroughSomeExamples) {
  Vector<4> vector;
  vector << -0.1, 0.2, -0.3, 0.4;
  Vector<4> vector_diff;
  vector_diff << -0.1, 0.02, -0.3, 0.4;
  Instruction instruction(
      VECTOR_CONST_SET_OLD_OP, 3, VectorDataSetter(vector));
  Instruction instruction_same(
      VECTOR_CONST_SET_OLD_OP, 3, VectorDataSetter(vector));
  Instruction instruction_diff_data(
      VECTOR_CONST_SET_OLD_OP, 3, VectorDataSetter(vector_diff));
  EXPECT_TRUE(instruction == instruction_same);
  EXPECT_TRUE(instruction != instruction_diff_data);
}

TEST(InstructionTest, EqualsOperatorConsidersOp) {
  vector<Op> ops = TestableOps();
  for (IntegerT i = 0; i < ops.size(); ++i) {
    Op op = ops[i];
    Instruction instr1(op, 0, 0, 0);
    Instruction instr2(op, 0, 0, 0);
    EXPECT_TRUE(instr1 == instr2);
    EXPECT_FALSE(instr1 != instr2);
  }
  for (IntegerT i = 0; i < ops.size(); ++i) {
    for (IntegerT j = i + 1; j < ops.size(); ++j) {
      Instruction instr1(ops[i], 0, 0, 0);
      Instruction instr2(ops[j], 0, 0, 0);
      EXPECT_FALSE(instr1 == instr2);
      EXPECT_TRUE(instr1 != instr2);
    }
  }
}

TEST(InstructionTest, EqualsOperatorConsidersStuffOtherThanOp) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  for (Op op : TestableOps()) {
    Instruction instr;
    instr.SetOpAndRandomizeParams(op, &rand_gen);
    Instruction same_instr(instr);
    EXPECT_TRUE(instr == same_instr);
    EXPECT_FALSE(instr != same_instr);

    Instruction other_instr(instr);
    other_instr.AlterParam(&rand_gen);
    if (Differences(instr, other_instr).empty()) {
      EXPECT_TRUE(instr == other_instr);
      EXPECT_FALSE(instr != other_instr);
    } else {
      EXPECT_FALSE(instr == other_instr);
      EXPECT_TRUE(instr != other_instr);
    }
  }
}

TEST(InstructionTest, RandomizesIn1) {
  CHECK_GE(kMaxScalarAddresses, 4);
  CHECK_GE(kMaxVectorAddresses, 3);
  CHECK_GE(kMaxMatrixAddresses, 2);
  RandomGenerator rand_gen = SimpleRandomGenerator();
  const AddressT range_start = 0;
  auto scalar_range = Range(range_start, kMaxScalarAddresses);
  auto vector_range = Range(range_start, kMaxVectorAddresses);
  auto matrix_range = Range(range_start, kMaxMatrixAddresses);
  for (const Op op : TestableOps()) {
    switch (op) {
      case NO_OP:
      case SCALAR_CONST_SET_OP:
      case VECTOR_CONST_SET_OLD_OP:
      case VECTOR_CONST_SET_OP:
      case MATRIX_ROW_CONST_SET_OLD_OP:
      case MATRIX_CONST_SET_OP:
      case SCALAR_GAUSSIAN_SET_OP:
      case VECTOR_GAUSSIAN_SET_OLD_OP:
      case VECTOR_GAUSSIAN_SET_OP:
      case MATRIX_GAUSSIAN_SET_OLD_OP:
      case MATRIX_GAUSSIAN_SET_OP:
      case SCALAR_UNIFORM_SET_OP:
      case VECTOR_UNIFORM_SET_OP:
      case MATRIX_UNIFORM_SET_OP:
      case SCALAR_BETA_SET_OP:
      case VECTOR_BETA_SET_OP:
      case MATRIX_BETA_SET_OP:
      case SCALAR_PRINT_OP:
      case VECTOR_PRINT_OP:
      case MATRIX_PRINT_OP:
        EXPECT_DEATH({RandomizeIn1(op, &rand_gen);}, "Invalid op");
        break;
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
      case SCALAR_VECTOR_PRODUCT_OP:
      case SCALAR_MATRIX_PRODUCT_OP:
      case SCALAR_RECIPROCAL_OP:
      case SCALAR_BROADCAST_OP:
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeIn1(op, &rand_gen);}),
            scalar_range, scalar_range));
        break;
      case VECTOR_SUM_OP:
      case VECTOR_DIFF_OP:
      case VECTOR_PRODUCT_OP:
      case VECTOR_DIVISION_OP:
      case VECTOR_MIN_OP:
      case VECTOR_MAX_OP:
      case VECTOR_ABS_OP:
      case VECTOR_HEAVYSIDE_OP:
      case VECTOR_RELU_OP:
      case VECTOR_INNER_PRODUCT_OP:
      case VECTOR_OUTER_PRODUCT_OP:
      case VECTOR_NORM_OP:
      case VECTOR_MEAN_OP:
      case VECTOR_ST_DEV_OP:
      case VECTOR_RECIPROCAL_OP:
      case VECTOR_COLUMN_BROADCAST_OP:
      case VECTOR_ROW_BROADCAST_OP:
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeIn1(op, &rand_gen);}),
            vector_range, vector_range));
        break;
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
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeIn1(op, &rand_gen);}),
            matrix_range, matrix_range));
        break;
      case UNSUPPORTED_OP:
        LOG(FATAL) << "Unsupported op." << std::endl;
      // Do not add default clause. All ops should be supported here.
    }
  }
}

TEST(InstructionTest, RandomizesIn2) {
  CHECK_GE(kMaxScalarAddresses, 4);
  CHECK_GE(kMaxVectorAddresses, 3);
  CHECK_GE(kMaxMatrixAddresses, 2);
  RandomGenerator rand_gen = SimpleRandomGenerator();
  const AddressT range_start = 0;
  auto scalar_range = Range(range_start, kMaxScalarAddresses);
  auto vector_range = Range(range_start, kMaxVectorAddresses);
  auto matrix_range = Range(range_start, kMaxMatrixAddresses);
  for (const Op op : TestableOps()) {
    switch (op) {
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
      case VECTOR_ABS_OP:
      case VECTOR_HEAVYSIDE_OP:
      case VECTOR_RELU_OP:
      case VECTOR_CONST_SET_OLD_OP:
      case VECTOR_CONST_SET_OP:
      case MATRIX_ABS_OP:
      case MATRIX_HEAVYSIDE_OP:
      case MATRIX_ROW_CONST_SET_OLD_OP:
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
      case VECTOR_GAUSSIAN_SET_OLD_OP:
      case VECTOR_GAUSSIAN_SET_OP:
      case MATRIX_GAUSSIAN_SET_OLD_OP:
      case MATRIX_GAUSSIAN_SET_OP:
      case SCALAR_UNIFORM_SET_OP:
      case VECTOR_UNIFORM_SET_OP:
      case MATRIX_UNIFORM_SET_OP:
      case SCALAR_BETA_SET_OP:
      case VECTOR_BETA_SET_OP:
      case MATRIX_BETA_SET_OP:
      case SCALAR_RECIPROCAL_OP:
      case SCALAR_BROADCAST_OP:
      case VECTOR_RECIPROCAL_OP:
      case MATRIX_RECIPROCAL_OP:
      case MATRIX_ROW_NORM_OP:
      case MATRIX_COLUMN_NORM_OP:
      case VECTOR_COLUMN_BROADCAST_OP:
      case VECTOR_ROW_BROADCAST_OP:
      case SCALAR_PRINT_OP:
      case VECTOR_PRINT_OP:
      case MATRIX_PRINT_OP:
        EXPECT_DEATH({RandomizeIn2(op, &rand_gen);}, "Invalid op");
        break;
      case SCALAR_SUM_OP:
      case SCALAR_DIFF_OP:
      case SCALAR_PRODUCT_OP:
      case SCALAR_DIVISION_OP:
      case SCALAR_MIN_OP:
      case SCALAR_MAX_OP:
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeIn2(op, &rand_gen);}),
            scalar_range, scalar_range));
        break;
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
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeIn2(op, &rand_gen);}),
            vector_range, vector_range));
        break;
      case MATRIX_SUM_OP:
      case MATRIX_DIFF_OP:
      case MATRIX_PRODUCT_OP:
      case MATRIX_DIVISION_OP:
      case MATRIX_MIN_OP:
      case MATRIX_MAX_OP:
      case SCALAR_MATRIX_PRODUCT_OP:
      case MATRIX_MATRIX_PRODUCT_OP:
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeIn2(op, &rand_gen);}),
            matrix_range, matrix_range));
        break;
      case UNSUPPORTED_OP:
        LOG(FATAL) << "Unsupported op." << std::endl;
      // Do not add default clause. All ops should be supported here.
    }
  }
}

TEST(InstructionTest, RandomizesOut) {
  CHECK_GE(kMaxScalarAddresses, 4);
  CHECK_GE(kMaxVectorAddresses, 3);
  CHECK_GE(kMaxMatrixAddresses, 2);
  RandomGenerator rand_gen = SimpleRandomGenerator();
  auto scalar_range =
      Range(kFirstOutScalarAddress, kMaxScalarAddresses);
  auto vector_range =
      Range(kFirstOutVectorAddress, kMaxVectorAddresses);
  auto matrix_range =
      Range(kFirstOutMatrixAddress, kMaxMatrixAddresses);
  for (const Op op : TestableOps()) {
    switch (op) {
      case NO_OP:
      case SCALAR_PRINT_OP:
      case VECTOR_PRINT_OP:
      case MATRIX_PRINT_OP:
        EXPECT_DEATH({RandomizeOut(op, &rand_gen);}, "Invalid op");
        break;
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
      case VECTOR_INNER_PRODUCT_OP:
      case VECTOR_NORM_OP:
      case MATRIX_NORM_OP:
      case VECTOR_MEAN_OP:
      case VECTOR_ST_DEV_OP:
      case MATRIX_MEAN_OP:
      case MATRIX_ST_DEV_OP:
      case SCALAR_GAUSSIAN_SET_OP:
      case SCALAR_UNIFORM_SET_OP:
      case SCALAR_BETA_SET_OP:
      case SCALAR_RECIPROCAL_OP:
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeOut(op, &rand_gen);}),
            scalar_range, scalar_range));
        break;
      case VECTOR_SUM_OP:
      case VECTOR_DIFF_OP:
      case VECTOR_PRODUCT_OP:
      case VECTOR_DIVISION_OP:
      case VECTOR_MIN_OP:
      case VECTOR_MAX_OP:
      case VECTOR_ABS_OP:
      case VECTOR_HEAVYSIDE_OP:
      case VECTOR_RELU_OP:
      case VECTOR_CONST_SET_OLD_OP:
      case VECTOR_CONST_SET_OP:
      case SCALAR_VECTOR_PRODUCT_OP:
      case MATRIX_VECTOR_PRODUCT_OP:
      case MATRIX_ROW_MEAN_OP:
      case MATRIX_ROW_ST_DEV_OP:
      case VECTOR_GAUSSIAN_SET_OLD_OP:
      case VECTOR_GAUSSIAN_SET_OP:
      case VECTOR_UNIFORM_SET_OP:
      case VECTOR_BETA_SET_OP:
      case SCALAR_BROADCAST_OP:
      case VECTOR_RECIPROCAL_OP:
      case MATRIX_ROW_NORM_OP:
      case MATRIX_COLUMN_NORM_OP:
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeOut(op, &rand_gen);}),
            vector_range, vector_range));
        break;
      case MATRIX_SUM_OP:
      case MATRIX_DIFF_OP:
      case MATRIX_PRODUCT_OP:
      case MATRIX_DIVISION_OP:
      case MATRIX_MIN_OP:
      case MATRIX_MAX_OP:
      case MATRIX_ABS_OP:
      case MATRIX_HEAVYSIDE_OP:
      case MATRIX_ROW_CONST_SET_OLD_OP:
      case MATRIX_CONST_SET_OP:
      case VECTOR_OUTER_PRODUCT_OP:
      case SCALAR_MATRIX_PRODUCT_OP:
      case MATRIX_TRANSPOSE_OP:
      case MATRIX_MATRIX_PRODUCT_OP:
      case MATRIX_GAUSSIAN_SET_OLD_OP:
      case MATRIX_GAUSSIAN_SET_OP:
      case MATRIX_UNIFORM_SET_OP:
      case MATRIX_BETA_SET_OP:
      case MATRIX_RECIPROCAL_OP:
      case VECTOR_COLUMN_BROADCAST_OP:
      case VECTOR_ROW_BROADCAST_OP:
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>([op, &rand_gen](){
              return RandomizeOut(op, &rand_gen);}),
            matrix_range, matrix_range));
        break;
      case UNSUPPORTED_OP:
        LOG(FATAL) << "Unsupported op." << std::endl;
      // Do not add default clause. All ops should be supported here.
    }
  }
}

TEST(InstructionTest, RandomizesData) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  auto feature_index_range = Range(kFirstFeaturesIndex, FeatureIndexT(4));
  for (const Op op : TestableOps()) {
    switch (op) {
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
      case VECTOR_RELU_OP:
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
      case SCALAR_RECIPROCAL_OP:
      case SCALAR_BROADCAST_OP:
      case VECTOR_RECIPROCAL_OP:
      case MATRIX_RECIPROCAL_OP:
      case MATRIX_ROW_NORM_OP:
      case MATRIX_COLUMN_NORM_OP:
      case VECTOR_COLUMN_BROADCAST_OP:
      case VECTOR_ROW_BROADCAST_OP:
      case MATRIX_ROW_ST_DEV_OP:
      case SCALAR_PRINT_OP:
      case VECTOR_PRINT_OP:
      case MATRIX_PRINT_OP: {
        Instruction instr;
        EXPECT_DEATH({RandomizeData(op, &rand_gen, &instr);}, "Invalid op");
        break;
      }
      case SCALAR_CONST_SET_OP: {
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(10.0 * instr.GetActivationData()));
            }),
            Range<IntegerT>(-100, 101), Range<IntegerT>(-10, 11)));
        break;
      }
      case VECTOR_CONST_SET_OLD_OP:
        // Deprecated. Data ranges tested in instruction_manual_test.cc.
        break;
      case VECTOR_CONST_SET_OP: {
        EXPECT_TRUE(IsEventually(
            function<FeatureIndexT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return FloatToIndex(instr.GetFloatData0(), 4);
            }),
            feature_index_range, feature_index_range));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(10.0 * instr.GetFloatData1()));
            }),
            Range<IntegerT>(-10, 11), Range<IntegerT>(-10, 11)));
        break;
      }
      case MATRIX_ROW_CONST_SET_OLD_OP: {
        // Deprecated.
        // Data row entry ranges tested in instruction_manual_test.cc.
        EXPECT_TRUE(IsEventually(
            function<AddressT(void)>(
                [&](){return SetOpAndRandomizeParams(op, &rand_gen).out_;}),
            Range(kFirstOutMatrixAddress, kMaxMatrixAddresses),
            Range(kFirstOutMatrixAddress, kMaxMatrixAddresses)));
        EXPECT_TRUE(IsEventually(
            function<FeatureIndexT(void)>(
                [&](){
                  Instruction instr(op, &rand_gen);
                  return instr.GetIndexData0();
                }),
            Range(FeatureIndexT(0), FeatureIndexT(4)),
            Range(FeatureIndexT(0), FeatureIndexT(4)), 10.0));
        break;
      }
      case MATRIX_CONST_SET_OP: {
        EXPECT_TRUE(IsEventually(
            function<FeatureIndexT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return FloatToIndex(instr.GetFloatData0(), 4);
            }),
            feature_index_range, feature_index_range));
        EXPECT_TRUE(IsEventually(
            function<FeatureIndexT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return FloatToIndex(instr.GetFloatData1(), 4);
            }),
            feature_index_range, feature_index_range));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(10.0 * instr.GetFloatData2()));
            }),
            Range<IntegerT>(-10, 11), Range<IntegerT>(-10, 11)));
        break;
      }
      case SCALAR_GAUSSIAN_SET_OP:
      case VECTOR_GAUSSIAN_SET_OP:
      case MATRIX_GAUSSIAN_SET_OP: {
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(10.0 * instr.GetFloatData0()));
            }),
            Range<IntegerT>(-10, 11), Range<IntegerT>(-10, 11)));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(10.0 * instr.GetFloatData1()));
            }),
            Range<IntegerT>(0, 11), Range<IntegerT>(0, 11)));
        break;
      }
      case VECTOR_GAUSSIAN_SET_OLD_OP:
      case MATRIX_GAUSSIAN_SET_OLD_OP:
        // Deprecated.
        // Data ranges tested in instruction_manual_test.cc.
        break;
      case SCALAR_UNIFORM_SET_OP:
      case VECTOR_UNIFORM_SET_OP:
      case MATRIX_UNIFORM_SET_OP: {
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(3.0 * instr.GetFloatData0()));
            }),
            Range<IntegerT>(-3, 4), Range<IntegerT>(-3, 4)));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(3.0 * instr.GetFloatData1()));
            }),
            Range<IntegerT>(-3, 4), Range<IntegerT>(-3, 4)));
        break;
      }
      case SCALAR_BETA_SET_OP:
      case VECTOR_BETA_SET_OP:
      case MATRIX_BETA_SET_OP: {
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(10.0 * instr.GetFloatData0()));
            }),
            Range<IntegerT>(0, 21), Range<IntegerT>(0, 21)));
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([op, &rand_gen](){
              Instruction instr;
              RandomizeData(op, &rand_gen, &instr);
              return static_cast<IntegerT>(
                  round(10.0 * instr.GetFloatData1()));
            }),
            Range<IntegerT>(0, 21), Range<IntegerT>(0, 21)));
        break;
      }
      case UNSUPPORTED_OP:
        LOG(FATAL) << "Unsupported op." << std::endl;
      // Do not add default clause. All ops should be supported here.
    }
  }
}

TEST(InstructionTest, RandomizesCorrectFields) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  for (const Op op : TestableOps()) {
    Instruction blank_instr = BlankInstruction(op);
    switch (op) {
      case NO_OP:
      case SCALAR_PRINT_OP:
      case VECTOR_PRINT_OP:
      case MATRIX_PRINT_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0}, {0}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference}, {kNoDifference}));
        break;
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
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2, 3}, {3}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentIn1, kDifferentIn2, kDifferentOut},
            {kDifferentIn1, kDifferentIn2, kDifferentOut}));
        break;
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
      case VECTOR_ABS_OP:
      case VECTOR_HEAVYSIDE_OP:
      case VECTOR_RELU_OP:
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
      case SCALAR_RECIPROCAL_OP:
      case SCALAR_BROADCAST_OP:
      case VECTOR_RECIPROCAL_OP:
      case MATRIX_RECIPROCAL_OP:
      case MATRIX_ROW_NORM_OP:
      case MATRIX_COLUMN_NORM_OP:
      case VECTOR_COLUMN_BROADCAST_OP:
      case VECTOR_ROW_BROADCAST_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2}, {2}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentIn1, kDifferentOut},
            {kDifferentIn1, kDifferentOut}));
        break;
      case SCALAR_CONST_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2}, {2}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentActivationData},
            {kDifferentOut, kDifferentActivationData}));
        break;
      case VECTOR_CONST_SET_OLD_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2, 3, 4, 5}, {5}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentVectorData0,
             kDifferentVectorData1, kDifferentVectorData2,
             kDifferentVectorData3},
            {kDifferentOut, kDifferentVectorData0,
             kDifferentVectorData1, kDifferentVectorData2,
             kDifferentVectorData3}));
        break;
      case VECTOR_CONST_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2, 3}, {3}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentFloatData0,
             kDifferentFloatData1},
            {kDifferentOut, kDifferentFloatData0, kDifferentFloatData1}));
        break;
      case MATRIX_ROW_CONST_SET_OLD_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2, 3, 4, 5, 6}, {6}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentIndexData0,
             kDifferentVectorData0, kDifferentVectorData1,
             kDifferentVectorData2, kDifferentVectorData3},
            {kDifferentOut, kDifferentIndexData0, kDifferentVectorData0,
             kDifferentVectorData1, kDifferentVectorData2,
             kDifferentVectorData3}));
        break;
      case MATRIX_CONST_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2, 3, 4}, {4}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentFloatData0,
             kDifferentFloatData1, kDifferentFloatData2},
            {kDifferentOut, kDifferentFloatData0, kDifferentFloatData1,
             kDifferentFloatData2}));
        break;
      case SCALAR_GAUSSIAN_SET_OP:
      case VECTOR_GAUSSIAN_SET_OP:
      case MATRIX_GAUSSIAN_SET_OP:
      case SCALAR_UNIFORM_SET_OP:
      case VECTOR_UNIFORM_SET_OP:
      case MATRIX_UNIFORM_SET_OP:
      case SCALAR_BETA_SET_OP:
      case VECTOR_BETA_SET_OP:
      case MATRIX_BETA_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2, 3}, {3}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentFloatData0,
             kDifferentFloatData1},
            {kDifferentOut, kDifferentFloatData0, kDifferentFloatData1}));
        break;
      case VECTOR_GAUSSIAN_SET_OLD_OP:
      case MATRIX_GAUSSIAN_SET_OLD_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {0, 1, 2}, {2}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  blank_instr, SetOpAndRandomizeParams(op, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentActivationData},
            {kDifferentOut, kDifferentActivationData}));
        break;
      case UNSUPPORTED_OP:
        LOG(FATAL) << "Unsupported op." << std::endl;
      // Do not add default clause. All ops should be supported here.
    }
  }
}

TEST(InstructionTest, AltersCorrectFields) {
  RandomGenerator rand_gen = SimpleRandomGenerator();
  for (const Op op : TestableOps()) {
    const Instruction instr = SetOpAndRandomizeParams(op, &rand_gen);
    switch (op) {
      case NO_OP:
      case SCALAR_PRINT_OP:
      case VECTOR_PRINT_OP:
      case MATRIX_PRINT_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0}, {0}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference}, {kNoDifference}));
        break;
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
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentIn1, kDifferentIn2, kDifferentOut},
            {kDifferentIn1, kDifferentIn2, kDifferentOut}));
        break;
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
      case VECTOR_ABS_OP:
      case VECTOR_HEAVYSIDE_OP:
      case VECTOR_RELU_OP:
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
      case SCALAR_RECIPROCAL_OP:
      case SCALAR_BROADCAST_OP:
      case VECTOR_RECIPROCAL_OP:
      case MATRIX_RECIPROCAL_OP:
      case MATRIX_ROW_NORM_OP:
      case MATRIX_COLUMN_NORM_OP:
      case VECTOR_COLUMN_BROADCAST_OP:
      case VECTOR_ROW_BROADCAST_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentIn1, kDifferentOut},
            {kDifferentIn1, kDifferentOut}));
        break;
      case SCALAR_CONST_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentActivationData},
            {kDifferentOut, kDifferentActivationData}));
        break;
      case VECTOR_CONST_SET_OLD_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentVectorData0,
             kDifferentVectorData1, kDifferentVectorData2,
             kDifferentVectorData3},
            {kDifferentOut, kDifferentVectorData0, kDifferentVectorData1,
             kDifferentVectorData2, kDifferentVectorData3}));
        break;
      case VECTOR_CONST_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut,
             kDifferentFloatData0, kDifferentFloatData1},
            {kDifferentOut, kDifferentFloatData0, kDifferentFloatData1}));
        break;
      case MATRIX_ROW_CONST_SET_OLD_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentVectorData0,
             kDifferentVectorData1, kDifferentVectorData2,
             kDifferentVectorData3},
            {kDifferentOut, kDifferentVectorData0,
             kDifferentVectorData1, kDifferentVectorData2,
             kDifferentVectorData3}));
        break;
      case MATRIX_CONST_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentFloatData0,
             kDifferentFloatData1, kDifferentFloatData2},
            {kDifferentOut, kDifferentFloatData0,
             kDifferentFloatData1, kDifferentFloatData2}));
        break;
      case SCALAR_GAUSSIAN_SET_OP:
      case VECTOR_GAUSSIAN_SET_OP:
      case MATRIX_GAUSSIAN_SET_OP:
      case SCALAR_BETA_SET_OP:
      case VECTOR_BETA_SET_OP:
      case MATRIX_BETA_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentFloatData0,
             kDifferentFloatData1},
            {kDifferentOut, kDifferentFloatData0, kDifferentFloatData1}));
        break;
      case SCALAR_UNIFORM_SET_OP:
      case VECTOR_UNIFORM_SET_OP:
      case MATRIX_UNIFORM_SET_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            // We allow modifying 2 because sometimes the order of the `low` and
            // `high` must be flipped (e.g. if the `low` is mutated to be above
            // the `high`).
            {0, 1, 2},
            {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentFloatData0,
             kDifferentFloatData1},
            {kDifferentOut, kDifferentFloatData0, kDifferentFloatData1}));
        break;
      case VECTOR_GAUSSIAN_SET_OLD_OP:
      case MATRIX_GAUSSIAN_SET_OLD_OP:
        EXPECT_TRUE(IsEventually(
            function<IntegerT(void)>([&](){
              return CountDifferences(
                  instr, AlterParam(instr, &rand_gen));}),
            {0, 1}, {1}));
        EXPECT_TRUE(IsEventually(
            function<DiffId(void)>([&](){
              return RandomDifference(
                  instr, AlterParam(instr, &rand_gen));}),
            {kNoDifference, kDifferentOut, kDifferentActivationData},
            {kDifferentOut, kDifferentActivationData}));
        break;
      case UNSUPPORTED_OP:
        LOG(FATAL) << "Unsupported op." << std::endl;
      // Do not add default clause. All ops should be supported here.
    }
  }
}

TEST(DiscretizationTest, WorksCorrectly) {
  EXPECT_EQ(Undiscretize(Discretize(kActivationAsDataMin)),
            kActivationAsDataMin);
  EXPECT_EQ(Undiscretize(Discretize(-1.6875)),
            -1.6875);
  EXPECT_EQ(Undiscretize(Discretize(0.0)),
            0.0);
  EXPECT_EQ(Undiscretize(Discretize(kActivationAsDataMax)),
            kActivationAsDataMax);
}

TEST(DiscretizationTest, ClipsCorrectly) {
  EXPECT_EQ(Undiscretize(Discretize(-10.5)),
            kActivationAsDataMin);
  EXPECT_EQ(Undiscretize(Discretize(-2.1)),
            kActivationAsDataMin);
  EXPECT_EQ(Undiscretize(Discretize(2.1)),
            kActivationAsDataMax);
  EXPECT_EQ(Undiscretize(Discretize(10.5)),
            kActivationAsDataMax);
}

TEST(DiscretizationTest, ApproximatesCorrectly) {
  EXPECT_EQ(Undiscretize(Discretize(-0.6 * kActivationAsDataStep)),
            -kActivationAsDataStep);
  EXPECT_EQ(Undiscretize(Discretize(-0.4 * kActivationAsDataStep)),
            0.0);
  EXPECT_EQ(Undiscretize(Discretize(0.4 * kActivationAsDataStep)),
            0.0);
  EXPECT_EQ(Undiscretize(Discretize(0.6 * kActivationAsDataStep)),
            kActivationAsDataStep);
}

TEST(InstructionTest, SerializesCorrectly) {
  mt19937 bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  for (Op op : TestableOps()) {
    Instruction instr_src(op, &rand_gen);
    Instruction instr_dest;
    instr_dest.Deserialize(instr_src.Serialize());
    EXPECT_EQ(instr_src, instr_dest);
  }
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
