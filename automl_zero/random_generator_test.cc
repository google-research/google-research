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

#include "random_generator.h"

#include <cmath>
#include <unordered_set>

#include "definitions.h"
#include "test_util.h"
#include "gtest/gtest.h"
#include "absl/container/node_hash_set.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::std::round;
using ::std::function;

TEST(RandomGeneratorTest, GaussianFloatProducesAllValues) {
  MTRandom bit_gen(GenerateRandomSeed());
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return round(rand_gen.GaussianFloat(0.0, 10.0));}),
      Range<IntegerT>(-100, 101),
      Range<IntegerT>(-10, 11)));
}

TEST(RandomGeneratorTest, UniformIntegerProducesAllValues) {
  MTRandom bit_gen(GenerateRandomSeed());
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return rand_gen.UniformInteger(2, 10);}),
      Range<IntegerT>(2, 10),
      Range<IntegerT>(2, 10)));
}

TEST(RandomGeneratorTest, UniformRandomSeedTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&]() {
        return std::round(rand_gen.UniformRandomSeed() % 20);}),
      Range<IntegerT>(0, 20), Range<IntegerT>(0, 20)));
}

TEST(RandomGeneratorTest, UniformDoubleProducesAllValues) {
  MTRandom bit_gen(GenerateRandomSeed());
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return round(rand_gen.UniformDouble(0.2, 1.6) * 10.0);}),
      Range<IntegerT>(2, 17),
      Range<IntegerT>(2, 17)));
}

TEST(RandomGeneratorTest, UniformFloatProducesAllValues) {
  MTRandom bit_gen(GenerateRandomSeed());
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return round(rand_gen.UniformFloat(0.2, 1.6) * 10.0);}),
      Range<IntegerT>(2, 17),
      Range<IntegerT>(2, 17)));
}

TEST(RandomGeneratorTest, UniformProbabilityProducesAllValues) {
  MTRandom bit_gen(GenerateRandomSeed());
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return round(rand_gen.UniformProbability(0.2, 0.8) * 10.0);}),
      Range<IntegerT>(2, 9),
      Range<IntegerT>(2, 9)));
}

TEST(RandomGeneratorTest, UniformStringProducesAllValues) {
  MTRandom bit_gen(GenerateRandomSeed());
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<std::string(void)>([&]() { return rand_gen.UniformString(1); }),
      absl::node_hash_set<std::string>{},
      {"a", "z", "A", "Z", "0", "9", "_", "~"}));
}

TEST(RandomGeneratorTest, FeatureIndexTest_Fis4) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<FeatureIndexT(void)>(
          [&](){return rand_gen.FeatureIndex(4);}),
      Range<FeatureIndexT>(0, 4),
      Range<FeatureIndexT>(0, 4)));
}

TEST(RandomGeneratorTest, FeatureIndexTest_Fis8) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<FeatureIndexT(void)>(
          [&](){return rand_gen.FeatureIndex(8);}),
      Range<FeatureIndexT>(0, 8),
      Range<FeatureIndexT>(0, 8)));
}

TEST(RandomGeneratorTest, ScalarAddressTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<AddressT(void)>([&](){return rand_gen.ScalarInAddress();}),
      Range<AddressT>(0, kMaxScalarAddresses),
      Range<AddressT>(0, kMaxScalarAddresses)));
}

TEST(RandomGeneratorTest, VectorAddressTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<AddressT(void)>([&](){return rand_gen.VectorInAddress();}),
      Range<AddressT>(0, kMaxVectorAddresses),
      Range<AddressT>(0, kMaxVectorAddresses)));
}

TEST(RandomGeneratorTest, MatrixAddressTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<AddressT(void)>([&](){return rand_gen.MatrixInAddress();}),
      Range<AddressT>(0, kMaxMatrixAddresses),
      Range<AddressT>(0, kMaxMatrixAddresses)));
}

TEST(RandomGeneratorTest, ScalarOutAddressTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<AddressT(void)>([&](){return rand_gen.ScalarOutAddress();}),
      Range<AddressT>(
          kFirstOutScalarAddress, kMaxScalarAddresses),
      Range<AddressT>(
          kFirstOutScalarAddress, kMaxScalarAddresses)));
}

TEST(RandomGeneratorTest, VectorOutAddressTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<AddressT(void)>([&](){return rand_gen.VectorOutAddress();}),
      Range<AddressT>(
          kFirstOutVectorAddress, kMaxVectorAddresses),
      Range<AddressT>(
          kFirstOutVectorAddress, kMaxVectorAddresses)));
}

TEST(RandomGeneratorTest, MatrixOutAddressTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<AddressT(void)>([&](){return rand_gen.MatrixOutAddress();}),
      Range<AddressT>(
          kFirstOutMatrixAddress, kMaxMatrixAddresses),
      Range<AddressT>(
          kFirstOutMatrixAddress, kMaxMatrixAddresses)));
}

TEST(RandomGeneratorTest, Choice2Test) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<Choice2T(void)>([&](){return rand_gen.Choice2();}),
      {kChoice0of2, kChoice1of2},
      {kChoice0of2, kChoice1of2}));
}

TEST(RandomGeneratorTest, Choice3Test) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<Choice3T(void)>([&](){return rand_gen.Choice3();}),
      {kChoice0of3, kChoice1of3, kChoice2of3},
      {kChoice0of3, kChoice1of3, kChoice2of3}));
}

TEST(RandomGeneratorTest, UniformTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return rand_gen.UniformPopulationSize(10);}),
      Range<IntegerT>(0, 10),
      Range<IntegerT>(0, 10)));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&]() {
        return std::round(rand_gen.UniformActivation(-5.0, 10.0));}),
      Range<IntegerT>(-5, 11), Range<IntegerT>(-5, 11)));
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&]() {
        return std::round(rand_gen.UniformRandomSeed() % 20);}),
      Range<IntegerT>(0, 20), Range<IntegerT>(0, 20)));
}

template<FeatureIndexT F>
IntegerT FillUniformVectorHelper(
    const FeatureIndexT index, RandomGenerator* rand_gen) {
  Vector<F> vector;
  rand_gen->FillUniform<F>(-5.0, 10.0, &vector);
  return std::round(vector(index));
}

TEST(RandomGeneratorTest, FillUniformVectorTest_Fis4) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index = rand_gen.FeatureIndex(4);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillUniformVectorHelper<4>(index, &rand_gen);}),
      Range<IntegerT>(-5, 11),
      Range<IntegerT>(-5, 11)));
}

TEST(RandomGeneratorTest, FillUniformVectorTest_Fis8) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index = rand_gen.FeatureIndex(8);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillUniformVectorHelper<8>(index, &rand_gen);}),
      Range<IntegerT>(-5, 11),
      Range<IntegerT>(-5, 11)));
}

template<FeatureIndexT F>
IntegerT FillUniformMatrixHelper(
    const FeatureIndexT index_x, const FeatureIndexT index_y,
    RandomGenerator* rand_gen) {
  Matrix<F> matrix;
  rand_gen->FillUniform<F>(-5.0, 10.0, &matrix);
  return std::round(matrix(index_x, index_y));
}

TEST(RandomGeneratorTest, FillUniformMatrixTest_Fis2) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index_x = rand_gen.FeatureIndex(2);
  FeatureIndexT index_y = rand_gen.FeatureIndex(2);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillUniformMatrixHelper<2>(index_x, index_y, &rand_gen);}),
      Range<IntegerT>(-5, 11),
      Range<IntegerT>(-5, 11)));
}

TEST(RandomGeneratorTest, FillUniformMatrixTest_Fis4) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index_x = rand_gen.FeatureIndex(4);
  FeatureIndexT index_y = rand_gen.FeatureIndex(4);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillUniformMatrixHelper<4>(index_x, index_y, &rand_gen);}),
      Range<IntegerT>(-5, 11),
      Range<IntegerT>(-5, 11)));
}

TEST(RandomGeneratorTest, Gaussiandoubleest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return IntegerT(rand_gen.GaussianActivation(0.0, 10.0));}),
      Range<IntegerT>(-100, 101),
      Range<IntegerT>(-10, 11)));
}

template<FeatureIndexT F>
IntegerT FillGaussianVectorHelper(
    const FeatureIndexT index, RandomGenerator* rand_gen) {
  Vector<F> vector;
  rand_gen->FillGaussian<F>(0.0, 10.0, &vector);
  return round(vector(index));
}

TEST(RandomGeneratorTest, FillGaussianVectorTest_Fis4) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index = rand_gen.FeatureIndex(4);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillGaussianVectorHelper<4>(index, &rand_gen);}),
      Range<IntegerT>(-100, 101),
      Range<IntegerT>(-10, 11)));
}

TEST(RandomGeneratorTest, FillGaussianVectorTest_Fis8) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index = rand_gen.FeatureIndex(8);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillGaussianVectorHelper<8>(index, &rand_gen);}),
      Range<IntegerT>(-100, 101),
      Range<IntegerT>(-10, 11)));
}

template<FeatureIndexT F>
IntegerT FillGaussianMatrixHelper(
    const FeatureIndexT index_x, const FeatureIndexT index_y,
    RandomGenerator* rand_gen) {
  Matrix<F> matrix;
  rand_gen->FillGaussian<F>(0.0, 10.0, &matrix);
  return round(matrix(index_x, index_y));
}

TEST(RandomGeneratorTest, FillGaussianMatrixTest_Fis2) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index_x = rand_gen.FeatureIndex(2);
  FeatureIndexT index_y = rand_gen.FeatureIndex(2);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillGaussianMatrixHelper<2>(index_x, index_y, &rand_gen);}),
      Range<IntegerT>(-100, 101),
      Range<IntegerT>(-10, 11)));
}

TEST(RandomGeneratorTest, FillGaussianMatrixTest_Fis4) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index_x = rand_gen.FeatureIndex(4);
  FeatureIndexT index_y = rand_gen.FeatureIndex(4);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillGaussianMatrixHelper<4>(index_x, index_y, &rand_gen);}),
      Range<IntegerT>(-100, 101),
      Range<IntegerT>(-10, 11)));
}

TEST(RandomGeneratorTest, BetaTest) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return IntegerT(round(rand_gen.BetaActivation(0.5, 0.5) * 10.0));}),
      Range<IntegerT>(0, 11),
      Range<IntegerT>(0, 11)));
}

template<FeatureIndexT F>
IntegerT FillBetaVectorHelper(
    const FeatureIndexT index, RandomGenerator* rand_gen) {
  Vector<F> vector;
  rand_gen->FillBeta<F>(0.5, 0.5, &vector);
  return round(vector(index) * 10.0);
}

TEST(RandomGeneratorTest, FillBetaVectorTest_Fis4) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index = rand_gen.FeatureIndex(4);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillBetaVectorHelper<4>(index, &rand_gen);}),
      Range<IntegerT>(0, 11),
      Range<IntegerT>(0, 11)));
}

TEST(RandomGeneratorTest, FillBetaVectorTest_Fis8) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index = rand_gen.FeatureIndex(8);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillBetaVectorHelper<8>(index, &rand_gen);}),
      Range<IntegerT>(0, 11),
      Range<IntegerT>(0, 11)));
}

template<FeatureIndexT F>
IntegerT FillBetaMatrixHelper(
    const FeatureIndexT index_x, const FeatureIndexT index_y,
    RandomGenerator* rand_gen) {
  Matrix<F> matrix;
  rand_gen->FillBeta<F>(0.5, 0.5, &matrix);
  return round(matrix(index_x, index_y) * 10.0);
}

TEST(RandomGeneratorTest, FillBetaMatrixTest_Fis2) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index_x = rand_gen.FeatureIndex(2);
  FeatureIndexT index_y = rand_gen.FeatureIndex(2);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillBetaMatrixHelper<2>(index_x, index_y, &rand_gen);}),
      Range<IntegerT>(0, 11),
      Range<IntegerT>(0, 11)));
}

TEST(RandomGeneratorTest, FillBetaMatrixTest_Fis4) {
  MTRandom bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  FeatureIndexT index_x = rand_gen.FeatureIndex(4);
  FeatureIndexT index_y = rand_gen.FeatureIndex(4);
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        return FillBetaMatrixHelper<4>(index_x, index_y, &rand_gen);}),
      Range<IntegerT>(0, 11),
      Range<IntegerT>(0, 11)));
}

TEST(GenerateRandomSeedTest, GeneratesDifferentSeeds) {
  RandomSeedT seed1 = GenerateRandomSeed();
  usleep(100);
  RandomSeedT seed2 = GenerateRandomSeed();
  EXPECT_NE(seed1, seed2);
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
