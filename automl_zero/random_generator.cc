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

#include "random_generator.h"

#include "definitions.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace automl_zero {

using ::absl::GetCurrentTimeNanos;
using ::absl::make_unique;
using ::std::mt19937;
using ::std::numeric_limits;
using ::std::string;

RandomGenerator::RandomGenerator(mt19937* bit_gen) : bit_gen_(bit_gen) {}

float RandomGenerator::GaussianFloat(float mean, float stdev) {
  return ::absl::Gaussian<float>(*bit_gen_, mean, stdev);
}

IntegerT RandomGenerator::UniformInteger(IntegerT low, IntegerT high) {
  // TODO(ereal): change this to IntegerT and change the values provided by
  // LeanClient::PutGetAndCount. Probably affects random number generation.
  CHECK_GE(low, std::numeric_limits<int32_t>::min());
  CHECK_LE(high, std::numeric_limits<int32_t>::max());
  return ::absl::Uniform<int32_t>(*bit_gen_, low, high);
}

RandomSeedT RandomGenerator::UniformRandomSeed() {
  return absl::Uniform<RandomSeedT>(
      absl::IntervalOpen, *bit_gen_,
      1, std::numeric_limits<RandomSeedT>::max());
}

double RandomGenerator::UniformDouble(double low, double high) {
  return ::absl::Uniform<double>(*bit_gen_, low, high);
}

float RandomGenerator::UniformFloat(float low, float high) {
  return ::absl::Uniform<float>(*bit_gen_, low, high);
}

ProbabilityT RandomGenerator::UniformProbability(
    const ProbabilityT low, const ProbabilityT high) {
  return ::absl::Uniform<ProbabilityT>(*bit_gen_, low, high);
}

string RandomGenerator::UniformString(const size_t size) {
  string random_string;
  for (size_t i = 0; i < size; ++i) {
    char random_char;
    const IntegerT char_index = UniformInteger(0, 64);
    if (char_index < 26) {
      random_char = char_index + 97;  // Maps 0-25 to 'a'-'z'.
    } else if (char_index < 52) {
      random_char = char_index - 26 + 65;  // Maps 26-51 to 'A'-'Z'.
    } else if (char_index < 62) {
      random_char = char_index - 52 + 48;  // Maps 52-61 to '0'-'9'.
    } else if (char_index == 62) {
      random_char = '_';
    } else if (char_index == 63) {
      random_char = '~';
    } else {
      LOG(FATAL) << "Code should not get here." << std::endl;
    }
    random_string.push_back(random_char);
  }
  return random_string;
}

FeatureIndexT RandomGenerator::FeatureIndex(
    const FeatureIndexT features_size) {
  // TODO(ereal): below should have FeatureIndexT instead of InstructionIndexT;
  // affects random number generation.
  return absl::Uniform<InstructionIndexT>(*bit_gen_, 0, features_size);
}

AddressT RandomGenerator::ScalarInAddress() {
  return absl::Uniform<AddressT>(*bit_gen_, 0, kMaxScalarAddresses);
}

AddressT RandomGenerator::VectorInAddress() {
  return absl::Uniform<AddressT>(*bit_gen_, 0, kMaxVectorAddresses);
}

AddressT RandomGenerator::MatrixInAddress() {
  return absl::Uniform<AddressT>(*bit_gen_, 0, kMaxMatrixAddresses);
}

AddressT RandomGenerator::ScalarOutAddress() {
  return absl::Uniform<AddressT>(
      *bit_gen_, kFirstOutScalarAddress, kMaxScalarAddresses);
}

AddressT RandomGenerator::VectorOutAddress() {
  return absl::Uniform<AddressT>(
      *bit_gen_, kFirstOutVectorAddress, kMaxVectorAddresses);
}

AddressT RandomGenerator::MatrixOutAddress() {
  return absl::Uniform<AddressT>(
      *bit_gen_, kFirstOutMatrixAddress, kMaxMatrixAddresses);
}

Choice2T RandomGenerator::Choice2() {
  return static_cast<Choice2T>(absl::Uniform<IntegerT>(*bit_gen_, 0, 2));
}

Choice3T RandomGenerator::Choice3() {
  return static_cast<Choice3T>(absl::Uniform<IntegerT>(*bit_gen_, 0, 3));
}

IntegerT RandomGenerator::UniformPopulationSize(
    IntegerT high) {
  return static_cast<IntegerT>(absl::Uniform<uint32_t>(*bit_gen_, 0, high));
}

double RandomGenerator::UniformActivation(
    double low, double high) {
  return absl::Uniform<double>(absl::IntervalOpen, *bit_gen_, low, high);
}

double RandomGenerator::GaussianActivation(
    const double mean, const double stdev) {
  return ::absl::Gaussian<double>(*bit_gen_, mean, stdev);
}

double RandomGenerator::BetaActivation(
    const double alpha, const double beta) {
  return ::absl::Beta<double>(*bit_gen_, alpha, beta);
}

RandomGenerator::RandomGenerator()
    : bit_gen_owned_(make_unique<mt19937>(GenerateRandomSeed())),
      bit_gen_(bit_gen_owned_.get()) {}

RandomSeedT GenerateRandomSeed() {
  RandomSeedT seed = 0;
  while (seed == 0) {
    seed = static_cast<RandomSeedT>(
        GetCurrentTimeNanos() % numeric_limits<RandomSeedT>::max());
  }
  return seed;
}

}  // namespace automl_zero
