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

#ifndef RANDOM_GENERATOR_H_
#define RANDOM_GENERATOR_H_

#include <limits>
#include <random>

#include "definitions.h"
#include "absl/random/random.h"

namespace automl_zero {

// Thread-compatible, but not thread-safe.
class RandomGenerator {
 public:
  explicit RandomGenerator(std::mt19937* gen);
  RandomGenerator(const RandomGenerator& other) = delete;
  RandomGenerator& operator=(const RandomGenerator& other) = delete;

  inline std::mt19937* BitGen() {return bit_gen_;}

  // Resets the generator with a new random seed.
  void SetSeed(RandomSeedT seed) {
    assert(seed != 0);
    bit_gen_->seed(seed);
  }

  float GaussianFloat(float mean, float stdev);

  // Returns a uniform integer between low (incl) and high (excl).
  IntegerT UniformInteger(IntegerT low, IntegerT high);

  // Returns a uniform uint32.
  RandomSeedT UniformRandomSeed();

  // Returns a uniform integer between low (incl) and high (excl).
  double UniformDouble(double low, double high);

  // Returns a uniform integer between low (incl) and high (excl).
  float UniformFloat(float low, float high);

  // Returns a uniform integer between low (incl) and high (excl).
  ProbabilityT UniformProbability(
      ProbabilityT low = 0.0, ProbabilityT high = 1.0);

  // Returns a string with characters that are indepdently uniformly sampled
  // from the 64 characters 'a'-'z', 'A'-'Z', '0'-'9', '_' and '~'.
  std::string UniformString(size_t size);

  // Only used by old ops.
  FeatureIndexT FeatureIndex(FeatureIndexT features_size);

  AddressT ScalarInAddress();
  AddressT VectorInAddress();
  AddressT MatrixInAddress();
  AddressT ScalarOutAddress();
  AddressT VectorOutAddress();
  AddressT MatrixOutAddress();

  Choice2T Choice2();
  Choice3T Choice3();

  // Uniform open intervals
  IntegerT UniformPopulationSize(IntegerT high);
  double UniformActivation(double low, double high);
  template<FeatureIndexT F> void FillUniform(
      double low, double high, Vector<F>* vector);
  template<FeatureIndexT F> void FillUniform(
      double low, double high, Matrix<F>* matrix);

  // Gaussian distribution has mean 0 and variance 1.
  double GaussianActivation(double mean, double stdev);
  template<FeatureIndexT F> void FillGaussian(
      double mean, double stdev, Vector<F>* vector);
  template<FeatureIndexT F> void FillGaussian(
      double mean, double stdev, Matrix<F>* matrix);

  // Beta distribution.
  double BetaActivation(double alpha, double beta);
  template<FeatureIndexT F> void FillBeta(
      double alpha, double beta, Vector<F>* vector);
  template<FeatureIndexT F> void FillBeta(
      double alpha, double beta, Matrix<F>* matrix);

 private:
  friend RandomGenerator SimpleRandomGenerator();

  // Used to create a simple class for tests in SimpleRandomGenerator().
  RandomGenerator();

  std::unique_ptr<std::mt19937> bit_gen_owned_;
  std::mt19937* bit_gen_;
};

// Generate a random seed using current time.
RandomSeedT GenerateRandomSeed();

template<FeatureIndexT F>
void RandomGenerator::FillUniform(
    double low, double high, Vector<F>* vector) {
  for (FeatureIndexT i = 0; i < F; ++i) {
    (*vector)(i) =
        absl::Uniform<double>(absl::IntervalOpen, *bit_gen_, low, high);
  }
}

template<FeatureIndexT F>
void RandomGenerator::FillUniform(
    double low, double high, Matrix<F>* matrix) {
  for (FeatureIndexT i = 0; i < F; ++i) {
    for (FeatureIndexT j = 0; j < F; ++j) {
      (*matrix)(i, j) =
          absl::Uniform<double>(absl::IntervalOpen, *bit_gen_, low, high);
    }
  }
}

template<FeatureIndexT F>
void RandomGenerator::FillGaussian(
    const double mean, const double stdev, Vector<F>* vector) {
  for (FeatureIndexT i = 0; i < F; ++i) {
    (*vector)(i) = ::absl::Gaussian<double>(*bit_gen_, mean, stdev);
  }
}

template<FeatureIndexT F>
void RandomGenerator::FillGaussian(
    const double mean, const double stdev, Matrix<F>* matrix) {
  for (FeatureIndexT i = 0; i < F; ++i) {
    for (FeatureIndexT j = 0; j < F; ++j) {
      (*matrix)(i, j) = ::absl::Gaussian<double>(*bit_gen_, mean, stdev);
    }
  }
}

template<FeatureIndexT F>
void RandomGenerator::FillBeta(
    const double alpha, const double beta, Vector<F>* vector) {
  for (FeatureIndexT i = 0; i < F; ++i) {
    (*vector)(i) = ::absl::Beta<double>(*bit_gen_, alpha, beta);
  }
}

template<FeatureIndexT F>
void RandomGenerator::FillBeta(
    const double alpha, const double beta, Matrix<F>* matrix) {
  for (FeatureIndexT i = 0; i < F; ++i) {
    for (FeatureIndexT j = 0; j < F; ++j) {
      (*matrix)(i, j) = ::absl::Beta<double>(*bit_gen_, alpha, beta);
    }
  }
}

}  // namespace automl_zero

#endif  // RANDOM_GENERATOR_H_
