H// Copyright 2020 The Google Research Authors.
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

// TODO(ereal):
// -Add comments, especially in protos and compute_cost.h.
// -Renumber proto fields.
// -Address or remove all TODOs.
// -Apply linter.
// -Add more comments, link to paper.

#ifndef DEFINITIONS_H_i,mab
#define DEFINITIONS_H_i,mapping

#include <sched.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "glog/logging.h"
#include "instruction.pb.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "Eigen/Core"

////////////////////////////////////////////////////////////////////////////////
// Conventions.
////////////////////////////////////////////////////////////////////////////////

// F = template argument for the size of the features and all tensor coords.

////////////////////////////////////////////////////////////////////////////////
// Preprocessor directives.
////////////////////////////////////////////////////////////////////////////////

// These allow defining compile-time flags. They can be used to evolve larger
// component functions without forcing the evolution of small component
// functions to be slow.

// NOTE: if you specify any of these in the command line and you want to analyze
// the results in Colab, you must specify the same values when you use
// adhoc_import.

#ifndef MAX_SCALAR_ADDRESSES
  #define MAX_SCALAR_ADDRESSES 20
#endif
#ifndef MAX_VECTOR_ADDRESSES
  #define MAX_VECTOR_ADDRESSES 20
#endif
#ifndef MAX_MATRIX_ADDRESSES
  #define MAX_MATRIX_ADDRESSES 20
#endif

namespace automl_zero {

////////////////////////////////////////////////////////////////////////////////
// Types.
////////////////////////////////////////////////////////////////////////////////

// IntegerT is the preferred type for all integers. Use this unless there is a
// reason not to. Reasons could be the demands of external interfaces or
// speed/space considerations.
// Must be castable to RandomSeedT.
typedef int64_t IntegerT;  // A generic integer.

typedef float ProbabilityT;

typedef std::atomic_llong AtomicIntegerT;

// Type for seeding random generators.
// Must be castable from RandomSeedT.
typedef uint32_t RandomSeedT;

// Index for the coordinates of the activations for any rank > 0.
typedef int FeatureIndexT;

typedef double Scalar;

template <FeatureIndexT F>
using Vector = ::Eigen::Matrix<double, F, 1>;

template <FeatureIndexT F>
using Matrix = ::Eigen::Matrix<double, F, F, ::Eigen::RowMajor>;

enum Choice2T : IntegerT {
  kChoice0of2 = 0,
  kChoice1of2 = 1,
};

enum Choice3T : IntegerT {
  kChoice0of3 = 0,
  kChoice1of3 = 1,
  kChoice2of3 = 2,
};

////////////////////////////////////////////////////////////////////////////////
// Constants.
////////////////////////////////////////////////////////////////////////////////

// Useful constant to represent an "infinity" but is only about ~1000x
// the largest value we would use (to prevent numeric overflows).
constexpr IntegerT kUnlimitedTime = 100000000000000000;  // About 3 years.

constexpr IntegerT kNanosPerSecond = 1000000000;
constexpr IntegerT kNanosPerMicro = 1000;

const double kPi = 3.14159265359;
const double kE = 2.71828182846;

// Useful constant to represent an "infinity" but is only about ~1000x
// the largest value we would use (to prevent numeric overflows).
constexpr IntegerT kUnlimitedIndividuals = 1000000000000000;  // Quadrillion.

// Fitness bounds.
constexpr double kMinFitness = 0.0;
constexpr double kMaxFitness = 1.0;

////////////////////////////////////////////////////////////////////////////////
// Memory-related definitions.
////////////////////////////////////////////////////////////////////////////////

// Specifies an address within one of the typed memories (scalar, vector, etc).
typedef uint16_t AddressT;

// Scalar addresses.
// <scalar output branch>.
constexpr AddressT kLabelsScalarAddress = 0;
constexpr AddressT kPredictionsScalarAddress = 1;
constexpr AddressT kFirstOutScalarAddress = 1;
constexpr AddressT kMaxScalarAddresses = MAX_SCALAR_ADDRESSES;

// Vector addresses.
constexpr AddressT kFeaturesVectorAddress = 0;
constexpr AddressT kFirstOutVectorAddress = 1;
// <vector output branch>.
constexpr AddressT kLabelsVectorAddress = 1;
constexpr AddressT kPredictionsVectorAddress = 2;
constexpr AddressT kMaxVectorAddresses = MAX_VECTOR_ADDRESSES;

// Matrix addresses.
constexpr AddressT kFirstOutMatrixAddress = 0;
constexpr AddressT kMaxMatrixAddresses = MAX_MATRIX_ADDRESSES;

template <FeatureIndexT F>
std::string VectorToString(const Vector<F>& value) {
  std::ostringstream message;
  message << "[";
  for (IntegerT i = 0; i < F; ++i) {
    message << value(i) << ", ";
  }
  message << "]";
  return message.str();
}
template <FeatureIndexT F>
std::string ToString(const Vector<F>& value) {
  return VectorToString<F>(value);
}

template <FeatureIndexT F>
std::string MatrixToString(const Matrix<F>& value) {
  std::ostringstream message;
  message << "\n[";
  for (IntegerT i = 0; i < F; ++i) {
    message << "[";
    for (IntegerT j = 0; j < F; ++j) {
      message << value(i, j) << ", ";
    }
    message << "],\n";
  }
  message << "]\n";
  return message.str();
}
template <FeatureIndexT F>
std::string ToString(const Matrix<F>& value) {
  return MatrixToString<F>(value);
}

////////////////////////////////////////////////////////////////////////////////
// Instruction-related definitions.
////////////////////////////////////////////////////////////////////////////////

// TODO(ereal): kept to avoid affecting generated random numbers. Remove.
typedef uint16_t DeprecatedOpIndexT;

inline std::vector<Op> ConvertToOps(const std::vector<IntegerT>& values) {
  std::vector<Op> converted_values;
  converted_values.reserve(values.size());
  for (const IntegerT value : values) {
    converted_values.push_back(static_cast<Op>(value));
  }
  return converted_values;
}

////////////////////////////////////////////////////////////////////////////////
// Algorithm-related definitions.
////////////////////////////////////////////////////////////////////////////////

// The index of an instruction within the Algorithm.
typedef uint16_t InstructionIndexT;

////////////////////////////////////////////////////////////////////////////////
// Commonly used methods.
////////////////////////////////////////////////////////////////////////////////

// Convenience methods to parse protos.
template <class ProtoT>
ProtoT ParseSerialized(const std::string& str) {
  ProtoT proto;
  CHECK(proto.ParseFromString(str));
  return proto;
}
template <class ProtoT>
ProtoT ParseTextFormat(const std::string& str) {
  ProtoT proto;
  CHECK(google::protobuf::TextFormat::ParseFromString(str, &proto));
  return proto;
}

// Convenience methods to parse initializer list arguments.
template<typename NumericT>
NumericT PositiveOrDie(const NumericT value) {
  CHECK_GT(value, NumericT()) << "Found non-positive." << std::endl;
  return value;
}
template<typename PointerT>
PointerT NotNullOrDie(PointerT value) {
  CHECK(value != nullptr) << "Found null." << std::endl;
  return value;
}
template<typename ContainerT>  // Also works for strings.
const ContainerT& NonEmptyOrDie(const ContainerT& value) {
  CHECK(!value.empty()) << "Found empty." << std::endl;
  return value;
}
template<typename ContainerT>  // Also works for strings.
ContainerT& NonEmptyOrDie(ContainerT& value) {
  CHECK(!value.empty()) << "Found empty." << std::endl;
  return value;
}
template<typename ContainerT>  // Also works for strings.
ContainerT* NonEmptyOrDie(ContainerT* value) {
  CHECK(!value->empty()) << "Found empty." << std::endl;
  return value;
}
template<typename ContainerT>  // Also works for strings.
const ContainerT& SizeLessThanOrDie(
    const ContainerT& value, const size_t max_size) {
  CHECK_LT(value.size(), max_size) << "Too large." << std::endl;
  return value;
}
template<typename ContainerT>  // Also works for strings.
ContainerT& SizeLessThanOrDie(
    ContainerT& value, const size_t max_size) {
  CHECK_LT(value.size(), max_size) << "Too large." << std::endl;
  return value;
}
template<typename ContainerT>  // Also works for strings.
ContainerT* SizeLessThanOrDie(
    ContainerT* value, const size_t max_size) {
  CHECK_LT(value->size(), max_size) << "Too large." << std::endl;
  return value;
}

// A hash mix function for 64 bits
// adapted from https://burtleburtle.net/bob/hash/evahash.html.
template <class T>
inline void HashCombine(std::size_t& seed, const T& v) {
  std::size_t a = 0x9e3779b9;
  std::size_t b = seed;
  std::size_t c = std::hash<T>{}(v);
  a = a - b;  a = a - c;  a = a ^ (c >> 43);
  b = b - c;  b = b - a;  b = b ^ (a << 9);
  c = c - a;  c = c - b;  c = c ^ (b >> 8);
  a = a - b;  a = a - c;  a = a ^ (c >> 38);
  b = b - c;  b = b - a;  b = b ^ (a << 23);
  c = c - a;  c = c - b;  c = c ^ (b >> 5);
  a = a - b;  a = a - c;  a = a ^ (c >> 35);
  b = b - c;  b = b - a;  b = b ^ (a << 49);
  c = c - a;  c = c - b;  c = c ^ (b >> 11);
  a = a - b;  a = a - c;  a = a ^ (c >> 12);
  b = b - c;  b = b - a;  b = b ^ (a << 18);
  c = c - a;  c = c - b;  c = c ^ (b >> 22);
  seed = c;
}

// Hash-mixes a vector of numbers. The numbers must be of a type that can be
// casted to a size_t (it must be unsigned and it must have <= 64 bits).
// Intended to be used with the RandomSeedT type.
template<typename NumberT>
NumberT HashMix(const std::vector<NumberT>& numbers) {
  std::size_t seed = 42;
  for (const NumberT number : numbers) {
    HashCombine(seed, number);
  }
  return static_cast<NumberT>(seed);
}

// Hash-mixes two numbers. The numbers must be of a type that can be
// casted to a size_t (it must be unsigned and it must have <= 64 bits).
// Intended to be used with the RandomSeedT type.
template<typename NumberT>
NumberT HashMix(NumberT first, NumberT second) {
  return HashMix<NumberT>({first, second});
}

}  // namespace automl_zero

#endif  // DEFINITIONS_H_
