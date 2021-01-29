// Copyright 2021 The Google Research Authors.
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

#include "npy_array/npy_array.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>

#include "array/array.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace npy_array {
namespace {

template <typename T>
void VerifyTwoImagesAreSame(const T& a1, const T& a2) {
  EXPECT_GT(a2.size(), 0) << "Testing rank " << a2.shape().rank();
  EXPECT_EQ(a1.shape().rank(), a2.shape().rank());
  for (int d = 0; d < a1.shape().rank(); ++d) {
    EXPECT_EQ(a1.shape().dim(d).extent(), a2.shape().dim(d).extent())
        << "Testing rank " << a2.shape().rank()
        << ", mismatch on the dimension " << d;
  }
  EXPECT_EQ(std::memcmp(a1.data(), a2.data(), a1.size()), 0)
      << "Testing rank " << a2.shape().rank();
}

template <typename T>
auto Distribution() {
  if constexpr (std::is_integral_v<T>) {
    return std::uniform_int_distribution<T>(std::numeric_limits<T>::min(),
                                            std::numeric_limits<T>::max());
  } else {
    return std::uniform_real_distribution<float>(0.0f, 1.0f);
  }
}

template <typename T, size_t Rank>
auto RandomArray(nda::shape_of_rank<Rank> size) {
  nda::array_of_rank<T, Rank> array(size);
  std::mt19937 gen;
  auto distribution = Distribution<T>();
  array.for_each_value([&](T& v) { v = distribution(gen); });
  return array;
}

template <typename T>
auto RandomScalarArray() {
  std::mt19937 gen;
  auto distribution = Distribution<T>();
  return nda::array_of_rank<T, 0>({}, distribution(gen));
}

template <typename T>
void VerifyRoundTrip(const T& arr1) {
  std::string s = npy_array::SerializeToNpyString(arr1.cref());
  const auto arr2 =
      npy_array::DeserializeFromNpyString<typename T::value_type,
                                          typename T::shape_type>(s);
  VerifyTwoImagesAreSame(arr1, arr2);
}

}  // namespace

TEST(Npy, NpyLoadRoundtrip) {
  // Rank 0.
  {
    VerifyRoundTrip(RandomScalarArray<float>());
    VerifyRoundTrip(RandomScalarArray<int>());
    VerifyRoundTrip(RandomScalarArray<int16_t>());
    VerifyRoundTrip(RandomScalarArray<uint32_t>());
    VerifyRoundTrip(RandomScalarArray<uint16_t>());
  }
  // Rank 1.
  {
    VerifyRoundTrip(RandomArray<float, 1>({3}));
    VerifyRoundTrip(RandomArray<int, 1>({4}));
    VerifyRoundTrip(RandomArray<int16_t, 1>({4}));
    VerifyRoundTrip(RandomArray<uint32_t, 1>({5}));
    VerifyRoundTrip(RandomArray<uint16_t, 1>({5}));
  }
  // Rank 2.
  {
    VerifyRoundTrip(RandomArray<float, 2>({13, 3}));
    VerifyRoundTrip(RandomArray<int, 2>({3, 13}));
    VerifyRoundTrip(RandomArray<int16_t, 2>({3, 13}));
    VerifyRoundTrip(RandomArray<uint32_t, 2>({11, 9}));
    VerifyRoundTrip(RandomArray<uint16_t, 2>({11, 9}));
  }
  // Rank 3.
  {
    VerifyRoundTrip(RandomArray<float, 3>({8, 6, 3}));
    VerifyRoundTrip(RandomArray<int, 3>({2, 11, 9}));
    VerifyRoundTrip(RandomArray<int16_t, 3>({2, 11, 9}));
    VerifyRoundTrip(RandomArray<uint32_t, 3>({4, 1, 7}));
    VerifyRoundTrip(RandomArray<uint16_t, 3>({4, 1, 7}));
  }
}

}  // namespace npy_array
