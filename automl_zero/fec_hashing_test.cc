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

#include "fec_hashing.h"

#include <limits>

#include "definitions.h"
#include "gtest/gtest.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::std::function;
using ::std::vector;
using ::testing::Test;

typedef
    function<size_t(const vector<double>&, const vector<double>&,
                    size_t, IntegerT)>
    HashFunction;

class HashFunctionTest : public Test {
 protected:
  void VerifyProducesEqualHashesForEqualVectors(
      const HashFunction& hash_function) {
    EXPECT_EQ(hash_function({0.6}, {0.0}, 0, 1000),
              hash_function({0.6}, {0.0}, 0, 1000));
    EXPECT_EQ(hash_function({0.6, 0.2}, {0.8, 2.5}, 1, 1000),
              hash_function({0.6, 0.2}, {0.8, 2.5}, 1, 1000));
    EXPECT_EQ(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 2, 1000),
              hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 2, 1000));
  }

  void VerifyDetectsDifferenceInSingleValue(const HashFunction& hash_function) {
    EXPECT_NE(hash_function({0.6}, {0.0}, 0, 1000),
              hash_function({0.601}, {0.0}, 0, 1000));
    EXPECT_NE(hash_function({0.6, 0.2}, {0.8, 2.5}, 1, 1000),
              hash_function({0.6, 0.2}, {0.81, 2.5}, 1, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 2, 1000),
              hash_function({0.6, 0.2, 0.0001}, {0.8, 2.5, 10.1}, 2, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 3, 1000),
              hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 30, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 3, 1000),
              hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 3, 1001));
  }

  void VerifyDetectsDifferenceInMultipleValues(
      const HashFunction& hash_function) {
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 0, 1000),
              hash_function({0.6, 0.1, 0.0}, {0.8, 2.6, 10.1}, 0, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 1, 1000),
              hash_function({0.6, 0.1, 0.1}, {0.8, 2.5, 10.1}, 1, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 2, 1000),
              hash_function({0.6, 0.2, 0.0}, {0.8, 2.6, 10.0}, 2, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 3, 1000),
              hash_function({0.6, 0.2, 0.0}, {0.8, 2.6, 10.0}, 4, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 3, 1000),
              hash_function({0.6, 0.2, 0.0}, {0.8, 2.6, 10.0}, 4, 10000));
  }

  void VerifyDetectsDifferenceInNumberOfValues(
      const HashFunction& hash_function) {
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 0, 1000),
              hash_function({0.6, 0.2}, {0.8, 2.5, 10.1}, 0, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 0, 1000),
              hash_function({0.6, 0.2, 0.0}, {0.8, 2.5}, 0, 1000));
    EXPECT_NE(hash_function({0.6, 0.2, 0.0}, {0.8, 2.5, 10.1}, 0, 1000),
              hash_function({0.6, 0.2}, {0.8, 2.5}, 0, 1000));
  }

  void VerifyHandlesLargeValues(const HashFunction& hash_function) {
    EXPECT_EQ(hash_function({0.6, 1000000000.0}, {0.8, 2.5}, 0, 1000),
              hash_function({0.6, 1000000000.0}, {0.8, 2.5}, 0, 1000));
    EXPECT_NE(hash_function({0.6, 1000000000.0}, {0.8, 2.5}, 0, 1000),
              hash_function({0.7, 1000000000.0}, {0.8, 2.5}, 0, 1000));
  }

  void VerifyHandlesInfinity(const HashFunction& hash_function) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(hash_function({0.6, inf}, {0.8, 2.5}, 0, 1000),
              hash_function({0.6, inf}, {0.8, 2.5}, 0, 1000));
    EXPECT_NE(hash_function({0.6, inf}, {0.8, 2.5}, 0, 1000),
              hash_function({0.7, inf}, {0.8, 2.5}, 0, 1000));
    EXPECT_NE(hash_function({0.6, inf}, {0.8, 2.5}, 0, 1000),
              hash_function({0.6, 1.0}, {0.8, 2.5}, 0, 1000));
  }

  void VerifyHandlesNan(const HashFunction& hash_function) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(hash_function({0.6, nan}, {0.8, 2.5}, 0, 1000),
              hash_function({0.6, nan}, {0.8, 2.5}, 0, 1000));
    EXPECT_NE(hash_function({0.6, nan}, {0.8, 2.5}, 0, 1000),
              hash_function({0.7, nan}, {0.8, 2.5}, 0, 1000));
    EXPECT_NE(hash_function({0.6, nan}, {0.8, 2.5}, 0, 1000),
              hash_function({0.6, 1.0}, {0.8, 2.5}, 0, 1000));
  }
};

TEST_F(HashFunctionTest, WellMixedHashWorksCorrectly) {
  VerifyProducesEqualHashesForEqualVectors(WellMixedHash);
  VerifyDetectsDifferenceInSingleValue(WellMixedHash);
  VerifyDetectsDifferenceInMultipleValues(WellMixedHash);
  VerifyDetectsDifferenceInNumberOfValues(WellMixedHash);
  VerifyHandlesLargeValues(WellMixedHash);
  VerifyHandlesInfinity(WellMixedHash);
  VerifyHandlesNan(WellMixedHash);
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
