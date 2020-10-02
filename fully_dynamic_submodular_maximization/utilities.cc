// Copyright 2020 The Authors.
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

//
//  Utilities
//

// Provides some utilites.

#include "utilities.h"

using std::cerr;
using std::vector;

std::mt19937 RandomHandler::generator_;

// Formats numbers like 1078546 -> 1,078,546.
std::string PrettyNum(int64_t number) {
  std::string pretty_number = std::to_string(number);
  for (int i = static_cast<int>(pretty_number.size()) - 3; i > 0; i -= 3) {
    pretty_number.insert(pretty_number.begin() + i, ',');
  }
  return pretty_number;
}

void Fail(const std::string& error) {
  std::cerr << error << std::endl;
  exit(1);
}

// Returns a sequence: small, ..., large.
// Where we roughly have a[i+1]/a[i] = base (we use base = 1+eps).
vector<double> LogSpace(double small, double large, double base) {
  if (small > large) {
    vector<double> log_space = LogSpace(large, small, base);
    reverse(log_space.begin(), log_space.end());
    return log_space;
  }
  assert(base > 1);
  int steps =
      static_cast<int>(ceil((log(large) - log(small)) / log(base) - 1e-6));
  double step = pow(large / small, 1.0 / steps);
  vector<double> log_space = {small};
  for (int i = 0; i < steps; ++i) {
    log_space.push_back(log_space.back() * step);
  }
  return log_space;
}
