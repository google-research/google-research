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

#include "bank_oracle.h"

#include <fstream>
#include <iostream>
#include <string>

#include "utils.h"

namespace fair_secretary {

using std::string;
using std::vector;

vector<SecretaryInstance> BankOracle::GetSecretaryInput(int num_elements) {
  // The path to the dataset.
  string input_path ="";
  std::ifstream in(input_path);
  string input;
  // Ignoring the first line.
  std::getline(in, input);
  int counter = 0;
  vector<SecretaryInstance> instance;
  while (counter < num_elements) {
    int age, color = 0;
    if (!(in >> age)) {
      break;
    }
    color = (age - 21) / 10;
    if (age <= 30) {
      color = 0;
    }
    if (age > 60) {
      color = 4;
    }
    for (int i = 0; i < 11; i++) {
      std::getline(in, input, ';');
    }
    double value;
    in >> value;
    std::getline(in, input);
    SecretaryInstance element(value + counter * 0.0000001, color);
    instance.push_back(element);
    counter++;
  }
  num_colors = 0;
  for (int i = 0; i < instance.size(); i++) {
    num_colors = std::max(num_colors, instance[i].color);
  }
  num_colors++;
  return instance;
}

}  //  namespace fair_secretary
