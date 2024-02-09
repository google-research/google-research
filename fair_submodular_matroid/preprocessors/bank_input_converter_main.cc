// Copyright 2023 The Authors.
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

#include <iostream>
#include <string>
#include <vector>

// Converts the format of the bank dataset to the desired one.
int main() {
  std::vector<std::vector<double>> a;
  std::string s;
  std::cin >> s;
  int age;
  while (std::cin >> age) {
    a.push_back(std::vector<double>());
    a[a.size() - 1].push_back(age);
    char ch;
    int counter = 0;
    while (counter < 5) {
      std::cin >> ch;
      if (ch == ';') counter++;
    }
    int b;
    std::cin >> b;
    a[a.size() - 1].push_back(b);
    counter = 0;
    while (counter < 4) {
      std::cin >> ch;
      if (ch == ';') counter++;
    }
    std::cin >> b;
    a[a.size() - 1].push_back(b);
    counter = 0;
    while (counter < 2) {
      std::cin >> ch;
      if (ch == ';') counter++;
    }
    std::cin >> b;
    a[a.size() - 1].push_back(b);
    std::cin >> ch >> b;
    a[a.size() - 1].push_back(b);
    std::cin >> ch >> b;
    a[a.size() - 1].push_back(b);
    std::cin >> ch >> b;
    a[a.size() - 1].push_back(b);
    std::getline(std::cin, s);
  }
  std::cout << a.size() << " " << a[0].size() << std::endl;
  for (int i = 0; i < a.size(); i++) {
    for (int j = 0; j < a[i].size(); j++) std::cout << a[i][j] << " ";
    std::cout << std::endl;
  }
  return 0;
}
