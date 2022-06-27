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

#include "secretary_eval.h"

#include <iostream>

#include "utils.h"

namespace fair_secretary {

using std::cout;
using std::endl;
using std::vector;

void SecretaryEval::Eval(const vector<SecretaryInstance>& instance,
                         const vector<SecretaryInstance>& answer,
                         int num_colors) {
  vector<double> max_value(num_colors, 0);
  vector<int> correct_answer(num_colors, 0);
  vector<int> num_answer(num_colors, 0);
  double total_max = 0;
  int not_picked = 0;
  int total_correct_answer = 0;
  for (int i = 0; i < instance.size(); i++) {
    max_value[instance[i].color] =
        std::max(max_value[instance[i].color], instance[i].value);
  }
  for (int i = 0; i < num_colors; i++) {
    total_max = std::max(total_max, max_value[i]);
  }
  cout << "Color Distribution: " << std::endl;
  vector<int> color(num_colors, 0);
  for (int i = 0; i < instance.size(); i++) {
    color[instance[i].color]++;
  }
  for (int i = 0; i < num_colors; i++) {
    cout << color[i] << " ";
  }
  cout << endl;
  for (const auto& element : answer) {
    if (element.color == -1) {
      not_picked++;
      continue;
    }
    num_answer[element.color]++;
    if (max_value[element.color] - element.value < 0.0000001) {
      correct_answer[element.color]++;
    }
    if (total_max - element.value < 0.0000001) {
      total_correct_answer++;
    }
  }
  cout << "Answer Distribution:" << endl;
  for (int i = 0; i < num_colors; i++) {
    cout << num_answer[i] << " ";
  }
  cout << endl;
  cout << "Correct Answer Distribution:" << endl;
  for (int i = 0; i < num_colors; i++) {
    cout << correct_answer[i] << " ";
  }
  cout << endl;
  cout << "Total Correct Answer: " << total_correct_answer << endl;
  cout << "Probability Correct Answer: "
       << static_cast<double>(total_correct_answer) / answer.size() << endl;
  cout << "Total Not Picked: " << not_picked << endl;
}

void SecretaryEval::ThEval(const vector<SecretaryInstance>& instance,
                           const vector<vector<SecretaryInstance>>& answers,
                           int num_colors) {
  double total_max = 0;
  int not_picked = 0;
  vector<double> total_correct_answer(answers.size(), 0);
  for (int i = 0; i < instance.size(); i++) {
    total_max = std::max(total_max, instance[i].value);
  }
  for (int i = 0; i < answers.size(); i++) {
    for (const auto& element : answers[i]) {
      if (element.color == -1) {
        not_picked++;
        continue;
      }
      if (total_max - element.value < 0.0000001) {
        total_correct_answer[i]++;
      }
    }
    cout << static_cast<double>(total_correct_answer[i]) / answers[i].size()
         << " ";
  }
  cout << endl;
}

void SecretaryEval::InnerUnbalanced(
    const vector<SecretaryInstance>& instance, const SecretaryInstance& ans,
    vector<int>& correct_answer, vector<int>& num_answer, vector<int>& max_dist,
    const int num_colors, int& not_picked, int& total_correct_answer) {
  vector<double> max_value(num_colors, 0);
  double total_max = 0;

  int max_color = 0;
  for (const auto& element : instance) {
    max_value[element.color] =
        std::max(max_value[element.color], element.value);
    double old_max = total_max;
    total_max = std::max(total_max, element.value);
    if (old_max < total_max) {
      max_color = element.color;
    }
  }
  max_dist[max_color]++;
  if (ans.color == -1) {
    not_picked++;
    return;
  }
  num_answer[ans.color]++;
  if (max_value[ans.color] - ans.value < 10) {
    correct_answer[ans.color]++;
  }
  if (total_max - ans.value < 10) {
    total_correct_answer++;
  }
}

}  // namespace fair_secretary
