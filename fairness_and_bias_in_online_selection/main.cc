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

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <random>
#include <type_traits>

#include "bank_oracle.h"
#include "clustering.h"
#include "distributions.h"
#include "fair-prophet.h"
#include "fair-secretary.h"
#include "pokec_oracle.h"
#include "random_handler.h"
#include "secretary_eval.h"
#include "synthetic_data.h"
#include "unfair-prophet.h"
#include "unfair-secretary.h"
#include "utils.h"

namespace fair_secretary {

using std::vector;
using std::cout;
using std::endl;

// The number of repetitions for the experiments.
const int num_rep = 100;

// General helper functions.
void MyShuffle(vector<SecretaryInstance>& elements) {
  for (int j = 0; j < 2; j++) {
    for (int i = 0; i < elements.size(); i++) {
      int x = RandomHandler::eng_() % elements.size();
      std::swap(elements[i], elements[x]);
    }
  }
}

double Average(const vector<SecretaryInstance>& elements) {
  double sum = 0;
  for (const auto& element : elements) {
    sum += std::max(0.0, element.value);
  }
  sum /= elements.size();
  return sum;
}

vector<double> GetThreshold(const vector<double>& p) {
  vector<double> t(p.size(), 0);
  int k = p.size();
  t[k - 1] = pow((1 - (k - 1) * p[k - 1]), (1.0 / (k - 1)));
  for (int i = k - 2; i > 0; i--) {
    double sum = 0;
    for (int r = 0; r <= i; r++) {
      sum += p[r];
    }
    sum /= i;
    t[i] = t[i + 1] * pow((sum - p[i]) / (sum - p[i + 1]), 1.0 / i);
  }
  t[0] = t[1] * exp(p[1] / p[0] - 1);
  return t;
}

vector<double> ComputeThreshold(int max_size) {
  vector<double> t(max_size + 1, 0);
  for (int i = 1; i < max_size; i++) {
    vector<double> p(i + 1, 1.0 / (i + 1));
    t[i + 1] = GetThreshold(p)[0];
  }
  return t;
}

// Experiments helper functions.
void SecretaryExperiment(vector<SecretaryInstance> instance, int num_color,
                         const vector<double>& threshold) {
  vector<SecretaryInstance> answer;
  vector<SecretaryInstance> answer_unfair;
  vector<SecretaryInstance> answer_single;
  FairSecretaryAlgorithm fair_sec_algo(
      vector<int>(num_color, instance.size() * threshold[num_color]),
      num_color);
  UnfairSecretaryAlgorithm unfair_sec_algo;
  for (int i = 0; i < num_rep; i++) {
    MyShuffle(instance);
    answer.push_back(fair_sec_algo.ComputeSolution(instance));
    answer_unfair.push_back(unfair_sec_algo.ComputeSolution(instance));
    answer_single.push_back(unfair_sec_algo.ComputeSolutionSingleColor(
        instance, vector<double>(num_color, 1.0 / num_color)));
  }
  SecretaryEval::Eval(instance, answer, num_color);
  cout << "Unfair Results." << endl;
  SecretaryEval::Eval(instance, answer_unfair, num_color);
  SecretaryEval::Eval(instance, answer_single, num_color);
}

void SecretaryThresholdExperiment(vector<SecretaryInstance> instance,
                                  int num_color) {
  vector<vector<SecretaryInstance>> answers;
  for (double i = 0; i < 20; i++) {
    answers.push_back(vector<SecretaryInstance>());
    FairSecretaryAlgorithm fair_sec_algo(
        vector<int>(num_color, instance.size() * (0.05 * i)), num_color);
    for (int j = 0; j < num_rep; j++) {
      MyShuffle(instance);
      answers[i].push_back(fair_sec_algo.ComputeSolution(instance));
    }
  }
  SecretaryEval::ThEval(instance, answers, num_color);
}

// Single threshold experiments.

// This is the Feedback maximization experiment for the bank dataset in the
// paper. Please make sure that you update the path to the input in
// `bank_oracle.cc` file. One can modify the way that the input is read there
// as well.
void BankSecretaryExperiment(const vector<double>& threshold) {
  BankOracle bank_oracle;
  vector<SecretaryInstance> instance =
      bank_oracle.GetSecretaryInput(/*num_elements=*/100000);
  SecretaryExperiment(instance, bank_oracle.num_colors, threshold);
}

// Please make sure that you update the path to the input in `clustering.cc`
// file. One can modify the way that the input is read there as well.
void ClusteringSecretaryExperiment(const vector<double>& threshold) {
  ClusteringOracle clustering_oracle;
  vector<SecretaryInstance> instance =
      clustering_oracle.GetSecretaryInput(/*num_elements=*/100000);
  SecretaryExperiment(instance, clustering_oracle.num_colors, threshold);
}

// This is the Influence maximization experiment for the pokec dataset in the
// paper. Please make sure that you update the path to the input in
// `pokec_oracle.cc` file. One can modify the way that the input is read there
// as well.
void InfMaxSecretaryExperiment(const vector<double>& threshold) {
  PokecOracle pokec_data;
  vector<SecretaryInstance> instance = pokec_data.GetSecretaryInput();
  SecretaryExperiment(instance, pokec_data.num_colors, threshold);
}

// This is the Syntatic experiment for the secretary problem in the paper. The
// fields that are needed to be set are explained inside the function.
void SyntaticSecretaryExperiment(const vector<double>& threshold) {
  SyntheticData syn_data;
  // The number of elements from each color. Here we have 4 colors, first color
  // has 10 elements, second color has 100, third color has 1000, and fourth
  // color has 10000. Please update as desired.
  vector<SecretaryInstance> instance =
      syn_data.GetSecretaryInput({10, 100, 1000, 10000});
  SecretaryExperiment(instance, syn_data.num_colors, threshold);
}

// Multi threshold experiments. Same as the experiments above but runs for
// multiple thresholds instead of the best one.
void BankSecretaryThresholdExperiment() {
  BankOracle bank_oracle;
  vector<SecretaryInstance> instance =
      bank_oracle.GetSecretaryInput(/*num_elements=*/100000);
  SecretaryThresholdExperiment(instance, bank_oracle.num_colors);
}

void SyntaticSecretaryThresholdExperiment() {
  SyntheticData syn_data;
  // The number of elements from each color.
  vector<SecretaryInstance> instance =
      syn_data.GetSecretaryInput({1000, 1000, 1000, 1000, 1000});
  SecretaryThresholdExperiment(instance, syn_data.num_colors);
}

// This is Syntatic experiment for secretary problem that instead of selecting
// with equal probability from each color, selects proportional to a given
// probability for each color. Notice that it does not guarantee to pick with
// given probability.
void UnbalanceThresholdSecretary() {
  SyntheticData syn_data;
  // Indicates the number of elements from each color.  Here we have 4 colors,
  // first color has 10 elements, second color has 100, third color has 1000,
  // and fourth color has 10000. Please update as desired.
  vector<int> sizes = {10, 100, 1000, 10000};
  int sum_sizes = 0;
  for (const auto size : sizes) {
    sum_sizes += size;
  }
  // Indicates the probabilities of picking from each of the colors 1 to 4.
  vector<double> prob = {0.3, 0.25, 0.25, 0.2};
  vector<double> th_d = GetThreshold(prob);
  vector<int> threshold;
  int num_colors = prob.size();
  threshold.reserve(th_d.size());
  const int num_algos = 3;
  for (int i = 0; i < th_d.size(); i++) {
    threshold.push_back(th_d[i] * sum_sizes);
    cout << threshold.back() << endl;
  }
  FairSecretaryAlgorithm fair_sec_algo(threshold, prob.size());
  UnfairSecretaryAlgorithm unfair_algo;
  vector<vector<int>> correct_answer(num_algos, vector<int>(num_colors, 0));
  vector<vector<int>> num_answer(num_algos, vector<int>(num_colors, 0));
  vector<vector<int>> max_dist(num_algos, vector<int>(num_colors, 0));
  int not_picked[num_algos] = {0, 0, 0};
  int total_correct_answer[num_algos] = {0, 0, 0};
  for (int i = 0; i < num_rep; i++) {
    vector<SecretaryInstance> instance =
        syn_data.GetSecretaryInput(sizes, prob);
    MyShuffle(instance);
    // Runs this experiment for the following three algorithms.
    SecretaryInstance ans[3] = {
        fair_sec_algo.ComputeSolution(instance),
        unfair_algo.ComputeSolution(instance),
        unfair_algo.ComputeSolutionSingleColor(instance, prob)};
    for (int i = 0; i < num_algos; i++) {
      SecretaryEval::InnerUnbalanced(instance, ans[i], correct_answer[i],
                                     num_answer[i], max_dist[i], num_colors,
                                     not_picked[i], total_correct_answer[i]);
    }
  }
  for (int j = 0; j < num_algos; j++) {
    cout << "Max Distribution:" << endl;
    for (int i = 0; i < num_colors; i++) {
      cout << max_dist[j][i] << " ";
    }
    cout << endl;
    cout << "Answer Distribution:" << endl;
    for (int i = 0; i < num_colors; i++) {
      cout << num_answer[j][i] << " ";
    }
    cout << endl;
    cout << "Correct Answer Distribution:" << endl;
    for (int i = 0; i < num_colors; i++) {
      cout << correct_answer[j][i] << " ";
    }
    cout << endl;
    cout << "Total Correct Answer:" << total_correct_answer[j] << endl;
    cout << "Total Not Picked: " << not_picked[j] << endl;
  }
}

// This is Syntatic experiment for prophet problem. We support two distributions
// Uniform (`unif_dist`) and Binomial distributions.
void SyntaticProphetExperiment() {
  SyntheticData syn_data;
  UniformDistribution unif_dist;
  BinomialDistribution bi_dist;
  // Number of elements to be sampled from the distributions.
  int size = 50;
  unif_dist.Init(size, 1.0);
  bi_dist.Init(size, 1.0 / 2);
  bi_dist.ComputeMaxDist(size);
  // Chose the desired distributions to be used. The numbers are sampled from
  // these two distributions, half from each.
  vector<std::reference_wrapper<RandomDistribution>> distributions = {
      unif_dist, unif_dist};
  vector<int> sizes(size, 1);
  vector<double> q(size, 1.0 / size);
  vector<SecretaryInstance> answer, answer_iid;
  vector<SecretaryInstance> instance;
  vector<SecretaryInstance> answer_unfair1, answer_unfair2, answer_unfair3,
      answer_unfair4;
  FairProphetAlgorithm fair_algo;
  UnfairProphetAlgorithm unfair_algo;
  for (int i = 0; i < num_rep; i++) {
    instance = syn_data.GetProphetInput(sizes.size(), distributions);
    answer.push_back(fair_algo.ComputeSolution(instance, distributions, q));
    answer_iid.push_back(
        fair_algo.ComputeSolutionIID(instance, distributions, q));
    answer_unfair1.push_back(
        unfair_algo.ComputeSolutionOneHalf(instance, distributions, q));
    answer_unfair2.push_back(
        unfair_algo.ComputeSolutionOneMinusOneE(instance, distributions, q));
    answer_unfair3.push_back(
        unfair_algo.ComputeSolutionThreeForth(instance, distributions, q));
    answer_unfair4.push_back(
        unfair_algo.ComputeSolutionDiffEq(instance, distributions, q));
  }
  SecretaryEval::Eval(instance, answer, sizes.size());
  cout << "Average Value: " << Average(answer) << endl;
  SecretaryEval::Eval(instance, answer_iid, sizes.size());
  cout << "Average Value: " << Average(answer_iid) << endl;
  cout << "Unfair Results." << endl;
  SecretaryEval::Eval(instance, answer_unfair1, sizes.size());
  cout << "Average Value: " << Average(answer_unfair1) << endl;
  SecretaryEval::Eval(instance, answer_unfair2, sizes.size());
  cout << "Average Value: " << Average(answer_unfair2) << endl;
  SecretaryEval::Eval(instance, answer_unfair3, sizes.size());
  cout << "Average Value: " << Average(answer_unfair3) << endl;
  SecretaryEval::Eval(instance, answer_unfair4, sizes.size());
  cout << "Average Value: " << Average(answer_unfair4) << endl;
}
}  // namespace fair_secretary

int main() {
  // Computes the optimum thresholds for the threshold-based algorithms.
  std::vector<double> threshold = fair_secretary::ComputeThreshold(20);

  // Please select one of the algorithm from this field to be executed.
  // Bofore running the experiments makes sure the paths to the dataset are set
  // in the oracle class. The details are provided before the declaration of
  // each function above.

  // Experiments for the Secretary problem.
  // fair_secretary::BankSecretaryExperiment(threshold);
  // fair_secretary::SyntaticSecretaryExperiment(threshold);
  // fair_secretary::ClusteringSecretaryExperiment(threshold);
  // fair_secretary::InfMaxSecretaryExperiment(threshold);
  // fair_secretary::SyntaticSecretaryThresholdExperiment();
  // fair_secretary::BankSecretaryThresholdExperiment();

  // This is also an experiment for the Secretaty problem. In this experiment
  // we indicate the probability that we want to select from each color.
  // The variables can be set inside the function.
  // fair_secretary::UnbalanceThresholdSecretary();

  // Experiments for Prophet Problem.
  // fair_secretary::SyntaticProphetExperiment();
  return 0;
}
