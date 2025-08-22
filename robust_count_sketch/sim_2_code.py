# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attack round dependency."""

import math

import numpy as np
from numpy.random import binomial
from numpy.random import default_rng
from numpy.random import normal


class AttackRoundDependency:
  """Attack round dependency."""

  # pylint: disable=dangerous-default-value
  def __init__(self,
               nof_repetitions=20,
               sd_list=[1.0],
               nof_sim_keys=3,
               tail_size=1000,
               b=30,
               l_list=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
               seed=None):
    self.nof_repetitions = nof_repetitions
    self.nof_sim_keys = nof_sim_keys
    self.sd_list = sd_list
    self.tail_size = tail_size
    self.l_list = l_list
    self.b = b
    self.seed = seed
    self.rng = default_rng(seed)
    self.results = np.zeros(
        (self.nof_repetitions, len(self.l_list), len(self.sd_list)),
        dtype=float)

  def __repr__(self):
    return ("test parameters:\n number of repetitions = {0}; \n num of keys "
            "simulated = {1};\n attack tail size = {2};\n ell values= {3}; sd "
            "values {6}\n b = {4}; seed = {5};").format(self.nof_repetitions,
                                                        self.nof_sim_keys,
                                                        self.tail_size,
                                                        self.l_list, self.b,
                                                        self.seed, self.sd_list)

  def set_l(self, curr_l):
    self.l = curr_l

  def set_sd(self, curr_sd):
    self.sd = curr_sd

  def draw_sketch(self):
    self.sim_keys_hash = self.rng.choice(
        self.b, size=(self.l, self.nof_sim_keys))
    self.sim_keys_sign = self.rng.choice(
        2, size=(self.l, self.nof_sim_keys)) * 2 - 1

  def check_parameters(self, nof_checks=100):
    # pass if all sketch draws return exact estimates for v
    v = self.generate_v()
    results = np.zeros(nof_checks, dtype=int)
    for i in range(nof_checks):
      self.draw_sketch()
      estimates_v = self.decode_v(self.encode_v(v))
      results[i] = np.absolute(estimates_v - v).sum()
    return results.sum() == 0

  def record_attack_round_dependency(self, file_pref="sim_2"):
    """Saves the results into CSV files."""
    self.results = np.zeros(
        (self.nof_repetitions, len(self.l_list), len(self.sd_list)),
        dtype=float)
    for l_idx in range(len(self.l_list)):
      self.set_l(self.l_list[l_idx])
      for sd_idx in range(len(self.sd_list)):
        self.set_sd(self.sd_list[sd_idx])
        for rep in range(self.nof_repetitions):
          self.draw_sketch()
          self.results[rep, l_idx, sd_idx] = self.simulate_median_attack()
      # self.results[:, l_idx, :]
      print("saving file for ell = {0}".format(self.l_list[l_idx]))
      file_name = "./results/{0}_ell_{1}.csv".format(file_pref,
                                                     self.l_list[l_idx])
      np.savetxt(file_name, self.results[:, l_idx, :], delimiter=",")

  def simulate_median_attack(self):
    """Simulates attack on the median estimator."""
    counters_a = np.zeros((self.l, self.nof_sim_keys), dtype=float)
    key_0_bias = 0
    key_1_bias = 0
    win_round = 1
    nof_collected = 0
    nof_sd = self.sd
    while abs(key_0_bias) < nof_sd or abs(key_1_bias) < nof_sd:
      #  query
      counters_z = self.get_tail_contribution()
      counters_z_median = np.median(counters_z, axis=0)
      # collection decision
      if counters_z_median[0] > counters_z_median[1]:
        counters_a = counters_a + counters_z
        nof_collected += 1
      elif counters_z_median[0] < counters_z_median[1]:
        counters_a = counters_a - counters_z
        nof_collected += 1
      # update the keys stopping signal
      factor = 1
      if nof_collected > 0:
        factor = math.sqrt(nof_collected * self.tail_size / self.b)
      estimates_a = np.median(counters_a, axis=0)
      key_0_bias = estimates_a[0] / factor
      key_1_bias = estimates_a[1] / factor
      win_round += 1
    print(
        "For l = {3}, sd = {4}, attack wins after {2} rounds. Bias: key 0 = {0}, key 1 = {1}."
        .format(key_0_bias, key_1_bias, win_round, self.l, nof_sd))
    return win_round

  def get_tail_contribution(self):
    """BLRW distribution contribution."""
    contribution = np.zeros((self.l, self.nof_sim_keys), dtype=float)
    normal_mean = 0
    normal_sd = 1
    for j in range(self.l):
      line_contribution = np.zeros(3, dtype=float)
      line_contribution[0] = math.sqrt(binomial(
          self.tail_size, 1 / self.b)) * normal(normal_mean, normal_sd)
      line_contribution[1] = math.sqrt(binomial(
          self.tail_size, 1 / self.b)) * normal(normal_mean, normal_sd)
      line_contribution[2] = math.sqrt(binomial(
          self.tail_size, 1 / self.b)) * normal(normal_mean, normal_sd)
      contribution[j] = line_contribution
    return contribution


# test 1:  5 repetitions of SD = 4
test = AttackRoundDependency(nof_repetitions=5, sd_list=[4])
print(test)
test.record_attack_round_dependency(file_pref="sim_2_sd_4")

# test 2: 20 repetitions of SD = 1
test = AttackRoundDependency(nof_repetitions=20, sd_list=[1])
print(test)
test.record_attack_round_dependency(file_pref="sim_2_sd_1")
