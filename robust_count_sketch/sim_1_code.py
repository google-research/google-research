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

"""Attack on CountSketch and median estimator."""

import math

import numpy as np
from numpy.random import default_rng


class CountsketchMedianAttack:
  """Attack on the classic CountSketch."""

  def __init__(self,
               num_of_rounds=100,
               nof_repetitions=10,
               nof_sim_keys=11,
               tail_size=1000,
               b=30,
               l=100,
               seed=None):
    self.num_of_rounds = num_of_rounds
    self.nof_repetitions = nof_repetitions
    self.nof_sim_keys = nof_sim_keys
    self.tail_size = tail_size
    self.l = l
    self.b = b
    self.seed = seed
    self.rng = default_rng(seed)
    self.results = np.zeros((self.num_of_rounds, self.nof_sim_keys),
                            dtype=float)

  def __repr__(self):
    return ("test parameters:\n number of rounds = {0}; number of repetitions ="
            " {6}\n num of keys simulated = {1};\n attack tail size = {2};\n l "
            "= {3};\n b = {4}; seed = {5};").format(self.num_of_rounds,
                                                    self.nof_sim_keys,
                                                    self.tail_size, self.l,
                                                    self.b, self.seed,
                                                    self.nof_repetitions)

  def draw_sketch(self):
    self.sim_keys_hash = self.rng.choice(
        self.b, size=(self.l, self.nof_sim_keys))
    self.sim_keys_sign = self.rng.choice(
        2, size=(self.l, self.nof_sim_keys)) * 2 - 1

  def generate_v(self, mk_factor=10, lk_factor=20):
    sd = math.sqrt(self.tail_size / self.b)
    lk_weight = int(sd * lk_factor)
    mk_weight = int(sd * mk_factor)
    v = np.ones(self.nof_sim_keys, dtype=int)
    v[0:2] = mk_weight
    v[2:] = lk_weight
    return v

  def encode_v(self, v):
    counters_v = np.zeros((self.l, self.b), dtype=int)
    for line in range(self.l):
      for key in range(self.nof_sim_keys):
        counters_v[line,
                   self.sim_keys_hash[line,
                                      key]] += v[key] * self.sim_keys_sign[line,
                                                                           key]
    return counters_v

  def decode_v(self, counters):
    weak_estimates = np.zeros((self.l, self.nof_sim_keys), dtype=int)
    for line in range(self.l):
      for key in range(self.nof_sim_keys):
        weak_estimates[line, key] = counters[line, self.sim_keys_hash[
            line, key]] * self.sim_keys_sign[line, key]
    # for even length axis: returns average of the two medians
    estimates = np.median(weak_estimates, axis=0)
    return estimates

  def check_parameters(self, nof_checks=100):
    # pass if all sketch draws return exact estimates for v
    v = self.generate_v()
    results = np.zeros(nof_checks, dtype=int)
    for i in range(nof_checks):
      self.draw_sketch()
      estimates_v = self.decode_v(self.encode_v(v))
      results[i] = np.absolute(estimates_v - v).sum()
    print("parameters OK")
    return results.sum() == 0

  def gencode_z(self):
    counters_z = np.zeros((self.l, self.b), dtype=int)
    for line in range(self.l):
      tail_hash = self.rng.choice(self.b, size=self.tail_size)
      tail_sign = self.rng.choice(2, size=self.tail_size) * 2 - 1
      for key in range(self.tail_size):
        counters_z[line, tail_hash[key]] += tail_sign[key]
    return counters_z

  def simulate_median_attack(self, files_pref="sim_1", new_seed=None):
    """Runs the attack and writes the results into CSV files."""
    if new_seed is not None:
      self.seed = new_seed
      self.rng = default_rng(new_seed)
    self.results = np.zeros((self.num_of_rounds, 3, self.nof_repetitions),
                            dtype=float)
    # here we add looping over 10 repetitions
    for rep in range(self.nof_repetitions):
      self.draw_sketch()
      nof_collected = 0
      v = self.generate_v()
      counters_a = np.zeros((self.l, self.b), dtype=int)
      for r in range(self.num_of_rounds):
        # query
        counters_v = self.encode_v(v)
        counters_z = self.gencode_z()
        estimates_q = self.decode_v(counters_v + counters_z)
        key_0_reported = estimates_q[0] >= sorted(estimates_q)[1]
        key_1_reported = estimates_q[1] >= sorted(estimates_q)[1]
        # collecting desicion
        if key_0_reported and (not key_1_reported):
          counters_a = counters_a + counters_z
          nof_collected += 1
        if key_1_reported and (not key_0_reported):
          counters_a = counters_a - counters_z
          nof_collected += 1
        # saving the keys signal
        factor = 1
        if nof_collected > 0:
          factor = math.sqrt(nof_collected * self.tail_size / self.b)
        estimates_a = self.decode_v(counters_a)
        self.results[r, :, rep] = estimates_a[0:3] / factor
      # here we ouput a file
      print("saving file for rep = {0}".format(rep))
      file_name = "./results/{0}_rep_{1}.csv".format(files_pref, rep)
      np.savetxt(file_name, self.results[:, :, rep], delimiter=",")
    return self.results


# test 1: 10 repetitions of 10000 rounds attack on skeches of size l=100, b=30.
rounds = 10000
test_1 = CountsketchMedianAttack(num_of_rounds=rounds)
print(test_1)

if test_1.check_parameters():
  test_1.simulate_median_attack(files_pref="sim_1_run")
