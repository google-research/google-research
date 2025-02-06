# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Experiments on 4 Sized Bags Large Margin."""
import random

import cvxpy as cp
import numpy as np
import pandas as pd

np.random.seed(123879)

random.seed(234)

D_list = [10, 40]  # dimension

N_list = [50, 100]  # no. of total bags

K = 25  # no. of runs

G = 5  # no. of gaussian roundings

mu = 0

sigma = 1


def hardsign(x):
    """
    Compute the hard sign function.
    The hard sign function returns 1 if x is positive or 0, and -1 otherwise.

    Args:
    x (float): Input value

    Returns:
    int: 1 if x is positive or 0, -1 otherwise
    """
    return 1 if np.sign(x)==1 else -1

# pylint: disable=redefined-outer-name
def propofbag(x, y, z, r):
    """
    Calculate the proportion of bag value using the hard sign function.

    This function calculates the sum of hard sign values for the dot products
    of three vectors (x, y, z) with vector r.

    Args:
    x (numpy.ndarray): First input vector
    y (numpy.ndarray): Second input vector
    z (numpy.ndarray): Third input vector
    r (numpy.ndarray): Vector for dot products

    Returns:
    int: Sum of hard sign values for the dot products
    """
    return hardsign(np.dot(x, r)) + hardsign(np.dot(y, r)) + hardsign(np.dot(z, r))


df = pd.DataFrame(columns=[
    "D", "N", "k", "n_non_mon", "n_mon", "threshold", "satbag", "frac_perf"
])

for D in D_list:
  for N in N_list:
    for k in range(K):

      print("D, N, k: ", D, N, k)

      # generate a random LTF (vector in N+1 dims)
      r = np.random.normal(mu, sigma, D)
      # get a unit vector
      r = r / np.linalg.norm(r)

      n_nonmon = 0

      n_mon = 0

      for i in range(N):
        if np.random.randint(8) == 0:
          n_mon = n_mon + 1
        else:
          n_nonmon = n_nonmon + 1

      list_of_bags = []
      C = cp.Variable((D, D), symmetric=True)
      constraints = [C >> 0]
      constraints += [cp.trace(C) == 1]

      list_of_matrices = []

      #

      for i in range(n_nonmon):
        list_of_matrices.append([])
        for j in range(6):
          list_of_matrices[i].append(cp.Variable((D, D), symmetric=True))
          constraints += [list_of_matrices[i][j] >> 0]
          constraints += [(C - list_of_matrices[i][j]) >> 0]

        # Add Valid Constraints 0 - (1,2),
        # 1 - (1,3), 2 - (1,4), 3 - (2,3), 4 - (2,4), 5 - (3,4)

        constraints += [(list_of_matrices[i][0] + list_of_matrices[i][1] +
                         list_of_matrices[i][2] - C) >> 0]
        constraints += [(list_of_matrices[i][0] + list_of_matrices[i][3] +
                         list_of_matrices[i][4] - C) >> 0]
        constraints += [(list_of_matrices[i][1] + list_of_matrices[i][3] +
                         list_of_matrices[i][5] - C) >> 0]
        constraints += [(list_of_matrices[i][2] + list_of_matrices[i][4] +
                         list_of_matrices[i][5] - C) >> 0]

      test_list = []
      i = 0
      while i < N + 100:
        # ith bag
        # First get a unit vector close to the plane r.x = 0
        u_1 = np.random.normal(mu, sigma, D)

        u_2 = np.random.normal(mu, sigma, D)

        u_3 = np.random.normal(mu, sigma, D)

        u_4 = np.random.normal(mu, sigma, D)

        x1 = u_1 / np.linalg.norm(u_1)

        x2 = u_2 / np.linalg.norm(u_2)

        x3 = u_3 / np.linalg.norm(u_3)

        x4 = u_4 / np.linalg.norm(u_4)

        if i >= N:

          x_list = [x1, x2, x3, x4]

          rind_x = random.randrange(4)

          test_list.append(
              dict({
                  "x": x_list[rind_x],
                  "hardsign": hardsign(np.dot(x_list[rind_x], r))
              }))

          i = i + 1
          continue

        if i >= n_nonmon:  # monochromatic bag

          if hardsign(np.dot(x2, r)) != hardsign(np.dot(x1, r)):
            x2 = (-1) * x2

          if hardsign(np.dot(x3, r)) != hardsign(np.dot(x1, r)):
            x3 = (-1) * x3

          if hardsign(np.dot(x4, r)) != hardsign(np.dot(x1, r)):
            x4 = (-1) * x4

          print("hardsign(np.dot(x1, r)): ", hardsign(np.dot(x1, r)))
          print("hardsign(np.dot(x2, r)): ", hardsign(np.dot(x2, r)))
          print("hardsign(np.dot(x3, r)): ", hardsign(np.dot(x3, r)))
          print("hardsign(np.dot(x4, r)): ", hardsign(np.dot(x4, r)))

          # sanity check

          if abs(propofbag(x1, x2, x3, x4, r)) != 4:
            print("Something wrong mon")
            print(
                dict({
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "x4": x4,
                    "propofbag": propofbag(x1, x2, x3, x4, r)
                }))
            exit(0)

          x1x2_trans = np.outer(x1, x2)
          x1x3_trans = np.outer(x1, x3)
          x2x3_trans = np.outer(x2, x3)

          x1x4_trans = np.outer(x1, x4)
          x4x3_trans = np.outer(x4, x3)
          x2x4_trans = np.outer(x2, x4)

          # Add monochromatic bag constraints
          constraints += [cp.trace(C @ x1x2_trans) >= 0.0000001]
          constraints += [cp.trace(C @ x1x3_trans) >= 0.0000001]
          constraints += [cp.trace(C @ x2x3_trans) >= 0.0000001]

          constraints += [cp.trace(C @ x1x4_trans) >= 0.0000001]
          constraints += [cp.trace(C @ x4x3_trans) >= 0.0000001]
          constraints += [cp.trace(C @ x2x4_trans) >= 0.0000001]

        else:  # non-monochromatic bag

          if abs(propofbag(x1, x2, x3, x4, r)) == 4:
            if random.randrange(7) <= 2:
              x1 = (-1) * x1
              x2 = (-1) * x2
            else:
              x1 = (-1) * x1

          print("hardsign(np.dot(x1, r)): ", hardsign(np.dot(x1, r)))
          print("hardsign(np.dot(x2, r)): ", hardsign(np.dot(x2, r)))
          print("hardsign(np.dot(x3, r)): ", hardsign(np.dot(x3, r)))
          print("hardsign(np.dot(x4, r)): ", hardsign(np.dot(x4, r)))

          ## sanity check

          if abs(propofbag(x1, x2, x3, x4, r)) == 4:
            print("Something wrong nonmon")
            print(
                dict({
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "x4": x4,
                    "propofbag": propofbag(x1, x2, x3, x4, r)
                }))
            exit(0)

          x1x2_trans = np.outer(x1, x2)
          x1x3_trans = np.outer(x1, x3)
          x2x3_trans = np.outer(x2, x3)

          x1x4_trans = np.outer(x1, x4)
          x3x4_trans = np.outer(x3, x4)
          x2x4_trans = np.outer(x2, x4)

          # Add Valid Constraints

          constraints += [cp.trace(list_of_matrices[i][0] @ x1x2_trans) <= 0.0]
          constraints += [cp.trace(list_of_matrices[i][1] @ x1x3_trans) <= 0.0]
          constraints += [cp.trace(list_of_matrices[i][2] @ x1x4_trans) <= 0.0]
          constraints += [cp.trace(list_of_matrices[i][3] @ x2x3_trans) <= 0.0]
          constraints += [cp.trace(list_of_matrices[i][4] @ x2x4_trans) <= 0.0]
          constraints += [cp.trace(list_of_matrices[i][5] @ x3x4_trans) <= 0.0]

        bag = dict({
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "propofbag": propofbag(x1, x2, x3, x4, r)
        })

        list_of_bags.append(bag)

        i = i + 1

      # solve the SDP
      prob = cp.Problem(cp.Minimize(cp.trace(C)), constraints)
      prob.solve(solver=cp.SCS, verbose=True, acceleration_lookback=0)

      # Get the eigen decompistion of C

      w, v = np.linalg.eigh(C.value)

      normalized_max_eigen = w[D - 1] / (np.sum(w))

      # get the performance of tildeac for each Gaussian rounding

      best_total_bags_satisfied = 0

      best_total_bags_satisfied_random_thresh = 0

      best_tildec = None

      best_random_thresh = None

      for i in range(G):

        nonmon_sat = 0

        mon_sat = 0

        mon_oppsat = 0

        total_bags_satisfied = 0

        nonmon_sat_random_thresh = 0

        mon_sat_random_thresh = 0

        mon_oppsat_random_thresh = 0

        total_bags_satisfied_random_thresh = 0

        tildec = np.dot(v * np.sqrt(w), np.random.normal(mu, sigma, D))

        random_thresh = np.random.normal(mu, sigma, D)

        for i, bag in enumerate(list_of_bags):
          x1 = bag["x1"]
          x2 = bag["x2"]
          x3 = bag["x3"]
          x3 = bag["x4"]
          if abs(bag["propofbag"]) == 4:
            if bag["propofbag"] == propofbag(x1, x2, x3, x4, tildec):
              mon_sat = mon_sat + 1
            else:
              if bag["propofbag"] == (-1) * propofbag(x1, x2, x3, x4, tildec):
                mon_oppsat = mon_oppsat + 1

            ####
            if bag["propofbag"] == propofbag(x1, x2, x3, x4, random_thresh):
              mon_sat_random_thresh = mon_sat_random_thresh + 1
            else:
              if (bag["propofbag"] == (-1) *
                  propofbag(x1, x2, x3, x4, random_thresh)):
                mon_oppsat_random_thresh = mon_oppsat_random_thresh + 1
            ###

          else:
            if abs(propofbag(x1, x2, x3, x4, tildec)) != 4:
              nonmon_sat = nonmon_sat + 1

            ##
            if abs(propofbag(x1, x2, x3, x4, random_thresh)) != 4:
              nonmon_sat_random_thresh = nonmon_sat_random_thresh + 1

        if mon_oppsat > mon_sat:
          total_bags_satisfied = mon_oppsat + nonmon_sat
          tildec = (-1) * tildec
        else:
          total_bags_satisfied = mon_sat + nonmon_sat

        if best_total_bags_satisfied < total_bags_satisfied:
          best_total_bags_satisfied = total_bags_satisfied
          best_tildec = tildec

##
        if mon_oppsat_random_thresh > mon_sat_random_thresh:
          total_bags_satisfied_random_thresh = mon_oppsat_random_thresh + nonmon_sat_random_thresh
          random_thresh = (-1) * random_thresh
        else:
          total_bags_satisfied_random_thresh = mon_sat_random_thresh + nonmon_sat_random_thresh

        if best_total_bags_satisfied_random_thresh < total_bags_satisfied_random_thresh:
          best_total_bags_satisfied_random_thresh = total_bags_satisfied_random_thresh
          best_random_thresh = random_thresh
##

## Test part

      if len(test_list) != 100:
        print("Test list length not 100, exiting")
        exit(0)

      test_sat_tildec = 0
      test_sat_random_thresh = 0
      for point in test_list:
        x_point = point["x"]
        x_sign = point["hardsign"]
        if hardsign(np.dot(best_tildec, x_point)) == x_sign:
          test_sat_tildec = test_sat_tildec + 1
        if hardsign(np.dot(best_random_thresh, x_point)) == x_sign:
          test_sat_random_thresh = test_sat_random_thresh + 1

      dict_to_add = dict({
          "D": D,
          "N": N,
          "k": k,
          "n_non_mon": n_nonmon,
          "n_mon": n_mon,
          "satbag": best_total_bags_satisfied,
          "satbag_random_thresh": best_total_bags_satisfied_random_thresh,
          "frac_rank_A": normalized_max_eigen,
          "test_sat_tildec": test_sat_tildec,
          "test_sat_random_thresh": test_sat_random_thresh
      })

      df = pd.DataFrame(dict_to_add, index=[0])

      print(dict_to_add)

      if D == 10 and N == 50 and k == 0:
        df.to_csv("large_margin_4_sized_results.csv", index=False)
      else:
        df.to_csv(
            "large_margin_4_sized_results.csv",
            index=False,
            mode="a",
            header=False)
