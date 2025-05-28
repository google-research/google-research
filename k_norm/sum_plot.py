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

"""Script for running sum simulation and saving its plot."""

import matplotlib.pyplot as plt
import numpy as np

from k_norm import lp_mechanism
from k_norm import sum_mechanism


epsilon = 1
d = 50
eulerian_numbers = sum_mechanism.compute_eulerian_numbers(d)
l0_bounds = np.arange(10, 31)
linf_bound = 1
num_trials = 1000
avg_errors = np.zeros((3, len(l0_bounds), num_trials))
error_norm = 2
for i, l0_bound in enumerate(l0_bounds):
  print(l0_bound)
  for trial in range(num_trials):
    avg_errors[0, i, trial] = np.linalg.norm(
        lp_mechanism.lp_mechanism(np.zeros(d), 1, l0_bound, epsilon),
        ord=error_norm,
    )
    avg_errors[1, i, trial] = np.linalg.norm(
        lp_mechanism.lp_mechanism(np.zeros(d), np.inf, 1, epsilon),
        ord=error_norm,
    )
    avg_errors[2, i, trial] = np.linalg.norm(
        sum_mechanism.sum_mechanism(
            np.zeros(d), eulerian_numbers, l0_bound, epsilon
        ),
        ord=error_norm,
    )
results = np.mean(avg_errors, axis=2)

plt.plot(
    l0_bounds,
    results[0],
    label="$\ell_1$",  # pylint: disable=anomalous-backslash-in-string
    linestyle="dashed",
    color="teal",
    linewidth=2,
)
plt.plot(
    l0_bounds,
    results[1],
    label="$\ell_\infty$", # pylint: disable=anomalous-backslash-in-string
    linestyle="dotted",
    color="olive",
    linewidth=2,
)
plt.plot(
    l0_bounds,
    results[2],
    label="induced",
    linestyle="solid",
    color="orange",
    linewidth=2,
)
plt.xlabel("$l_0$ bound")
plt.ylabel("mean $\ell_2$ error") # pylint: disable=anomalous-backslash-in-string
plt.title("sum, $d=" + str(d) + "$")
plt.legend()
plt.savefig("sum_simulation.pdf", bbox_inches="tight")
