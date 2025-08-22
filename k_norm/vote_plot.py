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

"""Script for running vote simulation and saving its plot."""

import matplotlib.pyplot as plt
import numpy as np

from k_norm import lp_mechanism
from k_norm import vote_mechanism


epsilon = 1
d_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
num_trials = 10000
avg_errors = np.zeros((3, len(d_range), num_trials))
error_norm = 2
for i, d in enumerate(d_range):
  print(d)
  for trial in range(num_trials):
    avg_errors[0, i, trial] = np.linalg.norm(
        lp_mechanism.lp_mechanism(
            np.zeros(d), 1, np.linalg.norm(np.arange(d), ord=1), epsilon
        ),
        ord=error_norm
    )
    avg_errors[1, i, trial] = np.linalg.norm(
        lp_mechanism.lp_mechanism(np.zeros(d), np.inf, d - 1, epsilon),
        ord=error_norm
    )
    avg_errors[2, i, trial] = np.linalg.norm(
        vote_mechanism.vote_mechanism(np.zeros(d), epsilon), ord=error_norm
    )
results = np.mean(avg_errors, axis=2)

plt.plot(
    d_range,
    results[0],
    label="$\ell_1$", # pylint: disable=anomalous-backslash-in-string
    linestyle="dashed",
    color="teal",
    linewidth=2,
)
plt.plot(
    d_range,
    results[1],
    label="$\ell_\infty$", # pylint: disable=anomalous-backslash-in-string
    linestyle="dotted",
    color="olive",
    linewidth=2,
)
plt.plot(
    d_range,
    results[2],
    label="induced",
    linestyle="solid",
    color="orange",
    linewidth=2,
)
plt.xlabel("$d$")
plt.ylabel("mean $\ell_2$ error") # pylint: disable=anomalous-backslash-in-string
plt.title("ranked vote")
plt.legend()
plt.savefig("vote_simulation.pdf", bbox_inches="tight")
