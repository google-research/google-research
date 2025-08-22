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

"""Script for generating the data and plots in the paper."""

import matplotlib as plt
import numpy as np

from dp_l2 import experiments
from dp_l2 import l2

# Run error, time, and sigma experiments.
eps = 1
delta = 1e-5
d_range = range(1, 101)
num_rs = 1000
num_e1_rs = 1000
num_samples = 1000
sigma_times, mean_errors, sample_times, l2_sigmas = (
    experiments.errors_and_times_experiment(
        eps, delta, d_range, num_rs, num_e1_rs, num_samples
    )
)

# Generate error plots.
plt.plot(
    # We only plot d <= 8 because, for larger d, the Laplace error exceeds the
    # Gaussian error.
    d_range[:8],
    np.divide(mean_errors["Laplace"], mean_errors["Gaussian"])[:8],
    label="laplace",
    linestyle="dotted",
    linewidth=3,
    color="#1b1d17",
)
plt.plot(
    # The other two plots are normalized by Gaussian error, so the Gaussian
    # plot is a constant 1.
    d_range,
    np.ones(len(d_range)),
    label="gaussian",
    linestyle="dashed",
    linewidth=3,
    color="#e9e4a4",
)
plt.plot(
    d_range,
    np.divide(mean_errors["l2"], mean_errors["Gaussian"]),
    label="$\ell_2$", # pylint: disable=anomalous-backslash-in-string
    linewidth=3,
    color="#8984d8",
)
plt.ylabel("normalized mean squared $\ell_2$ error", fontsize=12) # pylint: disable=anomalous-backslash-in-string
plt.yticks(fontsize=12)
plt.xlabel("dimension $d$", fontsize=12)
plt.xticks(fontsize=12)
plt.title("normalized error", fontsize=14)
plt.legend(fontsize=12)
title = "errors"
plt.savefig(title + ".pdf", bbox_inches="tight")
plt.close()

# Generate time plots.
plt.plot(
    d_range,
    sigma_times["Laplace"],
    linestyle="dotted",
    linewidth=3,
    color="#1b1d17",
    label="laplace",
)
plt.plot(
    d_range,
    sigma_times["Gaussian"],
    linestyle="dashed",
    linewidth=3,
    color="#e9e4a4",
    label="gaussian",
)
plt.plot(
    d_range,
    sigma_times["l2"],
    linewidth=3,
    color="#8984d8",
    label="$\ell_2$", # pylint: disable=anomalous-backslash-in-string
)
plt.xlabel("dimension $d$", fontsize=12)
plt.ylabel("time (s)", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale("log")
plt.title("$\sigma$ computation time", fontsize=14) # pylint: disable=anomalous-backslash-in-string
plt.legend(fontsize=12, loc=(0.65, 0.2))
title = "sigma_times"
plt.savefig(title + ".pdf", bbox_inches="tight")
plt.close()

plt.plot(
    d_range,
    sample_times["Laplace"],
    linestyle="dotted",
    linewidth=3,
    color="#1b1d17",
    label="laplace",
)
plt.plot(
    d_range,
    sample_times["Gaussian"],
    linestyle="dashed",
    linewidth=3,
    color="#e9e4a4",
    label="gaussian",
)
plt.plot(
    d_range,
    sample_times["l2"],
    linewidth=3,
    color="#8984d8",
    label="$\ell_2$", # pylint: disable=anomalous-backslash-in-string
)
plt.xlabel("dimension $d$", fontsize=12)
plt.ylabel("time (s)", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("sampling time", fontsize=14)
plt.legend(fontsize=12)
title = "sample_times"
plt.savefig(title + ".pdf", bbox_inches="tight")
plt.close()

# Generate l_2 mechanism pure DP parameter plot.
# By Lemma 2.4 in the paper, the pure DP parameter of the l_2 mechanism is
# 1 / sigma.
plt.plot(d_range, np.divide(1, l2_sigmas), linewidth=3, color="#8984d8")
plt.xlabel("dimension $d$", fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel("pure DP parameter", fontsize=12)
plt.yticks(fontsize=12)
plt.title("$\ell_2$ mechanism pure DP guarantees", fontsize=14) # pylint: disable=anomalous-backslash-in-string
title = "pure_dp"
plt.savefig(title + ".pdf", bbox_inches="tight")
plt.close()

# Run empirical privacy experiment.
eps = 1
delta = 0.01
num_rs = 1000
num_e1_rs = 1000
d_range = range(1, 102, 5)
empirical_sigmas = np.zeros(len(d_range))
theory_sigmas = np.zeros(len(d_range))
for d_idx, d in enumerate(d_range):
  empirical_sigmas[d_idx] = experiments.empirical_get_l2_sigma(d, eps, delta)
  theory_sigmas[d_idx] = l2.get_l2_sigma(d, eps, delta, num_rs, num_e1_rs)

# Generate empirical privacy plot.
plt.plot(
    d_range,
    empirical_sigmas,
    linestyle=":",
    color="#8984d8",
    linewidth=3,
    label="empirical",
)
plt.plot(
    d_range,
    theory_sigmas,
    color="#8984d8",
    linewidth=3,
    label="theory",
)
plt.xlabel("dimension $d$", fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel("parameter $\sigma$", fontsize=12) # pylint: disable=anomalous-backslash-in-string
plt.yticks(fontsize=12)
plt.title("empirical privacy analysis tightness", fontsize=14)
plt.legend(fontsize=12)
title = "empirical_privacy"
plt.savefig(title + ".pdf", bbox_inches="tight")
plt.close()
