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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test least square variants."""
from incontext import algos
from incontext import sampler_lib
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.io import gfile

if __name__ == "__main__":

  sampler = sampler_lib.Sampler(128, 10, 128, noise_std=0.01)
  data = sampler.sample(n=64)
  seqs, coefficients, xs, ys = data  # [np.repeat(d, 64, axis=0) for d in data]
  print(coefficients[0])
  print(coefficients[1])
  ax = plt.axes()
  _, ls_errors = algos.least_square_predictor_with_errors(xs, ys)
  ax.plot(
      np.arange(1,
                len(ls_errors) + 1),
      ls_errors,
      color="blue",
      label="least_square")
  _, ls_errors = algos.least_square_predictor_with_errors(
      xs, ys, precision=sampler.get_precision())
  ax.plot(
      np.arange(1,
                len(ls_errors) + 1),
      ls_errors,
      color="green",
      label="fake_least_square")
  ax.legend()
  with gfile.GFile("test.jpeg", "wb") as handle:
    plt.savefig(handle, dpi=150)
