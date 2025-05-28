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

"""Simple script to plot the output of bcs.py."""

import sys

from matplotlib import pyplot as plt
import numpy as np

if len(sys.argv) != 3:
  print("Usage: plot.py data-saved-by-bcs.tsv num_trials", file=sys.stderr)
  sys.exit(1)

data_file_name = sys.argv[1]
# Pass in the same value used in bcs.py
# We divide here to keep nice integers in data files instead of floats.
num_trials = int(sys.argv[2])

table = np.loadtxt(data_file_name)
print(table)

fig = plt.figure()  # figsize=(6, 3))
plt.title("Success Probability")
if "heavy" in data_file_name:
  plt.xlabel("Heaviest Value")
elif "depth" in data_file_name:
  plt.xlabel("Depth (d/b)")
elif "width" in data_file_name:
  plt.xlabel("Width (b)")
else:
  print(
      "Can't determine whether data file was heavy or depth sweep " +
      data_file_name,
      file=sys.stderr)
  sys.exit(1)

plt.ylabel("Success Probability")
x = table[:, 0]
columns = [
    "CountSketchMedian", "CountSketchSigns", "BCountSketchMedian",
    "BCountSketchSigns"
]
for i in range(len(columns)):
  y = table[:, 1 + i] / num_trials
  var = y * (1 - y) / (num_trials - 1)
  stdev = var**0.5
  # plt.plot(x, y, label=columns[i])
  print(columns[i])
  print(y)
  print(stdev)
  plt.errorbar(x, y, stdev, label=columns[i])
plt.legend(loc="lower right")
plt.show()

tmp = data_file_name.rsplit(".", 1)
if len(tmp) != 2:
  print("Couldn't split input file name", data_file_name, file=sys.stderr)
  sys.exit(2)
out_file_name = tmp[0] + ".png"
fig.savefig(out_file_name, dpi=fig.dpi)
print("Wrote", out_file_name)
