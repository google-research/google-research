# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Plot hinton graph."""

# coding=utf-8
import numpy as np
chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


class BarHack(str):

  def __str__(self):
    return self.internal

  def __len__(self):
    return 1


def plot(arr, max_val=None):
  """Plot bars to show categorical distribution."""
  if max_val is None:
    max_arr = arr
    max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))

  opts = np.get_printoptions()
  np.set_printoptions(edgeitems=500)
  fig = np.array2string(
      arr,
      formatter={
          "float_kind": lambda x: visual(x, max_val),
          "int_kind": lambda x: visual(x, max_val)
      },
      max_line_width=5000)
  np.set_printoptions(**opts)

  return fig


def visual(val, max_val):
  """Visualize a single number."""
  val = np.clip(val, 0, max_val)
  if abs(val) == max_val:
    step = len(chars) - 1
  else:
    step = int(abs(float(val) / max_val) * len(chars))
  colourstart = ""
  colourend = ""
  if val < 0:
    colourstart, colourend = "\033[90m", "\033[0m"
  return colourstart + chars[step] + colourend
