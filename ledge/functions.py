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

import numpy as np


# All function outputs are scaled and offset so they have an output
# range of [-1, 1].  The input x must be in [0, 1), which spans
# exactly one cycle.
def unitRelu(x):
  return np.maximum((x - 0.5) * 2, 0) * 2 - 1

def unitPieceLinear(x):
  return x * 2 - 1

def unitTriangle(x):
  return (0.25 - np.abs(x - 0.5)) * 4

def unitSquare(x):
  return (x <= 0.5) * 2 - 1.0

def unitParabola(x):
  return (x * 2 - 1) ** 2 * 2 - 1


def applyFunc(start, period, samples_wanted, func_name):
  """Returns discrete samples of a function.

  Args:
    start: (float) Time offset of the returned points. The Phi from
      the commands. In the range [-1, 1) for a single cycle.
    period: (float) Temporal scale of the returned points. The lambda from
      the commands. A period of -1 means 0.01 cycles per x. A period of
      1 is 10x slower or 0.001 cycles per x. Values outside this range
      are also fine.
    samples_wanted: (int) Returned samples are from x=range(samples_wanted).
    func_name: (str) Name of the func to sample.

  Returns: (list of float) y-values for the samples_wanted.
  """
  # The commands come from the DNN in [-1, 1] range. Rescale them
  # to be meaningful to a periodic function.
  start = (start + 1) / 2
  period = (period + 1) / 2 * 9 + 1
  funcs = {
      'relu': unitRelu,
      'pieceLinear': unitPieceLinear,
      'triangle': unitTriangle,
      'square': unitSquare,
      'parabola': unitParabola,
  }

  fn = funcs[func_name]
  res = fn((np.arange(samples_wanted) / period / 100 + start) % 1.0)
  return res
