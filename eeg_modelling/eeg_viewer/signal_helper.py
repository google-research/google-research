# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python2, python3
"""Contains helper functions for signal sampling."""

import numpy as np
from scipy.signal import decimate


def downsample_attribution(signal, ratio=10):
  """Downsamples a 1D or 2D signal after applying anti-aliasing filter."""
  return decimate(signal, q=ratio, ftype='iir')


def find_quantile(signal, q):
  """Finds the qth quantile of a signal."""
  signal = np.sort(signal.flatten())
  ind = int((signal.shape[0] - 1) * q)
  return signal[ind]


def threshold_attribution(signal, q=0.999):
  """Suppresses largest signals bt applying a threshold."""
  s = np.abs(signal)
  quantile = find_quantile(s, q)
  ratio = np.abs(signal / quantile)
  return signal / np.maximum(1.0, ratio)
