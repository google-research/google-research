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

"""Utilities for building ecosystem."""

import numpy as np
import scipy.stats as stats


def sample_from_simplex(rng, dim):
  """Uniformly samples a probability vector from a simplex of dimension dim."""
  alpha = [1] * dim
  return rng.dirichlet(alpha)


def sample_from_unit_ball(rng, dim):
  """Uniformly samples a vector from a unit ball."""
  vec = rng.randn(dim)
  return vec / np.sqrt(np.sum(vec**2))


def sample_from_truncated_normal(mean, std, clip_a, clip_b, size=None):
  """Samples from a truncated normal of mean and std within [clip_a, clip_b]."""
  a, b = (clip_a - mean) / std, (clip_b - mean) / std
  r = stats.truncnorm.rvs(a, b, size=size)
  return r * std + mean
