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

"""Tools for collecting performance data."""

from typing import Callable, Any

import jax


def compute_num_params(params_cpu):
  return sum(p.size for p in jax.tree.flatten(params_cpu)[0])


def compute_num_flops(f, optimize, *a, **kw):
  m = jax.jit(f).lower(*a, **kw)
  if optimize:
    analysis = m.compile(m).cost_analysis()  # pytype: disable=wrong-arg-types  # jax-api-types
  else:
    analysis = m.cost_analysis()
  return int(analysis['flops'])
