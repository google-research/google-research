# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
  return sum(p.size for p in jax.tree_flatten(params_cpu)[0])


def compute_num_flops(f, optimize, *a, **kw):
  m = jax.xla_computation(f)(*a, **kw)
  client = jax.lib.xla_bridge.get_backend()
  if optimize:
    m = client.compile(m).hlo_modules()[0]
  else:
    m = m.as_hlo_module()
  analysis = jax.lib.xla_extension.hlo_module_cost_analysis(client, m)
  return int(analysis['flops'])
