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

"""Utilities for working with JAX pytrees."""

import jax
import numpy as np


def summarize_tree(tree):
  """Summarize a pytree."""

  def _describe(x):
    if (hasattr(x, "shape") and x.shape and not isinstance(x, jax.core.Tracer)):
      flat = np.asarray(x).reshape([-1])
      if flat.shape[0] == 0:
        return f"{np.dtype(x.dtype).name}{x.shape}"
      info = f"{np.dtype(x.dtype).name}{x.shape}"
      if np.issubdtype(x.dtype, np.floating):
        info += (f" {np.mean(flat):.2} ±{np.std(flat):.2} "
                 f"[{np.min(flat):.2}, {np.max(flat):.2}]")
        if np.any(flat == 0):
          info += f" nz:{np.count_nonzero(flat) / len(flat):.2}"
        if np.any(np.isnan(flat)):
          info += f" nan:{np.count_nonzero(np.isnan(flat)) / len(flat):.2}"
        if np.any(flat == np.inf):
          info += f" inf:{np.count_nonzero(flat == np.inf) / len(flat):.2}"
        if np.any(flat == -np.inf):
          info += f" -inf:{np.count_nonzero(flat == -np.inf) / len(flat):.2}"
      elif np.issubdtype(x.dtype, np.integer):
        info += (f" [{np.min(flat)}, {np.max(flat)}] "
                 f"nz:{np.count_nonzero(flat) / len(flat):.2}")
      elif np.issubdtype(x.dtype, np.bool_):
        info += f" T:{np.count_nonzero(flat) / len(flat):.2}"

      return info
    else:
      return x

  return jax.tree_map(_describe, tree)
