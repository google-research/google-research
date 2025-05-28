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

"""xm_utils"""

import jax.numpy as jnp
import numpy as np


def clip(x, clip_norm=1.0):
  divisor = jnp.maximum(jnp.linalg.norm(x) / clip_norm, 1.)
  return x / divisor


# To be used with jax.jit
def eval_step(wopt, test_x_np_list, hidden_dims, num_labels):
  nc = 0.0
  t = 0.0
  theta_np = jnp.reshape(wopt, (-1, hidden_dims))
  for l in range(num_labels):
    l_p = jnp.argmax(
        jnp.einsum('ld,nd->nl', theta_np, jnp.array(test_x_np_list[l])),
        axis=1)
    t += len(test_x_np_list[l])
    nc += jnp.sum(l_p == l)
  return nc / t


def to_flat_np(xs, labels, num_labels):
  xs_np = list(map(lambda x: x.numpy(), xs))
  labels = list(map(lambda x: x.numpy(), labels))
  x_np = np.concatenate(xs_np, axis=0)
  y_np = np.concatenate(labels, axis=0)
  x_list = [[] for _ in range(num_labels)]
  for x, y in zip(x_np, y_np):
    x_list[y].append(x)
  return x_list
