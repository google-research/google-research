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

"""Simple neural networks in JAX.

Used only for NCP, should be deleted.
"""
import jax
from jax import jit
from jax import vmap
import jax.numpy as np


def random_layer_params(m, n, key, scale=2e-1):
  return (jax.nn.initializers.kaiming_normal()(key, [n, m]) * scale,
          np.zeros((n,)))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = jax.random.split(key, len(sizes))
  return tuple(
      random_layer_params(m, n, k)
      for m, n, k in zip(sizes[:-1], sizes[1:], keys))


def relu(x):
  return np.maximum(0, x)


def nn_fwd(X, params, activation=None, final_activation=None):
  if activation is None:
    activation = relu
  if final_activation is None:
    final_activation = lambda x: x

  activations = X
  for w, b in params[:-1]:
    outputs = np.dot(w, activations) + b
    activations = activation(outputs)

  final_w, final_b = params[-1]
  return final_activation(np.dot(final_w, activations) + final_b)


batch_nn_fwd = jit(vmap(nn_fwd, in_axes=(0, None)))
