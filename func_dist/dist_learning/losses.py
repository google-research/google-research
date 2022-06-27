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

"""Defines loss functions for distance regression and feature learning.
"""

import jax.numpy as jnp
import optax


def mean_error(x, y):
  return jnp.mean(jnp.abs(x - y))


def mse(x, y):
  return jnp.mean(jnp.square(x - y))


def discrimination_loss(logits, labels):
  return jnp.mean(
      optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))


def adversarial_discrimination_loss(logits, labels):
  return jnp.mean(
      optax.sigmoid_binary_cross_entropy(logits=1.0 - logits, labels=labels))

