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

"""CNN haiku models."""

from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

Batch = Tuple[jnp.ndarray, jnp.ndarray]


def lenet_fn(batch):
  """Network inspired by LeNet-5."""
  x, _ = batch
  x = x.astype(jnp.float32)
  cnn = hk.Sequential([
      hk.Conv2D(output_channels=32, kernel_shape=5, padding="SAME"),
      jax.nn.relu,
      hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
      hk.Conv2D(output_channels=64, kernel_shape=5, padding="SAME"),
      jax.nn.relu,
      hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
      hk.Conv2D(output_channels=128, kernel_shape=5, padding="SAME"),
      hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
      hk.Flatten(),
      hk.Linear(1000),
      jax.nn.relu,
      hk.Linear(1000),
      jax.nn.relu,
      hk.Linear(10),
  ])
  return cnn(x)


def resnet(batch, is_training):
  """ResNet-18."""
  x, _ = batch
  x = x.astype(jnp.float32)
  net = hk.nets.ResNet18(10, resnet_v2=True)
  return net(x, is_training=is_training)
