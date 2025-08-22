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

"""Encoder definitions and util functions."""

from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import objax

from logit_dp.logit_dp import resnet


def get_forward_fn_from_module(module):
  """Converts a Haiku module to a forward function."""
  if not callable(module):
    raise ValueError('Haiku module should be callable.')

  def module_forward_fn(x):
    model = module()
    return model(x)

  return hk.transform(module_forward_fn)


class KaimingNormal(hk.initializers.Initializer):
  """Mimics objax.nn.init.kaiming_normal."""

  def __init__(self, gain=1.0):
    self.gain = gain

  def __call__(self, shape, dtype):
    del dtype
    return objax.nn.init.kaiming_normal(tuple(shape), self.gain)


class XavierNormal(hk.initializers.Initializer):
  """Mimics objax.nn.init.xavier_normal."""

  def __init__(self, gain=1.0):
    self.gain = gain

  def __call__(self, shape, dtype):
    del dtype
    return objax.nn.init.xavier_normal(tuple(shape), self.gain)


# Model definitions.
class SmallEmbeddingNet(hk.Module):
  """Embedding network for CIFAR10 dataset."""

  def __init__(self, embedding_dim=8, name=None):
    super().__init__(name=name)
    # Use initializers that are equivalent to the ones used in OBJAX.
    self.conv_initializer = KaimingNormal()
    self.convs = hk.Sequential([
        hk.Conv2D(
            output_channels=8,
            kernel_shape=(3, 3),
            stride=2,
            data_format='NCHW',
            w_init=self.conv_initializer,
        ),
        jax.nn.relu,
        hk.Conv2D(
            output_channels=16,
            kernel_shape=(3, 3),
            stride=2,
            data_format='NCHW',
            w_init=self.conv_initializer,
        ),
        jax.nn.relu,
        hk.Conv2D(
            output_channels=32,
            kernel_shape=(3, 3),
            stride=2,
            data_format='NCHW',
            w_init=self.conv_initializer,
        ),
        jax.nn.relu,
    ])
    self.embedding_initializer = XavierNormal()
    self.embeddings = hk.Linear(
        output_size=embedding_dim, w_init=self.embedding_initializer
    )

  def __call__(self, x):
    # Apply convolutional stack.
    x = self.convs(x)
    # Apply spatial global pooling.(if it was a batch we would use 2,3?
    # since we need instance based for pairwise gradients we assume x is a
    # single example. )
    x = jnp.mean(x, axis=(1, 2))
    # Calculate embeddings, and normalise them to unit length.
    embeddings = self.embeddings(x)
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings


class FineTuningEmbeddingNet(SmallEmbeddingNet):
  """Classification task for CIFAR100."""

  def __init__(self):
    super().__init__()
    self.classification_layer = hk.Linear(
        output_size=20, w_init=self.embedding_initializer
    )

  def __call__(self, x):
    # Apply convolutional stack.
    x = self.convs(x)
    # Apply spatial global pooling.(if it was a batch we would use 2,3?
    # since we need instance based for pairwise gradients we assume x is a
    # single example. )
    x = jnp.mean(x, axis=(1, 2))
    # Calculate embeddings, and normalise them to unit length.
    embeddings = self.embeddings(x)
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    logits = self.classification_layer(embeddings)
    return logits


class ResNet18(resnet.ResNet18):
  """Resnet18 based embedding network for CIFAR10 dataset."""

  def __init__(self, embedding_dim):
    super().__init__(num_classes=embedding_dim)

  def __call__(self, x):
    # Calculate embeddings, and normalise them to unit length.
    embeddings = super().__call__(inputs=x)
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings
