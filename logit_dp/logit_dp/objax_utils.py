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

"""Utility functions for running Objax models"""

from typing import Iterable

import jax.numpy as jnp
import objax
from objax.functional import loss as objax_loss
import objax.zoo.resnet_v2


class ObjaxEmbeddingNet(objax.Module):
  """Small embedding net Objax model."""

  def __init__(self, embedding_dim=8):
    self.conv_initializer = objax.nn.init.kaiming_normal
    nc = [8, 16, 32]
    self.convs = objax.nn.Sequential([
        objax.nn.Conv2D(
            nin=3, nout=nc[0], k=3, strides=2, w_init=self.conv_initializer
        ),
        objax.functional.relu,
        objax.nn.Conv2D(
            nin=nc[0],
            nout=nc[1],
            k=3,
            strides=2,
            w_init=self.conv_initializer,
        ),
        objax.functional.relu,
        objax.nn.Conv2D(
            nin=nc[1],
            nout=nc[2],
            k=3,
            strides=2,
            w_init=self.conv_initializer,
        ),
        objax.functional.relu,
    ])
    self.embedding_initializer = objax.nn.init.xavier_normal
    self.embeddings = objax.nn.Linear(
        nin=nc[2], nout=embedding_dim, w_init=self.embedding_initializer
    )

  def __call__(self, x, training):
    # Apply convolutional stack.
    x = self.convs(x)
    # Apply spatial global pooling. Axes 2 and 3 correspond to height and width,
    # respectively, whereas axes 0 and 1 correspond to the batch number and
    # channel, respectively.
    x = jnp.mean(x, axis=(2, 3))
    # Calculate embeddings, and normalise them to unit length.
    embeddings = self.embeddings(x)
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings


class ObjaxFinetuningNet(ObjaxEmbeddingNet):
  """Small embedding net Objax model."""

  def __init__(self, embedding_dim=8, hidden_dims=[64, 32, 20]):
    super().__init__()
    classification_layers = []
    num_layers = len(hidden_dims)
    hidden_dims.insert(0, embedding_dim)
    for i in range(num_layers):
      classification_layers.append(
          objax.nn.Linear(
              nin=hidden_dims[i],
              nout=hidden_dims[i + 1],
              w_init=self.embedding_initializer,
          )
      )
      if i < num_layers - 1:
        # Avoid relu on last layer
        classification_layers.append(objax.functional.relu)
    self.classification_layers = objax.nn.Sequential(classification_layers)

  def __call__(self, x, training):
    embeddings = super().__call__(x, training)
    logits = self.classification_layers(embeddings)
    return logits


class StatelessBatchNorm(objax.module.Module):
  """Stateless version of objax.nn.BatchNorm."""

  def __init__(
      self, dims, redux, eps = 1e-5
  ):
    super().__init__()
    dims = tuple(dims)
    self.eps = eps
    self.redux = tuple(redux)
    self.beta = objax.variable.TrainVar(jnp.zeros(dims))
    self.gamma = objax.variable.TrainVar(jnp.ones(dims))

  def __call__(
      self, x, training
  ):
    del training
    m = x.mean(self.redux, keepdims=True)
    v = ((x - m) ** 2).mean(self.redux, keepdims=True)
    return (
        self.gamma.value * (x - m) * objax.functional.rsqrt(v + self.eps)
        + self.beta.value
    )

  def __repr__(self):
    args = dict(dims=self.beta.value.shape, redux=self.redux, eps=self.eps)
    args = ', '.join(f'{x}={y}' for x, y in args.items())
    return f'{objax.util.class_name(self)}({args})'


class StatelessBatchNorm2D(StatelessBatchNorm):
  """Stateless version of objax.nn.BatchNorm2D."""

  def __init__(self, nin, eps = 1e-5):
    super().__init__((1, nin, 1, 1), (0, 2, 3), eps)

  def __repr__(self):
    return (
        f'{objax.util.class_name(self)}(nin={self.beta.value.shape[1]},'
        f' eps={self.eps})'
    )


class ObjaxResNet18(objax.zoo.resnet_v2.ResNet18):
  """ResNet18 implementation from `objax.zoo`."""

  def __init__(self, in_channels=3, num_classes=10, use_batch_norm=False):
    if use_batch_norm:
      normalization_fn = StatelessBatchNorm2D
    else:
      # Equivalent to the identity operator.
      normalization_fn = lambda _: objax.nn.Dropout(1.0)
    super().__init__(
        in_channels, num_classes, normalization_fn=normalization_fn
    )
    self.in_channels = in_channels
    self.num_classes = num_classes

  def __call__(self, *args, **kwargs):
    embeddings = super().__call__(*args, **kwargs)
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings


def make_objax_loss_function(model, temperature, extended_batch_axis=False):
  """Binary cross-entropy loss function for Objax_models."""

  @objax.Function.with_vars(model.vars())
  def loss(batch):
    # Run both anchors and positives through model.
    if extended_batch_axis:
      batch = jnp.squeeze(batch)
    anchor_embeddings = model(batch[:, 0, :], training=True)
    positive_embeddings = model(batch[:, 1, :], training=True)
    # Calculate cosine similarity between anchors and positives. As they have
    # been normalised this is just the pair wise dot products.
    similarities = jnp.einsum(
        'ae,pe->ap', anchor_embeddings, positive_embeddings
    )
    # Since we intend to use  these as logits we scale them by a temperature.
    # This value would normally be chosen as a hyper parameter.
    similarities /= temperature
    # We use these similarities as logits for a softmax. The labels for
    # this call are just the sequence [0, 1, 2, ..., batch_size since we
    # want the main diagonal values, which correspond to the anchor/positive
    # pairs, to be high. This loss will move embeddings for the
    # anchor/positive pairs together and move all other pairs apart.
    batch_size = batch.shape[0]
    sparse_labels = jnp.arange(batch_size)
    return jnp.mean(
        objax_loss.cross_entropy_logits_sparse(
            logits=similarities, labels=sparse_labels
        )
    )

  return loss


def make_objax_finetuning_loss_function(model):
  """Binary cross-entropy loss function for Objax_models."""

  cross_entrpy_loss_fn = objax.functional.loss.cross_entropy_logits_sparse

  @objax.Function.with_vars(model.vars())
  def loss(batch, labels):
    logits = model(batch, training=True)
    return cross_entrpy_loss_fn(logits, labels=labels).mean()

  return loss
