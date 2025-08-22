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

"""Trainer for various models."""

from flax import optim
import jax
import jax.numpy as jnp
import numpy as np

from rs_gnn import models


class BestKeeper:
  """Keeps best performance and model params during training."""

  def __init__(self, min_or_max):
    self.min_or_max = min_or_max
    self.best_result = np.inf if min_or_max == 'min' else 0.0
    self.best_params = None

  def print_(self, epoch, result):
    if self.min_or_max == 'min':
      print('Epoch:', epoch, 'Loss:', result)
    elif self.min_or_max == 'max':
      print('Epoch:', epoch, 'Accu:', result)

  def update(self, epoch, result, params, print_=True):
    """Updates the best performance and model params if necessary."""
    if print_:
      self.print_(epoch, result)

    if self.min_or_max == 'min' and result < self.best_result:
      self.best_result = result
      self.best_params = params
    elif self.min_or_max == 'max' and result > self.best_result:
      self.best_result = result
      self.best_params = params

  def get(self):
    return self.best_params


def create_model(flags, model_name, inp, rng):
  """Creates model given model_name."""
  new_rng, init_rng, drop_rng = jax.random.split(rng, num=3)
  if model_name == 'gcn':
    features = [flags.hid_dim, flags.num_classes]
    model = models.GCN(features, flags.drop_rate, 'PReLU')
    init = model.init({'params': init_rng, 'dropout': drop_rng}, inp)
  elif model_name == 'rsgnn':
    model = models.RSGNN(flags.hid_dim, flags.num_reps)
    init = model.init({'params': init_rng, 'dropout': drop_rng}, inp, inp)
  return model, init, new_rng


def create_optimizer(flags, init_params, w_decay=0.0):
  optimizer = optim.Adam(learning_rate=flags.lr, weight_decay=w_decay)
  optimizer = optimizer.create(init_params)
  return jax.device_put(optimizer)


def train_rsgnn(flags, graph, rng):
  """Trainer function for RS-GNN."""
  n_nodes = graph.n_node[0]
  labels = jnp.concatenate([jnp.ones(n_nodes), -jnp.ones(n_nodes)])
  model, init_params, rng = create_model(flags, 'rsgnn', graph, rng)
  optimizer = create_optimizer(flags, init_params)

  @jax.jit
  def corrupt_graph(rng):
    return graph._replace(nodes=jax.random.permutation(rng, graph.nodes))

  @jax.jit
  def train_step(optimizer, graph, c_graph, drop_rng):
    def loss_fn(params):
      _, _, _, cluster_loss, logits = model.apply(
          params, graph, c_graph, rngs={'dropout': drop_rng})
      dgi_loss = -jnp.sum(jax.nn.log_sigmoid(labels * logits))
      return dgi_loss + flags.lambda_ * cluster_loss
    loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad), loss

  best_keeper = BestKeeper('min')
  for epoch in range(1, flags.epochs + 1):
    rng, drop_rng, corrupt_rng = jax.random.split(rng, num=3)
    c_graph = corrupt_graph(corrupt_rng)
    optimizer, loss = train_step(optimizer, graph, c_graph, drop_rng)
    if epoch % flags.valid_each == 0:
      best_keeper.update(epoch, loss, optimizer.target)

  _, _, rep_ids, _, _ = model.apply(
      best_keeper.get(), graph, c_graph, train=False)

  return np.array(rep_ids)


def train_gcn(flags, graph, labels, rng, splits):
  """Trainer function for a classification GCN."""
  model, init_params, rng = create_model(flags, 'gcn', graph, rng)
  optimizer = create_optimizer(flags, init_params, flags.w_decay)

  @jax.jit
  def train_step(optimizer, drop_rng):
    def loss_fn(params):
      logits = model.apply(params, graph, rngs={'dropout': drop_rng})
      log_prob = jax.nn.log_softmax(logits)
      return -jnp.sum(log_prob[splits.train] * labels[splits.train])
    loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    return optimizer.apply_gradient(grad), loss

  @jax.jit
  def accuracy(params, mask):
    logits = model.apply(params, graph, train=False)
    correct = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
    return jnp.sum(correct * mask) / jnp.sum(mask)

  best_keeper = BestKeeper('max')
  for epoch in range(1, flags.epochs + 1):
    rng, drop_rng = jax.random.split(rng)
    optimizer, _ = train_step(optimizer, drop_rng)
    if epoch % flags.valid_each == 0:
      accu = accuracy(optimizer.target, splits.valid)
      best_keeper.update(epoch, accu, optimizer.target)

  accu = accuracy(best_keeper.get(), splits.test)
  return float(accu)
