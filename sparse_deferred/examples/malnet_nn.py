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

"""MLP implementation of MalNet classification using `sparse_deferred`."""

import os

from absl import app
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.utils.extmath import randomized_svd
import tensorflow as tf
import tqdm

import sparse_deferred as sd
from sparse_deferred.algorithms import auto_hop
from sparse_deferred.examples import data
import sparse_deferred.jax as sdjnp
from sparse_deferred.structs import graph_struct


os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

jax.config.update('jax_enable_x64', True)
InMemoryDB = graph_struct.InMemoryDB
MALNET_TINY_NUM_CLASSES = 5
BATCH_NUM = 5
RANK = 50


def activation_fn(activation = 'leaky_relu'):
  """Helper function for FLAX activation function. Defaults to leaky_relu."""
  if not hasattr(nn.activation, activation):
    raise ValueError(f'Activation {activation} is not part of `nn.activation`')
  return getattr(nn.activation, activation)


def mlp(dims, hidden_activation, output_dim, use_bias=True):
  """Helper function for multi-layer perceptron (MLP)."""
  layers = []
  for dim in dims:
    layers.append(nn.Dense(dim, use_bias=use_bias))
    layers.append(activation_fn(hidden_activation))
  layers.append(nn.Dense(output_dim, use_bias=False))
  return nn.Sequential(layers)


def make_propagated_db(db):
  """Returns a new InMemoryDB with propagated tensors."""
  new_db = InMemoryDB()
  for i in tqdm.tqdm(range(db.size)):
    g = db.get_item_with_engine(sdjnp.engine, i)
    new_db.add(amend_graph_struct_with_compute_propagated_tensors(g))
  new_db.finalize()
  return new_db


def main(unused_argv):
  train_npz = os.path.expanduser(
      os.path.join('~', 'data', 'malnet_tiny', 'train.npz')
  )
  val_npz = os.path.expanduser(
      os.path.join('~', 'data', 'malnet_tiny', 'val.npz')
  )
  test_npz = os.path.expanduser(
      os.path.join('~', 'data', 'malnet_tiny', 'test.npz')
  )
  if not os.path.exists(train_npz):
    train, val, test = data.get_malnet_tiny_dataset(
        sdjnp.engine, add_features='one_hot_degree'
    )
    open(train_npz, 'wb+')
    open(val_npz, 'wb+')
    open(test_npz, 'wb+')
    train.save(train_npz)
    val.save(val_npz)
    test.save(test_npz)
  else:
    train = InMemoryDB.from_file(train_npz)
    # val = InMemoryDB.from_file(val_npz)
    test = InMemoryDB.from_file(test_npz)
  # train = make_propagated_db(train)
  # test = make_propagated_db(test)
  # val = make_propagated_db(val)

  train = compute_z_normalization(train)
  test = compute_z_normalization(test)

  graphs = []
  for i in range(BATCH_NUM):
    graph = train.get_item_with_engine(sdjnp.engine, i)
    graph = graph_struct.GraphStruct.new(
        nodes={
            'my_nodes': graph.nodes['my_nodes'],
        },
        edges=graph.edges,
        schema=graph.schema,
    ).add_pooling(sdjnp.engine, graph.nodes['g'])
    graphs.append(graph)
  graph = graph_struct.combine_graph_structs(sdjnp.engine, *graphs)

  # feature_dim = compute_propagated_tensors(graph).shape[-1]

  model = NN(
      num_classes=MALNET_TINY_NUM_CLASSES,
      num_hidden_layers=3,
      hidden_dim=4096,
      # feature_dim=feature_dim,
      activation='leaky_relu',
  )
  key = jax.random.PRNGKey(0)
  params = model.init(key, graph)

  opt = optax.chain(
      optax.clip_by_global_norm(1),
      optax.adam(learning_rate=1e-3),
  )
  opt_state = opt.init(params)

  # @jax.jit
  def loss_fn(params, graph):
    h = model.apply(params, graph)
    label = jax.nn.one_hot(graph.nodes['g']['y'], MALNET_TINY_NUM_CLASSES)
    return jnp.sum(optax.losses.softmax_cross_entropy(h, label))

  l2reg = 1e-3

  def l2_loss(x):
    return jnp.sum(x * x) * l2reg

  # @jax.jit
  def train_step(params, opt_state, graph):
    loss, grads = jax.value_and_grad(loss_fn)(params, graph)
    loss += jnp.sum(
        jnp.asarray(
            jax.tree_util.tree_leaves(jax.tree_util.tree_map(l2_loss, params))
        )
    )  # warning: computes l2 loss over bias as well
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  ds = (
      tf.data.Dataset.range(train.size).repeat(10)
      .shuffle(1000, reshuffle_each_iteration=True).batch(BATCH_NUM))

  def calculate_test_accuracy(num_examples=test.size):
    correct = 0
    for i in tqdm.tqdm(range(num_examples)):
      graph = test.get_item_with_engine(sdjnp.engine, i)
      graph = graph_struct.GraphStruct.new(
          nodes={'my_nodes': graph.nodes['my_nodes']},
          edges=graph.edges,
          schema=graph.schema,
      ).add_pooling(sdjnp.engine, graph.nodes['g'])
      z = jnp.argmax(model.apply(params, graph), axis=-1)
      correct += jnp.sum(z == graph.nodes['g']['y'])
    acc = correct / num_examples
    return acc

  for _, indices in enumerate(tqdm.tqdm(ds)):
    graphs = []
    for i in indices:
      graph = train.get_item_with_engine(sdjnp.engine, i)
      graph = graph_struct.GraphStruct.new(
          nodes={'my_nodes': graph.nodes['my_nodes']},
          edges=graph.edges,
          schema=graph.schema,
      ).add_pooling(sdjnp.engine, graph.nodes['g'])
      graphs.append(graph)
    graph = graph_struct.combine_graph_structs(sdjnp.engine, *graphs)
    params, opt_state, _ = train_step(params, opt_state, graph)

  print('Accuracy is', calculate_test_accuracy())


def compute_z_normalization(db, rank = RANK):
  """Computes z-normalization of each graph in an InMemoryDB."""
  means = []
  stddevs = []
  sizes = []
  for i in tqdm.tqdm(range(db.size)):
    g = db.get_item_with_engine(sdjnp.engine, i)
    x = compute_propagated_tensors(g)
    means.append(jnp.mean(x, axis=0))
    stddevs.append(jnp.std(x, axis=0))
    sizes.append(x.shape[0])

  d = InMemoryDB()
  global_mean = jnp.average(jnp.array(means), axis=0, weights=jnp.array(sizes))
  global_size = jnp.sum(jnp.array(sizes))
  numerator = jnp.sum(
      jnp.array([
          (jnp.square(stddevs[i]) * sizes[i])
          + (sizes[i] * jnp.square(means[i] - global_mean))
          for i in range(len(sizes))
      ])
  )
  global_stddev = jnp.sqrt(jnp.true_divide(numerator, global_size))
  for i in tqdm.tqdm(range(db.size)):
    g = db.get_item_with_engine(sdjnp.engine, i)
    x = compute_propagated_tensors(g)
    z = np.array((x - global_mean) / global_stddev)
    u, s, vt = randomized_svd(z, rank, random_state=None)
    vt = vt[:, :rank]
    g = g.update(
        nodes={'my_nodes': {'x': ((u * s) @ vt)}}
    )
    d.add(g)

  d.finalize()
  return d


def compute_propagated_tensors(
    graph, pagerank_alpha = 0.85):
  """Computes propagated tensors for a graph, using the auto_hop library."""
  adj = graph.adj(sdjnp.engine, 'my_edges')
  transition = adj.normalize_left()
  inv_degree = sdjnp.engine.deferred_diag(jnp.reciprocal(adj.rowsums() + 1e-5))

  n = graph.get_num_nodes(sdjnp.engine, 'my_nodes')
  m = sd.prod(
      [transition, sdjnp.engine.deferred_diag((jnp.ones(n) * pagerank_alpha))]
  )
  cols, rows = graph.edges['my_edges'][0]
  vals = jnp.array([((1 - pagerank_alpha) / n) for _ in range(len(rows))])
  pr = sd.sum([m, sd.SparseMatrix(sdjnp.engine,
                                  indices=(rows, cols),
                                  dense_shape=(n, n),
                                  values=vals)])
  #  while jnp.linalg.norm(w - pr) >= 1e-10:
  #  w = pr
  #  pr = m @ w + (1 - self.alpha)

  deepwalk = sd.prod([inv_degree, adj])

  tensors = auto_hop.recursive_propagate(  # SimpleGCN model.
      {'x': graph.nodes['my_nodes']['x']},
      {('x', 'x'): [adj, transition, pr, deepwalk]},
      'x',
      hops=2,
  )
  return jnp.concatenate(
      [jnp.array(t) for t in tensors], axis=-1)  #, dtype=jnp.float64)


def amend_graph_struct_with_compute_propagated_tensors(
    graph):
  return graph.update(
      nodes={
          'my_nodes': {
              'tensors': compute_propagated_tensors(graph)
              }
          }
      )


class NN(nn.Module):
  """Standard MLP for MalNet classification."""

  num_classes: int
  num_hidden_layers: int = 1
  hidden_dim: int = 32
  activation: str = 'leaky_relu'

  def setup(self):
    self.mlp = mlp(
        [self.hidden_dim] * self.num_hidden_layers,
        self.activation,
        self.num_classes,
    )

  def __call__(self, graph):
    tensors = graph.nodes['my_nodes']['x']
    pooling_adj = graph.adj(
        sdjnp.engine, 'g_my_nodes'
    )  # Shape (TotalNodes, #Graphs)

    return pooling_adj.T.normalize_right() @ self.mlp(
        tensors
    )  # (#Graphs, #Classes)


class SVDNN(nn.Module):
  """Standard MLP with SVD for MalNet classification."""
  num_classes: int
  feature_dim: int
  num_hidden_layers: int = 1
  hidden_dim: int = 32
  activation: str = 'leaky_relu'
  rank: int = RANK

  def setup(self):
    self.mlp = mlp(
        [self.hidden_dim] * self.num_hidden_layers,
        self.activation,
        self.num_classes,
    )
    self.w = self.param(
        'w', nn.initializers.lecun_normal(), (self.feature_dim, 64)
    )

  def low_rank(self, graph):
    h = compute_propagated_tensors(graph)
    u, s, vt = self._svd(h, self.rank)
    us = u * s
    return vt, us

  def _svd(self, x, rank):
    u, s, vt = randomized_svd(x, rank, random_state=None)
    return jnp.array(u), jnp.array(s), jnp.array(vt)

  def __call__(self, graph):
    vt, us = self.low_rank(graph)
    pooling_adj = graph.adj(
        sdjnp.engine, 'g_my_nodes'
    )  # Shape (TotalNodes, #Graphs)

    term = vt @ self.w
    term = us @ term

    result = pooling_adj.T.normalize_right() @ self.mlp(
        term
    )  # (#Graphs, #Classes)
    return result


if __name__ == '__main__':
  app.run(main)
