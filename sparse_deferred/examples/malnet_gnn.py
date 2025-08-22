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

"""GIN implementation of MalNet classification using `sparse_deferred`."""

import os

from absl import app
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tqdm

import sparse_deferred as sd
from sparse_deferred.algorithms import auto_hop
from sparse_deferred.examples import data
from sparse_deferred.examples import models
import sparse_deferred.jax as sdjnp
from sparse_deferred.structs import graph_struct

InMemoryDB = graph_struct.InMemoryDB
MALNET_TINY_NUM_CLASSES = 5
BATCH_NUM = 5


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

  # train = create_propagated_db(train)
  # test = create_propagated_db(test)

  graph = train.get_item_with_engine(sdjnp.engine, 0)
  # graphs = []
  # for i in range(BATCH_NUM):
  #   graph = train.get_item_with_engine(sdjnp.engine, i)
  #   graph = graph_struct.GraphStruct.new(
  #       nodes={
  #           'my_nodes': graph.nodes['my_nodes'],
  #       },
  #       edges=graph.edges,
  #       schema=graph.schema,
  #   ).add_pooling(sdjnp.engine, graph.nodes['g'])
  #   graphs.append(graph)
  # graph = graph_struct.combine_graph_structs(sdjnp.engine, *graphs)

  model = GPS(
      num_classes=MALNET_TINY_NUM_CLASSES,
      # node_dim=graph.nodes['my_nodes']['x'].shape[1],
      num_hidden_layers=5,
      hidden_dim=64,
  )
  key = jax.random.PRNGKey(0)
  params = model.init(key, graph)

  opt = optax.chain(
      optax.clip_by_global_norm(1),
      optax.adam(learning_rate=1e-3),
  )
  opt_state = opt.init(params)

  @jax.jit
  def loss_fn(params, graph):
    h = model.apply(params, graph, mutable=['batch_stats'])[0]
    # label = jax.nn.one_hot(graph.nodes['g']['y'], MALNET_TINY_NUM_CLASSES)
    # return jnp.sum(optax.losses.softmax_cross_entropy(h, label))
    label = jnp.reshape(
        jax.nn.one_hot(graph.nodes['g']['y'], MALNET_TINY_NUM_CLASSES),
        (MALNET_TINY_NUM_CLASSES,),
    )
    return optax.losses.softmax_cross_entropy(h, label)

  l2reg = 1e-3

  def l2_loss(x):
    return jnp.sum(x * x) * l2reg

  @jax.jit
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
      tf.data.Dataset.range(train.size)
      .repeat(10)
      .shuffle(1000, reshuffle_each_iteration=True)
      # .batch(BATCH_NUM)
  )

  # for i in tqdm.tqdm(range(train.size)):
  for i in tqdm.tqdm(ds):
    params, opt_state, _ = train_step(
        params,
        opt_state,
        train.get_item_with_engine(sdjnp.engine, i)
    )
  # for step, indices in enumerate(tqdm.tqdm(ds)):
  #   graphs = []
  #   for i in indices:
  #     graph = train.get_item_with_engine(sdjnp.engine, i)
  #     graph = graph_struct.GraphStruct.new(
  #         nodes={
  #             'my_nodes': graph.nodes['my_nodes'],
  #         },
  #         edges=graph.edges,
  #         schema=graph.schema,
  #     ).add_pooling(sdjnp.engine, graph.nodes['g'])
  #     graphs.append(graph)
  #   graph = graph_struct.combine_graph_structs(sdjnp.engine, *graphs)
  #   params, opt_state, _ = train_step(params, opt_state, graph)

  def calculate_test_accuracy(num_examples=test.size):
    correct = 0
    for i in tqdm.tqdm(range(num_examples)):
      graph = test.get_item_with_engine(sdjnp.engine, i)
      z = jnp.argmax(nn.activation.log_softmax(model.apply(params, graph)))
      if z == graph.nodes['g']['y']:
        correct += 1
    return correct / test.size

  # def calculate_test_accuracy(num_examples=test.size):
  #   correct = 0
  #   for i in tqdm.tqdm(range(num_examples)):
  #     graph = test.get_item_with_engine(sdjnp.engine, i)
  #     graph = graph_struct.GraphStruct.new(
  #         nodes={'my_nodes': graph.nodes['my_nodes']},
  #         edges=graph.edges,
  #         schema=graph.schema,
  #     ).add_pooling(sdjnp.engine, graph.nodes['g'])
  #     z = jnp.argmax(model.apply(params, graph), axis=-1)
  #     correct += jnp.sum(z == graph.nodes['g']['y'])
  #   acc = correct / num_examples
  #   return acc

  print('Accuracy is', calculate_test_accuracy())


def compute_propagated_tensors(
    graph, pagerank_alpha = 0.85
):
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
  pr = sd.sum([
      m,
      sd.SparseMatrix(
          sdjnp.engine, indices=(rows, cols), dense_shape=(n, n), values=vals
      ),
  ])
  # while jnp.linalg.norm(w - pr) >= 1e-10:
  #   w = pr
  #   pr = m @ w + (1 - self.alpha)

  deepwalk = sd.prod([inv_degree, adj])

  tensors = auto_hop.recursive_propagate(  # SimpleGCN model.
      {'x': graph.nodes['my_nodes']['x']},
      {('x', 'x'): [adj, transition, pr, deepwalk]},
      'x',
      hops=2,
  )
  return jnp.concatenate(
      [jnp.array(t) for t in tensors], axis=-1
  )  # , dtype=jnp.float64)


def create_propagated_db(db):
  """Returns a new InMemoryDB with propagated tensors."""
  new_db = InMemoryDB()
  for i in tqdm.tqdm(range(db.size)):
    g = db.get_item_with_engine(sdjnp.engine, i)
    tensors = compute_propagated_tensors(g)
    g.update(nodes={'my_nodes': {'x': tensors}})
    new_db.add((g))
  new_db.finalize()
  return new_db


class GIN(nn.Module):
  """Graph information network: https://arxiv.org/pdf/1810.00826."""

  num_classes: int
  num_hidden_layers: int = 1
  hidden_dim: int = 32
  epsilon: float = 0.1  # See GIN paper (link above)

  def setup(self):
    layer_dims = [self.hidden_dim] * self.num_hidden_layers
    self.layers = [nn.Dense(dim, use_bias=False) for dim in layer_dims]
    self.out = nn.Dense(self.num_classes, use_bias=False)
    self.batch_norm = nn.BatchNorm(use_running_average=True)

  def __call__(self, graph):
    x = graph.nodes['my_nodes']['x']
    adj = graph.adj(sdjnp.engine, 'my_edges')
    adj = adj.add_eye(1 + self.epsilon)  # self connections with 1+eps weight.

    for i, layer in enumerate(self.layers):
      x = layer(adj @ x)
      x = self.batch_norm(x)
      if i < self.num_hidden_layers:
        x = nn.relu(x)

    x = jnp.sum(x, axis=0)
    return self.out(x)


class GCN(nn.Module):
  """Graph convolutional network: https://arxiv.org/pdf/1609.02907."""

  num_classes: int
  num_hidden_layers: int = 1
  hidden_dim: int = 32

  def setup(self):
    layer_dims = [self.hidden_dim] * self.num_hidden_layers
    self.layers = [nn.Dense(dim, use_bias=False) for dim in layer_dims]
    # self.out = nn.Dense(self.num_classes, use_bias=False)

  def __call__(self, graph):
    x = graph.nodes['my_nodes']['x']
    adj = graph.adj(sdjnp.engine, 'my_edges')

    for i, layer in enumerate(self.layers):
      x = layer(adj @ x)
      if i < self.num_hidden_layers:
        x = nn.relu(x)

    return x
    # return self.out(adj @ x)


class GPS(nn.Module):
  """GraphGPS: https://arxiv.org/pdf/2205.12454."""

  num_classes: int
  # node_dim: int
  num_hidden_layers: int = 3
  hidden_dim: int = 64

  def setup(self):
    self.pre_dense = nn.Sequential(
        [nn.Dense(self.hidden_dim, use_bias=False), nn.relu]
    )
    self.post_dense = nn.Sequential(
        [nn.Dense(self.hidden_dim, use_bias=False), nn.relu]
    )
    self.attention = models.GTLayer(self.hidden_dim)
    self.gcn = GCN(self.num_classes, self.num_hidden_layers, self.hidden_dim)
    self.layer_norm1 = nn.LayerNorm()
    self.layer_norm2 = nn.LayerNorm()
    self.out = nn.Dense(self.num_classes, use_bias=False)

  def __call__(self, graph):
    x = graph.nodes['my_nodes']['x']
    x = self.pre_dense(x)
    x += self.gcn(graph)
    x = self.layer_norm1(x)
    x += self.attention(x)
    x = self.layer_norm2(x)
    x = self.post_dense(x)
    return self.out(jnp.sum(x, axis=0))

if __name__ == '__main__':
  app.run(main)
