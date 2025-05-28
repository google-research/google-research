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

"""FLAX implementation of GIN training using `sparse_deferred`."""

from absl import app
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from sparse_deferred.examples import data
import sparse_deferred.jax as sdjnp
from sparse_deferred.structs import graph_struct


def main(argv):
  del argv
  allx, edge_list, node_labels, test_idx = data.get_planetoid_dataset('cora')

  allx = jnp.array(allx)
  edge_list = jnp.array(edge_list)
  num_classes = node_labels.max() + 1
  node_labels = jax.nn.one_hot(node_labels, num_classes)
  test_idx = jnp.array(test_idx)

  graph = graph_struct.GraphStruct.new(
      nodes={
          'my_nodes': {'x': allx},
          'graph': {
              'test_idx': test_idx,
              'train_idx': jnp.arange(num_classes * 20),  # Per Kipf et al.
          },
      },
      edges={'my_edges': ((edge_list[0], edge_list[1]), {})},
      schema={
          # my_edges connects my_nodes --> my_nodes.
          'my_edges': ('my_nodes', 'my_nodes')
      })

  model = GIN(num_classes=node_labels.shape[1])
  key = jax.random.PRNGKey(0)
  params = model.init(key, graph)

  opt = optax.chain(
      optax.clip_by_global_norm(1),
      optax.adam(learning_rate=1e-3),
  )
  opt_state = opt.init(params)

  def loss_fn(params, graph, label):
    h = model.apply(params, graph)
    train_ids = graph.nodes['graph']['train_idx']
    h = h[train_ids]
    label = label[train_ids]
    return jnp.mean(-nn.activation.log_softmax(h) * label)

  l2reg = 1e-3
  def l2_loss(x):
    return jnp.sum(x * x) * l2reg

  def train_step(params, opt_state, graph, label):
    loss, grads = jax.value_and_grad(loss_fn)(params, graph, label)
    loss += jnp.sum(
        jnp.asarray(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(l2_loss, params)
            )
        )
    )  # warning: computes l2 loss over bias as well
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  epochs = 50
  for _ in range(epochs):
    params, opt_state, _ = train_step(
        params,
        opt_state,
        graph,
        node_labels,
    )

  z = model.apply(params, graph)
  acc = jnp.argmax(node_labels[test_idx], axis=-1) == jnp.argmax(
      z[test_idx], axis=-1
  )
  print('Accuracy is', jnp.mean(acc))


class GIN(nn.Module):
  """Graph information network model: https://arxiv.org/pdf/1810.00826.pdf."""

  num_classes: int
  num_hidden_layers: int = 1
  hidden_dim: int = 32
  epsilon: float = 0.1  # See GIN paper (link above)

  def setup(self):
    layer_dims = [self.hidden_dim] * self.num_hidden_layers
    layer_dims.append(self.num_classes)
    self.layers = [nn.Dense(dim, use_bias=False) for dim in layer_dims]

  def __call__(self, graph):
    x = graph.nodes['my_nodes']['x']
    adj = graph.adj(sdjnp.engine, 'my_edges')
    adj = adj.add_eye(1 + self.epsilon)  # self connections with 1+eps weight.

    for i, layer in enumerate(self.layers):
      x = layer(adj @ x)
      if i < self.num_hidden_layers:
        x = nn.relu(x)
    return x


if __name__ == '__main__':
  app.run(main)
