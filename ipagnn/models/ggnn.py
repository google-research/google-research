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

"""Gated Graph Neural Network."""

from absl import logging  # pylint: disable=unused-import
from flax.deprecated import nn
import jax
from jax import lax
import jax.numpy as jnp

from ipagnn.modules import common_modules

Embed = common_modules.Embed
StackedRNNCell = common_modules.StackedRNNCell


def create_lstm_cells(n):
  """Creates a list of n LSTM cells."""
  cells = []
  for i in range(n):
    cell = nn.LSTMCell.partial(
        gate_fn=nn.sigmoid,
        activation_fn=nn.tanh,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.orthogonal(),
        bias_init=nn.initializers.zeros,
        name=f'lstm_{i}',
    )
    cells.append(cell)
  return cells


class GGNNLayer(nn.Module):
  """A single layer of GGNN message passing."""

  def apply(
      self,
      statement_embeddings,
      source_indices,
      dest_indices,
      edge_types,
      num_nodes,
      hidden_size,
      config):
    """Apply graph attention transformer layer."""
    gru_cell = nn.recurrent.GRUCell.shared(name='gru_cell')

    num_edge_types = 6
    num_edges = edge_types.shape[0]
    edge_dense = nn.Dense.partial(  # Used for creating key/query/values.
        name='edge_dense',
        features=num_edge_types * hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    # statement_embeddings.shape: num_nodes, hidden_size
    source_embeddings = statement_embeddings[source_indices]
    # source_embeddings.shape: num_edges, hidden_size

    new_source_embeddings_all_types = edge_dense(source_embeddings)
    # new_statement_embeddings_all_types.shape:
    #   num_edges, (num_edge_types*hidden_size)
    new_source_embeddings_by_type = (
        new_source_embeddings_all_types.reshape(
            (-1, num_edge_types, hidden_size)))
    # new_source_embeddings_by_type.shape:
    #    num_edges, num_edge_types, hidden_size
    new_source_embeddings = (
        new_source_embeddings_by_type[jnp.arange(num_edges), edge_types, :])
    # new_source_embeddings.shape: num_edges, hidden_size

    proposed_statement_embeddings = jax.ops.segment_sum(
        data=new_source_embeddings,
        segment_ids=dest_indices,
        num_segments=num_nodes,
    )
    # proposed_statement_embeddings.shape: num_nodes, hidden_state

    _, outputs = gru_cell(proposed_statement_embeddings, statement_embeddings)
    return outputs


class GGNN(nn.Module):
  """GGNN model."""

  def apply(self, inputs, info, config, train=False, cache=None):
    start_indexes = inputs['start_index']  # pylint: disable=unused-variable
    exit_indexes = inputs['exit_index']
    steps_all = jnp.squeeze(inputs['steps'], axis=-1)
    # steps_all.shape: batch_size
    edge_types = inputs['edge_types']
    source_indices = inputs['source_indices']
    dest_indices = inputs['dest_indices']
    vocab_size = info.features[info._builder.key('statements')].vocab_size  # pylint: disable=protected-access
    output_token_vocabulary_size = info.output_vocab_size
    hidden_size = config.model.hidden_size
    data = inputs['data'].astype('int32')
    unused_batch_size, num_nodes, unused_statement_length = data.shape

    max_steps = int(1.5 * info.max_diameter)

    # Init parameters
    def emb_init(key, shape, dtype=jnp.float32):
      return jax.random.uniform(
          key, shape, dtype,
          -config.initialization.maxval,
          config.initialization.maxval)

    embed = Embed.shared(num_embeddings=vocab_size,
                         features=hidden_size,
                         emb_init=emb_init,
                         name='embed')

    cells = create_lstm_cells(config.model.rnn_cell.layers)
    lstm = StackedRNNCell.shared(cells=cells)
    initial_state = lstm.initialize_carry(
        jax.random.PRNGKey(0), cells, (), hidden_size)

    def embed_statement(token_embeddings):
      # token_embeddings.shape: 4, hidden_size
      _, results = lax.scan(lstm, initial_state, token_embeddings)
      return results[-1]
    embed_all_statements_single_example = jax.vmap(embed_statement)
    embed_all_statements = jax.vmap(embed_all_statements_single_example)

    output_dense = nn.Dense.shared(
        name='output_dense',
        features=output_token_vocabulary_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    node_embeddings = embed(data)
    # node_embeddings.shape:
    #     batch_size, num_nodes, statement_length, hidden_size
    statement_embeddings = embed_all_statements(node_embeddings)
    # statement_embeddings.shape: batch_size, num_nodes, hidden_size

    gnn_layer_single_example = GGNNLayer.shared(
        num_nodes=num_nodes,
        hidden_size=hidden_size,
        config=config)
    gnn_layer = jax.vmap(gnn_layer_single_example)

    # statement_embeddings.shape: batch_size, num_nodes, hidden_size
    for step in range(max_steps):
      new_statement_embeddings = gnn_layer(
          statement_embeddings,
          source_indices,
          dest_indices,
          edge_types)
      # steps_all.shape: batch_size
      valid = jnp.expand_dims(step < steps_all, axis=(1, 2))
      # valid.shape: batch_size, 1, 1
      statement_embeddings = jnp.where(
          valid,
          new_statement_embeddings,
          statement_embeddings)

    def get_final_state(statement_embeddings, exit_index):
      return statement_embeddings[exit_index]
    final_states = jax.vmap(get_final_state)(statement_embeddings, exit_indexes)
    # final_states.shape: batch_size, hidden_size
    logits = output_dense(final_states)
    # logits.shape: batch_size, output_token_vocabulary_size
    logits = jnp.expand_dims(logits, axis=1)
    # logits.shape: batch_size, 1, output_token_vocabulary_size
    return logits
