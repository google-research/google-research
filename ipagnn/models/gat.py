# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Graph Attention Network."""

from absl import logging  # pylint: disable=unused-import
from flax import nn
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


def leaky_relu(x, alpha=0.2):
  return jnp.maximum(alpha * x, x)


class GATLayer(nn.Module):
  """A single layer of Graph Attention."""

  def apply(
      self,
      inputs,
      source_indices,
      dest_indices,
      edge_types,
      num_nodes,
      hidden_size,
      config):
    """Apply graph attention transformer layer."""
    attention_dense = nn.Dense.partial(  # Used for creating key/query/values.
        name='attention-dense',
        features=hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    attention_dense2 = nn.Dense.partial(  # Used for computing attention logits.
        name='attention-dense2',
        features=1,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))

    def emb_init(key, shape, dtype=jnp.float32):
      return jax.random.uniform(
          key, shape, dtype,
          -config.initialization.maxval,
          config.initialization.maxval)
    edge_type_embedding_layer = Embed.partial(
        name='edge_type_embedding_layer',
        num_embeddings=6,
        features=hidden_size,
        emb_init=emb_init)

    # inputs.shape: num_nodes, hidden_size
    keys = attention_dense(inputs)
    queries = keys
    values = keys
    # keys.shape: num_nodes, hidden_size

    # edge_types.shape: num_edges,
    edge_embeddings = edge_type_embedding_layer(edge_types)
    # edge_embeddings.shape: num_edges, hidden_size

    # source_indices.shape: num_edges
    keys_source = keys[source_indices]
    # keys_source.shape: num_edges, hidden_size
    queries_dest = queries[dest_indices]
    # queries_dest.shape: num_edges, hidden_size
    wh = jnp.concatenate(
        [keys_source + edge_embeddings, queries_dest], axis=-1)
    # wh.shape: num_edges, 2 * hidden_size

    attention_logits = leaky_relu(attention_dense2(wh))
    # attention_logits.shape: num_edges
    attention_weights = unsorted_segment_softmax(
        attention_logits,
        dest_indices,
        num_nodes
    )
    # attention_weights.shape: num_edges, 1

    values_source = values[source_indices]
    # values_source.shape: num_edges, hidden_size
    weighted_values = (values_source + edge_embeddings) * attention_weights
    # weighted_values.shape: num_edges, hidden_size
    outputs = jax.ops.segment_sum(
        data=weighted_values,
        segment_ids=dest_indices,
        num_segments=num_nodes,
    )
    # outputs.shape: num_nodes, hidden_state
    return outputs


def segment_max(data, segment_ids, num_segments):
  """Computes the max within segments of an array.

  Args:
    data: an array with the values to be aggregated by max.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis). Values can be repeated and
      need not be sorted.
    num_segments: an int with positive value indicating the number of segments.
  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing
    the max of the associated segments.
  """
  num_segments = int(num_segments)

  full_zeros = jnp.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
  full_ones = jnp.ones((num_segments,) + data.shape[1:], dtype=data.dtype)

  valid_updates = jax.ops.index_update(full_zeros, segment_ids,
                                       jnp.ones_like(data, dtype=data.dtype))
  updated = jax.ops.index_max(full_ones * -jnp.inf, segment_ids, data)

  return jnp.where(valid_updates, updated, full_zeros)


def unsorted_segment_softmax(logits, segment_ids, num_segments):
  """Returns softmax over each segment.

  Args:
    logits: Logits of elements, of shape `[dim_0, ...]`.
    segment_ids: Segmentation of logits, which elements will participate in in
      the same softmax. Shape `[dim_0]`.
    num_segments: Scalar number of segments, typically `max(segment_ids)`.

  Returns:
    Probabilities of the softmax, shape `[dim_0, ...]`.
  """
  segment_max_ = segment_max(logits, segment_ids, num_segments)
  broadcast_segment_max = segment_max_[segment_ids]
  shifted_logits = logits - broadcast_segment_max

  # Sum and get the probabilities.
  exp_logits = jnp.exp(shifted_logits)
  exp_sum = jax.ops.segment_sum(exp_logits, segment_ids, num_segments)
  broadcast_exp_sum = exp_sum[segment_ids]
  probs = exp_logits / broadcast_exp_sum
  return probs


class GAT(nn.Module):
  """GAT model."""

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

    gat_layer_single_example = GATLayer.shared(
        num_nodes=num_nodes,
        hidden_size=hidden_size,
        config=config,
        name='gat_layer_single_example')
    gat_layer = jax.vmap(gat_layer_single_example)

    # statement_embeddings.shape: batch_size, num_nodes, hidden_size
    for step in range(max_steps):
      new_statement_embeddings = gat_layer(
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
